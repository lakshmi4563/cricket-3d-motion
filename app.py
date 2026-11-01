"""
Cricket3D_Motion — Streamlit Dashboard
Main entry point. Multi-page layout via st.navigation / sidebar.

Pages:
1. Method Comparison      — existing + new phase_aware method
2. Clip Analysis          — per-clip phase bands, speed, angles, impact marker
3. Reconstruction Metrics — error vs K chart + per-clip table

Run:
    cd Cricket3D_Motion
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Cricket3D Motion",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Method Comparison",
        "🎬 Clip Analysis",
        "📈 Reconstruction Metrics",
    ],
    index=0,
)

# ──────────────────────────────────────────────────────────
# Imports (deferred to avoid top-level errors)
# ──────────────────────────────────────────────────────────
from pathlib import Path
import json
import numpy as np
import sys

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ──────────────────────────────────────────────────────────
# Helpers & Caching
# ──────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _get_video_files() -> list[Path]:
    return sorted((BASE_DIR / "outputs" / "sample_clips").rglob("*.avi"))


@st.cache_resource(show_spinner="Loading MediaPipe…")
def _get_detector():
    """Return a shared PoseLandmarker using canonical settings from
    pose_detector_factory — identical to evaluate_reconstruction.py."""
    from pose_detector_factory import create_pose_detector, print_detector_settings
    model_path = BASE_DIR / "models" / "pose_landmarker_lite.task"
    if not model_path.exists():
        return None
    print_detector_settings()
    return create_pose_detector(model_path)


@st.cache_data(show_spinner="Extracting poses…", max_entries=20)
def _extract_poses_cached(video_path: str) -> list[dict]:
    """Extract per-frame poses using shared factory (same settings as
    evaluate_reconstruction.py). Keyed by video_path string."""
    from motion_analysis import decode_all_frames
    from pose_detector_factory import extract_poses_from_frames
    detector = _get_detector()
    if detector is None:
        return []
    frames = decode_all_frames(Path(video_path))
    return extract_poses_from_frames(frames, detector)


@st.cache_data(show_spinner="Computing motion scores…", max_entries=20)
def _compute_energies(video_path: str) -> list[float]:
    from motion_analysis import decode_all_frames, compute_consecutive_motion_scores
    frames = decode_all_frames(Path(video_path))
    _, energies = compute_consecutive_motion_scores(frames)
    return energies


# ──────────────────────────────────────────────────────────
# PAGE: Home
# ──────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏏 Cricket3D_Motion — Analysis Dashboard")
    st.markdown("""
    This dashboard extends the existing Cricket3D_Motion pipeline with:
    
    | Feature | Status |
    |---|---|
    | Phase-aware frame selection (4th method) | ✅ |
    | Reconstruction error evaluation (MPJPE vs K) | ✅ |
    | Impact frame estimation (body motion only) | ✅ |
    | Manual impact labelling tool | ✅ (`tools/label_impact.py`) |
    | Unit tests | ✅ (`tests/test_pipeline.py`) |
    
    ### Quick Start
    ```bash
    # Run unit tests first
    python tests/test_pipeline.py

    # Reconstruction evaluation (needs MediaPipe model)
    python src/evaluate_reconstruction.py

    # Impact estimation evaluation
    python tools/evaluate_impact.py

    # Label impact frames manually
    streamlit run tools/label_impact.py

    # This dashboard
    streamlit run app.py
    ```
    
    ### Clips Available
    """)
    vfs = _get_video_files()
    for vf in vfs:
        st.caption(f"• {vf.parent.name} / {vf.name}")

    st.info(f"**{len(vfs)} clips** found. All clips are 24 frames @ ~30 fps.")


# ──────────────────────────────────────────────────────────
# PAGE: Method Comparison
# ──────────────────────────────────────────────────────────
elif page == "📊 Method Comparison":
    st.title("📊 Frame Selection Method Comparison")
    st.markdown(
        "Compares uniform, motion-aware, hybrid, and **phase-aware** (new) "
        "selection methods on a chosen clip. All use the same frame budget K."
    )

    vfs = _get_video_files()
    if not vfs:
        st.error("No clips found.")
        st.stop()

    clip_labels = [f"{vf.parent.name}/{vf.name}" for vf in vfs]
    chosen      = st.selectbox("Clip", clip_labels)
    chosen_vf   = vfs[clip_labels.index(chosen)]
    K           = st.slider("Frame budget K", 4, 16, value=6, step=2)

    with st.spinner("Running all four methods…"):
        try:
            import cv2
            from motion_analysis import (
                decode_all_frames,
                compute_consecutive_motion_scores,
                select_uniform_frames,
                select_motion_frames,
            )
            from phase_aware_selection import select_hybrid_frames, select_phase_aware_frames

            frames = decode_all_frames(chosen_vf)
            N = len(frames)
            _, energies = compute_consecutive_motion_scores(frames)

            poses = _extract_poses_cached(str(chosen_vf))

            sel_u = select_uniform_frames(N, k=K)
            sel_m = select_motion_frames(energies, k=K)
            sel_h = select_hybrid_frames(energies, k=K)
            sel_p, phase_info = select_phase_aware_frames(poses, energies, k=K)

        except Exception as e:
            st.error(f"Error running selection: {e}")
            st.stop()

    methods = {
        "Uniform":      sel_u,
        "Motion-Aware": sel_m,
        "Hybrid":       sel_h,
        "Phase-Aware":  sel_p,
    }

    tab_names = list(methods.keys())
    tabs = st.tabs(tab_names)

    for tab, (method_name, sel) in zip(tabs, methods.items()):
        with tab:
            st.subheader(f"{method_name} — K={K}")
            st.caption(f"Selected frames: {sel}")

            # Show selected frames as thumbnails
            cols = st.columns(min(K, 8))
            for ci, fidx in enumerate(sel[:8]):
                with cols[ci]:
                    frame_rgb = cv2.cvtColor(frames[fidx], cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"F{fidx}", width="stretch")

            if method_name == "Phase-Aware":
                st.markdown("---")
                rel = phase_info.get("reliable", False)
                fallback = phase_info.get("fallback_used", False)
                st.metric("Phase detection reliable", "✅ Yes" if rel else "⚠️ No")
                st.metric("Fallback used", "Yes (hybrid)" if fallback else "No")
                if phase_info.get("reliability_note"):
                    st.caption(f"Note: {phase_info['reliability_note']}")
                if rel and phase_info.get("phases"):
                    st.json(phase_info["phases"])


# ──────────────────────────────────────────────────────────
# PAGE: Clip Analysis
# ──────────────────────────────────────────────────────────
elif page == "🎬 Clip Analysis":
    st.title("🎬 Per-Clip Phase & Impact Analysis")
    st.markdown(
        "Shows detected phases as coloured bands on the wrist-speed and "
        "elbow-angle plots, marks the estimated impact frame, and shows "
        "the reliability flag."
    )

    vfs = _get_video_files()
    if not vfs:
        st.error("No clips found.")
        st.stop()

    clip_labels = [f"{vf.parent.name}/{vf.name}" for vf in vfs]
    chosen      = st.selectbox("Clip", clip_labels)
    chosen_vf   = vfs[clip_labels.index(chosen)]

    # ── Option to ignore edge frames for peak detection ───────────────────────
    ignore_edges = st.sidebar.checkbox(
        "Ignore first/last 2 frames for peak detection",
        value=False,
        help="Excludes edge frames from wrist-speed peak search to avoid initialisation artifacts.",
    )
    VIS_THRESHOLD_ANGLE = 0.5  # elbow angle masked when any landmark below this

    with st.spinner("Extracting poses and detecting phases…"):
        try:
            from phase_detection import detect_phases
            from impact_estimation import estimate_impact_frame

            poses = _extract_poses_cached(str(chosen_vf))
            phase_info  = detect_phases(poses)
            impact_info = estimate_impact_frame(poses, phase_info=phase_info)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    N = len(poses)
    wrist_idx_app = phase_info.get("dominant_wrist_idx", 15)

    # Build per-frame interpolated mask (wrist visibility < 0.2 => was NaN before gap-fill)
    _interp_mask = []
    for pose in poses:
        if not pose.get("detected", False):
            _interp_mask.append(True)
        else:
            lms = pose.get("landmarks", [])
            v = float(lms[wrist_idx_app].get("visibility", 0.0)) if wrist_idx_app < len(lms) else 0.0
            _interp_mask.append(v < 0.2)

    # Metadata strip
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", N)
    col2.metric("Dominant Wrist", phase_info.get("dominant_wrist_side", "?").capitalize())
    col3.metric("Phase Detection", "✅ Reliable" if phase_info.get("reliable") else "⚠️ Uncertain")
    col4.metric("Interp. Frames", phase_info.get("n_interpolated_frames", "?"))

    if not phase_info.get("reliable"):
        st.warning(
            f"Phase detection is **uncertain** for this clip. "
            f"Reason: {phase_info.get('reliability_note', 'unknown')}"
        )

    if not _PLOTLY:
        st.warning("Plotly not available. Cannot render interactive charts.")
        st.stop()

    phases = phase_info.get("phases", {})
    speed  = phase_info.get("wrist_speed", [])
    elbows = phase_info.get("elbow_angle", [])
    peak_f = phase_info.get("peak_speed_frame")

    # Impact frame: hide impact line and show "No reliable estimate" when uncertain
    impact_frame     = impact_info.get("estimated_impact_frame")
    impact_uncertain = impact_info.get("uncertain", True)

    PHASE_COLOURS = {
        "stance":         "rgba(100,180,100,0.18)",
        "backlift":       "rgba(255,200,50,0.18)",
        "downswing":      "rgba(255,120,50,0.25)",
        "impact_region":  "rgba(220,50,50,0.30)",
        "follow_through": "rgba(100,150,255,0.18)",
    }

    def _add_phase_bands(fig, phases, y_range):
        for name, bounds in phases.items():
            colour = PHASE_COLOURS.get(name, "rgba(128,128,128,0.15)")
            fig.add_vrect(
                x0=bounds["start"], x1=bounds["end"],
                fillcolor=colour, layer="below", line_width=0,
                annotation_text=name.replace("_", " ").title(),
                annotation_position="top left", annotation_font_size=10,
            )

    def _add_impact_line(fig, fidx):
        if fidx is not None:
            fig.add_vline(
                x=fidx, line_dash="dash", line_color="#FF3333",
                annotation_text=f"Impact ~{fidx}", annotation_position="top right",
            )

    # ── Plot 1: Wrist Speed ──────────────────────────────────────────────────
    speed_list = list(speed)
    edge = 2 if ignore_edges else 0

    speed_symbols = [
        "circle-open" if (_interp_mask[t] or (ignore_edges and (t < edge or t >= N - edge)))
        else "circle"
        for t in range(N)
    ]

    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(
        x=list(range(N)), y=speed_list,
        mode="lines+markers", name="Wrist Speed",
        line={"color": "#00C4CC", "width": 2},
        marker={"size": 6, "symbol": speed_symbols,
                "color": "#00C4CC", "line": {"width": 2, "color": "#00C4CC"}},
    ))

    # Recompute peak after edge exclusion
    peak_display = peak_f
    if ignore_edges and speed_list and N > 2 * edge:
        valid = [s if not (_interp_mask[t] or t < edge or t >= N - edge) else -1
                 for t, s in enumerate(speed_list)]
        if any(v >= 0 for v in valid):
            peak_display = int(np.argmax(valid))

    if peak_display is not None and peak_display < len(speed_list):
        fig_speed.add_trace(go.Scatter(
            x=[peak_display], y=[speed_list[peak_display]],
            mode="markers",
            name="Speed Peak" + (" (edge-excl.)" if ignore_edges else ""),
            marker={"color": "yellow", "size": 12, "symbol": "star"},
        ))

    _add_phase_bands(fig_speed, phases, [0, max(speed_list) if speed_list else 1])
    if not impact_uncertain:
        _add_impact_line(fig_speed, impact_frame)
    fig_speed.update_layout(
        title="Dominant Wrist Speed — hollow markers = interpolated or edge-excluded frames",
        xaxis_title="Frame", yaxis_title="Speed",
        template="plotly_dark", height=340,
    )
    st.plotly_chart(fig_speed, width="stretch")

    # ── Plot 2: Elbow Angle (mask vis < 0.5) ─────────────────────────────────
    side_app    = phase_info.get("dominant_wrist_side", "left")
    shoulder_lm = 11 if side_app == "left" else 12
    elbow_lm    = 13 if side_app == "left" else 14
    wrist_lm    = wrist_idx_app

    elbows_masked = []
    elbow_interp  = []
    for t, pose in enumerate(poses):
        raw = elbows[t] if t < len(elbows) else None
        if raw is None or not pose.get("detected", False):
            elbows_masked.append(None); elbow_interp.append(True); continue
        lms   = pose.get("landmarks", [])
        vis_s = float(lms[shoulder_lm].get("visibility", 0.0)) if shoulder_lm < len(lms) else 0.0
        vis_e = float(lms[elbow_lm  ].get("visibility", 0.0)) if elbow_lm    < len(lms) else 0.0
        vis_w = float(lms[wrist_lm  ].get("visibility", 0.0)) if wrist_lm    < len(lms) else 0.0
        if min(vis_s, vis_e, vis_w) < VIS_THRESHOLD_ANGLE:
            elbows_masked.append(None); elbow_interp.append(True)
        else:
            elbows_masked.append(raw); elbow_interp.append(_interp_mask[t])

    elbow_vals_masked = [v for v in elbows_masked if v is not None]
    if elbow_vals_masked:
        fig_elbow = go.Figure()
        x_m   = [i for i, v in enumerate(elbows_masked) if v is not None]
        e_sym = ["circle-open" if elbow_interp[i] else "circle" for i in x_m]
        fig_elbow.add_trace(go.Scatter(
            x=x_m, y=elbow_vals_masked,
            mode="lines+markers", name="Elbow Angle (vis≥0.5)",
            line={"color": "#FF9F43", "width": 2},
            marker={"size": 6, "symbol": e_sym,
                    "color": "#FF9F43", "line": {"width": 2, "color": "#FF9F43"}},
        ))
        _add_phase_bands(fig_elbow, phases, [min(elbow_vals_masked), max(elbow_vals_masked)])
        if not impact_uncertain:
            _add_impact_line(fig_elbow, impact_frame)
        fig_elbow.update_layout(
            title="Dominant Elbow Angle (°) — hollow = interpolated frame, blank = vis<0.5",
            xaxis_title="Frame", yaxis_title="Angle (°)",
            template="plotly_dark", height=340,
        )
        st.plotly_chart(fig_elbow, width="stretch")
        n_masked = sum(1 for v in elbows_masked if v is None)
        if n_masked:
            st.caption(
                f"⚠️ {n_masked} frame(s) excluded: shoulder/elbow/wrist visibility < {VIS_THRESHOLD_ANGLE}."
            )
    else:
        st.info("Elbow angle unavailable (all frames below visibility threshold 0.5).")

    # ── Impact info panel ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Impact Frame Estimation")
    icol1, icol2, icol3 = st.columns(3)
    if impact_uncertain:
        icol1.metric("Estimated Frame", "No reliable estimate")
    else:
        icol1.metric("Estimated Frame", impact_frame if impact_frame is not None else "—")
    icol2.metric("Confidence", f"{impact_info.get('confidence', 0):.2f}")
    icol3.metric("Uncertain", "Yes" if impact_uncertain else "No")
    st.caption(f"Signal agreement: {impact_info.get('signal_agreement', '?')} | "
               f"{impact_info.get('method_note', '')}")
    if impact_uncertain:
        st.warning(
            "Impact estimate is **uncertain** — signals disagree or are too weak. "
            "No impact line is drawn on the plots above."
        )


# ──────────────────────────────────────────────────────────
# PAGE: Reconstruction Metrics
# ──────────────────────────────────────────────────────────
elif page == "📈 Reconstruction Metrics":
    st.title("📈 Reconstruction Error vs Frame Budget K")
    st.markdown(
        "Loaded from pre-computed `reports/reconstruction_evaluation.json`. "
        "Run `python src/evaluate_reconstruction.py` to regenerate."
    )

    report_path = BASE_DIR / "reports" / "reconstruction_evaluation.json"
    data = _load_json(report_path)

    if data is None:
        st.warning(
            "No reconstruction evaluation report found. "
            "Run:\n```\npython src/evaluate_reconstruction.py\n```"
        )
        st.stop()

    if isinstance(data, list):
        per_clip = data
        all_ks = set()
        for clip in per_clip:
            all_ks.update(clip.get("per_k_results", {}).keys())
        k_int_list = sorted([int(k) for k in all_ks if k.isdigit()])
        K_BUDGETS = k_int_list if k_int_list else [4, 6, 8, 12, 16]

        agg = {}
        methods_list = ["uniform", "motion_aware", "hybrid", "phase_aware", "greedy_oracle"]
        for m in methods_list:
            agg[m] = {}
            for k in K_BUDGETS:
                k_str = str(k)
                mpjpes = []
                recalls = []
                angles = []
                for clip in per_clip:
                    mdata = clip.get("per_k_results", {}).get(k_str, {}).get(m)
                    if mdata:
                        if mdata.get("mpjpe") is not None:
                            mpjpes.append(mdata["mpjpe"])
                        if mdata.get("phase_recall") is not None:
                            recalls.append(mdata["phase_recall"])
                        if mdata.get("angle_error") is not None:
                            angles.append(mdata["angle_error"])
                agg[m][k_str] = {
                    "mpjpe": {
                        "mean": float(np.mean(mpjpes)) if mpjpes else None,
                        "std": float(np.std(mpjpes)) if mpjpes else 0.0,
                    },
                    "phase_recall": {
                        "mean": float(np.mean(recalls)) if recalls else None,
                        "std": float(np.std(recalls)) if recalls else 0.0,
                    },
                    "angle_error": {
                        "mean": float(np.mean(angles)) if angles else None,
                        "std": float(np.std(angles)) if angles else 0.0,
                    },
                }
    else:
        agg      = data.get("aggregate", {})
        per_clip = data.get("per_clip", [])
        K_BUDGETS = [6, 8, 12, 16, 24]

    st.info(
        f"**{len(per_clip)} clips** evaluated. "
        "Note: sample size is small — standard deviations are large. "
        "Uniform sampling is a strong baseline; differences between methods "
        "may not be statistically significant."
    )

    methods_display = {
        "uniform":      "Uniform",
        "motion_aware": "Motion-Aware",
        "hybrid":       "Hybrid",
        "phase_aware":  "Phase-Aware",
    }
    colours = {
        "uniform":      "#1f77b4",
        "motion_aware": "#ff7f0e",
        "hybrid":       "#2ca02c",
        "phase_aware":  "#d62728",
    }

    if not _PLOTLY:
        st.warning("Plotly not available.")
        st.stop()

    # ── MPJPE chart (D10) ──
    st.subheader("Mean Per-Joint Position Error (MPJPE) vs K")
    fig = go.Figure()
    for m, label in methods_display.items():
        ks, means, stds = [], [], []
        for K in K_BUDGETS:
            stat = agg.get(m, {}).get(str(K), {}).get("mpjpe", {})
            mv = stat.get("mean")
            sv = stat.get("std") or 0
            if mv is None:
                continue
            ks.append(K); means.append(mv); stds.append(sv)
        if not ks:
            continue
        uppers = [m_ + s for m_, s in zip(means, stds)]
        lowers = [m_ - s for m_, s in zip(means, stds)]
        fig.add_trace(go.Scatter(
            x=ks + ks[::-1], y=uppers + lowers[::-1],
            fill="toself", fillcolor=colours[m], opacity=0.15,
            line={"color": "rgba(0,0,0,0)"}, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=ks, y=means, mode="lines+markers", name=label,
            line={"color": colours[m], "width": 2}, marker={"size": 8},
            error_y={"type": "data", "array": stds, "visible": True},
        ))
    # D10: add greedy_oracle as dashed grey upper-bound line
    oracle_ks, oracle_means = [], []
    for K in K_BUDGETS:
        stat = agg.get("greedy_oracle", {}).get(str(K), {}).get("mpjpe", {})
        mv = stat.get("mean")
        if mv is not None:
            oracle_ks.append(K); oracle_means.append(mv)
    if oracle_ks:
        fig.add_trace(go.Scatter(
            x=oracle_ks, y=oracle_means,
            mode="lines+markers", name="Upper bound (uses ground truth)",
            line={"color": "#888888", "width": 2, "dash": "dash"},
            marker={"size": 6, "symbol": "diamond"},
        ))
    fig.update_layout(
        xaxis={"title": "K (frames)", "tickmode": "array", "tickvals": K_BUDGETS},
        yaxis={"title": "MPJPE (normalised 2D)"},
        template="plotly_dark", height=420,
        legend={"orientation": "h", "y": -0.35, "x": 0.0, "xanchor": "left"},
    )
    st.plotly_chart(fig, width="stretch")

    # ── Phase recall chart ──
    st.subheader("Key-Phase Recall vs K")
    fig2 = go.Figure()
    for m, label in methods_display.items():
        ks, means = [], []
        for K in K_BUDGETS:
            stat = agg.get(m, {}).get(str(K), {}).get("phase_recall", {})
            mv = stat.get("mean")
            if mv is None:
                continue
            ks.append(K); means.append(mv)
        if not ks:
            continue
        fig2.add_trace(go.Scatter(
            x=ks, y=means, mode="lines+markers", name=label,
            line={"color": colours[m], "width": 2}, marker={"size": 8},
        ))
    fig2.update_layout(
        xaxis={"title": "K (frames)", "tickmode": "array", "tickvals": K_BUDGETS},
        yaxis={"title": "Phase Recall (fraction of phases covered)", "range": [0, 1.05]},
        template="plotly_dark", height=360,
    )
    st.plotly_chart(fig2, width="stretch")
    st.caption(
        "⚠️ **Circularity note (Phase-Aware only):** Phase recall is partly circular "
        "for the Phase-Aware method. The same rule-based detector defines phase boundaries "
        "AND selects the frames, so higher recall is expected by construction and does not "
        "imply independent validation."
    )

    # ── Per-clip table ──
    st.subheader("Per-Clip Results")
    default_k = 6 if 6 in K_BUDGETS else K_BUDGETS[0]
    K_sel = st.select_slider("K budget for table", options=K_BUDGETS, value=default_k)

    rows = []
    for clip in per_clip:
        k_data = clip["per_k_results"].get(str(K_sel))
        if k_data is None:
            continue
        phase_reliable = bool(clip.get("phase_reliable") or clip.get("phase_success"))
        pa_data   = k_data.get("phase_aware", {}) or {}
        pa_fallbk = pa_data.get("fallback", False)
        row = {
            "Clip":                 f"{clip['class']}/{clip['clip_name']}",
            "Phase reliable":       "✅" if phase_reliable else "⚠️",
            "Phase-Aware Fallback": "Yes" if pa_fallbk else "No",
        }
        for m, label in methods_display.items():
            mdata = k_data.get(m) or {}
            mpjpe = mdata.get("mpjpe")
            # Convert MPJPE to formatted string
            if mpjpe is None:
                row[f"{label} MPJPE"] = "n/a (no phases detected)" if (m == "phase_aware" and not phase_reliable) else "n/a"
            else:
                row[f"{label} MPJPE"] = f"{mpjpe:.4f}"

            # Convert recall columns to strings before st.dataframe so Arrow does not fail
            pr = mdata.get("phase_recall")
            if pr is None:
                row[f"{label} Recall"] = "n/a (no phases detected)" if (m == "phase_aware" and not phase_reliable) else "n/a"
            else:
                row[f"{label} Recall"] = f"{pr:.2f}"
        rows.append(row)

    if rows:
        if _PANDAS:
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch")
        else:
            st.json(rows)
    else:
        st.info(f"No results for K={K_sel}. Run `python src/evaluate_reconstruction.py`.")

    # ── Honest comparison note ──
    st.markdown("---")
    st.markdown(
        "> **Honest Comparison Note:** Uniform sampling is the strongest "
        "> baseline on MPJPE in several clips. Phase-aware selection improves "
        "> phase recall but does not always reduce reconstruction error, "
        "> especially for short (24-frame) clips where uniform spacing already "
        "> covers all phases. These findings are reported without adjustment."
    )
