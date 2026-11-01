"""
Reconstruction Error Evaluation for Frame Selection Methods.

Run from the project root (the folder that contains app.py):
    python src\\evaluate_reconstruction.py
    python src\\evaluate_reconstruction.py --clips "path\\to\\clips"

What this script does
- Runs MediaPipe Pose on EVERY frame of each clip (ground truth).
- For each method and each frame budget K, selects K frames, linearly
  interpolates the skeleton between them, and compares to the ground truth.
- Methods: uniform, motion_aware, hybrid, phase_aware, plus greedy_oracle.
  greedy_oracle sees the ground truth, so it is an UPPER BOUND, not a
  competitor. Never count it when saying which method "won".
- Metrics: MPJPE, joint-angle error, key-phase recall, impact-frame distance,
  information per frame, runtime.

Important limits (state these in your report)
- MPJPE is in normalised image coordinates (0-1), not metres.
- Ground truth is MediaPipe's own pose with short gaps linearly interpolated.
  This measures sampling fidelity, not true pose accuracy.
- Pose estimation runs on all frames before selection, so selecting frames
  does NOT reduce pose-estimation compute in this experiment.
- Key-phase recall is partly circular for phase_aware, because the same phase
  detector defines the phases and picks the frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Imports from the rest of the project
# ---------------------------------------------------------------------------
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from motion_analysis import (  # noqa: E402
    select_uniform_frames,
    select_motion_frames,
    compute_consecutive_motion_scores,
    decode_all_frames,
)
from phase_detection import detect_phases  # noqa: E402
from phase_aware_selection import select_phase_aware_frames  # noqa: E402
from impact_estimation import estimate_impact_frame  # noqa: E402

try:  # hybrid selector: try the module that currently exports it
    from phase_aware_selection import select_hybrid_frames  # noqa: E402
except ImportError:  # pragma: no cover
    from hybrid_frame_selection import select_hybrid_frames  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MP_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOTLY_AVAILABLE = False

try:
    from scipy.stats import wilcoxon
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K_BUDGETS = [3, 4, 5, 6, 8, 12, 16]  # K=24 is trivial for 24-frame clips
METHODS = ["uniform", "motion_aware", "hybrid", "phase_aware", "greedy_oracle"]
COMPETING_METHODS = ["motion_aware", "hybrid", "phase_aware"]  # excludes oracle
METRICS = ["mpjpe", "angle_error", "phase_recall", "impact_dist",
           "info_per_frame", "runtime_ms"]
MIN_DETECTION_RATE = 0.5  # skip clips where pose is found in fewer frames

# MediaPipe Pose landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _r(x, nd: int = 6):
    """Round a number, converting NaN/None to None (JSON friendly)."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(xf):
        return None
    return round(xf, nd)


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _interp_nans(y: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs in a 1-D array (edges are held constant)."""
    y = np.asarray(y, dtype=float).copy()
    valid = np.where(~np.isnan(y))[0]
    if len(valid) == 0:
        return np.zeros(len(y))
    if len(valid) == len(y):
        return y
    if valid[0] > 0:
        y[:valid[0]] = y[valid[0]]
    if valid[-1] < len(y) - 1:
        y[valid[-1] + 1:] = y[valid[-1]]
    nans = np.isnan(y)
    if nans.any():
        idx = np.arange(len(y))
        y[nans] = np.interp(idx[nans], idx[~nans], y[~nans])
    return y


def _as_int_list(sel) -> list[int]:
    return sorted({int(s) for s in sel})


# ---------------------------------------------------------------------------
# Pose extraction and ground truth
# ---------------------------------------------------------------------------
def extract_clip_poses_eval(frames: list, detector) -> list[dict]:
    """Run MediaPipe on all frames in a clip (IMAGE mode).

    Kept under this name because tools/evaluate_impact.py imports it.
    Returns a list of {"detected": bool, "landmarks": [ {x,y,z,visibility,presence}, ... ]}.
    """
    poses: list[dict] = []
    for frame_bgr in frames:
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lm_list = []
                for pt in result.pose_landmarks[0]:
                    lm_list.append({
                        "x": float(pt.x),
                        "y": float(pt.y),
                        "z": float(pt.z),
                        "visibility": float(getattr(pt, "visibility", 0.0) or 0.0),
                        "presence": float(getattr(pt, "presence", 0.0) or 0.0),
                    })
                poses.append({"detected": True, "landmarks": lm_list})
            else:
                poses.append({"detected": False, "landmarks": []})
        except Exception as exc:  # keep going, but tell the user once
            if not getattr(extract_clip_poses_eval, "_warned", False):
                print(f"    [WARN] pose detection error: {exc!r}")
                extract_clip_poses_eval._warned = True
            poses.append({"detected": False, "landmarks": []})
    return poses


def build_ground_truth_matrix(poses: list[dict]) -> np.ndarray:
    """Ground truth (N, 33, 2). Low-visibility or missing landmarks are
    linearly interpolated per joint and coordinate."""
    n = len(poses)
    gt = np.full((n, 33, 2), np.nan)
    for t, pose in enumerate(poses):
        if not pose.get("detected", False):
            continue
        for j, lm in enumerate(pose.get("landmarks", [])[:33]):
            if lm.get("visibility", 0.0) >= 0.2:
                gt[t, j, 0] = lm["x"]
                gt[t, j, 1] = lm["y"]
    for j in range(33):
        for c in range(2):
            gt[:, j, c] = _interp_nans(gt[:, j, c])
    return gt


def interpolate_skeleton(gt: np.ndarray, selected_indices: list[int]) -> np.ndarray:
    """Linearly interpolate the skeleton between the selected frames."""
    n = len(gt)
    sel = _as_int_list(selected_indices)
    interp = np.empty_like(gt)
    x = np.arange(n, dtype=float)
    xp = np.array(sel, dtype=float)
    for j in range(33):
        for c in range(2):
            interp[:, j, c] = np.interp(x, xp, gt[sel, j, c])
    return interp


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_mpjpe(gt: np.ndarray, interp: np.ndarray) -> float:
    diff = gt - interp
    per = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
    if not np.any(~np.isnan(per)):
        return float("nan")
    return float(np.nanmean(per))


def _angle_series(arr: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Angle (degrees) at joint b formed by a-b-c, for each frame."""
    v1 = arr[:, a, :] - arr[:, b, :]
    v2 = arr[:, c, :] - arr[:, b, :]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_a = np.sum(v1 * v2, axis=1) / (n1 * n2)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_a))
    ang[(n1 < 1e-9) | (n2 < 1e-9)] = np.nan
    return ang


def compute_joint_angle_error(gt: np.ndarray, interp: np.ndarray,
                              side: str = "left") -> float:
    """Mean absolute angle error (degrees) for elbow, shoulder and knee."""
    if str(side).lower().startswith("r"):
        triples = [(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
                   (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
                   (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)]
    else:
        triples = [(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                   (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
                   (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)]
    errs = []
    for tri in triples:
        g = _angle_series(gt, *tri)
        i = _angle_series(interp, *tri)
        mask = ~np.isnan(g) & ~np.isnan(i)
        if mask.any():
            errs.append(float(np.mean(np.abs(g[mask] - i[mask]))))
    return float(np.mean(errs)) if errs else float("nan")


def _phase_bounds(b):
    """Accept {"start":..,"end":..} or (start, end); return ints or None."""
    if isinstance(b, dict):
        s, e = b.get("start"), b.get("end")
    elif isinstance(b, (list, tuple)) and len(b) >= 2:
        s, e = b[0], b[1]
    else:
        return None
    if s is None or e is None:
        return None
    try:
        return int(s), int(e)
    except (TypeError, ValueError):
        return None


def compute_key_phase_recall(phases, selected: list[int]) -> float:
    """Fraction of detected phases that contain at least one selected frame.
    NOTE: partly circular for phase_aware (same detector defines phases)."""
    if not isinstance(phases, dict) or not phases:
        return float("nan")
    sel = set(int(s) for s in selected)
    total = 0
    covered = 0
    for bounds in phases.values():
        pb = _phase_bounds(bounds)
        if pb is None:
            continue
        total += 1
        if any(pb[0] <= s <= pb[1] for s in sel):
            covered += 1
    return covered / total if total else float("nan")


def compute_impact_coverage(impact_frame, selected: list[int]):
    """Distance in frames from the impact frame to the nearest selected frame."""
    if impact_frame is None:
        return None
    try:
        imp = int(impact_frame)
    except (TypeError, ValueError):
        return None
    return int(min(abs(int(s) - imp) for s in selected))


def compute_info_per_frame(gt: np.ndarray, selected: list[int],
                           method_error: float) -> float:
    """IPF = (1 - method_error / baseline_error) / K, where the baseline keeps
    only the first and last frame."""
    n, k = len(gt), len(selected)
    if k == 0 or np.isnan(method_error):
        return float("nan")
    base = compute_mpjpe(gt, interpolate_skeleton(gt, [0, n - 1]))
    if np.isnan(base) or base < 1e-9:
        return float("nan")
    return float((1.0 - method_error / base) / k)


# ---------------------------------------------------------------------------
# Greedy oracle (upper bound; uses ground truth)
# ---------------------------------------------------------------------------
def greedy_oracle_selection(gt: np.ndarray, k: int) -> list[int]:
    n = len(gt)
    if k >= n:
        return list(range(n))
    selected = {0, n - 1}
    while len(selected) < k:
        best_frame, best_err = None, float("inf")
        for cand in range(n):
            if cand in selected:
                continue
            err = compute_mpjpe(gt, interpolate_skeleton(gt, sorted(selected | {cand})))
            if np.isnan(err):
                err = float("inf")
            if best_frame is None or err < best_err:
                best_frame, best_err = cand, err
        selected.add(best_frame)
    return sorted(selected)


# ---------------------------------------------------------------------------
# Scoring one selection
# ---------------------------------------------------------------------------
def _score(gt, sel, side, phases, impact_frame, runtime_ms):
    sel = _as_int_list(sel)
    interp = interpolate_skeleton(gt, sel)
    err = compute_mpjpe(gt, interp)
    return {
        "selected": sel,
        "mpjpe": _r(err),
        "angle_error": _r(compute_joint_angle_error(gt, interp, side), 4),
        "phase_recall": _r(compute_key_phase_recall(phases, sel), 4),
        "impact_dist": compute_impact_coverage(impact_frame, sel),
        "info_per_frame": _r(compute_info_per_frame(gt, sel, err)),
        "runtime_ms": _r(runtime_ms, 4),
    }


# ---------------------------------------------------------------------------
# Impact labels
# ---------------------------------------------------------------------------
def load_manual_impact(base_dir: Path, clip_class: str, clip_name: str):
    path = base_dir / "reports" / "impact_ground_truth.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except Exception:
        return None
    candidates = [f"{clip_class}/{clip_name}", f"{clip_class}\\{clip_name}", clip_name]
    for key in candidates:
        if key in labels:
            val = labels[key]
            if isinstance(val, dict):
                val = val.get("impact_frame")
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_all_clips(video_files: list[Path], detector, base_dir: Path) -> list[dict]:
    results: list[dict] = []
    for vf in video_files:
        clip_name = vf.name
        clip_class = vf.parent.name
        print(f"\n  Processing {clip_class}/{clip_name} ...", flush=True)

        frames = decode_all_frames(vf)
        n = len(frames)
        print(f"    Total frames: {n}")
        if n < 5:
            print(f"    [SKIP] only {n} frames")
            continue

        _, frame_energies = compute_consecutive_motion_scores(frames)

        poses = extract_clip_poses_eval(frames, detector)
        detected = sum(1 for p in poses if p.get("detected"))
        print(f"    Pose detected in {detected}/{n} frames")
        if detected < max(3, int(MIN_DETECTION_RATE * n)):
            print("    [SKIP] too few detected frames for a reliable ground truth")
            continue
        gt = build_ground_truth_matrix(poses)

        phase_info = detect_phases(poses) or {}
        phases = phase_info.get("phases", {})
        side = phase_info.get("dominant_wrist_side", "left")
        phase_success = bool(phase_info.get("reliable", False))

        impact_info = estimate_impact_frame(poses, phase_info=phase_info) or {}
        impact_est = impact_info.get("estimated_impact_frame")
        manual = load_manual_impact(base_dir, clip_class, clip_name)
        impact_frame = manual if manual is not None else impact_est

        clip_result = {
            "clip_name": clip_name,
            "class": clip_class,
            "total_frames": n,
            "detected_frames": detected,
            "phase_success": phase_success,
            "impact_frame_estimated": impact_est,
            "impact_frame_manual": manual,
            "impact_frame_used": impact_frame,
            "impact_uncertain": impact_info.get("uncertain", True),
            "phase_aware_fallback_ks": [],
            "phase_aware_errors": 0,
            "per_k_results": {},
        }

        for k in K_BUDGETS:
            if k >= n:  # K >= clip length would select every frame (trivial)
                continue
            kr: dict[str, dict] = {}

            t0 = time.perf_counter()
            sel_u = select_uniform_frames(n, k=k)
            kr["uniform"] = _score(gt, sel_u, side, phases, impact_frame,
                                   (time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            sel_m = select_motion_frames(frame_energies, k=k)
            kr["motion_aware"] = _score(gt, sel_m, side, phases, impact_frame,
                                        (time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            sel_h = select_hybrid_frames(frame_energies, k=k)
            kr["hybrid"] = _score(gt, sel_h, side, phases, impact_frame,
                                  (time.perf_counter() - t0) * 1000)

            # --- Phase-aware: returns (selected_frames, phase_info) ---
            fallback = True
            error_flag = False
            sel_p = sel_h
            t0 = time.perf_counter()
            try:
                out = select_phase_aware_frames(poses, frame_energies, k)
                if isinstance(out, tuple) and len(out) == 2:
                    sel_p, phase_info_p = out
                else:  # older version that returns only the frame list
                    sel_p, phase_info_p = out, {}
                fallback = bool((phase_info_p or {}).get("fallback_used", False))
            except Exception as exc:
                error_flag = True
                print(f"    [ERROR] phase_aware crashed at K={k} ({exc!r}); "
                      f"using hybrid for this K")
                sel_p = sel_h
            runtime_p = (time.perf_counter() - t0) * 1000
            if fallback and not error_flag:
                print(f"    [FALLBACK] phase_aware fell back to hybrid "
                      f"({clip_name}, K={k})")
            kr["phase_aware"] = _score(gt, sel_p, side, phases, impact_frame,
                                       runtime_p)
            kr["phase_aware"]["fallback"] = fallback or error_flag
            kr["phase_aware"]["error"] = error_flag
            if fallback or error_flag:
                clip_result["phase_aware_fallback_ks"].append(k)
            if error_flag:
                clip_result["phase_aware_errors"] += 1

            t0 = time.perf_counter()
            sel_g = greedy_oracle_selection(gt, k)
            kr["greedy_oracle"] = _score(gt, sel_g, side, phases, impact_frame,
                                         (time.perf_counter() - t0) * 1000)

            clip_result["per_k_results"][k] = kr
            line = " ".join(f"{m[:1].upper()}={kr[m]['mpjpe']}" for m in METHODS)
            print(f"    K={k:>2} | {line}")

        results.append(clip_result)
    return results


# ---------------------------------------------------------------------------
# Aggregation and statistics
# ---------------------------------------------------------------------------
def _phase_aware_active(r: dict) -> bool:
    """True if phase_aware genuinely ran (no fallback or crash at any K)."""
    return not r.get("phase_aware_fallback_ks")


def aggregate_results(results: list[dict], phase_success_only: bool = False) -> dict:
    clips = [r for r in results if (_phase_aware_active(r) or not phase_success_only)]
    agg: dict = {}
    for k in K_BUDGETS:
        collected = {m: {x: [] for x in METRICS} for m in METHODS}
        n_clips = 0
        for clip in clips:
            kr = clip["per_k_results"].get(k)
            if not kr:
                continue
            n_clips += 1
            for m in METHODS:
                for x in METRICS:
                    v = kr[m].get(x)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        collected[m][x].append(float(v))
        if n_clips == 0:
            continue
        agg[k] = {"n_clips": n_clips}
        for m in METHODS:
            agg[k][m] = {
                x: ((float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan")))
                for x, v in collected[m].items()
            }
    return agg


def paired_comparison(results: list[dict], active_only: bool = False) -> dict:
    """Each competing method vs uniform. Wins = better than uniform on that
    metric for that clip. For info_per_frame and phase_recall higher is better.
    active_only=True keeps only clips where phase_aware genuinely ran."""
    higher_better = {"info_per_frame", "phase_recall"}
    if active_only:
        results = [r for r in results if _phase_aware_active(r)]
    comp: dict = {}
    for k in K_BUDGETS:
        comp[k] = {}
        for meth in COMPETING_METHODS:
            comp[k][meth] = {}
            for metric in ["mpjpe", "angle_error", "phase_recall",
                           "impact_dist", "info_per_frame"]:
                diffs, wins, losses, ties = [], 0, 0, 0
                for clip in results:
                    kr = clip["per_k_results"].get(k)
                    if not kr:
                        continue
                    mv, uv = kr[meth].get(metric), kr["uniform"].get(metric)
                    if mv is None or uv is None:
                        continue
                    d = float(mv) - float(uv)
                    diffs.append(d)
                    if abs(d) < 1e-12:
                        ties += 1
                    elif (d > 0) == (metric in higher_better):
                        wins += 1
                    else:
                        losses += 1
                p = None
                if len(diffs) >= 5 and _SCIPY_AVAILABLE and any(abs(d) > 1e-12 for d in diffs):
                    try:
                        p = float(wilcoxon(diffs).pvalue)
                    except Exception:
                        p = None
                comp[k][meth][metric] = {
                    "n": len(diffs),
                    "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
                    "wilcoxon_p": p,
                    "wins": wins, "losses": losses, "ties": ties,
                }
    return comp


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_table(metric: str, agg: dict, title: str, fmt: str = "{:.4f}"):
    print(f"\n{title}")
    print(f"{'K':<4}{'n':<4}" + "".join(f"{m:>20}" for m in METHODS))
    for k in K_BUDGETS:
        if k not in agg:
            continue
        row = f"{k:<4}{agg[k]['n_clips']:<4}"
        for m in METHODS:
            mean, std = agg[k][m][metric]
            if np.isnan(mean):
                row += f"{'n/a':>20}"
            else:
                row += f"{(fmt.format(mean) + '+-' + fmt.format(std)):>20}"
        print(row)


def print_paired(comp: dict, title: str = "PAIRED COMPARISON VS UNIFORM"):
    print("\n" + "=" * 80)
    print(f"{title} (wins = better than uniform on that clip)")
    print("p-values only shown with >= 5 clips; with few clips treat them as indicative.")
    print("=" * 80)
    for k in K_BUDGETS:
        if k not in comp:
            continue
        print(f"\n--- K={k} ---")
        for meth, metrics in comp[k].items():
            print(f"  {meth}")
            for metric, s in metrics.items():
                if s["n"] == 0:
                    continue
                p = "NA" if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.3g}"
                print(f"    {metric:<15} n={s['n']:<3} mean diff={s['mean_diff']:+.4f}  "
                      f"W/L/T={s['wins']}/{s['losses']}/{s['ties']}  p={p}")


# ---------------------------------------------------------------------------
# Finding clips and the model
# ---------------------------------------------------------------------------
def find_videos(project_root: Path, clips_dir: str | None) -> list[Path]:
    if clips_dir:
        roots = [Path(clips_dir)]
    else:
        roots = [project_root / "outputs" / "sample_clips", project_root / "outputs"]
    found: dict[Path, Path] = {}
    for root in roots:
        if root.is_dir():
            for p in root.rglob("*.avi"):
                found.setdefault(p.resolve(), p)
        if found:
            break  # use the first root that has clips
    return sorted(found.values(), key=lambda p: (p.parent.name, p.name))


def find_model(project_root: Path) -> Path | None:
    preferred = project_root / "models" / "pose_landmarker_lite.task"
    if preferred.is_file():
        return preferred
    for p in project_root.rglob("*.task"):
        return p
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", default=None, help="folder containing .avi clips")
    args = parser.parse_args()

    if not _MP_AVAILABLE or cv2 is None:
        print("[ERROR] mediapipe and opencv-python are required.")
        return

    base_dir = _src.parent
    videos = find_videos(base_dir, args.clips)
    if not videos:
        print("[ERROR] No .avi clips found. Use --clips \"path\\to\\clips\".")
        return

    model = find_model(base_dir)
    if model is None:
        print("[ERROR] No pose_landmarker .task model found under the project folder.")
        return

    print("=" * 80)
    print("RECONSTRUCTION ERROR EVALUATION")
    print(f"Clips found: {len(videos)} | K budgets: {K_BUDGETS}")
    print(f"Model: {model}")
    print("=" * 80)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,  # detect() needs IMAGE mode
    )
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    try:
        results = evaluate_all_clips(videos, detector, base_dir)
    finally:
        detector.close()

    total = len(results)
    if total == 0:
        print("\n[ERROR] No clips were evaluated (see SKIP messages above).")
        return

    print("\n" + "=" * 80)
    print(f"Clips evaluated: {total} (unique)")
    if total < 15:
        print("NOTE: small sample; differences may not be statistically meaningful.")
    for k in K_BUDGETS:
        print(f"  K={k:<2}: eligible clips = {sum(1 for r in results if r['total_frames'] > k)}")
    fb = sum(1 for r in results if r["phase_aware_fallback_ks"])
    crashes = sum(r["phase_aware_errors"] for r in results)
    active = total - fb
    print(f"phase_aware fell back to hybrid (any K): {fb}/{total} clips")
    print(f"phase_aware genuinely ran on: {active}/{total} clips")
    print(f"phase_aware crashes (counted per clip and K): {crashes}")
    if active == 0:
        print("WARNING: phase_aware never ran, so its columns equal hybrid. "
              "Do not interpret them.")
    labelled = sum(1 for r in results if r["impact_frame_manual"] is not None)
    print(f"Clips with manual impact label: {labelled}/{total}")
    print("greedy_oracle uses ground truth: upper bound only, not a competitor.")
    print("=" * 80)

    agg_all = aggregate_results(results, phase_success_only=False)
    agg_ok = aggregate_results(results, phase_success_only=True)

    for metric, fmt, name in [
        ("mpjpe", "{:.4f}", "MPJPE (normalised image coords, lower is better)"),
        ("angle_error", "{:.2f}", "Joint-angle error in degrees (lower is better)"),
        ("phase_recall", "{:.2f}", "Key-phase recall (higher is better; partly circular for phase_aware)"),
        ("impact_dist", "{:.2f}", "Frames from impact to nearest selected frame (lower is better)"),
        ("info_per_frame", "{:.4f}", "Information per frame (higher is better)"),
        ("runtime_ms", "{:.3f}", "Selection runtime in ms"),
    ]:
        print_table(metric, agg_all, f"=== {name} - all clips ===", fmt)
    print_table("mpjpe", agg_ok,
                "=== MPJPE - only clips where phase_aware genuinely ran ===")

    comp = paired_comparison(results)
    print_paired(comp, "PAIRED COMPARISON VS UNIFORM - all clips")
    if 0 < active < total:
        comp_active = paired_comparison(results, active_only=True)
        print_paired(comp_active,
                     "PAIRED COMPARISON VS UNIFORM - clips where phase_aware ran")

    reports = base_dir / "reports"
    reports.mkdir(exist_ok=True)

    out_json = reports / "reconstruction_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\n[OK] JSON: {out_json}")

    rows = []
    for clip in results:
        for k, kr in clip["per_k_results"].items():
            for meth, vals in kr.items():
                rows.append({
                    "clip": clip["clip_name"], "class": clip["class"],
                    "total_frames": clip["total_frames"],
                    "phase_success": clip["phase_success"],
                    "K": k, "method": meth,
                    "fallback": vals.get("fallback"),
                    **{x: vals.get(x) for x in METRICS},
                })
    out_csv = reports / "reconstruction_evaluation.csv"
    if _PANDAS_AVAILABLE:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join("" if v is None else str(v) for v in r.values()) + "\n")
    print(f"[OK] CSV: {out_csv}")

    if _PLOTLY_AVAILABLE:
        fig = go.Figure()
        for meth in METHODS:
            ks, means, stds = [], [], []
            for k in K_BUDGETS:
                if k in agg_all:
                    mean, std = agg_all[k][meth]["mpjpe"]
                    if not np.isnan(mean):
                        ks.append(k)
                        means.append(mean)
                        stds.append(std)
            fig.add_trace(go.Scatter(
                x=ks, y=means, mode="lines+markers", name=meth,
                line=dict(dash="dash") if meth == "greedy_oracle" else None,
                error_y=dict(type="data", array=stds, visible=True)))
        fig.update_layout(title="MPJPE vs K (all clips; greedy_oracle = upper bound)",
                          xaxis_title="K (frame budget)",
                          yaxis_title="MPJPE (normalised image coords)")
        html_path = reports / "reconstruction_error_vs_k.html"
        fig.write_html(str(html_path))
        print(f"[OK] Chart: {html_path}")

    print("\nNOTE: MPJPE is in normalised image coordinates (0-1), not metres.")
    print("Ground truth = MediaPipe pose on every frame (short gaps interpolated).")
    print("This measures sampling fidelity, not true pose accuracy.")


if __name__ == "__main__":
    main()