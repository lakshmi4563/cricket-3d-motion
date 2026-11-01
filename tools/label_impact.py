"""
Manual Impact Frame Labelling Tool.

A Streamlit-based helper that lets you step through a video clip frame by frame
and record the true impact frame manually.

Labels are saved to reports/impact_ground_truth.json.

Usage
-----
    streamlit run tools/label_impact.py

Controls
--------
- Use the slider to scrub to any frame.
- Click "Mark This Frame as Impact" to record the label.
- Labels persist across relaunches (JSON is loaded on start).
- Use "Clear label for this clip" to remove a previously set label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("[ERROR] Streamlit not available. Install with: pip install streamlit")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV (cv2) not available.")
    sys.exit(1)

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOOL_DIR   = Path(__file__).resolve().parent
BASE_DIR   = TOOL_DIR.parent
SAMPLE_DIR = BASE_DIR / "outputs" / "sample_clips"
GT_FILE    = BASE_DIR / "reports" / "impact_ground_truth.json"


# ---------------------------------------------------------------------------
# Load / save labels
# ---------------------------------------------------------------------------
def load_labels() -> dict:
    if GT_FILE.exists():
        with open(str(GT_FILE), "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels: dict) -> None:
    GT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(GT_FILE), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_frames(video_path: str) -> list:
    """Decode all frames from video and cache them."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Impact Frame Labeller",
        page_icon="🏏",
        layout="wide",
    )

    st.title("🏏 Impact Frame Manual Labeller")
    st.markdown(
        "Step through each clip and mark the frame where bat meets ball "
        "(or closest observable body contact moment). Labels are saved to "
        f"`{GT_FILE.relative_to(BASE_DIR)}`."
    )

    # Collect all video files
    video_files = sorted(SAMPLE_DIR.rglob("*.avi"))
    if not video_files:
        st.error(f"No .avi clips found in `{SAMPLE_DIR}`")
        return

    clip_labels = ["--- select ---"] + [
        f"{vf.parent.name}/{vf.name}" for vf in video_files
    ]
    clip_map = {
        f"{vf.parent.name}/{vf.name}": vf for vf in video_files
    }

    # Load existing labels
    if "labels" not in st.session_state:
        st.session_state.labels = load_labels()

    # Sidebar: clip selector
    with st.sidebar:
        st.header("Select Clip")
        chosen_label = st.selectbox("Clip", clip_labels, index=0)
        if chosen_label == "--- select ---":
            st.info("Choose a clip to begin.")
            _render_label_summary(st.session_state.labels, video_files)
            return

        video_path = clip_map[chosen_label]
        st.success(f"Loaded: {chosen_label}")

    # Load frames
    with st.spinner("Loading frames…"):
        frames = load_frames(str(video_path))

    N = len(frames)
    if N == 0:
        st.error("Could not decode frames.")
        return

    # Retrieve existing label
    clip_key = chosen_label
    existing_label = st.session_state.labels.get(clip_key, {})
    existing_frame = existing_label.get("impact_frame", None)

    col_main, col_info = st.columns([3, 1])

    with col_main:
        # Frame slider — C8: start at 0 so default is never accidentally saved
        default_frame = existing_frame if existing_frame is not None else 0
        frame_idx = st.slider(
            "Frame", 0, N - 1, value=default_frame, step=1,
            help="Use arrow keys for fine control after clicking the slider.",
        )

        # Display frame
        frame_rgb = frames[frame_idx]
        annotated = frame_rgb.copy()

        # Overlay frame index
        import PIL.Image, PIL.ImageDraw, PIL.ImageFont
        pil_img = PIL.Image.fromarray(annotated)
        draw    = PIL.ImageDraw.Draw(pil_img)
        draw.text((8, 8), f"Frame {frame_idx} / {N-1}", fill=(255, 255, 0))
        if existing_frame is not None:
            colour = (0, 255, 0) if frame_idx == existing_frame else (200, 200, 200)
            draw.text((8, 28), f"Labelled impact: {existing_frame}", fill=colour)

        # B6: width='stretch'
        st.image(pil_img, caption=f"{chosen_label}  —  Frame {frame_idx} / {N - 1}", width="stretch")

    with col_info:
        st.subheader("Label Controls")

        # C7: Mark button — st.rerun() to refresh stale state
        if st.button(f"✅ Mark Frame {frame_idx} as Impact", type="primary"):
            st.session_state.labels[clip_key] = {
                "clip":         chosen_label,
                "impact_frame": frame_idx,
                "total_frames": N,
            }
            save_labels(st.session_state.labels)
            st.success(f"Saved! Impact frame = {frame_idx}")
            st.rerun()

        # C8: Skip button
        if st.button("⏭️ Skip clip (contact not visible)"):
            if clip_key in st.session_state.labels:
                del st.session_state.labels[clip_key]
                save_labels(st.session_state.labels)
            st.info("Clip skipped — no label saved.")
            st.rerun()

        # Current label display
        if existing_frame is not None:
            st.info(f"Current label: frame **{existing_frame}**")
        else:
            st.warning("No label yet for this clip.")

        # Clear button — C7: st.rerun() after clear
        if existing_frame is not None:
            if st.button("🗑️ Clear label for this clip"):
                del st.session_state.labels[clip_key]
                save_labels(st.session_state.labels)
                st.info("Label cleared.")
                st.rerun()

        st.markdown("---")
        st.metric("Frames", N)
        st.metric("Clips labelled", len(st.session_state.labels))

        # Keyboard hint
        st.markdown(
            "**Tip:** Click the slider, then use ← → arrow keys to step "
            "one frame at a time."
        )

    # Label summary table
    _render_label_summary(st.session_state.labels, video_files)


def _render_label_summary(labels: dict, video_files: list):
    """Render a table of all labels in the sidebar."""
    st.markdown("---")
    st.subheader("All Labels")
    if not labels:
        st.caption("No labels recorded yet.")
        return
    for clip_key, info in sorted(labels.items()):
        st.caption(f"`{clip_key}` → frame **{info['impact_frame']}**")

    labelled = len(labels)
    total    = len(video_files)
    st.caption(f"{labelled}/{total} clips labelled.")


if __name__ == "__main__":
    main()
