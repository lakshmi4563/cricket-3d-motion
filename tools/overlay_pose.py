"""
Overlay MediaPipe pose detections onto video frames and detect tracking jumps.

Runs the shared detector from src/pose_detector_factory.py on every clip
and writes annotated frames (skeleton, frame number, mean landmark visibility)
to outputs/pose_overlays/<clip>/frame_XX.png.

Computes the frame-to-frame jump of the hip midpoint and flags jumps larger
than 0.15 in normalised coordinates (indicating a possible person switch).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
_tools_dir = Path(__file__).resolve().parent
_project_root = _tools_dir.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import cv2
import numpy as np
import mediapipe as mp
from evaluate_reconstruction import decode_all_frames
from pose_detector_factory import create_pose_detector

# Standard MediaPipe pose connections
POSE_CONNECTIONS = [
    # Arms
    (11, 13), (13, 15),  # Left shoulder-elbow-wrist
    (12, 14), (14, 16),  # Right shoulder-elbow-wrist
    (11, 12),            # Shoulders
    # Torso
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 24),            # Hips
    # Legs
    (23, 25), (25, 27),  # Left hip-knee-ankle
    (24, 26), (26, 28),  # Right hip-knee-ankle
]


def overlay_clip(
    clip_path: Path,
    output_dirs: list[Path],
    detector,
    jump_threshold: float = 0.15,
) -> dict:
    """
    Process one video clip: run shared detector, annotate frames, write PNGs,
    and compute hip midpoint jumps.
    """
    frames = decode_all_frames(clip_path)
    for out_d in output_dirs:
        out_d.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    hip_records = []
    detected_count = 0

    for t, frame_bgr in enumerate(frames):
        canvas = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = detector.detect(mp_img)

        detected = False
        hip_center = None
        vis_mean = 0.0

        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            detected = True
            detected_count += 1
            lms = res.pose_landmarks[0]
            pts_px = [(int(p.x * w), int(p.y * h)) for p in lms]
            vis_list = [float(getattr(p, "visibility", 0.0) or 0.0) for p in lms]
            vis_mean = float(np.mean(vis_list))

            # Bounding box
            xs = [pt[0] for pt in pts_px]
            ys = [pt[1] for pt in pts_px]
            x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
            y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))

            # Hip midpoint (landmarks 23 left hip, 24 right hip)
            left_hip = lms[23]
            right_hip = lms[24]
            hx = float(left_hip.x + right_hip.x) / 2.0
            hy = float(left_hip.y + right_hip.y) / 2.0
            hip_center = (hx, hy)

            # Draw skeleton
            for j1, j2 in POSE_CONNECTIONS:
                if j1 < len(pts_px) and j2 < len(pts_px):
                    if vis_list[j1] >= 0.2 and vis_list[j2] >= 0.2:
                        cv2.line(canvas, pts_px[j1], pts_px[j2], (0, 255, 0), 2)

            # Draw joints
            for idx, pt in enumerate(pts_px):
                v = vis_list[idx]
                col = (0, 255, 0) if v >= 0.5 else (0, 165, 255)
                cv2.circle(canvas, pt, 3, col, -1)

            # Draw bounding box & hip midpoint
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (255, 200, 0), 1)
            hip_px = (int(hx * w), int(hy * h))
            cv2.circle(canvas, hip_px, 5, (0, 0, 255), -1)

        # Frame overlay text: frame number, detection status, mean landmark visibility
        header_txt = f"Frame {t:02d}/{len(frames)-1} | Det: {detected} | Mean Vis: {vis_mean:.2f}"
        if hip_center is not None:
            header_txt += f" | Hip: ({hip_center[0]:.2f}, {hip_center[1]:.2f})"
        cv2.putText(canvas, header_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Write PNG frames to output directories
        filename = f"frame_{t:02d}.png"
        for out_d in output_dirs:
            cv2.imwrite(str(out_d / filename), canvas)

        hip_records.append((t, hip_center))

    # Compute hip midpoint jumps
    # 1. Across consecutive frames where both t-1 and t have detected poses
    consec_jumps = []
    for i in range(1, len(hip_records)):
        t_prev, hip_prev = hip_records[i - 1]
        t_curr, hip_curr = hip_records[i]
        if hip_prev is not None and hip_curr is not None:
            jump = float(np.hypot(hip_curr[0] - hip_prev[0], hip_curr[1] - hip_prev[1]))
            if jump > jump_threshold:
                consec_jumps.append((t_curr, round(jump, 4)))

    # 2. Across consecutive detected frames (even if intervening frames were missing)
    detected_jumps = []
    last_hip = None
    for t, hip in hip_records:
        if hip is not None:
            if last_hip is not None:
                jump = float(np.hypot(hip[0] - last_hip[0], hip[1] - last_hip[1]))
                if jump > jump_threshold:
                    detected_jumps.append((t, round(jump, 4)))
            last_hip = hip

    return {
        "clip_rel": f"{clip_path.parent.name}/{clip_path.name}",
        "total_frames": len(frames),
        "detected_frames": detected_count,
        "consec_jumps": consec_jumps,
        "detected_jumps": detected_jumps,
    }


def main():
    clips_dir = _project_root / "outputs" / "sample_clips"
    output_root = _project_root / "outputs" / "pose_overlays"
    clips = sorted(clips_dir.rglob("*.avi"))

    model_path = _project_root / "models" / "pose_landmarker_lite.task"
    detector = create_pose_detector(model_path)

    print("=" * 80)
    print("RUNNING POSE OVERLAY & PERSON-SWITCH JUMP DIAGNOSTICS")
    print(f"Clips found: {len(clips)} | Output root: {output_root}")
    print("=" * 80)

    results = []
    try:
        for vf in clips:
            rel = f"{vf.parent.name}/{vf.name}"
            # Write to both outputs/pose_overlays/<class>/<clip_stem> and outputs/pose_overlays/<clip_stem>
            out_clip_dir1 = output_root / vf.parent.name / vf.stem
            out_clip_dir2 = output_root / vf.stem
            res = overlay_clip(vf, [out_clip_dir1, out_clip_dir2], detector)
            results.append(res)
    finally:
        detector.close()

    print("\n" + "=" * 80)
    print("SUMMARY PER CLIP:")
    print("=" * 80)
    for r in results:
        flagged_consec = [f for f, j in r["consec_jumps"]]
        flagged_all = [f for f, j in r["detected_jumps"]]
        print(f"Clip: {r['clip_rel']}")
        print(f"  - Frames with pose detected: {r['detected_frames']}/{r['total_frames']}")
        print(f"  - Flagged jumps > 0.15 (consecutive detected pairs): {len(r['detected_jumps'])} -> Frames: {flagged_all}")
        if r["consec_jumps"] != r["detected_jumps"]:
            print(f"    (Strict consecutive frames t-1 to t: {len(r['consec_jumps'])} -> Frames: {flagged_consec})")

    return results


if __name__ == "__main__":
    main()
