"""
Impact Estimation Accuracy Evaluation.

Compares estimated impact frames (from src/impact_estimation.py) against
manually labelled ground-truth frames (in reports/impact_ground_truth.json).

Computes (per clip and overall):
    - timing error = |estimated_frame - ground_truth_frame|  (in frames)
    - mean, median, max timing error
    - fraction where |error| <= 2 frames ("near miss" threshold)

Writes results to:
    reports/impact_estimation_summary.json

IMPORTANT: Accuracy is only reported for clips that have been labelled.
If no labels exist, the script reports estimation results only (no error).
The script does NOT claim accuracy beyond what the labelled clips show.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parents[1]
SRC_DIR     = BASE_DIR / "src"
REPORTS_DIR = BASE_DIR / "reports"
GT_FILE     = REPORTS_DIR / "impact_ground_truth.json"
MODEL_PATH  = BASE_DIR / "models" / "pose_landmarker_lite.task"
SAMPLE_DIR  = BASE_DIR / "outputs" / "sample_clips"
OUT_FILE    = REPORTS_DIR / "impact_estimation_summary.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase_detection import detect_phases
from impact_estimation import estimate_impact_frame
from motion_analysis import decode_all_frames
from evaluate_reconstruction import extract_clip_poses_eval

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


def main():
    # Load ground-truth labels
    gt_labels: dict = {}
    if GT_FILE.exists():
        with open(str(GT_FILE), "r", encoding="utf-8") as f:
            gt_labels = json.load(f)
        print(f"[INFO] Loaded {len(gt_labels)} ground-truth labels from {GT_FILE.name}")
    else:
        print(f"[WARN] No ground-truth file at {GT_FILE}. "
              "Run `streamlit run tools/label_impact.py` to label clips first.")
        print("       Running estimation only (no error computation).")

    # Collect clips
    video_files = sorted(SAMPLE_DIR.rglob("*.avi"))
    if not video_files:
        print(f"[ERROR] No .avi clips in {SAMPLE_DIR}")
        sys.exit(1)

    if not _MP_AVAILABLE:
        print("[ERROR] MediaPipe not available.")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    # Init MediaPipe
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    print("=" * 70)
    print("IMPACT FRAME ESTIMATION EVALUATION")
    print("=" * 70)

    per_clip = []
    errors   = []

    try:
        for vf in video_files:
            clip_key   = f"{vf.parent.name}/{vf.name}"
            clip_class = vf.parent.name
            clip_name  = vf.name

            t0 = time.perf_counter()
            frames = decode_all_frames(vf)
            poses  = extract_clip_poses_eval(frames, detector)
            t_pose = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            phase_info  = detect_phases(poses)
            impact_info = estimate_impact_frame(poses, phase_info=phase_info)
            t_est  = (time.perf_counter() - t1) * 1000

            est_frame = impact_info.get("estimated_impact_frame")
            uncertain = impact_info.get("uncertain", True)
            conf      = impact_info.get("confidence", 0.0)

            record: dict = {
                "clip":             clip_key,
                "total_frames":     len(frames),
                "estimated_frame":  est_frame,
                "uncertain":        uncertain,
                "confidence":       conf,
                "signal_agreement": impact_info.get("signal_agreement"),
                "method_note":      impact_info.get("method_note", ""),
                "pose_time_ms":     round(t_pose, 2),
                "estimation_time_ms": round(t_est, 2),
                "ground_truth_frame": None,
                "timing_error_frames": None,
                "within_2_frames": None,
            }

            gt_info = gt_labels.get(clip_key, {})
            gt_frame = gt_info.get("impact_frame", None)
            record["ground_truth_frame"] = gt_frame

            if gt_frame is not None and est_frame is not None:
                err = abs(est_frame - gt_frame)
                record["timing_error_frames"] = err
                record["within_2_frames"]     = err <= 2
                errors.append(err)
                err_str = f"err={err}f"
            elif gt_frame is None:
                err_str = "no_label"
            else:
                err_str = "est=uncertain"

            print(
                f"  {clip_key:<40} est={est_frame!s:<5} conf={conf:.2f} "
                f"unc={uncertain!s:<6} {err_str}"
            )

            per_clip.append(record)

    finally:
        detector.close()
        print("\nMediaPipe closed cleanly.")

    # Aggregate statistics
    summary: dict = {
        "note": (
            "Accuracy is reported ONLY for clips that have been manually labelled. "
            f"Labelled clips: {len(errors)}/{len(per_clip)}. "
            "Do not extrapolate results beyond these clips."
        ),
        "n_clips":         len(per_clip),
        "n_labelled":      len(errors),
        "n_unlabelled":    len(per_clip) - len(errors),
        "per_clip":        per_clip,
    }

    if errors:
        summary["timing_error_stats"] = {
            "mean_frames":   round(float(np.mean(errors)), 3),
            "median_frames": round(float(np.median(errors)), 3),
            "max_frames":    int(np.max(errors)),
            "within_2_frames_pct": round(
                sum(1 for e in errors if e <= 2) / len(errors) * 100, 1
            ),
        }
        print("\n--- Timing Error Summary (labelled clips only) ---")
        print(f"  Mean  : {summary['timing_error_stats']['mean_frames']:.2f} frames")
        print(f"  Median: {summary['timing_error_stats']['median_frames']:.2f} frames")
        print(f"  Max   : {summary['timing_error_stats']['max_frames']} frames")
        print(f"  Within ±2 frames: {summary['timing_error_stats']['within_2_frames_pct']}%")
    else:
        summary["timing_error_stats"] = None
        print("\n[INFO] No labelled clips yet — timing error cannot be computed.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_FILE), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Saved: {OUT_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

