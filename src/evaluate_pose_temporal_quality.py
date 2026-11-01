"""
Temporal Body-Pose Quality Evaluation for Uniform vs. Motion-Aware Frame Selection.

Extracts normalized 2D landmark coordinates and visibility scores across all 24 frames
of each of the 12 cricket video clips to form a pseudo-ground-truth reference sequence.
Compares:
  - Reference sequence (all 24 frames)
  - Uniform subset (6 frames)
  - Motion-aware subset (6 frames)

Metrics evaluated:
1. Pose Detection Coverage: % of selected frames with a detected human pose.
2. Mean Landmark Visibility: Average visibility score of detected landmarks.
3. Temporal Span & Coverage:
   - Temporal span: max(selected_idx) - min(selected_idx) normalized by (N - 1).
   - Mean / Max temporal gap between consecutive selected frames.
4. Nearest-Selected-Pose Error (NSPE):
   For each reference frame t that has a valid detected pose, find the temporally nearest
   frame s in the selected subset that ALSO has a valid detected pose.
   Compute the Mean Per-Joint Distance (MPJD) in normalized 2D space:
     dist(t, s) = (1 / |V|) * sum_{j in V} sqrt((x_{t,j} - x_{s,j})^2 + (y_{t,j} - y_{s,j})^2)
   where V is the set of joints where visibility >= 0.5 in BOTH frames.
   If no overlapping visible joints or no valid selected pose exists, handle explicitly.
   NOTE: This measures temporal representativeness against the unconstrained 24-frame video,
   NOT true 3D biomechanical motion-capture accuracy.

Outputs:
- reports/pose_temporal_quality.json (Per-clip detailed metrics and errors)
- reports/pose_temporal_quality_summary.json (Aggregate and per-class summary)
"""

from pathlib import Path
import json
import time
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV (cv2) is not installed.")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"[ERROR] MediaPipe Tasks import failed: {e}")
    sys.exit(1)


def decode_clip_frames(video_path: Path):
    """Decode all frames of a clip into a list of BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)

    cap.release()
    return frames


def extract_clip_poses(frames, detector):
    """
    Runs MediaPipe Tasks PoseLandmarker on all frames in sequence.
    Returns:
        dict: {frame_idx: {"detected": bool, "landmarks": list of dicts, "inference_time_ms": float}}
    """
    poses = {}
    for idx, frame_bgr in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        t_start = time.perf_counter()
        result = detector.detect(mp_image)
        inf_ms = round((time.perf_counter() - t_start) * 1000, 2)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm_list = []
            for pt in result.pose_landmarks[0]:
                lm_list.append({
                    "x": float(pt.x),
                    "y": float(pt.y),
                    "z": float(pt.z),
                    "visibility": float(getattr(pt, "visibility", 0.0)),
                    "presence": float(getattr(pt, "presence", 0.0)),
                })
            poses[idx] = {
                "detected": True,
                "landmarks": lm_list,
                "inference_time_ms": inf_ms,
            }
        else:
            poses[idx] = {
                "detected": False,
                "landmarks": [],
                "inference_time_ms": inf_ms,
            }
    return poses


def compute_temporal_coverage(selected_indices, total_frames=24):
    """
    Computes temporal span, max gap, and mean gap between selected frame indices.
    """
    if not selected_indices:
        return {"temporal_span": 0.0, "max_gap": total_frames, "mean_gap": float(total_frames)}

    sorted_idx = sorted(selected_indices)
    span = (sorted_idx[-1] - sorted_idx[0]) / max(total_frames - 1, 1)

    if len(sorted_idx) > 1:
        gaps = [sorted_idx[i + 1] - sorted_idx[i] for i in range(len(sorted_idx) - 1)]
        max_gap = int(max(gaps))
        mean_gap = round(float(np.mean(gaps)), 2)
    else:
        max_gap = total_frames
        mean_gap = float(total_frames)

    return {
        "temporal_span_ratio": round(span, 4),
        "min_frame": sorted_idx[0],
        "max_frame": sorted_idx[-1],
        "max_frame_gap": max_gap,
        "mean_frame_gap": mean_gap,
    }


def compute_nearest_pose_error(ref_poses, selected_indices, total_frames=24):
    """
    Computes Nearest-Selected-Pose Error (NSPE) across all valid reference frames.
    For each reference frame t that has a valid detected pose:
      - Finds the temporally nearest index s in selected_indices that ALSO has a valid detected pose.
      - Calculates mean normalized 2D euclidean error across overlapping visible keypoints (visibility >= 0.5).
    Returns detailed frame-by-frame errors and summary averages.
    """
    valid_selected = [s for s in selected_indices if ref_poses.get(s, {}).get("detected", False)]

    if not valid_selected:
        return {
            "error_computable": False,
            "reason": "No valid pose detected in any selected frame.",
            "mean_nspe": None,
            "median_nspe": None,
            "evaluated_ref_frames": 0,
            "per_frame_errors": [],
        }

    per_frame_errors = []
    distances = []

    for t in range(total_frames):
        ref_p = ref_poses.get(t, {})
        if not ref_p.get("detected", False):
            continue  # Skip reference frames where pose failed

        ref_lm = ref_p["landmarks"]

        # Find nearest selected frame index in temporal space |t - s|
        nearest_s = min(valid_selected, key=lambda s: abs(t - s))
        sel_lm = ref_poses[nearest_s]["landmarks"]

        # Compute error on mutually visible landmarks (visibility >= 0.5)
        joint_dists = []
        for j in range(min(len(ref_lm), len(sel_lm))):
            r_pt = ref_lm[j]
            s_pt = sel_lm[j]
            if r_pt["visibility"] >= 0.5 and s_pt["visibility"] >= 0.5:
                d = np.sqrt((r_pt["x"] - s_pt["x"]) ** 2 + (r_pt["y"] - s_pt["y"]) ** 2)
                joint_dists.append(float(d))

        if joint_dists:
            frame_err = float(np.mean(joint_dists))
            distances.append(frame_err)
            per_frame_errors.append({
                "ref_frame": t,
                "matched_selected_frame": nearest_s,
                "temporal_distance_frames": abs(t - nearest_s),
                "num_overlapping_joints": len(joint_dists),
                "mean_norm_distance": round(frame_err, 4),
            })
        else:
            # Poses detected but no overlapping visible joints above 0.5
            per_frame_errors.append({
                "ref_frame": t,
                "matched_selected_frame": nearest_s,
                "temporal_distance_frames": abs(t - nearest_s),
                "num_overlapping_joints": 0,
                "mean_norm_distance": None,
                "note": "No mutually visible joints >= 0.5"
            })

    if distances:
        mean_err = round(float(np.mean(distances)), 4)
        median_err = round(float(np.median(distances)), 4)
    else:
        mean_err, median_err = None, None

    return {
        "error_computable": True,
        "mean_nspe": mean_err,
        "median_nspe": median_err,
        "evaluated_ref_frames": len(distances),
        "total_valid_ref_frames": sum(1 for t in range(total_frames) if ref_poses.get(t, {}).get("detected")),
        "per_frame_errors": per_frame_errors,
    }


def main():
    base_dir = Path(__file__).resolve().parents[1]
    motion_json_path = base_dir / "reports" / "motion_analysis.json"
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not motion_json_path.exists():
        raise FileNotFoundError(f"Missing motion analysis file: {motion_json_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with open(motion_json_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)

    print("=" * 96)
    print("EVALUATING TEMPORAL POSE PRESERVATION: UNIFORM VS MOTION-AWARE (CPU)")
    print(f"Total Clips: {len(clips_data)} | Reference Frames per Clip: 24 | Evaluated Sequences: 3 per clip")
    print("=" * 96)

    # Initialize MediaPipe PoseLandmarker
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    clip_evaluations = []

    try:
        for idx, clip in enumerate(clips_data):
            clip_name = clip["clip_name"]
            clip_class = clip["class"]
            rel_path = clip["relative_video_path"]
            video_file = base_dir / rel_path

            u_indices = clip["uniform_selected_indices"]
            m_indices = clip["motion_selected_indices"]

            # Decode all 24 frames for full reference extraction
            all_frames = decode_clip_frames(video_file)
            total_frames = len(all_frames)

            # Extract poses across all 24 frames
            clip_poses = extract_clip_poses(all_frames, detector)

            # 1. Reference sequence stats
            ref_detected_frames = [f for f in range(total_frames) if clip_poses[f]["detected"]]
            ref_detection_rate = round((len(ref_detected_frames) / total_frames) * 100, 2)
            ref_visibilities = [
                pt["visibility"]
                for f in ref_detected_frames
                for pt in clip_poses[f]["landmarks"]
            ]
            ref_mean_vis = round(float(np.mean(ref_visibilities)), 4) if ref_visibilities else 0.0

            # 2. Uniform selection metrics
            u_detected_frames = [f for f in u_indices if clip_poses[f]["detected"]]
            u_detection_rate = round((len(u_detected_frames) / len(u_indices)) * 100, 2)
            u_visibilities = [
                pt["visibility"]
                for f in u_detected_frames
                for pt in clip_poses[f]["landmarks"]
            ]
            u_mean_vis = round(float(np.mean(u_visibilities)), 4) if u_visibilities else 0.0
            u_temporal = compute_temporal_coverage(u_indices, total_frames)
            u_nspe = compute_nearest_pose_error(clip_poses, u_indices, total_frames)

            # 3. Motion-aware selection metrics
            m_detected_frames = [f for f in m_indices if clip_poses[f]["detected"]]
            m_detection_rate = round((len(m_detected_frames) / len(m_indices)) * 100, 2)
            m_visibilities = [
                pt["visibility"]
                for f in m_detected_frames
                for pt in clip_poses[f]["landmarks"]
            ]
            m_mean_vis = round(float(np.mean(m_visibilities)), 4) if m_visibilities else 0.0
            m_temporal = compute_temporal_coverage(m_indices, total_frames)
            m_nspe = compute_nearest_pose_error(clip_poses, m_indices, total_frames)

            clip_record = {
                "clip_name": clip_name,
                "class": clip_class,
                "total_frames": total_frames,
                "reference_sequence": {
                    "detected_frames_count": len(ref_detected_frames),
                    "detection_rate_pct": ref_detection_rate,
                    "mean_visibility": ref_mean_vis,
                    "missed_frames": [f for f in range(total_frames) if not clip_poses[f]["detected"]],
                },
                "uniform_selection": {
                    "indices": u_indices,
                    "detected_frames_count": len(u_detected_frames),
                    "detection_rate_pct": u_detection_rate,
                    "mean_visibility": u_mean_vis,
                    "temporal_coverage": u_temporal,
                    "nearest_pose_error": {
                        "mean_nspe": u_nspe["mean_nspe"],
                        "median_nspe": u_nspe["median_nspe"],
                        "evaluated_ref_frames": u_nspe["evaluated_ref_frames"],
                    },
                    "missing_detection_indices": [f for f in u_indices if not clip_poses[f]["detected"]],
                },
                "motion_aware_selection": {
                    "indices": m_indices,
                    "detected_frames_count": len(m_detected_frames),
                    "detection_rate_pct": m_detection_rate,
                    "mean_visibility": m_mean_vis,
                    "temporal_coverage": m_temporal,
                    "nearest_pose_error": {
                        "mean_nspe": m_nspe["mean_nspe"],
                        "median_nspe": m_nspe["median_nspe"],
                        "evaluated_ref_frames": m_nspe["evaluated_ref_frames"],
                    },
                    "missing_detection_indices": [f for f in m_indices if not clip_poses[f]["detected"]],
                },
                "nspe_details": {
                    "uniform": u_nspe.get("per_frame_errors", []),
                    "motion_aware": m_nspe.get("per_frame_errors", []),
                }
            }
            clip_evaluations.append(clip_record)

            label = f"{clip_class}/{clip_name}"
            u_err_str = f"{u_nspe['mean_nspe']:.4f}" if u_nspe["mean_nspe"] is not None else "N/A"
            m_err_str = f"{m_nspe['mean_nspe']:.4f}" if m_nspe["mean_nspe"] is not None else "N/A"
            print(f"[{idx+1:02d}/12] {label:<30} Ref Det: {len(ref_detected_frames)}/24 | NSPE (Uniform): {u_err_str} | NSPE (Motion): {m_err_str}")

    finally:
        detector.close()
        print("MediaPipe PoseLandmarker closed cleanly.")

    # Compute aggregate summary statistics
    all_u_nspe = [c["uniform_selection"]["nearest_pose_error"]["mean_nspe"] for c in clip_evaluations if c["uniform_selection"]["nearest_pose_error"]["mean_nspe"] is not None]
    all_m_nspe = [c["motion_aware_selection"]["nearest_pose_error"]["mean_nspe"] for c in clip_evaluations if c["motion_aware_selection"]["nearest_pose_error"]["mean_nspe"] is not None]

    all_u_vis = [c["uniform_selection"]["mean_visibility"] for c in clip_evaluations if c["uniform_selection"]["mean_visibility"] > 0]
    all_m_vis = [c["motion_aware_selection"]["mean_visibility"] for c in clip_evaluations if c["motion_aware_selection"]["mean_visibility"] > 0]

    all_u_det_pct = [c["uniform_selection"]["detection_rate_pct"] for c in clip_evaluations]
    all_m_det_pct = [c["motion_aware_selection"]["detection_rate_pct"] for c in clip_evaluations]

    all_u_span = [c["uniform_selection"]["temporal_coverage"]["temporal_span_ratio"] for c in clip_evaluations]
    all_m_span = [c["motion_aware_selection"]["temporal_coverage"]["temporal_span_ratio"] for c in clip_evaluations]

    all_u_max_gap = [c["uniform_selection"]["temporal_coverage"]["max_frame_gap"] for c in clip_evaluations]
    all_m_max_gap = [c["motion_aware_selection"]["temporal_coverage"]["max_frame_gap"] for c in clip_evaluations]

    summary = {
        "pilot_scope": {
            "total_clips": len(clip_evaluations),
            "total_reference_frames": len(clip_evaluations) * 24,
            "disclaimer": "Evaluated on CPU using MediaPipe estimated landmarks. MediaPipe 2D coordinates are not ground-truth motion capture. Nearest-Selected-Pose Error (NSPE) reflects temporal representativeness against full video poses, not true 3D biomechanical accuracy."
        },
        "aggregate_comparison": {
            "reference_sequence": {
                "total_frames": len(clip_evaluations) * 24,
                "total_detected_frames": sum(c["reference_sequence"]["detected_frames_count"] for c in clip_evaluations),
                "mean_detection_rate_pct": round(float(np.mean([c["reference_sequence"]["detection_rate_pct"] for c in clip_evaluations])), 2),
                "mean_landmark_visibility": round(float(np.mean([c["reference_sequence"]["mean_visibility"] for c in clip_evaluations])), 4),
            },
            "uniform_selection": {
                "mean_detection_rate_pct": round(float(np.mean(all_u_det_pct)), 2),
                "mean_landmark_visibility": round(float(np.mean(all_u_vis)), 4),
                "mean_temporal_span_ratio": round(float(np.mean(all_u_span)), 4),
                "mean_max_temporal_gap": round(float(np.mean(all_u_max_gap)), 2),
                "mean_nearest_pose_error_nspe": round(float(np.mean(all_u_nspe)), 4) if all_u_nspe else None,
            },
            "motion_aware_selection": {
                "mean_detection_rate_pct": round(float(np.mean(all_m_det_pct)), 2),
                "mean_landmark_visibility": round(float(np.mean(all_m_vis)), 4),
                "mean_temporal_span_ratio": round(float(np.mean(all_m_span)), 4),
                "mean_max_temporal_gap": round(float(np.mean(all_m_max_gap)), 2),
                "mean_nearest_pose_error_nspe": round(float(np.mean(all_m_nspe)), 4) if all_m_nspe else None,
            },
        },
        "per_class_comparison": {},
    }

    # Per-class breakdown
    classes = sorted(list(set(c["class"] for c in clip_evaluations)))
    for c in classes:
        c_clips = [clip for clip in clip_evaluations if clip["class"] == c]
        u_nspe_c = [clip["uniform_selection"]["nearest_pose_error"]["mean_nspe"] for clip in c_clips if clip["uniform_selection"]["nearest_pose_error"]["mean_nspe"] is not None]
        m_nspe_c = [clip["motion_aware_selection"]["nearest_pose_error"]["mean_nspe"] for clip in c_clips if clip["motion_aware_selection"]["nearest_pose_error"]["mean_nspe"] is not None]

        summary["per_class_comparison"][c] = {
            "num_clips": len(c_clips),
            "ref_detection_rate_pct": round(float(np.mean([clip["reference_sequence"]["detection_rate_pct"] for clip in c_clips])), 1),
            "uniform": {
                "detection_rate_pct": round(float(np.mean([clip["uniform_selection"]["detection_rate_pct"] for clip in c_clips])), 1),
                "mean_visibility": round(float(np.mean([clip["uniform_selection"]["mean_visibility"] for clip in c_clips])), 3),
                "mean_nspe": round(float(np.mean(u_nspe_c)), 4) if u_nspe_c else None,
            },
            "motion_aware": {
                "detection_rate_pct": round(float(np.mean([clip["motion_aware_selection"]["detection_rate_pct"] for clip in c_clips])), 1),
                "mean_visibility": round(float(np.mean([clip["motion_aware_selection"]["mean_visibility"] for clip in c_clips])), 3),
                "mean_nspe": round(float(np.mean(m_nspe_c)), 4) if m_nspe_c else None,
            }
        }

    # Save JSON files
    detail_file = reports_dir / "pose_temporal_quality.json"
    summary_file = reports_dir / "pose_temporal_quality_summary.json"

    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(clip_evaluations, f, indent=2)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 96)
    print("TEMPORAL POSE PRESERVATION: AGGREGATE SUMMARY (12 CLIPS, 288 REFERENCE FRAMES)")
    print("=" * 96)
    u_agg = summary["aggregate_comparison"]["uniform_selection"]
    m_agg = summary["aggregate_comparison"]["motion_aware_selection"]
    r_agg = summary["aggregate_comparison"]["reference_sequence"]

    print(f"{'Metric':<40} {'Reference (24f)':<18} {'Uniform (6f)':<18} {'Motion-Aware (6f)'}")
    print("-" * 96)
    print(f"{'Mean Detection Coverage (%)':<40} {r_agg['mean_detection_rate_pct']:<18.2f} {u_agg['mean_detection_rate_pct']:<18.2f} {m_agg['mean_detection_rate_pct']:.2f}%")
    print(f"{'Mean Landmark Visibility':<40} {r_agg['mean_landmark_visibility']:<18.4f} {u_agg['mean_landmark_visibility']:<18.4f} {m_agg['mean_landmark_visibility']:.4f}")
    print(f"{'Mean Temporal Span Ratio':<40} {'1.0000':<18} {u_agg['mean_temporal_span_ratio']:<18.4f} {m_agg['mean_temporal_span_ratio']:.4f}")
    print(f"{'Mean Max Temporal Gap (frames)':<40} {'1.0':<18} {u_agg['mean_max_temporal_gap']:<18.1f} {m_agg['mean_max_temporal_gap']:.1f}")
    print(f"{'Nearest-Selected-Pose Error (NSPE)':<40} {'0.0000':<18} {u_agg['mean_nearest_pose_error_nspe']:<18.4f} {m_agg['mean_nearest_pose_error_nspe']:.4f}")
    print("=" * 96)

    print("\nPER-CLASS TEMPORAL REPRESENTATIVENESS BREAKDOWN (NSPE):")
    print("-" * 96)
    print(f"{'Class':<20} {'Clips':<8} {'Ref Det %':<14} {'Uniform NSPE':<18} {'Motion-Aware NSPE':<18} {'NSPE Winner'}")
    print("-" * 96)
    for c in classes:
        p_c = summary["per_class_comparison"][c]
        u_err = p_c["uniform"]["mean_nspe"]
        m_err = p_c["motion_aware"]["mean_nspe"]
        u_s = f"{u_err:.4f}" if u_err is not None else "N/A"
        m_s = f"{m_err:.4f}" if m_err is not None else "N/A"
        if u_err is not None and m_err is not None:
            winner = "Motion-Aware" if m_err < u_err else ("Uniform" if u_err < m_err else "Tie")
        else:
            winner = "Inconclusive"
        print(f"{c:<20} {p_c['num_clips']:<8} {p_c['ref_detection_rate_pct']:<14.1f} {u_s:<18} {m_s:<18} {winner}")
    print("=" * 96)

    print(f"\n[OK] Detailed temporal evaluations: {detail_file.relative_to(base_dir)}")
    print(f"[OK] Summary temporal report: {summary_file.relative_to(base_dir)}")


if __name__ == "__main__":
    main()
