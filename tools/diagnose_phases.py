"""
Phase Detection Diagnostics.

Inspects every clip in outputs/sample_clips/ and evaluates:
1. Frame counts and MediaPipe pose detection yield.
2. Phase detection reliability and specific failure/warning reasons.
3. Identified phase boundaries (frame index intervals).
4. Fallback status when running select_phase_aware_frames at K=6 and K=8.

Saves structured results to reports/phase_diagnostics.json and prints a
concise per-clip report and a one-line summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Setup paths
_tools_dir = Path(__file__).resolve().parent
_project_root = _tools_dir.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import mediapipe as mp
from evaluate_reconstruction import (
    decode_all_frames,
    extract_clip_poses_eval,
    compute_consecutive_motion_scores,
)
from phase_detection import detect_phases
from phase_aware_selection import select_phase_aware_frames


def find_clips(base_dir: Path) -> list[Path]:
    clips_dir = base_dir / "outputs" / "sample_clips"
    if not clips_dir.exists():
        # Fallback to general outputs search
        clips_dir = base_dir / "outputs"
    clips = sorted(clips_dir.rglob("*.avi"))
    return clips


def run_diagnostics(base_dir: Path) -> dict:
    clips = find_clips(base_dir)
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model missing at {model_path}")

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "phase_diagnostics.json"

    diagnostics = []
    reliable_count = 0
    total_clips = len(clips)

    print("=" * 80)
    print("PHASE DETECTION DIAGNOSTICS")
    print(f"Total clips to analyze: {total_clips}")
    print("=" * 80)

    try:
        for vf in clips:
            rel_clip = f"{vf.parent.name}/{vf.name}"
            frames = decode_all_frames(vf)
            n_frames = len(frames)

            poses = extract_clip_poses_eval(frames, detector)
            n_detected = sum(1 for p in poses if p.get("detected", False))

            _, energies = compute_consecutive_motion_scores(frames)
            phase_info = detect_phases(poses) or {}

            is_reliable = bool(phase_info.get("reliable", False))
            reason = phase_info.get("reliability_note", "")
            if is_reliable:
                reliable_count += 1
                reason = "All phases detected with valid temporal ordering"
            elif not reason:
                reason = "Phase detection failed or signals ambiguous"

            raw_phases = phase_info.get("phases", {})
            phase_bounds = {
                p: {"start": bounds["start"], "end": bounds["end"]}
                for p, bounds in raw_phases.items()
            }

            # Test fallback at K=6 and K=8
            _, info_k6 = select_phase_aware_frames(poses, energies, k=6)
            fallback_k6 = bool(info_k6.get("fallback_used", False))

            _, info_k8 = select_phase_aware_frames(poses, energies, k=8)
            fallback_k8 = bool(info_k8.get("fallback_used", False))

            clip_entry = {
                "clip": rel_clip,
                "n_frames": n_frames,
                "n_detected": n_detected,
                "detection_rate": round(n_detected / n_frames, 3) if n_frames > 0 else 0.0,
                "reliable": is_reliable,
                "reason": reason,
                "dominant_wrist": phase_info.get("dominant_wrist_side"),
                "peak_speed_frame": phase_info.get("peak_speed_frame"),
                "phase_boundaries": phase_bounds,
                "fallback_k6": fallback_k6,
                "fallback_k8": fallback_k8,
            }
            diagnostics.append(clip_entry)

            print(f"\nClip: {rel_clip}")
            print(f"  Frames: {n_frames} total, {n_detected} with pose detected ({n_detected/n_frames:.1%})")
            print(f"  Reliable: {is_reliable}")
            print(f"  Reason: {reason}")
            if phase_bounds:
                bounds_str = ", ".join(f"{k}: [{v['start']}-{v['end']}]" for k, v in phase_bounds.items())
                print(f"  Phases: {bounds_str}")
            else:
                print("  Phases: None")
            print(f"  Fallback used: K=6 -> {fallback_k6}, K=8 -> {fallback_k8}")

    finally:
        detector.close()

    summary_text = f"Phase detection reliable on {reliable_count} of {total_clips} clips"
    payload = {
        "summary": summary_text,
        "total_clips": total_clips,
        "reliable_count": reliable_count,
        "unreliable_count": total_clips - reliable_count,
        "fallback_rate_pct": round(((total_clips - reliable_count) / total_clips) * 100, 1) if total_clips > 0 else 0.0,
        "clips": diagnostics,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[OK] Saved diagnostics to: {out_json}")
    print(summary_text)
    print("=" * 80)

    return payload


if __name__ == "__main__":
    run_diagnostics(_project_root)

