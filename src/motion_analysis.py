"""
Motion-Aware Frame Selection Baseline for Short Cricket Clips.
Calculates consecutive frame-difference motion scores, compares uniform vs motion-based
frame selection, and generates side-by-side comparison visualization grids.

Selection Algorithm:
1. Decode all frames and convert to grayscale.
2. Calculate Mean Absolute Difference (MAD) between consecutive frames (t and t+1):
   motion_score(t) = mean(|gray_{t+1} - gray_t|).
   We assign each frame t an associated motion energy:
   - For frame 0: motion_score(0)
   - For frame t in [1, N-2]: max(motion_score(t-1), motion_score(t))
   - For frame N-1: motion_score(N-2)
3. Uniform Selection (K=6 frames from N=24):
   Select indices at regular intervals: np.round(np.linspace(0, N-1, K)).
4. Motion-Based Selection (K=6 frames from N=24):
   To avoid choosing only adjacent high-motion peak frames or duplicates, we partition
   the clip into K temporally contiguous segments (e.g. 24 // 6 = 4 frames per segment).
   In each segment, we select the frame with the maximum associated motion score.
   This guarantees temporal distribution across the entire stroke (stance -> backlift ->
   impact -> follow-through) while capturing the most dynamic instant within each phase.
"""

from pathlib import Path
import json
import time
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV (cv2) is not installed in the current Python environment.")
    sys.exit(1)


def decode_all_frames(video_path: Path):
    """Decode all frames from a video clip into a list of BGR numpy arrays."""
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


def compute_consecutive_motion_scores(frames):
    """
    Compute mean absolute difference (MAD) between consecutive grayscale frames.
    Returns:
        mad_scores: list of N-1 float values representing motion between frame t and t+1.
        frame_motion_energies: list of N float values representing motion around frame t.
    """
    n_frames = len(frames)
    if n_frames < 2:
        return [0.0], [0.0] * n_frames

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    mad_scores = []
    for t in range(n_frames - 1):
        diff = cv2.absdiff(gray_frames[t + 1], gray_frames[t])
        score = float(np.mean(diff))
        mad_scores.append(round(score, 4))

    # Assign each frame a representative motion energy:
    # Frame 0 gets mad[0], last frame gets mad[-1], intermediate frames get max of adjacent differences
    frame_motion_energies = []
    for t in range(n_frames):
        if t == 0:
            frame_motion_energies.append(mad_scores[0])
        elif t == n_frames - 1:
            frame_motion_energies.append(mad_scores[-1])
        else:
            frame_motion_energies.append(max(mad_scores[t - 1], mad_scores[t]))

    return mad_scores, frame_motion_energies


def select_uniform_frames(total_frames: int, k: int = 6) -> list:
    """Uniform selection: regular step spacing across the clip."""
    if total_frames <= k:
        return list(range(total_frames))
    indices = np.round(np.linspace(0, total_frames - 1, k)).astype(int).tolist()
    # Deduplicate while preserving order if any edge cases occur
    return sorted(list(dict.fromkeys(indices)))


def select_motion_frames(frame_energies: list, k: int = 6) -> list:
    """
    Segment-constrained peak motion selection:
    Divides the sequence of N frames into K non-overlapping temporal segments.
    Within each temporal segment, picks the frame index with the highest motion score.
    This ensures global temporal coverage across all shot phases while capturing local peaks.
    """
    n = len(frame_energies)
    if n <= k:
        return list(range(n))

    # Partition indices into k roughly equal contiguous segments
    segment_splits = np.array_split(np.arange(n), k)
    selected_indices = []

    for seg in segment_splits:
        if len(seg) == 0:
            continue
        # Pick index with maximum energy in this segment
        best_idx_in_seg = seg[np.argmax([frame_energies[i] for i in seg])]
        selected_indices.append(int(best_idx_in_seg))

    # Safeguard: ensure distinct frames and sorted order
    selected_indices = sorted(list(dict.fromkeys(selected_indices)))

    # If any duplicates dropped count below k, fill in using highest remaining scores
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_indices]
        remaining.sort(key=lambda idx: frame_energies[idx], reverse=True)
        needed = k - len(selected_indices)
        selected_indices.extend(remaining[:needed])
        selected_indices.sort()

    return selected_indices


def render_comparison_grid(frames, uniform_idx, motion_idx, clip_class, clip_name, out_path: Path):
    """
    Renders a 2-row x 6-column image grid:
    - Row 1: Uniformly selected frames with frame indices.
    - Row 2: Motion-selected frames with frame indices and motion scores.
    """
    # Downscale individual frame thumbnails for a compact, clean output image (e.g. 298x180)
    thumb_w, thumb_h = 280, 168
    grid_k = len(uniform_idx)
    banner_h = 45
    row_label_w = 140

    canvas_w = row_label_w + (thumb_w * grid_k) + ((grid_k + 1) * 8)
    canvas_h = banner_h + (2 * thumb_h) + 24
    canvas = np.full((canvas_h, canvas_w, 3), 28, dtype=np.uint8)

    # Top title banner
    title = f"Clip: {clip_class} / {clip_name}  |  Uniform vs Motion-Aware Selection (K={grid_k})"
    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

    rows = [
        ("Uniform", uniform_idx, (210, 170, 70)),      # Cyan-ish blue in BGR
        ("Motion-Aware", motion_idx, (60, 200, 110)), # Mint green in BGR
    ]

    for r_idx, (label, idx_list, color) in enumerate(rows):
        y_top = banner_h + 8 + (r_idx * (thumb_h + 8))

        # Row label background card
        cv2.rectangle(canvas, (8, y_top), (row_label_w - 6, y_top + thumb_h), (42, 42, 42), -1)
        cv2.putText(canvas, label, (14, y_top + thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"({len(idx_list)} frames)", (14, y_top + thumb_h // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)

        # Place thumbnails
        for col_idx, f_idx in enumerate(idx_list):
            x_left = row_label_w + (col_idx * (thumb_w + 8))
            orig_frame = frames[f_idx]
            thumb = cv2.resize(orig_frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            # Border
            cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), color, 2)

            # Overlay frame label badge
            badge_text = f"Frame #{f_idx}"
            cv2.rectangle(thumb, (4, 4), (110, 26), (20, 20, 20), -1)
            cv2.putText(thumb, badge_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            canvas[y_top : y_top + thumb_h, x_left : x_left + thumb_w] = thumb

    cv2.imwrite(str(out_path), canvas)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    sample_dir = base_dir / "outputs" / "sample_clips"
    comparisons_dir = base_dir / "outputs" / "motion_comparisons"
    reports_dir = base_dir / "reports"

    comparisons_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(sample_dir.rglob("*.avi"))
    if not video_files:
        print(f"[ERROR] No video clips found in {sample_dir}")
        sys.exit(1)

    print("=" * 96)
    print("MOTION-AWARE FRAME SELECTION ANALYSIS (CPU)")
    print("=" * 96)
    print(f"{'Class / Clip':<28} {'Uniform Indices':<22} {'Motion Indices':<22} {'Differs':<7} {'Proc Time (ms)':>14}")
    print("-" * 96)

    all_reports = []

    for vf in video_files:
        start_t = time.perf_counter()
        clip_class = vf.parent.name
        clip_name = vf.name

        frames = decode_all_frames(vf)
        total_frames = len(frames)

        mad_scores, frame_energies = compute_consecutive_motion_scores(frames)

        uniform_indices = select_uniform_frames(total_frames, k=6)
        motion_indices = select_motion_frames(frame_energies, k=6)

        proc_time_ms = round((time.perf_counter() - start_t) * 1000, 2)

        differs_from_uniform = (uniform_indices != motion_indices)

        # Render comparison image if it doesn't already exist (e.g. for newly sampled clips)
        safe_class_name = clip_class.replace(" ", "_")
        comp_img_name = f"comparison_{safe_class_name}_{vf.stem}.jpg"
        comp_img_path = comparisons_dir / comp_img_name
        if not comp_img_path.exists():
            render_comparison_grid(
                frames=frames,
                uniform_idx=uniform_indices,
                motion_idx=motion_indices,
                clip_class=clip_class,
                clip_name=clip_name,
                out_path=comp_img_path
            )

        clip_report = {
            "clip_name": clip_name,
            "class": clip_class,
            "relative_video_path": str(vf.relative_to(base_dir)).replace("\\", "/"),
            "total_frames": total_frames,
            "fps": 30.0,
            "processing_time_ms": proc_time_ms,
            "differs_from_uniform": differs_from_uniform,
            "consecutive_mad_scores": mad_scores,
            "frame_motion_energies": [round(e, 4) for e in frame_energies],
            "uniform_selected_indices": uniform_indices,
            "motion_selected_indices": motion_indices,
            "comparison_image": str(comp_img_path.relative_to(base_dir)).replace("\\", "/"),
        }
        all_reports.append(clip_report)

        u_str = str(uniform_indices)
        m_str = str(motion_indices)
        label = f"{clip_class}/{clip_name}"
        diff_str = "YES" if differs_from_uniform else "NO"
        print(f"{label:<28} {u_str:<22} {m_str:<22} {diff_str:<7} {proc_time_ms:>10.2f} ms")

    print("=" * 96)

    # Calculate aggregate summary metrics
    total_clips = len(all_reports)
    diff_count = sum(1 for r in all_reports if r["differs_from_uniform"])
    mean_proc_time = round(float(np.mean([r["processing_time_ms"] for r in all_reports])), 2)

    eval_summary = {
        "total_clips_evaluated": total_clips,
        "classes_represented": sorted(list(set(r["class"] for r in all_reports))),
        "num_clips_differing_from_uniform": diff_count,
        "percentage_differing_from_uniform": round((diff_count / total_clips) * 100, 2),
        "mean_cpu_processing_time_ms": mean_proc_time,
        "note": "Evaluated on CPU without ground-truth 3D annotations. Demonstrates frame-selection divergence without claiming 3D reconstruction accuracy.",
        "per_clip_evaluations": [
            {
                "clip": f"{r['class']}/{r['clip_name']}",
                "uniform_indices": r["uniform_selected_indices"],
                "motion_indices": r["motion_selected_indices"],
                "differs_from_uniform": r["differs_from_uniform"],
                "processing_time_ms": r["processing_time_ms"],
            }
            for r in all_reports
        ],
    }

    out_json = reports_dir / "motion_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    eval_json = reports_dir / "selection_evaluation.json"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\n[OK] Frame analysis saved to: {out_json.relative_to(base_dir)}")
    print(f"[OK] Evaluation summary saved to: {eval_json.relative_to(base_dir)}")
    print(f"Total Clips: {total_clips} | Differ from Uniform: {diff_count}/{total_clips} ({round((diff_count/total_clips)*100, 1)}%) | Mean CPU Time: {mean_proc_time} ms")


if __name__ == "__main__":
    main()
