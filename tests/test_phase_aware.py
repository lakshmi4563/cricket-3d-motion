import sys
from pathlib import Path
import pytest

# Ensure src/ is on path
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import mediapipe as mp
from evaluate_reconstruction import (
    decode_all_frames,
    extract_clip_poses_eval,
    compute_consecutive_motion_scores,
)
from phase_aware_selection import select_phase_aware_frames, select_hybrid_frames


@pytest.fixture(scope="session")
def detector():
    model_path = project_root / "models" / "pose_landmarker_lite.task"
    if not model_path.exists():
        pytest.skip(f"MediaPipe model not found at {model_path}")
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    yield detector
    detector.close()


def test_select_phase_aware_frames_real_clip_no_fallback(detector):
    """
    Calls select_phase_aware_frames on a real clip's poses (Cover Drive/vid100_41.avi).
    Verifies it returns exactly k unique sorted frames, including first (0) and last (N-1),
    WITHOUT using the fallback.
    """
    clip_path = project_root / "outputs" / "sample_clips" / "Cover Drive" / "vid100_41.avi"
    assert clip_path.exists(), f"Sample clip not found: {clip_path}"

    frames = decode_all_frames(clip_path)
    n = len(frames)
    assert n >= 6, f"Expected at least 6 frames, got {n}"

    poses = extract_clip_poses_eval(frames, detector)
    _, frame_energies = compute_consecutive_motion_scores(frames)

    for k in [4, 6, 8, 12]:
        # 1. Test call with (poses, frame_energies, k)
        selected, info = select_phase_aware_frames(poses, frame_energies, k=k)
        assert len(selected) == k, f"Expected {k} frames, got {len(selected)}"
        assert len(set(selected)) == k, "Selected frames must be unique"
        assert selected == sorted(selected), "Selected frames must be sorted"
        assert selected[0] == 0, "Must include first frame (0)"
        assert selected[-1] == n - 1, f"Must include last frame ({n - 1})"
        assert info.get("fallback_used") is False, f"Expected fallback_used=False for k={k}, got {info}"
        assert info.get("reliable") is True, f"Expected reliable=True for k={k}"

        # 2. Test call with (poses, k) - interface without explicit energies
        selected_no_energy, info_no_energy = select_phase_aware_frames(poses, k)
        assert len(selected_no_energy) == k
        assert len(set(selected_no_energy)) == k
        assert selected_no_energy[0] == 0
        assert selected_no_energy[-1] == n - 1
        assert info_no_energy.get("fallback_used") is False


def test_select_phase_aware_frames_unreliable_clip_fallback(detector):
    """
    Test a clip where phase detection is unreliable (Straight Drive/vid100_45.avi).
    Verifies that it gracefully falls back to hybrid selection without crashing,
    still returning exactly k unique sorted frames with first and last included.
    """
    clip_path = project_root / "outputs" / "sample_clips" / "Straight Drive" / "vid100_45.avi"
    assert clip_path.exists(), f"Sample clip not found: {clip_path}"

    frames = decode_all_frames(clip_path)
    n = len(frames)
    poses = extract_clip_poses_eval(frames, detector)
    _, frame_energies = compute_consecutive_motion_scores(frames)

    k = 6
    selected, info = select_phase_aware_frames(poses, frame_energies, k=k)
    assert len(selected) == k
    assert len(set(selected)) == k
    assert selected[0] == 0
    assert selected[-1] == n - 1
    assert info.get("fallback_used") is True, "Expected fallback_used=True for unreliable clip"
    assert info.get("selection_method") == "hybrid_fallback"


def test_select_hybrid_frames_with_integer_or_list():
    """
    Ensure select_hybrid_frames accepts either a list of energies or an integer frame count.
    """
    energies = [1.0, 2.0, 0.5, 4.0, 3.0, 1.5, 0.2, 5.0]
    n = len(energies)
    k = 4

    # With list of energies
    sel_list = select_hybrid_frames(energies, k=k)
    assert len(sel_list) == k
    assert sel_list[0] == 0
    assert sel_list[-1] == n - 1

    # With integer frame count n
    sel_int = select_hybrid_frames(n, k=k)
    assert len(sel_int) == k
    assert sel_int[0] == 0
    assert sel_int[-1] == n - 1
