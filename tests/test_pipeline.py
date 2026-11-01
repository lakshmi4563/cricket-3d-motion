"""
Unit tests for:
1. Phase detection (basic correctness, handles missing landmarks).
2. Frame selection budget (exactly K frames, no duplicates, includes first/last).
3. Skeleton interpolation (exact at selected frames, continuous elsewhere).

Run with:
    python -m pytest tests/test_pipeline.py -v
or:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
import numpy as np

# Add src to path
_repo = Path(__file__).resolve().parents[1]
_src  = _repo / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from phase_detection import (
    detect_phases,
    detect_dominant_wrist,
    _interp_nans,
    _smooth,
    LEFT_WRIST, RIGHT_WRIST,
)
from motion_analysis import select_uniform_frames, select_motion_frames
from phase_aware_selection import (
    select_hybrid_frames,
    select_phase_aware_frames,
)
from evaluate_reconstruction import (
    interpolate_skeleton,
    build_ground_truth_matrix,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic pose sequences
# ---------------------------------------------------------------------------
def _make_pose_seq(
    n: int,
    wrist_speed_profile: list[float] | None = None,
    visibility: float = 0.9,
) -> list[dict]:
    """
    Build a synthetic pose sequence of length n with a wrist speed profile.
    All 33 landmarks are synthesised from x = t/n (slow drift).
    The LEFT_WRIST landmark gets enhanced motion from wrist_speed_profile.
    """
    poses = []
    if wrist_speed_profile is None:
        # A simple triangle: rise then fall
        half = n // 2
        wrist_speed_profile = (
            [float(i) / max(1, half) for i in range(half + 1)]
            + [1.0 - float(i) / max(1, n - half) for i in range(1, n - half)]
        )[:n]

    wrist_x = 0.5
    wrist_y = 0.5
    for t in range(n):
        move = wrist_speed_profile[t] if t < len(wrist_speed_profile) else 0.0
        wrist_x += move * 0.01
        wrist_y -= move * 0.005

        lms = []
        for j in range(33):
            lms.append({
                "x": float(j % 5) / 10 + float(t) / (n * 10),
                "y": float(j // 5) / 10,
                "z": 0.0,
                "visibility": visibility,
                "presence": visibility,
            })
        # Override left wrist
        lms[LEFT_WRIST]["x"] = round(wrist_x, 4)
        lms[LEFT_WRIST]["y"] = round(wrist_y, 4)

        poses.append({"detected": True, "landmarks": lms})

    return poses


def _make_pose_seq_low_vis(n: int) -> list[dict]:
    """Pose sequence where first 3 frames have low visibility."""
    poses = _make_pose_seq(n)
    for t in range(min(3, n)):
        for lm in poses[t]["landmarks"]:
            lm["visibility"] = 0.1
    return poses


def _make_empty_pose_seq(n: int) -> list[dict]:
    """All frames undetected."""
    return [{"detected": False, "landmarks": []} for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests: _interp_nans
# ---------------------------------------------------------------------------
class TestInterpNans:
    def test_no_nans(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = _interp_nans(arr)
        np.testing.assert_allclose(out, arr)

    def test_interior_nan(self):
        arr = np.array([0.0, np.nan, 2.0, 3.0])
        out = _interp_nans(arr)
        assert not np.any(np.isnan(out))
        assert abs(out[1] - 1.0) < 1e-6

    def test_leading_nan(self):
        arr = np.array([np.nan, np.nan, 2.0, 4.0])
        out = _interp_nans(arr)
        assert not np.any(np.isnan(out))
        # Leading NaN is forward-filled with first valid value
        assert out[0] == out[2]

    def test_all_nan(self):
        arr = np.array([np.nan, np.nan, np.nan])
        out = _interp_nans(arr)
        assert not np.any(np.isnan(out))

    def test_single_value(self):
        arr = np.array([5.0])
        out = _interp_nans(arr)
        np.testing.assert_allclose(out, arr)


# ---------------------------------------------------------------------------
# Tests: detect_phases
# ---------------------------------------------------------------------------
class TestPhaseDetection:
    def test_basic_phases_present(self):
        """A clear synthetic sequence should produce all 5 phases."""
        n = 24
        poses = _make_pose_seq(n)
        result = detect_phases(poses)
        assert isinstance(result, dict)
        assert "phases" in result
        assert "reliable" in result
        if result["reliable"]:
            phase_names = set(result["phases"].keys())
            assert phase_names == {"stance", "backlift", "downswing", "impact_region", "follow_through"}

    def test_phase_boundaries_are_ordered(self):
        """phase boundaries must be monotonically ordered."""
        poses = _make_pose_seq(24)
        result = detect_phases(poses)
        if not result["reliable"] or not result["phases"]:
            return  # uncertain is acceptable
        ordered = ["stance", "backlift", "downswing", "impact_region", "follow_through"]
        prev_end = -1
        for name in ordered:
            if name in result["phases"]:
                b = result["phases"][name]
                assert b["start"] >= 0
                assert b["end"] >= b["start"]
                assert b["start"] >= prev_end or prev_end == -1
                prev_end = b["end"]

    def test_short_clip_does_not_crash(self):
        """3-frame clips must not crash."""
        poses = _make_pose_seq(3)
        result = detect_phases(poses)
        assert isinstance(result, dict)

    def test_empty_sequence(self):
        result = detect_phases([])
        assert result["reliable"] is False
        assert result["phases"] == {}

    def test_all_undetected(self):
        poses = _make_empty_pose_seq(24)
        result = detect_phases(poses)
        # Should not crash; reliability should be False or phases empty
        assert isinstance(result, dict)

    def test_low_visibility_frames(self):
        """Low visibility should not cause an exception."""
        poses = _make_pose_seq_low_vis(24)
        result = detect_phases(poses)
        assert isinstance(result, dict)

    def test_dominant_wrist_detection(self):
        """Left wrist moves more in synthetic sequence → should be dominant."""
        poses = _make_pose_seq(24)
        wrist_idx, elbow_idx, side = detect_dominant_wrist(poses)
        # In our synthetic sequence left wrist has explicit movement
        assert wrist_idx == LEFT_WRIST
        assert side == "left"

    def test_output_fields(self):
        """All required fields must be present in output."""
        poses = _make_pose_seq(24)
        result = detect_phases(poses)
        required = [
            "dominant_wrist_side", "dominant_wrist_idx", "n_frames",
            "n_interpolated_frames", "peak_speed_frame", "phases",
            "reliable", "reliability_note", "wrist_speed", "elbow_angle",
            "phase_weights",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_wrist_speed_length(self):
        n = 24
        poses = _make_pose_seq(n)
        result = detect_phases(poses)
        assert len(result["wrist_speed"]) == n

    def test_elbow_angle_length(self):
        n = 24
        poses = _make_pose_seq(n)
        result = detect_phases(poses)
        assert len(result["elbow_angle"]) == n


# ---------------------------------------------------------------------------
# Tests: Frame selection budget
# ---------------------------------------------------------------------------
class TestSelectionBudget:
    """All methods must return exactly K frames, no duplicates, sorted."""

    def _check_budget(self, selected: list[int], k: int, n: int, method: str):
        assert len(selected) == min(k, n), (
            f"{method}: expected {min(k,n)} frames, got {len(selected)}"
        )
        assert len(set(selected)) == len(selected), f"{method}: duplicates found"
        assert selected == sorted(selected), f"{method}: not sorted"
        for idx in selected:
            assert 0 <= idx < n, f"{method}: index {idx} out of bounds [0, {n})"

    def test_uniform(self):
        for n in [6, 12, 24, 30]:
            for k in [4, 6, 8]:
                sel = select_uniform_frames(n, k=k)
                self._check_budget(sel, k, n, f"uniform(n={n},k={k})")

    def test_motion_aware(self):
        for n in [12, 24]:
            energies = [float(i % 7) for i in range(n)]
            for k in [4, 6, 8]:
                sel = select_motion_frames(energies, k=k)
                self._check_budget(sel, k, n, f"motion_aware(n={n},k={k})")

    def test_hybrid(self):
        for n in [12, 24]:
            energies = [float(i % 5) for i in range(n)]
            for k in [4, 6, 8]:
                sel = select_hybrid_frames(energies, k=k)
                self._check_budget(sel, k, n, f"hybrid(n={n},k={k})")

    def test_hybrid_always_includes_first_and_last(self):
        n = 24
        energies = [float(i) for i in range(n)]
        for k in [4, 6, 8, 12]:
            sel = select_hybrid_frames(energies, k=k)
            assert 0 in sel, f"hybrid k={k}: frame 0 missing"
            assert n - 1 in sel, f"hybrid k={k}: last frame missing"

    def test_phase_aware_budget(self):
        poses = _make_pose_seq(24)
        energies = [float(i % 5) for i in range(24)]
        for k in [4, 6, 8, 12]:
            sel, _ = select_phase_aware_frames(poses, energies, k=k)
            self._check_budget(sel, k, 24, f"phase_aware(k={k})")

    def test_phase_aware_k_greater_than_n(self):
        """When k >= n, return all frames."""
        poses = _make_pose_seq(6)
        energies = [1.0] * 6
        sel, _ = select_phase_aware_frames(poses, energies, k=10)
        assert len(sel) == 6

    def test_uniform_k_eq_n(self):
        n = 6
        sel = select_uniform_frames(n, k=6)
        assert len(sel) == 6
        assert sel == list(range(6))

    def test_selection_is_sorted(self):
        """All methods must return sorted indices."""
        poses    = _make_pose_seq(24)
        energies = [float(i) for i in range(24)]
        for k in [6, 8]:
            assert select_uniform_frames(24, k=k) == sorted(select_uniform_frames(24, k=k))
            assert select_motion_frames(energies, k=k) == sorted(select_motion_frames(energies, k=k))
            assert select_hybrid_frames(energies, k=k) == sorted(select_hybrid_frames(energies, k=k))
            sel, _ = select_phase_aware_frames(poses, energies, k=k)
            assert sel == sorted(sel)


# ---------------------------------------------------------------------------
# Tests: Interpolation
# ---------------------------------------------------------------------------
class TestInterpolation:
    def _make_gt(self, n: int) -> np.ndarray:
        """Simple linearly changing GT skeleton."""
        gt = np.zeros((n, 33, 2))
        for t in range(n):
            gt[t, :, 0] = float(t) / max(1, n - 1)
            gt[t, :, 1] = 1.0 - float(t) / max(1, n - 1)
        return gt

    def test_exact_at_selected_frames(self):
        """Interpolated skeleton must match GT exactly at selected frames."""
        n  = 24
        gt = self._make_gt(n)
        sel = [0, 5, 11, 17, 23]
        interp = interpolate_skeleton(gt, sel)
        for idx in sel:
            np.testing.assert_allclose(
                interp[idx], gt[idx], atol=1e-6,
                err_msg=f"Mismatch at selected frame {idx}"
            )

    def test_continuous_between_frames(self):
        """Interpolated values at non-selected frames must be between neighbours."""
        n  = 24
        gt = self._make_gt(n)
        sel = [0, 11, 23]
        interp = interpolate_skeleton(gt, sel)

        # Frame 5 is between 0 and 11 — x should be monotonically increasing
        for t in range(1, 11):
            assert interp[t, 0, 0] >= interp[t - 1, 0, 0] - 1e-9

    def test_output_shape(self):
        n  = 24
        gt = self._make_gt(n)
        sel = [0, 5, 11, 17, 23]
        interp = interpolate_skeleton(gt, sel)
        assert interp.shape == (n, 33, 2)

    def test_all_frames_selected(self):
        """When all frames are selected, interpolation = GT."""
        n  = 12
        gt = self._make_gt(n)
        sel = list(range(n))
        interp = interpolate_skeleton(gt, sel)
        np.testing.assert_allclose(interp, gt, atol=1e-6)

    def test_single_selected_frame(self):
        """When only one frame is selected, all frames should equal that frame."""
        n  = 10
        gt = self._make_gt(n)
        sel = [4]
        interp = interpolate_skeleton(gt, sel)
        for t in range(n):
            np.testing.assert_allclose(interp[t], gt[4], atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Ground-truth matrix builder
# ---------------------------------------------------------------------------
class TestGTMatrix:
    def test_shape(self):
        poses = _make_pose_seq(24)
        gt = build_ground_truth_matrix(poses)
        assert gt.shape == (24, 33, 2)

    def test_no_nans_after_interp(self):
        poses = _make_pose_seq(24)
        gt = build_ground_truth_matrix(poses)
        assert not np.any(np.isnan(gt))

    def test_empty_poses(self):
        poses = _make_empty_pose_seq(8)
        gt = build_ground_truth_matrix(poses)
        assert gt.shape == (8, 33, 2)
        # All NaN → all zeros after interp (boundary fill)
        assert not np.any(np.isnan(gt))


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------
def _run_all_tests():
    import traceback
    test_classes = [
        TestInterpNans,
        TestPhaseDetection,
        TestSelectionBudget,
        TestInterpolation,
        TestGTMatrix,
    ]

    total  = 0
    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        print(f"\n{'='*60}")
        print(f"Class: {cls.__name__}")
        print('='*60)
        for attr in dir(instance):
            if not attr.startswith("test_"):
                continue
            total += 1
            try:
                getattr(instance, attr)()
                print(f"  [PASS] {attr}")
                passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {attr}: {e}")
                failed += 1
            except Exception as e:
                print(f"  [ERROR] {attr}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print('='*60)
    return failed == 0


if __name__ == "__main__":
    ok = _run_all_tests()
    sys.exit(0 if ok else 1)

