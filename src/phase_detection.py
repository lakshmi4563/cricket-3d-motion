"""
Phase Detection for Cricket Batting Strokes (rule-based, no ML training).

Detects batting phases from MediaPipe landmark sequences:
    stance -> backlift -> downswing -> impact_region -> follow_through

Pipeline:
1. Determine dominant wrist (larger total path length or peak speed) —
   handles left- and right-handed batters automatically.
2. Smooth trajectories with Savitzky-Golay (window adaptive to clip length;
   falls back gracefully for short clips).
3. Interpolate short gaps where visibility is low; count how many frames
   were interpolated.
4. Compute wrist speed and elbow angle curves.
5. Label phase boundaries and return a per-clip reliability flag.

All outputs are pure Python / NumPy — no ML models, no GPU.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from scipy.signal import savgol_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (33-point model)
# ---------------------------------------------------------------------------
LEFT_WRIST  = 15
RIGHT_WRIST = 16
LEFT_ELBOW  = 13
RIGHT_ELBOW = 14
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP  = 23
RIGHT_HIP = 24
LEFT_KNEE  = 25
RIGHT_KNEE = 26
LEFT_ANKLE  = 27
RIGHT_ANKLE = 28

# Phase allocation weights (used downstream by phase_aware_selection)
PHASE_WEIGHTS = {
    "stance":         0.10,
    "backlift":       0.15,
    "downswing":      0.30,
    "impact_region":  0.30,
    "follow_through": 0.15,
}


# ---------------------------------------------------------------------------
# Helper: extract a single landmark coordinate array from pose sequence
# ---------------------------------------------------------------------------
def _extract_coord(
    poses: list[dict],
    landmark_idx: int,
    coord: str,  # "x", "y", or "z"
    vis_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one coordinate channel for one landmark across all frames.

    Returns
    -------
    values : shape (N,) – coordinate values (NaN where visibility is low)
    visibility : shape (N,) – per-frame visibility score
    """
    N = len(poses)
    values = np.full(N, np.nan)
    vis    = np.zeros(N)

    for t, pose in enumerate(poses):
        if not pose.get("detected", False):
            continue
        lms = pose.get("landmarks", [])
        if landmark_idx >= len(lms):
            continue
        pt = lms[landmark_idx]
        v  = float(pt.get("visibility", 0.0))
        vis[t] = v
        if v >= vis_threshold:
            values[t] = float(pt.get(coord, np.nan))

    return values, vis


# ---------------------------------------------------------------------------
# Helper: Savitzky-Golay smoothing with graceful fallback
# ---------------------------------------------------------------------------
def _smooth(arr: np.ndarray, clip_len: int) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to a 1-D array.
    Window is adaptive to clip_len:
        clip_len >= 13 → window=7, poly=3
        clip_len >= 9  → window=5, poly=2
        otherwise      → window=3, poly=1
    Falls back to a 3-point moving average if scipy is absent or the array
    is too short even for the minimal window.
    """
    n = len(arr)
    if n < 3:
        return arr.copy()

    if clip_len >= 13:
        win, poly = 7, 3
    elif clip_len >= 9:
        win, poly = 5, 2
    else:
        win, poly = 3, 1

    # Ensure window <= n and window is odd
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        win = 3
    poly = min(poly, win - 1)

    if _SCIPY_AVAILABLE and n >= win:
        try:
            # Fill NaN with linear interp for the filter, then restore NaN
            nan_mask = np.isnan(arr)
            filled = _interp_nans(arr.copy())
            smoothed = savgol_filter(filled, win, poly)
            smoothed[nan_mask] = np.nan
            return smoothed
        except Exception:
            pass  # Fall through to simple moving average

    # Simple 3-point moving average fallback
    result = arr.copy()
    for i in range(1, n - 1):
        result[i] = np.nanmean(arr[max(0, i-1):i+2])
    return result


# ---------------------------------------------------------------------------
# Helper: interpolate NaN gaps linearly
# ---------------------------------------------------------------------------
def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """
    Linear interpolation over NaN gaps in a 1-D array.
    Boundary NaNs are forward/backward filled from the nearest valid value.
    Returns a new array; does NOT modify arr in-place.
    """
    out = arr.copy()
    n = len(out)
    valid = np.where(~np.isnan(out))[0]
    if len(valid) == 0:
        return np.zeros(n)  # all NaN → zeros
    if len(valid) == n:
        return out

    # Forward fill left boundary
    if valid[0] > 0:
        out[:valid[0]] = out[valid[0]]
    # Backward fill right boundary
    if valid[-1] < n - 1:
        out[valid[-1]+1:] = out[valid[-1]]
    # Interpolate interior gaps
    nans = np.isnan(out)
    idx  = np.arange(n)
    out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])
    return out


def _count_interpolated(arr: np.ndarray) -> int:
    """Count how many positions were NaN before interpolation."""
    return int(np.sum(np.isnan(arr)))


# ---------------------------------------------------------------------------
# Dominant wrist detection
# ---------------------------------------------------------------------------
def detect_dominant_wrist(poses: list[dict]) -> tuple[int, int, str]:
    """
    Compare total path length of left and right wrists.

    Returns
    -------
    wrist_idx   : MediaPipe landmark index of the dominant wrist
    elbow_idx   : corresponding elbow landmark index
    side        : "left" or "right"
    """
    def _path_length(lm_idx: int) -> float:
        xs, _ = _extract_coord(poses, lm_idx, "x")
        ys, _ = _extract_coord(poses, lm_idx, "y")
        xs = _interp_nans(xs)
        ys = _interp_nans(ys)
        dx = np.diff(xs)
        dy = np.diff(ys)
        return float(np.sum(np.sqrt(dx**2 + dy**2)))

    path_left  = _path_length(LEFT_WRIST)
    path_right = _path_length(RIGHT_WRIST)

    if path_left >= path_right:
        return LEFT_WRIST, LEFT_ELBOW, "left"
    else:
        return RIGHT_WRIST, RIGHT_ELBOW, "right"


# ---------------------------------------------------------------------------
# Speed curve
# ---------------------------------------------------------------------------
def _wrist_speed(poses: list[dict], wrist_idx: int) -> np.ndarray:
    """
    Compute per-frame wrist speed (pixel-normalised units/frame) from
    smoothed 2D image landmarks.

    Returns array of shape (N,) – speed at each frame (frame 0 is 0).
    """
    N = len(poses)
    xs, _ = _extract_coord(poses, wrist_idx, "x")
    ys, _ = _extract_coord(poses, wrist_idx, "y")

    # Smooth then interpolate
    xs = _smooth(_interp_nans(xs), N)
    ys = _smooth(_interp_nans(ys), N)

    speed = np.zeros(N)
    dx = np.diff(xs)
    dy = np.diff(ys)
    speed[1:] = np.sqrt(dx**2 + dy**2)
    return speed


# ---------------------------------------------------------------------------
# Elbow angle curve
# ---------------------------------------------------------------------------
def _elbow_angle(
    poses: list[dict],
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
) -> np.ndarray:
    """
    Compute elbow flexion angle (degrees) from smoothed 2D image landmarks.
    Uses the law of cosines on shoulder-elbow-wrist triangle.
    Returns array shape (N,) with NaN where landmarks are missing.
    """
    N = len(poses)

    def _xy(lm_idx: int) -> tuple[np.ndarray, np.ndarray]:
        xs, _ = _extract_coord(poses, lm_idx, "x")
        ys, _ = _extract_coord(poses, lm_idx, "y")
        return _interp_nans(xs), _interp_nans(ys)

    sx, sy = _xy(shoulder_idx)
    ex, ey = _xy(elbow_idx)
    wx, wy = _xy(wrist_idx)

    sx = _smooth(sx, N); sy = _smooth(sy, N)
    ex = _smooth(ex, N); ey = _smooth(ey, N)
    wx = _smooth(wx, N); wy = _smooth(wy, N)

    angles = np.full(N, np.nan)
    for t in range(N):
        # Vectors: elbow→shoulder and elbow→wrist
        v1 = np.array([sx[t] - ex[t], sy[t] - ey[t]])
        v2 = np.array([wx[t] - ex[t], wy[t] - ey[t]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles[t] = float(np.degrees(np.arccos(cos_a)))

    return angles


# ---------------------------------------------------------------------------
# Ankle height (for front-foot stabilisation)
# ---------------------------------------------------------------------------
def _ankle_y(poses: list[dict], ankle_idx: int) -> np.ndarray:
    N = len(poses)
    ys, _ = _extract_coord(poses, ankle_idx, "y")
    return _smooth(_interp_nans(ys), N)


# ---------------------------------------------------------------------------
# Phase labelling (rule-based)
# ---------------------------------------------------------------------------
def _find_phase_boundaries(
    speed: np.ndarray,
    elbow_angle: np.ndarray,
    N: int,
) -> dict:
    """
    Rule-based phase segmentation from wrist speed and elbow angle.

    Rules (all indices are exclusive end-points):
        stance         : [0, backlift_start)
        backlift       : [backlift_start, downswing_start)
        downswing      : [downswing_start, impact_start)
        impact_region  : [impact_start, follow_start)
        follow_through : [follow_start, N)

    Heuristics
    ----------
    * peak_speed_frame  : argmax of speed
    * backlift_start    : first frame where speed exceeds 10% of max speed
                          (wrist starts rising)
    * downswing_start   : the local minimum of speed that precedes the peak
                          (bat reaches top of backlift, then accelerates)
    * impact_start      : peak_speed_frame - small window (impact zone)
    * follow_start      : first frame AFTER peak where speed drops below
                          50% of peak (bat deceleration)

    Returns dict with keys: phase_start, phase_end (inclusive), reliability.
    """
    smooth_speed = speed.copy()
    peak_frame = int(np.argmax(smooth_speed))
    peak_speed = float(smooth_speed[peak_frame])

    result = {
        "peak_speed_frame": peak_frame,
        "peak_speed_value": round(peak_speed, 6),
        "phases": {},
        "reliable": True,
        "reliability_note": "",
    }

    if peak_speed < 1e-7 or N < 4:
        result["reliable"] = False
        result["reliability_note"] = "Peak speed near zero or clip too short"
        return result

    threshold_10pct = 0.10 * peak_speed

    # --- backlift_start: first frame where speed > 10% of peak
    backlift_start = 0
    for t in range(N):
        if smooth_speed[t] >= threshold_10pct:
            backlift_start = t
            break

    # --- downswing_start: local minimum of speed between backlift_start and peak
    if peak_frame > backlift_start + 1:
        seg = smooth_speed[backlift_start:peak_frame]
        downswing_start = backlift_start + int(np.argmin(seg))
    else:
        downswing_start = backlift_start

    # Guard: downswing_start must be before peak
    if downswing_start >= peak_frame:
        downswing_start = max(0, peak_frame - 1)

    # --- impact_start: a window just before peak (≈ 1-2 frames)
    impact_window = max(1, N // 12)
    impact_start = max(downswing_start + 1, peak_frame - impact_window)

    # --- follow_start: after peak, speed drops to 50% of peak
    threshold_50pct = 0.50 * peak_speed
    follow_start = peak_frame + 1
    for t in range(peak_frame + 1, N):
        if smooth_speed[t] <= threshold_50pct:
            follow_start = t
            break
    else:
        follow_start = min(peak_frame + 2, N - 1)

    # Sanity-check ordering
    phases_ok = (
        0 <= backlift_start <= downswing_start <= impact_start <= follow_start < N
    )
    if not phases_ok:
        result["reliable"] = False
        result["reliability_note"] = "Phase boundary ordering failed — speed signal ambiguous"
        return result

    result["phases"] = {
        "stance":         {"start": 0,               "end": max(0, backlift_start - 1)},
        "backlift":       {"start": backlift_start,   "end": max(backlift_start, downswing_start - 1)},
        "downswing":      {"start": downswing_start,  "end": max(downswing_start, impact_start - 1)},
        "impact_region":  {"start": impact_start,     "end": min(follow_start, N - 1)},
        "follow_through": {"start": follow_start,     "end": N - 1},
    }

    # Flag uncertain if key phases are degenerate (0-length)
    for name, bounds in result["phases"].items():
        if bounds["end"] < bounds["start"]:
            result["phases"][name]["end"] = bounds["start"]  # clamp to single frame

    # Elbow angle check: if elbow data is all NaN → lower reliability but don't fail
    if np.all(np.isnan(elbow_angle)):
        result["reliability_note"] = "Elbow angle unavailable (landmark visibility); speed-only detection"

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_phases(poses: list[dict]) -> dict:
    """
    Main entry point: detect batting phases from a sequence of per-frame
    pose dicts (same format produced by evaluate_pose_temporal_quality.py).

    Parameters
    ----------
    poses : list of dicts, each with keys:
        "detected"  : bool
        "landmarks" : list of 33 dicts with x, y, z, visibility

    Returns
    -------
    dict with keys:
        "dominant_wrist_side"    : "left" | "right"
        "dominant_wrist_idx"     : int (MediaPipe index)
        "n_frames"               : int
        "n_interpolated_frames"  : int
        "peak_speed_frame"       : int | None
        "phases"                 : dict (see below) — empty if uncertain
        "reliable"               : bool
        "reliability_note"       : str
        "wrist_speed"            : list[float]
        "elbow_angle"            : list[float|None]
        "phase_weights"          : dict  (PHASE_WEIGHTS constant)

    phases dict structure (if reliable):
        {
          "stance":         {"start": int, "end": int},
          "backlift":       {"start": int, "end": int},
          "downswing":      {"start": int, "end": int},
          "impact_region":  {"start": int, "end": int},
          "follow_through": {"start": int, "end": int},
        }
    If detection is uncertain, phases == {} and reliable == False.
    """
    N = len(poses)
    if N == 0:
        return {
            "dominant_wrist_side": "unknown",
            "dominant_wrist_idx": LEFT_WRIST,
            "n_frames": 0,
            "n_interpolated_frames": 0,
            "peak_speed_frame": None,
            "phases": {},
            "reliable": False,
            "reliability_note": "Empty pose sequence",
            "wrist_speed": [],
            "elbow_angle": [],
            "phase_weights": PHASE_WEIGHTS,
        }

    # Detect dominant wrist
    wrist_idx, elbow_idx, side = detect_dominant_wrist(poses)
    shoulder_idx = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER

    # Count interpolated frames before smoothing
    raw_x, _ = _extract_coord(poses, wrist_idx, "x")
    n_interp  = _count_interpolated(raw_x)

    # Compute curves
    speed       = _wrist_speed(poses, wrist_idx)
    elbow_angle = _elbow_angle(poses, shoulder_idx, elbow_idx, wrist_idx)

    # Phase boundaries
    phase_result = _find_phase_boundaries(speed, elbow_angle, N)

    # Build clean elbow_angle list for output (None where NaN)
    elbow_out = [
        None if np.isnan(a) else round(float(a), 2)
        for a in elbow_angle
    ]

    return {
        "dominant_wrist_side":   side,
        "dominant_wrist_idx":    wrist_idx,
        "n_frames":              N,
        "n_interpolated_frames": n_interp,
        "peak_speed_frame":      phase_result["peak_speed_frame"],
        "peak_speed_value":      phase_result.get("peak_speed_value"),
        "phases":                phase_result.get("phases", {}),
        "reliable":              phase_result["reliable"],
        "reliability_note":      phase_result.get("reliability_note", ""),
        "wrist_speed":           [round(float(s), 6) for s in speed],
        "elbow_angle":           elbow_out,
        "phase_weights":         PHASE_WEIGHTS,
    }

