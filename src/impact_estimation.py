"""
Impact Frame Estimation from Body Motion (no ball tracking, no camera model).

Estimates the cricket impact frame using wrist speed, elbow extension,
and front-foot stabilisation signals from MediaPipe body landmarks.

Strategy
--------
1. Compute dominant wrist speed curve (smoothed).
2. Find the wrist-speed peak in the downswing phase (if phase info is
   available) or globally.
3. Cross-check with elbow extension: after impact the elbow tends to extend.
4. Cross-check with front-foot stabilisation: ankle-y variance drops near
   impact.
5. Combine signals into a confidence score; return "uncertain" when they
   disagree or signals are too weak.

Limitations
-----------
- No ball in the scene; estimation is from body motion only.
- Confidence scores are heuristic — not calibrated probabilities.
- Short clips (< 8 frames) are flagged uncertain.
- Do not interpret confidence as a binary correct/incorrect classifier.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from phase_detection import (
    detect_phases,
    _wrist_speed,
    _elbow_angle,
    _ankle_y,
    _interp_nans,
    _smooth,
    _extract_coord,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_impact_frame(
    poses: list[dict],
    phase_info: dict | None = None,
) -> dict:
    """
    Estimate the impact frame from body motion only.

    Parameters
    ----------
    poses       : list[dict] — per-frame MediaPipe pose records
    phase_info  : dict | None — output from detect_phases() for this clip.
                  If None, detect_phases() is called internally.

    Returns
    -------
    dict with keys:
        "estimated_impact_frame" : int | None
        "confidence"             : float in [0, 1]
        "uncertain"              : bool — True if signals disagree or are weak
        "method_note"            : str
        "wrist_speed_peak"       : int | None
        "elbow_extension_frame"  : int | None
        "foot_stable_frame"      : int | None
        "signal_agreement"       : str  ("high" | "medium" | "low")
    """
    N = len(poses)
    _uncertain = {
        "estimated_impact_frame": None,
        "confidence": 0.0,
        "uncertain": True,
        "method_note": "",
        "wrist_speed_peak": None,
        "elbow_extension_frame": None,
        "foot_stable_frame": None,
        "signal_agreement": "low",
    }

    if N < 4:
        return {**_uncertain, "method_note": f"Clip too short ({N} frames)"}

    # --- Phase info ---
    if phase_info is None:
        phase_info = detect_phases(poses)

    side = phase_info.get("dominant_wrist_side", "left")
    wrist_idx   = LEFT_WRIST  if side == "left" else RIGHT_WRIST
    elbow_idx   = LEFT_ELBOW  if side == "left" else RIGHT_ELBOW
    shoulder_idx = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    ankle_idx   = LEFT_ANKLE  if side == "left" else RIGHT_ANKLE

    # --- Wrist speed signal ---
    speed = _wrist_speed(poses, wrist_idx)

    # Restrict search to downswing + impact region if available
    phases = phase_info.get("phases", {})
    search_start = 0
    search_end   = N - 1

    if phases and phase_info.get("reliable", False):
        ds = phases.get("downswing", {})
        ir = phases.get("impact_region", {})
        if ds:
            search_start = ds.get("start", 0)
        if ir:
            search_end = ir.get("end", N - 1)

    search_speed = speed[search_start:search_end + 1]
    if len(search_speed) == 0 or np.max(search_speed) < 1e-7:
        return {**_uncertain, "method_note": "Wrist speed near zero in search window"}

    wrist_peak_local = int(np.argmax(search_speed))
    wrist_peak_frame = search_start + wrist_peak_local

    # Verify deceleration after peak (needed for impact signal)
    if wrist_peak_frame < N - 1:
        post_peak = speed[wrist_peak_frame:min(wrist_peak_frame + 4, N)]
        has_decel = len(post_peak) > 1 and float(post_peak[-1]) < float(post_peak[0]) * 0.8
    else:
        has_decel = False

    if not has_decel and not phases:
        return {
            **_uncertain,
            "method_note": "No clear deceleration after speed peak; uncertain",
            "wrist_speed_peak": wrist_peak_frame,
        }

    # --- Elbow extension signal ---
    elbow_angles = _elbow_angle(poses, shoulder_idx, elbow_idx, wrist_idx)
    elbow_ext_frame = None

    if not np.all(np.isnan(elbow_angles)):
        # Look for the local maximum of elbow angle after peak speed
        # (arm extends at/after impact)
        window_start = max(0, wrist_peak_frame - 2)
        window_end   = min(N - 1, wrist_peak_frame + 5)
        win_angles   = elbow_angles[window_start:window_end + 1]
        valid_mask   = ~np.isnan(win_angles)
        if np.any(valid_mask):
            local_max_local = int(
                np.nanargmax(np.where(valid_mask, win_angles, -999))
            )
            elbow_ext_frame = window_start + local_max_local

    # --- Foot stabilisation signal ---
    ankle_ys = _ankle_y(poses, ankle_idx)
    foot_stable_frame = None

    # Stable foot: variance over a sliding 3-frame window is minimal
    min_var = np.inf
    for t in range(1, N - 1):
        local_var = float(np.var(ankle_ys[max(0, t-1):t+2]))
        if local_var < min_var:
            min_var     = local_var
            foot_stable_frame = t

    # --- Combine signals ---
    candidate_frames = [f for f in [wrist_peak_frame, elbow_ext_frame, foot_stable_frame] if f is not None]
    if not candidate_frames:
        return {**_uncertain, "method_note": "No candidate impact frames found"}

    # Weighted average (speed peak has highest weight)
    weights = []
    frames  = []
    if wrist_peak_frame is not None:
        frames.append(wrist_peak_frame);  weights.append(3.0)
    if elbow_ext_frame is not None:
        frames.append(elbow_ext_frame);   weights.append(1.5)
    if foot_stable_frame is not None:
        frames.append(foot_stable_frame); weights.append(1.0)

    weighted_frame = int(round(np.average(frames, weights=weights)))
    weighted_frame = max(0, min(N - 1, weighted_frame))

    # Signal agreement
    spread = max(candidate_frames) - min(candidate_frames)
    if spread <= 2:
        agreement = "high"
    elif spread <= 5:
        agreement = "medium"
    else:
        agreement = "low"

    # Confidence heuristic
    # Signals add up, but low agreement hard-caps at 0.30 to reflect that the
    # weighted-average frame may be far from any actual signal peak.
    conf = 0.0
    if has_decel:
        conf += 0.50
    if elbow_ext_frame is not None:
        conf += 0.25
    if foot_stable_frame is not None:
        conf += 0.10
    if agreement == "high":
        conf += 0.15
    elif agreement == "medium":
        conf += 0.07

    conf = min(1.0, conf)
    # A5: low agreement overrides — cap confidence and force uncertain
    if agreement == "low":
        conf = min(conf, 0.30)
    uncertain = (agreement == "low") or (conf < 0.40) or (not has_decel)

    method_note = (
        f"Speed-peak frame {wrist_peak_frame}"
        + (f"; elbow ext frame {elbow_ext_frame}" if elbow_ext_frame is not None else "")
        + (f"; foot stable frame {foot_stable_frame}" if foot_stable_frame is not None else "")
        + f"; agreement={agreement}"
    )

    return {
        "estimated_impact_frame":  weighted_frame,
        "confidence":              round(conf, 3),
        "uncertain":               uncertain,
        "method_note":             method_note,
        "wrist_speed_peak":        wrist_peak_frame,
        "elbow_extension_frame":   elbow_ext_frame,
        "foot_stable_frame":       foot_stable_frame,
        "signal_agreement":        agreement,
    }

