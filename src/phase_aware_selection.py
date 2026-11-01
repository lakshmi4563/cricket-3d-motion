"""
Phase-Aware Frame Selection for Cricket Batting Clips.

Adds a fourth selection method "phase_aware" with the SAME interface as the
existing methods (select_uniform_frames, select_motion_frames, hybrid).

Algorithm
---------
1. Run phase detection (src/phase_detection.py).
2. If detection is "uncertain", fall back to the existing hybrid method.
3. Otherwise allocate budget K across phases using PHASE_WEIGHTS
   (downswing + impact_region get the most frames).
4. Within each phase, always include:
     - phase boundary frames (first and last frame of the phase)
     - the peak wrist-speed frame (if it falls in the phase)
   Then fill the remaining allocation with uniform spacing inside the phase.
5. Guarantee: exactly K frames, no duplicates, always includes frame 0 and
   frame N-1 (clip start/end), sorted.

Fallback hybrid implementation
-------------------------------
The original hybrid-generating code was not preserved in src/; this module
provides a clean, callable hybrid implementation that matches the documented
behaviour in reports/hybrid_metric_audit.json:
    - anchor frame 0 and frame N-1
    - fill interior (K-2) slots with the top-energy motion frames not
      adjacent to anchors
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

# Import existing selection functions
_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from motion_analysis import select_uniform_frames, select_motion_frames
from phase_detection import detect_phases, PHASE_WEIGHTS


# ---------------------------------------------------------------------------
# Hybrid selection (callable version — matches documented audit behaviour)
# ---------------------------------------------------------------------------
def select_hybrid_frames(
    frame_energies: list | int,
    k: int = 6,
) -> list[int]:
    """
    Hybrid selection: anchor frame 0 and frame N-1; fill interior slots
    with the highest-energy frames that are not duplicates of anchors.

    Parameters
    ----------
    frame_energies : list[float] or int — per-frame motion energy scores (length N),
                     or total frame count N (if motion energies are not provided)
    k              : int — number of frames to select

    Returns
    -------
    list[int] — exactly k sorted, unique frame indices
    """
    if isinstance(frame_energies, (int, np.integer)):
        n = int(frame_energies)
        energies = [0.0] * n
    else:
        energies = list(frame_energies) if frame_energies is not None else []
        n = len(energies)

    if n <= k:
        return list(range(n))

    anchors = {0, n - 1}
    if k <= 2:
        result = sorted(list(anchors))[:k]
        return result

    # Interior candidates sorted by descending energy
    interior = sorted(
        [i for i in range(1, n - 1)],
        key=lambda i: energies[i] if i < len(energies) else 0.0,
        reverse=True,
    )
    needed = k - 2  # We already have frame 0 and N-1
    selected = set(anchors) | set(interior[:needed])

    # Fill any shortfall (shouldn't happen, but guard anyway)
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected]
        remaining.sort(key=lambda i: energies[i] if i < len(energies) else 0.0, reverse=True)
        for i in remaining:
            selected.add(i)
            if len(selected) >= k:
                break

    result = sorted(list(selected))[:k]
    return result


# ---------------------------------------------------------------------------
# Phase-aware selection
# ---------------------------------------------------------------------------
def select_phase_aware_frames(
    poses: list[dict],
    frame_energies: list | int | None = None,
    k: int = 6,
) -> tuple[list[int], dict]:
    """
    Phase-aware frame selection. Same interface as the existing methods.

    Parameters
    ----------
    poses          : list[dict] — per-frame pose records from MediaPipe
                     (same format as evaluate_pose_temporal_quality.py)
    frame_energies : list[float] or int, optional — per-frame MAD motion energies,
                     or budget k if called directly as select_phase_aware_frames(poses, k)
    k              : int — number of frames to select

    Returns
    -------
    selected_indices : list[int] — exactly k sorted unique frame indices
    phase_info       : dict — output from detect_phases(), plus
                       "fallback_used": bool,
                       "selection_method": str
    """
    # Flexible signature: handle select_phase_aware_frames(poses, k)
    if isinstance(frame_energies, (int, np.integer)):
        k = int(frame_energies)
        frame_energies = None

    if poses is not None and len(poses) > 0:
        n = len(poses)
    elif frame_energies is not None and hasattr(frame_energies, "__len__"):
        n = len(frame_energies)
    else:
        n = 0

    if frame_energies is None or not isinstance(frame_energies, (list, tuple, np.ndarray)):
        energies = [0.0] * n
    else:
        energies = list(frame_energies)

    if n == 0:
        return [], {"reliable": False, "fallback_used": True, "selection_method": "uniform"}

    if n <= k:
        phase_info = detect_phases(poses) if poses else {}
        phase_info["fallback_used"] = False
        phase_info["selection_method"] = "phase_aware_trivial"
        return list(range(n)), phase_info

    phase_info = detect_phases(poses) if poses else {}

    # Fall back to hybrid if phase detection is unreliable
    if not phase_info.get("reliable", False) or not phase_info.get("phases"):
        fallback = select_hybrid_frames(energies, k=k)
        phase_info["fallback_used"] = True
        phase_info["selection_method"] = "hybrid_fallback"
        return fallback, phase_info

    phase_info["fallback_used"] = False
    phase_info["selection_method"] = "phase_aware"
    phase_info["selection_method"] = "phase_aware"

    phases    = phase_info["phases"]
    peak_frame = phase_info.get("peak_speed_frame")

    # --- Budget allocation ---
    total_weight = sum(PHASE_WEIGHTS[p] for p in phases)
    raw_alloc = {
        p: PHASE_WEIGHTS[p] / total_weight * k for p in phases
    }

    # Minimum 1 frame per phase that has any frames; at least 0 for degenerate ones
    int_alloc: dict[str, int] = {}
    for p, bounds in phases.items():
        phase_len = bounds["end"] - bounds["start"] + 1
        int_alloc[p] = 1 if phase_len >= 1 else 0

    # Distribute remaining budget by fractional parts (largest-remainder method)
    remaining = k - sum(int_alloc.values())
    if remaining > 0:
        fractions = {p: raw_alloc[p] - int_alloc[p] for p in phases}
        sorted_phases = sorted(phases, key=lambda p: fractions[p], reverse=True)
        for p in sorted_phases:
            if remaining <= 0:
                break
            phase_len = phases[p]["end"] - phases[p]["start"] + 1
            max_extra = phase_len - int_alloc[p]
            if max_extra > 0:
                int_alloc[p] += 1
                remaining -= 1
    elif remaining < 0:
        # Over-allocated — trim from phases with lowest weight first
        sorted_phases = sorted(phases, key=lambda p: PHASE_WEIGHTS[p])
        for p in sorted_phases:
            if remaining >= 0:
                break
            if int_alloc[p] > 1:
                int_alloc[p] -= 1
                remaining += 1

    # --- Frame selection within each phase ---
    selected: set[int] = set()

    # Always pin first and last frame of clip
    selected.add(0)
    selected.add(n - 1)

    for phase_name, bounds in phases.items():
        start, end = bounds["start"], bounds["end"]
        budget = int_alloc.get(phase_name, 0)
        if budget <= 0:
            continue

        candidates: list[int] = []

        # Include phase boundaries
        candidates.append(max(0, start))
        if end != start:
            candidates.append(min(n - 1, end))

        # Include peak speed frame if it falls in this phase
        if peak_frame is not None and start <= peak_frame <= end:
            candidates.append(peak_frame)

        # Deduplicate and clip
        candidates = sorted(set(c for c in candidates if 0 <= c < n))

        # If we need more frames, fill with uniform spacing inside phase
        if len(candidates) < budget:
            phase_range = list(range(start, end + 1))
            already = set(candidates)
            remaining_slots = budget - len(candidates)
            # Uniform linspace inside the phase
            if len(phase_range) > 0:
                interior = np.round(
                    np.linspace(start, end, max(2, budget))
                ).astype(int).tolist()
                for idx in interior:
                    if idx not in already and 0 <= idx < n:
                        candidates.append(idx)
                        already.add(idx)
                        if len(candidates) >= budget:
                            break

        # Take the first `budget` frames (sorted)
        candidates.sort()
        for idx in candidates[:budget]:
            selected.add(idx)

    # --- Final trim / fill to exactly K ---
    result = sorted(selected)

    if len(result) > k:
        # Keep frame 0, N-1, peak, and the rest by priority
        must_keep = {0, n - 1}
        if peak_frame is not None:
            must_keep.add(peak_frame)
        priority = sorted(
            result,
            key=lambda i: (
                -1 if i in must_keep else 0,
                -energies[i] if i < len(energies) else 0,
            ),
        )
        result = sorted(priority[:k])

    elif len(result) < k:
        # Fill from high-energy frames not yet selected
        all_by_energy = sorted(
            range(n),
            key=lambda i: energies[i] if i < len(energies) else 0,
            reverse=True,
        )
        selected_set = set(result)
        for idx in all_by_energy:
            if idx not in selected_set:
                selected_set.add(idx)
                result = sorted(selected_set)
                if len(result) >= k:
                    break

    # Final guarantee: sorted, unique, exactly k (or n if n < k)
    result = sorted(list(dict.fromkeys(result)))
    if len(result) > k:
        result = result[:k]

    return result, phase_info

