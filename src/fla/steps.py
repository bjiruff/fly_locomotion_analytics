from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .constants import FPS, PX_PER_MM


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_durations(binary_series: np.ndarray, target: int) -> List[float]:
    """Durations (ms) of all consecutive runs of `target` in a 1D binary array."""
    durations = []
    count = 0
    inside = False
    for i in range(1, len(binary_series)):
        if not inside and binary_series[i - 1] != target and binary_series[i] == target:
            inside = True
            count = 1
        elif inside and binary_series[i] == target:
            count += 1
        elif inside and binary_series[i] != target:
            durations.append(count / FPS * 1000)
            inside = False
            count = 0
    return durations


def _phase_durations_all_legs(
    binary_ranges: List[np.ndarray],
    target: int,
) -> List[List[float]]:
    """All leg durations mixed together, one list per walking bout."""
    result = []
    for binary in binary_ranges:
        binary = np.asarray(binary)
        bout_durations = []
        for col in range(binary.shape[1]):
            bout_durations.extend(_run_durations(binary[:, col], target))
        result.append(bout_durations)
    return result


def _phase_durations_per_leg(
    binary_ranges: List[np.ndarray],
    target: int,
) -> Tuple[List, ...]:
    """Per-leg durations as 6 separate lists, each containing per-bout sublists."""
    per_leg: List[List[List[float]]] = [[] for _ in range(6)]
    for binary in binary_ranges:
        binary = np.asarray(binary)
        for col in range(6):
            per_leg[col].append(_run_durations(binary[:, col], target))
    return tuple(per_leg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_swing_durations(binary_all_ranges: List[np.ndarray]) -> List[List[float]]:
    """Swing (leg-in-motion) durations in ms, mixed across legs, per walking bout."""
    return _phase_durations_all_legs(binary_all_ranges, target=1)


def get_stance_durations(binary_all_ranges: List[np.ndarray]) -> List[List[float]]:
    """Stance (leg-planted) durations in ms, mixed across legs, per walking bout."""
    return _phase_durations_all_legs(binary_all_ranges, target=0)


def get_swing_durations_individual(
    binary_all_ranges: List[np.ndarray],
) -> Tuple[List, ...]:
    """Swing durations per individual leg.

    Returns 6 lists (hindR, midR, foreR, hindL, midL, foreL), each containing
    one sublist of durations (ms) per walking bout.
    """
    return _phase_durations_per_leg(binary_all_ranges, target=1)


def get_stance_durations_individual(
    binary_all_ranges: List[np.ndarray],
) -> Tuple[List, ...]:
    """Stance durations per individual leg.

    Returns 6 lists (hindR, midR, foreR, hindL, midL, foreL), each containing
    one sublist of durations (ms) per walking bout.
    """
    return _phase_durations_per_leg(binary_all_ranges, target=0)


def remove_empty_data(
    average_speeds: np.ndarray,
    durations_leg: List[List],
) -> Tuple[List[float], List[List]]:
    """Remove bouts with no leg duration data.

    Returns corresponding speeds and non-empty duration lists.
    """
    if len(average_speeds) != len(durations_leg):
        raise IndexError("average_speeds and durations_leg must have the same length.")
    pairs = [(s, d) for s, d in zip(average_speeds, durations_leg) if len(d) > 0]
    if not pairs:
        return [], []
    speeds, durations = zip(*pairs)
    return list(speeds), list(durations)


def get_step_lengths(
    fly: dict,
    frames_moving_forward: np.ndarray,
    position_diffs: np.ndarray,
    units: str = "mm",
) -> np.ndarray:
    """Average step length per walking bout.

    A step length is the Euclidean distance a foot travels from lift-off to
    touch-down. Returns NaN for bouts where no complete step is detected.

    Args:
        fly: Body-part coordinate dict.
        frames_moving_forward: (N, 2) array of [start, end) frame indices.
        position_diffs: (n_frames, 6) array from get_position_diff.
        units: "mm" or "px".
    """
    px_threshold = 1
    leg_abs_keys = ["forelegR1", "midlegR1", "hindlegR1", "forelegL1", "midlegL1", "hindlegL1"]
    # position_diffs column indices matching these legs
    diff_cols = [2, 1, 0, 5, 4, 3]

    def _avg_step_for_leg(binary: np.ndarray, abs_coords: np.ndarray, start_idx: int) -> float:
        lengths = []
        first_coords = None
        for i in range(len(binary) - 1):
            if binary[i] == 0 and binary[i + 1] == 1:
                first_coords = abs_coords[start_idx + i] # Record coords at onset of leg movement.
            elif binary[i] == 1 and binary[i + 1] == 0 and first_coords is not None: # Leg stopped moving.
                last_coords = abs_coords[start_idx + i]
                dist = np.linalg.norm(last_coords - first_coords)
                if units == "mm":
                    dist /= PX_PER_MM
                lengths.append(dist)
                first_coords = None
        return float("nan") if not lengths else sum(lengths) / len(lengths)

    avg_step_lengths = []
    for start, end in frames_moving_forward:
        leg_avgs = []
        for key, col in zip(leg_abs_keys, diff_cols):
            col_diffs = position_diffs[start:end, col]
            binary = (col_diffs > px_threshold).astype(int)
            leg_avgs.append(_avg_step_for_leg(binary, fly[key], start))

        if any(math.isnan(v) for v in leg_avgs):
            avg_step_lengths.append(float("nan"))
        else:
            avg_step_lengths.append(sum(leg_avgs) / len(leg_avgs))

    return np.array(avg_step_lengths)


def get_period_durations(
    frames_moving_forward: np.ndarray,
    position_diffs: np.ndarray,
) -> List[float]:
    """Average stride period duration (ms) per walking bout.

    A stride period spans one complete cycle: each of the 6 legs has swung
    at least once starting from the first foreleg lift-off.

    Args:
        frames_moving_forward: (N, 2) array of [start, end) frame indices.
        position_diffs: (n_frames, 6) array from get_position_diff.
    """
    px_threshold = 1
    # column indices: hindR=0, midR=1, foreR=2, hindL=3, midL=4, foreL=5
    COL_FR, COL_FL = 2, 5
    COL_MR, COL_ML = 1, 4
    COL_HR, COL_HL = 0, 3

    avg_periods = []

    for start, end in frames_moving_forward:
        subset = position_diffs[start:end]
        foreR = subset[:, COL_FR] > px_threshold
        foreL = subset[:, COL_FL] > px_threshold
        midR  = subset[:, COL_MR] > px_threshold
        midL  = subset[:, COL_ML] > px_threshold
        hindR = subset[:, COL_HR] > px_threshold
        hindL = subset[:, COL_HL] > px_threshold

        period_durations = []
        fR = fL = mR = mL = hR = hL = False
        next_skip = 0

        for f_idx in range(len(foreR) - 1):
            if next_skip > 0:
                next_skip -= 1 # Prevent double-counting of same period cycles by skipping second foreleg to go into swing.
                continue

            if not (fR or fL):
                # Record foreleg movement as starting point of step period.
                if not foreR[f_idx] and foreR[f_idx + 1]:
                    fR = True
                elif not foreL[f_idx] and foreL[f_idx + 1]:
                    fL = True
                else:
                    continue
                # f_idx becomes first index of period.
                for n_idx in range(f_idx + 1, len(foreR) - 1):
                    if not fR and foreR[n_idx] and not foreR[n_idx + 1]:
                        fR = True
                        next_skip = n_idx - f_idx
                    if not fL and foreL[n_idx] and not foreL[n_idx + 1]:
                        fL = True
                        next_skip = n_idx - f_idx
                    if not mR and midR[n_idx] and not midR[n_idx + 1]:
                        mR = True
                    if not mL and midL[n_idx] and not midL[n_idx + 1]:
                        mL = True
                    if not hR and hindR[n_idx] and not hindR[n_idx + 1]:
                        hR = True
                    if not hL and hindL[n_idx] and not hindL[n_idx + 1]:
                        hL = True

                    if all([fR, mR, hR, fL, mL, hL]):
                        fR = fL = mR = mL = hR = hL = False
                        period_durations.append((n_idx - f_idx) / FPS * 1000) # milliseconds
                        break

        avg_periods.append(
            float("nan") if not period_durations
            else sum(period_durations) / len(period_durations)
        )

    return avg_periods


def get_footprint_alignment(
    fly: dict,
    frames_moving_forward: np.ndarray,
    position_diffs: np.ndarray,
    units: str = "mm",
) -> Tuple[np.ndarray, np.ndarray]:
    """Footprint alignment (std dev of projected touch-down positions) per bout.

    Measures how precisely the mid- and hind-leg land in the footprint left
    by the foreleg. Returns NaN for bouts with too few complete triplets.

    Args:
        fly: Body-part coordinate dict.
        frames_moving_forward: (N, 2) array of [start, end) frame indices.
        position_diffs: (n_frames, 6) array from get_position_diff.
        units: "mm" or "px".
    Returns:
        (stdevsR, stdevsL): Each shape (N,), one value per bout.
    """
    px_threshold = 1
    thorax = fly["thorax1"]

    def _alignment_for_side(
        fore_key: str,
        mid_key: str,
        hind_key: str,
        fore_col: int,
        mid_col: int,
        hind_col: int,
        start: int,
        end: int,
    ) -> float:
        fore_abs = fly[fore_key]
        mid_abs = fly[mid_key]
        hind_abs = fly[hind_key]

        fore_diffs = position_diffs[start:end, fore_col]
        mid_diffs  = position_diffs[start:end, mid_col]
        hind_diffs = position_diffs[start:end, hind_col]

        fore_bin = (fore_diffs > px_threshold).astype(int)
        mid_bin  = (mid_diffs  > px_threshold).astype(int)
        hind_bin = (hind_diffs > px_threshold).astype(int)

        stdevs = []
        for f_i in range(len(fore_bin) - 1):
            if not (fore_bin[f_i] == 1 and fore_bin[f_i + 1] == 0):
                continue
            for m_i in range(f_i, len(mid_bin) - 1):
                if not (mid_bin[m_i] == 1 and mid_bin[m_i + 1] == 0):
                    continue
                for h_i in range(m_i, len(hind_bin) - 1):
                    if not (hind_bin[h_i] == 1 and hind_bin[h_i + 1] == 0):
                        continue

                    tx1 = thorax[f_i + start]
                    tx2 = thorax[h_i + start]
                    dx = tx2[0] - tx1[0]
                    if dx == 0:
                        break
                    thorax_m = (tx2[1] - tx1[1]) / dx
                    thorax_b = tx1[1] - thorax_m * tx1[0]
                    leg_m = -(1 / thorax_m)

                    def _project(coord: np.ndarray) -> np.ndarray:
                        b = coord[1] - leg_m * coord[0]
                        A = np.array([[-thorax_m, 1], [-leg_m, 1]])
                        return np.linalg.solve(A, [thorax_b, b])

                    f_proj = _project(fore_abs[f_i + start])
                    m_proj = _project(mid_abs[m_i + start])
                    h_proj = _project(hind_abs[h_i + start])

                    mean = (f_proj + m_proj + h_proj) / 3
                    stdev = math.sqrt(
                        (np.sum((f_proj - mean) ** 2) +
                         np.sum((m_proj - mean) ** 2) +
                         np.sum((h_proj - mean) ** 2)) / 3
                    )
                    stdevs.append(stdev)
                    break
                break

        if not stdevs:
            return float("nan")
        total = sum(stdevs)
        if units == "mm":
            total /= PX_PER_MM
        return total / len(stdevs)

    stdevsR, stdevsL = [], []
    for start, end in frames_moving_forward:
        stdevsR.append(_alignment_for_side(
            "forelegR1", "midlegR1", "hindlegR1", 2, 1, 0, start, end
        ))
        stdevsL.append(_alignment_for_side(
            "forelegL1", "midlegL1", "hindlegL1", 5, 4, 3, start, end
        ))

    return np.array(stdevsR), np.array(stdevsL)
