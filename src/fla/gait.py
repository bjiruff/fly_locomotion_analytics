from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np

from .constants import FPS, PX_PER_MM, TRIPOD_STANCES, TETRAPOD_STANCES
from .kinematics import get_position_diff
from .locomotion import _thorax_displacements, _find_angle


def _classify_gait(legs: tuple) -> int:
    """Return 1 (tripod), -1 (tetrapod), or 0 (other) for a leg stance tuple."""
    if legs in TRIPOD_STANCES:
        return 1
    if legs in TETRAPOD_STANCES:
        return -1
    return 0


def get_gait_by_frame(swing_stance) -> list:
    """Classify each frame as tripod (1), tetrapod (-1), other (0), or NaN.

    Args:
        swing_stance: Iterable of 6-element tuples or NaN values.
    """
    result = []
    for frame in swing_stance:
        if isinstance(frame, float) and math.isnan(frame):
            result.append(float("nan"))
        else:
            result.append(_classify_gait(tuple(frame)))
    return result


def gait_index_grabber(gait_by_frame: list, sliding_window: int) -> list:
    """Sliding window mean gait index. NaN at boundaries and NaN-containing windows."""
    half = sliding_window // 2
    n = len(gait_by_frame)
    result = [float("nan")] * n

    for i in range(half, n - half):
        window = gait_by_frame[i - half: i + half + 1]
        if any(isinstance(v, float) and math.isnan(v) for v in window):
            continue
        result[i] = sum(window) / sliding_window

    return result


def get_average_gait_index_for_ranges(
    fly: dict,
    frame_ranges: np.ndarray,
    sigma: float = 1,
) -> List[float]:
    """Mean gait index (tripod vs tetrapod) for each walking bout.

    Args:
        fly: Body-part coordinate dict.
        frame_ranges: (N, 2) array of [start, end) frame indices.
        sigma: Smoothing sigma for position differences.
    Returns:
        List of average gait indices, one per bout. NaN for bouts where all
        frames were originally missing from tracking.
    """
    position_diffs = get_position_diff(fly, smoothing=True, sigma=sigma)
    nan_mask = fly.get("nan_frames")
    averages = []
    for start, end in frame_ranges:
        subset = position_diffs[start:end]
        if nan_mask is not None:
            subset = subset[~nan_mask[start:end]]
        if len(subset) == 0:
            averages.append(float("nan"))
            continue
        legs_binary = (subset > 1).astype(int)
        gaits = [_classify_gait(tuple(row)) for row in legs_binary]
        averages.append(sum(gaits) / len(gaits))
    return averages


def convert_ranges_to_binary(
    fly: dict,
    frame_ranges: np.ndarray,
    sigma: float = 1,
) -> List[np.ndarray]:
    """Convert walking bouts to per-frame binary swing/stance arrays.

    Args:
        fly: Body-part coordinate dict.
        frame_ranges: (N, 2) array of [start, end) frame indices.
        sigma: Smoothing sigma for position differences.
    Returns:
        List of (n_frames, 6) int arrays (1=swing, 0=stance), one per bout.
        Frames that were originally NaN in tracking are marked -1 so that
        downstream duration functions treat them as breaks rather than
        classifying linearly-interpolated motion as swing or stance.
    """
    position_diffs = get_position_diff(fly, smoothing=True, sigma=sigma)
    nan_mask = fly.get("nan_frames")
    result = []
    for start, end in frame_ranges:
        binary = (position_diffs[start:end] > 1).astype(int)
        if nan_mask is not None:
            binary[nan_mask[start:end]] = -1
        result.append(binary)
    return result


def get_instantaneous_speeds_and_stance(
    fly: dict,
    units: str = "mm",
    sigma: float = 1,
    stance_type: str = "leg",
    mm_limit: float = 20,
) -> np.ndarray:
    """Speed-stance pairs for all forward-walking frames within speed limits.

    Args:
        fly: Body-part coordinate dict.
        units: "body", "mm", or "px".
        sigma: Smoothing sigma.
        stance_type: "leg" (count of legs in swing) or "gait" (tripod/tetrapod/other).
        mm_limit: Upper speed limit in mm/s to filter flying, fighting, and sparse data.
    Returns:
        (M, 2) array of [speed, stance_value] rows.
    """
    if stance_type not in ("gait", "leg"):
        raise ValueError("stance_type must be 'gait' or 'leg'.")

    thorax1 = fly["thorax1"]
    head1 = fly["head1"]
    body_length = fly["body_length"]
    position_diffs = get_position_diff(fly, smoothing=True, sigma=sigma)
    displacements = _thorax_displacements(thorax1, sigma)

    if units == "body":
        speed_limit = body_length * 12.0
    elif units == "mm":
        speed_limit = PX_PER_MM * mm_limit
    else:
        speed_limit = 720.0

    nan_mask = fly.get("nan_frames")
    pairs = []
    for frame in range(1, len(thorax1)):
        if nan_mask is not None and nan_mask[frame]:
            continue
        if (
            np.any(np.isnan(position_diffs[frame]))
            or np.any(np.isnan(head1[frame]))
            or np.any(np.isnan(thorax1[frame]))
        ):
            continue

        movement = head1[frame] - head1[frame - 1]
        if np.linalg.norm(movement) < 0.005:
            facing_forward = True
        else:
            facing_forward = _find_angle(head1[frame] - thorax1[frame], movement) < 30

        speed_px = displacements[frame] * FPS
        if not (facing_forward and speed_px < speed_limit):
            continue

        legs = tuple((position_diffs[frame] > 1).astype(int))
        stance = _classify_gait(legs) if stance_type == "gait" else sum(legs)

        if units == "body":
            speed = speed_px / body_length
        elif units == "mm":
            speed = speed_px / PX_PER_MM
        else:
            speed = speed_px

        pairs.append((speed, stance))

    return np.array(pairs)


def get_gait_probability_distribution(
    speeds_and_stance: np.ndarray,
    bins: int,
    stance_type: str = "leg",
) -> Tuple[np.ndarray, object]:
    """Speed-binned probability distribution of stance types.

    Args:
        speeds_and_stance: (M, 2) array from get_instantaneous_speeds_and_stance.
        bins: Number of speed bins.
        stance_type: "leg" or "gait".
    Returns:
        bin_centers: (bins,) array of speed bin centers.
        percent_stance: Dict keyed by gait name (stance_type="gait") or list
                        indexed by legs-in-swing count (stance_type="leg").
    """
    if stance_type not in ("gait", "leg"):
        raise ValueError("stance_type must be 'gait' or 'leg'.")

    all_speeds = speeds_and_stance[:, 0]
    _, edges = np.histogram(all_speeds, bins=bins, range=(0, all_speeds.max()))
    # Clip to [0, bins-1] to handle edge values
    bin_indices = np.clip(np.digitize(all_speeds, edges) - 1, 0, bins - 1)

    if stance_type == "gait":
        percent_stance: object = {"tripod": [], "noncanonical": [], "tetrapod": []}
    else:
        percent_stance = [[] for _ in range(7)]

    for b in range(bins):
        mask = bin_indices == b
        if not mask.any():
            if stance_type == "gait":
                for key in percent_stance:
                    percent_stance[key].append(0.0)
            else:
                for s in range(7):
                    percent_stance[s].append(0.0)
            continue
        stances = speeds_and_stance[mask, 1]
        counts = Counter(stances.tolist())
        n = len(stances)
        if stance_type == "gait":
            percent_stance["tripod"].append(counts.get(1, 0) / n)
            percent_stance["noncanonical"].append(counts.get(0, 0) / n)
            percent_stance["tetrapod"].append(counts.get(-1, 0) / n)
        else:
            for s in range(7):
                percent_stance[s].append(counts.get(s, 0) / n)

    bin_centers = (edges[:-1] + edges[1:]) / 2
    return bin_centers, percent_stance
