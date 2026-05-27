from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .constants import FPS, PX_PER_MM


def _thorax_displacements(thorax1: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smoothed frame-to-frame thorax displacement in pixels.

    Returns shape (n_frames,). Index 0 is NaN.
    """
    diffs = np.linalg.norm(np.diff(thorax1, axis=0), axis=1)
    smoothed = gaussian_filter1d(diffs, sigma=sigma)
    return np.concatenate([[np.nan], smoothed])


def _find_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle in degrees between two 2D vectors."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0.0:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return math.degrees(math.acos(float(cos_theta)))


def get_frames_when_walking_forward(
    fly: dict,
    frame_req: int = 1,
    units: str = "mm",
    sigma: float = 1,
    max_gap: int = 2,
) -> np.ndarray:
    """Detect bouts where the fly walks forward above a speed threshold.

    Args:
        fly: Body-part coordinate dict from get_bodypart_coords.
        frame_req: Minimum consecutive frames required to count as a bout.
        units: Speed units -- "body" (body lengths/s), "mm", or "px".
        sigma: Gaussian smoothing sigma for thorax displacement.
        max_gap: Max frames below threshold to bridge within a bout.
    Returns:
        (N, 2) int array of [start, end) frame indices.
    """
    thorax1 = fly["thorax1"]
    head1 = fly["head1"]
    body_length = fly["body_length"]
    displacements = _thorax_displacements(thorax1, sigma)

    if units == "body":
        speed_threshold = body_length * 1.0
    elif units == "mm":
        speed_threshold = 2.0 * PX_PER_MM # ≥2mm of displacement is counted as movement.
    else:
        speed_threshold = 80.0

    frame_ranges = []
    first_index = streak = gap = 0
    last_met = met_req = False

    for frame in range(1, len(thorax1)):
        actually_nan = (
            np.any(np.isnan(head1[frame]))
            or np.any(np.isnan(thorax1[frame]))
            or np.isnan(displacements[frame])
        )
        is_last = frame == len(thorax1) - 1

        if actually_nan or is_last:
            if streak >= frame_req:
                # For real NaN frames close the bout before the NaN frame so
                # it is not included in the range. For the last-frame sentinel
                # (a valid frame that is simply the end of the recording) use
                # the standard formula which correctly includes it.
                end = frame if actually_nan else frame - gap + 1
                frame_ranges.append((first_index, end))
            last_met = met_req = False
            gap = streak = 0
            continue

        direction = head1[frame] - thorax1[frame]
        movement = head1[frame] - head1[frame - 1]

        # if np.linalg.norm(movement) < 0.005:
        #     moving_forward = True
        # else:
        moving_forward = _find_angle(direction, movement) < 30 # Direction should not change significantly from previous frame.

        fast_enough = displacements[frame] * FPS > speed_threshold

        if moving_forward and fast_enough:
            streak += 1 + (gap if gap <= max_gap else 0)
            gap = 0
            if streak >= frame_req and not met_req:
                first_index = frame - streak + 1
                met_req = True
            last_met = True
        elif last_met:
            gap += 1
            if gap > max_gap:
                if streak >= frame_req:
                    frame_ranges.append((first_index, frame - gap + 1))
                streak = gap = 0
                last_met = met_req = False

    return np.array(frame_ranges, dtype=int)


def get_average_speed_for_ranges(
    fly: dict,
    frame_ranges: np.ndarray,
    units: str = "mm",
    sigma: float = 1,
) -> np.ndarray:
    """Average locomotion speed for each walking bout.

    Args:
        fly: Body-part coordinate dict.
        frame_ranges: (N, 2) array of [start, end) frame indices.
        units: "body", "mm", or "px".
        sigma: Smoothing sigma for thorax displacement.
    Returns:
        Shape (N,) array of average speeds.
    """
    displacements = _thorax_displacements(fly["thorax1"], sigma)
    body_length = fly["body_length"]

    speeds = np.array([
        displacements[start:end].sum() / ((end - start) / FPS)
        for start, end in frame_ranges
    ])

    if units == "body":
        speeds /= body_length
    elif units == "mm":
        speeds /= PX_PER_MM

    return speeds


def get_raw_instantaneous_speeds(
    fly: dict,
    units: str = "mm",
    smoothing: bool = True,
    sigma: float = 1,
) -> np.ndarray:
    """Frame-to-frame speeds for all frames (no direction filter).

    Returns shape (n_frames - 1,).
    """
    diffs = np.linalg.norm(np.diff(fly["thorax1"], axis=0), axis=1) * FPS
    if units == "body":
        diffs /= fly["body_length"]
    elif units == "mm":
        diffs /= PX_PER_MM
    if smoothing:
        diffs = gaussian_filter1d(diffs, sigma=sigma)
    return diffs


def get_instantaneous_speeds(
    fly: dict,
    units: str = "mm",
    sigma: float = 1,
) -> List[float]:
    """Instantaneous speeds for forward-moving frames only (NaN otherwise).

    Returns a list of length n_frames; index 0 is always NaN.
    """
    thorax1 = fly["thorax1"]
    head1 = fly["head1"]
    body_length = fly["body_length"]

    speeds: List[float] = [np.nan]
    for frame in range(1, len(thorax1)):
        direction = head1[frame] - thorax1[frame]
        movement = head1[frame] - head1[frame - 1]
        if np.dot(direction, movement) <= 0:
            speeds.append(np.nan)
            continue
        speed = np.linalg.norm(thorax1[frame] - thorax1[frame - 1]) * FPS
        if units == "body":
            speed /= body_length
        elif units == "mm":
            speed /= PX_PER_MM
        speeds.append(speed)

    smoothed = gaussian_filter1d(speeds[1:], sigma=sigma)
    return [np.nan] + smoothed.tolist()
