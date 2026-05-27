from __future__ import annotations

import math
from typing import List

import numpy as np


def _signed_angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Signed angle in degrees from v1 to v2 (positive = counter-clockwise)."""
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / norms, -1.0, 1.0)
    angle = math.acos(float(cos_theta))
    if np.cross(v1, v2) > 0:
        angle = -angle
    return math.degrees(angle)


def get_all_angles(fly: dict, frames_moving_forward: np.ndarray) -> List[float]:
    """Turning angles (degrees) for every frame within forward-walking bouts.

    Each angle is between the fly's heading direction and its direction of
    movement. Positive = left turn, negative = right turn.

    Args:
        fly: Body-part coordinate dict.
        frames_moving_forward: (N, 2) array of [start, end) frame indices.
    Returns:
        List of signed angles in degrees.
    """
    head1 = fly["head1"]
    thorax1 = fly["thorax1"]
    angles = []

    for start, end in frames_moving_forward:
        for frame in range(start + 1, end):
            direction = head1[frame] - thorax1[frame]
            movement = head1[frame] - head1[frame - 1]
            angles.append(_signed_angle_2d(direction, movement))

    return angles
