from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def signed_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angle in degrees from a to b (positive = clockwise rotation).

    Args:
        a: Shape (n, 2).
        b: Shape (n, 2).
    Returns:
        Shape (n,) array of signed angles in degrees.
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    theta = np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))
    cross = np.cross(a, b, axis=1)
    sign = np.where(cross >= 0, -1.0, 1.0)
    return np.rad2deg(theta) * sign


def normalize_to_egocentric(
    x: np.ndarray,
    rel_to: Optional[np.ndarray] = None,
    ctr_ind: int = 0,
    fwd_ind: int = 1,
    return_angles: bool = False,
):
    """Normalize pose estimates to egocentric coordinates.

    Args:
        x: Pose of shape (joints, 2) or (time, joints, 2).
        rel_to: Reference pose to align to; defaults to x.
        ctr_ind: Index of the centroid joint (e.g. thorax).
        fwd_ind: Index of the forward joint (e.g. head).
        return_angles: If True, also return the rotation angles.
    Returns:
        Aligned poses of same shape as input, and optionally angles.
    """
    if rel_to is None:
        rel_to = x

    is_singleton = x.ndim == 2 and rel_to.ndim == 2
    if x.ndim == 2:
        x = x[np.newaxis]
    if rel_to.ndim == 2:
        rel_to = rel_to[np.newaxis]

    ctr = rel_to[..., ctr_ind, :]
    ego_fwd = rel_to[..., fwd_ind, :] - ctr

    ang = np.arctan2(ego_fwd[..., 1], ego_fwd[..., 0])
    ca, sa = np.cos(ang), np.sin(ang)

    rot = np.zeros([len(ca), 3, 3], dtype=ca.dtype)
    rot[..., 0, 0] = ca
    rot[..., 0, 1] = -sa
    rot[..., 1, 0] = sa
    rot[..., 1, 1] = ca
    rot[..., 2, 2] = 1

    x = x - ctr[:, np.newaxis]
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1) @ rot
    x = x[..., :2]

    if is_singleton:
        x = x[0]

    return (x, ang) if return_angles else x


def interpolate_nans(data: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """Linear interpolation over NaN gaps of at most max_gap consecutive frames.

    Gaps longer than max_gap, or at the start/end of the array with no valid
    value on both sides, are left as NaN. Operates in-place.
    """
    def _interp_1d(arr: np.ndarray) -> None:
        nans = np.isnan(arr)
        if not nans.any():
            return
        changes = np.diff(nans.astype(np.int8), prepend=0, append=0)
        run_starts = np.where(changes == 1)[0]
        run_ends   = np.where(changes == -1)[0]
        for s, e in zip(run_starts, run_ends):
            if (e - s) <= max_gap and s > 0 and e < len(arr):
                if not np.isnan(arr[s - 1]) and not np.isnan(arr[e]):
                    arr[s:e] = np.interp(
                        np.arange(s, e), [s - 1, e], [arr[s - 1], arr[e]]
                    )

    if data.ndim == 1:
        _interp_1d(data)
    elif data.ndim == 2:
        for col in range(data.shape[1]):
            _interp_1d(data[:, col])
    else:
        raise ValueError("Array must be 1D or 2D.")
    return data


def get_bodypart_coords(
    labels,
    track_index: int,
    video: Optional[int] = None,
    interpolate: bool = True,
    smoothing: bool = False,
    sigma: float = 1,
) -> dict:
    """Extract body-part coordinates for one track from a SLEAP Labels object.

    track_index is relative to the tracks present in the specified video (0 =
    first fly in that video, 1 = second, etc.), not the global labels.tracks list.

    Returns a dict with egocentric ("ego_*") and absolute coordinates for each
    body part, plus a scalar "body_length" in pixels.
    """
    trx = labels.numpy(video=video)
    valid = ~np.all(np.isnan(trx), axis=(0, 2, 3))
    trx = trx[:, valid]
    trx_sliced = trx[:, track_index]
    sk = labels.skeleton

    ctr_ind = sk.index("thorax1")
    fwd_ind = sk.index("head1")
    trx_ego = normalize_to_egocentric(trx_sliced, ctr_ind=ctr_ind, fwd_ind=fwd_ind)

    def _ego(name: str) -> np.ndarray:
        return trx_ego[:, sk.index(name)].copy()

    def _abs(name: str) -> np.ndarray:
        return trx_sliced[:, sk.index(name)].copy()

    _ego_leg_keys = (
        "ego_hindlegR1", "ego_midlegR1", "ego_forelegR1",
        "ego_hindlegL1", "ego_midlegL1", "ego_forelegL1",
    )
    body_data = {
        "ego_hindlegR1": _ego("hindlegR1"),
        "ego_midlegR1":  _ego("midlegR1"),
        "ego_forelegR1": _ego("forelegR1"),
        "ego_hindlegL1": _ego("hindlegL1"),
        "ego_midlegL1":  _ego("midlegL1"),
        "ego_forelegL1": _ego("forelegL1"),
        "ego_abdomen1":  _ego("abdomen1"),
        "ego_head1":     _ego("head1"),
        "hindlegR1": _abs("hindlegR1"),
        "midlegR1":  _abs("midlegR1"),
        "forelegR1": _abs("forelegR1"),
        "hindlegL1": _abs("hindlegL1"),
        "midlegL1":  _abs("midlegL1"),
        "forelegL1": _abs("forelegL1"),
        "head1":    _abs("head1"),
        "thorax1":  _abs("thorax1"),
    }

    if interpolate:
        for key in body_data:
            body_data[key] = interpolate_nans(body_data[key])

    # Frames still NaN after small-gap interpolation are large tracking gaps.
    # Downstream functions use this mask to exclude those frames rather than
    # treating linearly-interpolated motion as real data.
    nan_frames = np.zeros(len(trx_sliced), dtype=bool)
    for k in _ego_leg_keys:
        nan_frames |= np.any(np.isnan(body_data[k]), axis=1)
    body_data["nan_frames"] = nan_frames

    if smoothing:
        for key in body_data:
            if key == "nan_frames":
                continue
            body_data[key] = gaussian_filter(body_data[key], sigma=sigma, axes=0)

    # Body length: average egocentric head x-distance + average abdomen distance
    head_x = body_data["ego_head1"][:, 0]
    abdomen = body_data["ego_abdomen1"]
    body_data["body_length"] = float(
        np.nanmean(head_x) + np.nanmean(np.linalg.norm(abdomen, axis=1))
    )

    return body_data


def get_position_diff(
    fly: dict,
    smoothing: bool = True,
    sigma: float = 1,
) -> np.ndarray:
    """Frame-to-frame egocentric x-position differences for all 6 legs.

    Returns shape (n_frames, 6). Column order: hindR, midR, foreR, hindL, midL, foreL.
    Row 0 is NaN (no previous frame to diff against).
    """
    leg_keys = [
        "ego_hindlegR1", "ego_midlegR1", "ego_forelegR1",
        "ego_hindlegL1", "ego_midlegL1", "ego_forelegL1",
    ]
    x = np.column_stack([fly[k][:, 0] for k in leg_keys])  # (n_frames, 6)
    diffs = np.diff(x, axis=0, prepend=np.full((1, 6), np.nan))  # (n_frames, 6)
    if smoothing:
        diffs[1:] = gaussian_filter1d(diffs[1:], sigma=sigma, axis=0)
    return diffs
