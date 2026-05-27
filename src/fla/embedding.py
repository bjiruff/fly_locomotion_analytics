"""UMAP embedding of fly leg trajectories."""
from __future__ import annotations

import random as _random
from typing import List, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import umap

from .constants import FPS
from .kinematics import get_position_diff
from .locomotion import get_frames_when_walking_forward, get_average_speed_for_ranges
from .gait import convert_ranges_to_binary
from .steps import get_stance_durations


# Leg key order used when building window vectors.
# Produces interleaved layout: for each frame → (hindL_x, hindL_y, midL_x, ..., foreR_y)
_LEG_KEYS = [
    "ego_hindlegL1", "ego_midlegL1", "ego_forelegL1",
    "ego_hindlegR1", "ego_midlegR1", "ego_forelegR1",
]


# ── Internal helpers ────────────────────────────────────────────────────────────

def _extract_window_vector(fly: dict, center: int, half_window: int) -> np.ndarray:
    """Flattened egocentric leg trajectory window centered at `center`.

    Collects a window of shape (window_len, 6_legs, 2_coords) and flattens
    to 1D, giving length (2*half_window+1) * 12.
    """
    s, e = center - half_window, center + half_window + 1
    coords = np.stack([fly[k][s:e] for k in _LEG_KEYS])  # (6, W, 2)
    return coords.transpose(1, 0, 2).reshape(-1)           # (W*12,)


def _sampleable_frames(
    fly: dict,
    half_window: int,
    frame_req: int,
    sigma: float,
    max_gap: int,
) -> np.ndarray:
    """Frames within walking bouts that have enough padding for a full window.

    Any center frame whose window overlaps a frame that was originally NaN
    (before interpolation in get_bodypart_coords) is excluded, preventing
    linearly-interpolated tracking gaps from entering the embedding.
    """
    fwf = get_frames_when_walking_forward(
        fly, frame_req=frame_req, units="mm", sigma=sigma, max_gap=max_gap
    )
    frames = []
    for start, end in fwf:
        frames.extend(range(start + half_window, end - half_window))
    frames = np.array(frames, dtype=int)

    nan_mask = fly.get("nan_frames")
    if nan_mask is not None and nan_mask.any() and len(frames) > 0:
        cumsum = np.concatenate([[0], np.cumsum(nan_mask.astype(np.int32))])
        starts = np.maximum(0, frames - half_window)
        ends = np.minimum(len(nan_mask), frames + half_window + 1)
        frames = frames[cumsum[ends] - cumsum[starts] == 0]

    return frames


def _remove_overlap(frames: np.ndarray, half_window: int) -> np.ndarray:
    """Keep only frames whose windows do not overlap with the previous kept one."""
    if len(frames) == 0:
        return frames
    window_size = 2 * half_window + 1
    kept = [frames[0]]
    for f in frames[1:]:
        if f >= kept[-1] + window_size:
            kept.append(f)
    return np.array(kept, dtype=int)


def _sample_across(arrays: List[np.ndarray], k: int) -> List[np.ndarray]:
    """Draw k frames uniformly from the union of multiple fly frame arrays."""
    flat = [(item, i) for i, arr in enumerate(arrays) for item in arr]
    if k > len(flat):
        raise ValueError(
            f"num_samples ({k}) exceeds total available frames ({len(flat)})."
        )
    sampled = _random.sample(flat, k)
    result: List[list] = [[] for _ in arrays]
    for item, idx in sampled:
        result[idx].append(item)
    return [np.sort(np.array(r, dtype=int)) for r in result]


# ── Public API ──────────────────────────────────────────────────────────────────

def build_umap_dataset(
    flies: List[dict],
    half_window: int = 6,
    sampling: str = "random",
    num_samples: Optional[int] = None,
    allow_overlap: bool = True,
    frame_req: int = 20,
    sigma: float = 1.0,
    max_gap: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build UMAP input matrix from short leg trajectory windows.

    Each data point captures egocentric leg coordinates within a centered
    frame window and flattens them to a 1D vector of length
    ``(2*half_window+1) * 12``.

    Args:
        flies: Body-part coordinate dicts from ``get_bodypart_coords``.
        half_window: Half-width of the window in frames (full width = 2*half_window+1).
        sampling: ``"all"`` uses every sampleable window; ``"random"`` subsamples.
        num_samples: Total windows to sample; required when ``sampling="random"``.
        allow_overlap: If ``False``, adjacent windows must be non-overlapping.
        frame_req: Minimum frames for ``get_frames_when_walking_forward``.
        sigma: Smoothing sigma for walking detection.
        max_gap: ``max_gap`` for walking detection.

    Returns:
        vectors: ``(N, D)`` feature matrix.
        window_ranges: ``(N, 2)`` int array of ``[start, end)`` indices for each window,
            used later to compute per-window coloring values.
        fly_slices: ``(n_flies, 2)`` int array; row ``i`` gives ``[start, end)`` into
            vectors/window_ranges for fly ``i``.
    """
    if sampling not in ("all", "random"):
        raise ValueError('sampling must be "all" or "random".')
    if sampling == "random" and num_samples is None:
        raise ValueError("num_samples is required when sampling='random'.")

    sampleable = [
        _sampleable_frames(fly, half_window, frame_req, sigma, max_gap)
        for fly in flies
    ]

    triggers = _sample_across(sampleable, num_samples) if sampling == "random" else sampleable
    if not allow_overlap:
        triggers = [_remove_overlap(t, half_window) for t in triggers]

    all_vectors, all_ranges, fly_slices = [], [], []
    cursor = 0

    for fly, trig in zip(flies, triggers):
        fly_slices.append((cursor, cursor + len(trig)))
        cursor += len(trig)
        if len(trig) == 0:
            continue
        all_vectors.append(
            np.stack([_extract_window_vector(fly, f, half_window) for f in trig])
        )
        all_ranges.append(
            np.column_stack([trig - half_window, trig + half_window + 1])
        )

    if not all_vectors:
        raise ValueError("No walking windows could be extracted from any fly.")

    return (
        np.concatenate(all_vectors),
        np.concatenate(all_ranges),
        np.array(fly_slices),
    )


def normalize_window_vectors(vectors: np.ndarray) -> np.ndarray:
    """Z-score normalize each leg coordinate time series within each window.

    The 1D vector layout repeats 12 values per frame. Each of
    the 12 coordinate series is normalized independently across the frames of
    that window so that position offsets and scale differences do not dominate
    the embedding.

    Args:
        vectors: ``(N, D)`` array where D is divisible by 12.

    Returns:
        Normalized copy of vectors.
    """
    if vectors.shape[1] % 12 != 0:
        raise ValueError("Vector length must be divisible by 12.")
    out = vectors.astype(float)
    for offset in range(12):
        idx = np.arange(offset, out.shape[1], 12)
        vals = out[:, idx]
        mean = vals.mean(axis=1, keepdims=True)
        std = vals.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        out[:, idx] = (vals - mean) / std
    return out


def compute_window_coloring(
    flies: List[dict],
    window_ranges: np.ndarray,
    fly_slices: np.ndarray,
    movement_parameter: str = "gait",
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute a scalar coloring value for every UMAP window.

    Args:
        flies: Body-part coordinate dicts.
        window_ranges: ``(N, 2)`` array from ``build_umap_dataset``.
        fly_slices: ``(n_flies, 2)`` array from ``build_umap_dataset``.
        movement_parameter: ``"gait"`` (mean legs in swing), ``"stance"``
            (mean stance duration in ms), or ``"speed"`` (mean mm/s).
        sigma: Smoothing sigma for feature computation.

    Returns:
        ``(N,)`` array. ``NaN`` where the parameter cannot be computed
        (e.g. too few steps for stance duration). Filter with
        ``valid = ~np.isnan(coloring)`` before embedding if needed.
    """
    if movement_parameter not in ("gait", "stance", "speed"):
        raise ValueError('movement_parameter must be "gait", "stance", or "speed".')

    coloring = np.full(len(window_ranges), np.nan)

    for fly, (s, e) in zip(flies, fly_slices):
        if s == e:
            continue
        fly_ranges = window_ranges[s:e]  # (n_windows, 2)

        if movement_parameter == "gait":
            pos_diffs = get_position_diff(fly, smoothing=True, sigma=sigma)
            coloring[s:e] = [
                (pos_diffs[r0:r1] > 1).sum(axis=1).mean()
                for r0, r1 in fly_ranges
            ]

        elif movement_parameter == "stance":
            binary_ranges = convert_ranges_to_binary(fly, fly_ranges, sigma=sigma)
            dur_lists = get_stance_durations(binary_ranges)
            coloring[s:e] = [
                np.mean(d) if d else np.nan for d in dur_lists
            ]

        elif movement_parameter == "speed":
            coloring[s:e] = get_average_speed_for_ranges(
                fly, fly_ranges, units="mm", sigma=sigma
            )

    return coloring


def fit_umap(
    vectors: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    n_components: int = 3,
    metric: str = "euclidean",
    random_state: int = 1,
) -> np.ndarray:
    """Fit UMAP and return the low-dimensional embedding.

    Args:
        vectors: ``(N, D)`` feature matrix.
        n_neighbors: Controls local vs global structure (higher = more global).
        min_dist: Minimum distance between points in the embedding (0 = tight clusters).
        n_components: Embedding dimensionality (2 or 3).
        metric: Distance metric.
        random_state: Seed for reproducibility.

    Returns:
        ``(N, n_components)`` embedding array.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


def plot_umap_colored(
    embedding: np.ndarray,
    coloring: np.ndarray,
    colorscale: str = "Viridis",
    colorbar_title: str = "",
    mark_size: float = 3,
    opacity: float = 0.8,
    camera: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
) -> go.Figure:
    """3D scatter plot colored by a continuous scalar variable.

    Args:
        embedding: ``(N, 3)`` embedding array.
        coloring: ``(N,)`` scalar values (e.g. gait, speed).
        colorscale: Plotly colorscale name (e.g. ``"Viridis"``, ``"matter"``).
        colorbar_title: Label shown next to the colorbar.
        mark_size: Marker size in pixels.
        opacity: Marker opacity in [0, 1].
        camera: ``(x, y, z)`` eye position for the 3D view.
        cmin, cmax: Fixed colorscale bounds; if ``None``, uses data range.

    Returns:
        Plotly ``Figure``.
    """
    fig = go.Figure(
        data=go.Scatter3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            mode="markers",
            marker=dict(
                size=mark_size,
                color=coloring,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
                opacity=opacity,
                cmin=cmin,
                cmax=cmax,
            ),
        ),
        layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0)),
    )
    fig.update_layout(
        scene_camera=dict(eye=dict(x=camera[0], y=camera[1], z=camera[2]))
    )
    return fig


def plot_umap_by_group(
    embedding: np.ndarray,
    group_slices: List[Tuple[int, int]],
    group_labels: List[str],
    colors: List[str],
    opacities: Optional[List[float]] = None,
    mark_size: float = 2,
    camera: Tuple[float, float, float] = (1.5, 1.5, 1.5),
) -> go.Figure:
    """3D scatter with one trace per group, colored by group identity.

    Args:
        embedding: ``(N, 3)`` embedding array.
        group_slices: ``[(start, end), ...]`` row ranges into embedding per group.
        group_labels: Display name for each group.
        colors: Hex color strings, one per group.
        opacities: Opacity per group; defaults to 0.8 for all.
        mark_size: Marker size.
        camera: ``(x, y, z)`` eye position.

    Returns:
        Plotly ``Figure``.
    """
    if opacities is None:
        opacities = [0.8] * len(group_labels)

    fig = go.Figure()
    for (start, end), label, color, opacity in zip(
        group_slices, group_labels, colors, opacities
    ):
        fig.add_trace(go.Scatter3d(
            x=embedding[start:end, 0],
            y=embedding[start:end, 1],
            z=embedding[start:end, 2],
            mode="markers",
            name=label,
            marker=dict(size=mark_size, color=color, opacity=opacity),
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(title="Group"),
        scene_camera=dict(eye=dict(x=camera[0], y=camera[1], z=camera[2])),
    )
    return fig


def plot_umap_by_individual(
    embedding: np.ndarray,
    individual_slices: List[Tuple[int, int]],
    mark_size: float = 3,
) -> go.Figure:
    """3D scatter where each individual fly is assigned a unique color.

    Args:
        embedding: ``(N, 3)`` embedding array.
        individual_slices: ``[(start, end), ...]`` row ranges per fly.
        mark_size: Marker size.

    Returns:
        Plotly ``Figure``.
    """
    palette = px.colors.qualitative.Dark24
    fig = go.Figure()
    for i, (start, end) in enumerate(individual_slices):
        fig.add_trace(go.Scatter3d(
            x=embedding[start:end, 0],
            y=embedding[start:end, 1],
            z=embedding[start:end, 2],
            mode="markers",
            name=f"fly {i}",
            marker=dict(
                size=mark_size,
                color=palette[i % len(palette)],
                opacity=0.8,
            ),
        ))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(title="Individual"),
    )
    return fig
