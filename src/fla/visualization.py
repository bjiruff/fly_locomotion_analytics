from __future__ import annotations

import copy
import math

import numpy as np
import matplotlib.pyplot as plt

from .constants import LEG_LABELS


def plot_swing_stance(swing_stance) -> None:
    """Rasterplot of swing (black) and stance (white) phases for all 6 legs.

    Args:
        swing_stance: Sequence of 6-element tuples or NaN values, one per frame.
    """
    ss = copy.deepcopy(list(swing_stance))

    nan_marker = (0.2,) * 6
    for i, frame in enumerate(ss):
        if isinstance(frame, float) and math.isnan(frame):
            ss[i] = nan_marker

    img = 1 - np.array(ss).T  # invert: 1 → black (swing), 0 → white (stance)

    plt.figure(figsize=(3.8, 2.8))
    plt.imshow(img, cmap="gray", aspect="auto")
    plt.xticks([])
    plt.yticks(ticks=np.arange(6), labels=LEG_LABELS)
