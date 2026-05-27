"""Run this script once to generate analysis.ipynb."""
import json, uuid

def code(src): return {"cell_type":"code","execution_count":None,"id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":src}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Fly Locomotion Analysis

Load SLEAP `.slp` files, extract locomotion features, and generate all figures.

**Quick start**
1. Edit the `GROUPS` dict in the *Configuration* cell with your file paths.
2. Run all cells top-to-bottom (`Kernel → Restart & Run All`).
3. Figures are saved to `figures/` as PDFs and displayed inline.\
"""))

# ── Imports ────────────────────────────────────────────────────────────────────
cells.append(code("""\
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sleap_io as sio
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# If fla is not installed, add the repo src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src"))

from fla import (
    get_bodypart_coords, get_position_diff,
    get_frames_when_walking_forward, get_average_speed_for_ranges,
    get_raw_instantaneous_speeds, get_all_angles,
    get_average_gait_index_for_ranges, get_instantaneous_speeds_and_stance,
    get_probability_distribution, convert_ranges_to_binary,
    get_swing_durations, get_stance_durations,
    get_step_lengths, get_period_durations, get_footprint_alignment,
    plot_swing_stance, linear, power_law,
    FPS,
)

os.makedirs("figures", exist_ok=True)\
"""))

# ── Configuration ──────────────────────────────────────────────────────────────
cells.append(md("""\
## Configuration

Edit `GROUPS` to point to your `.slp` files.  Each entry under `"files"` is a
`(path, track_index)` tuple — track index is the fly's index within that recording.\
"""))

cells.append(code("""\
# ── EDIT THIS CELL ─────────────────────────────────────────────────────────────
GROUPS = {
    "Old Males": {
        "color": "#1750AC",
        "files": [
            ("path/to/IM062.slp", 0),
            ("path/to/IM062.slp", 1),
            ("path/to/IM066.slp", 0),
            # add more (path, track_index) pairs as needed
        ],
    },
    "Young Males": {
        "color": "#86CEFA",
        "files": [
            ("path/to/IM089.slp", 0),
            ("path/to/IM089.slp", 1),
        ],
    },
    "Old Females": {
        "color": "#C63287",
        "files": [
            ("path/to/IM061.slp", 0),
            ("path/to/IM061.slp", 1),
        ],
    },
    "Young Females": {
        "color": "#FF99DA",
        "files": [
            ("path/to/IM095.slp", 0),
            ("path/to/IM095.slp", 1),
        ],
    },
}

ALPHA = 0.5          # scatter plot opacity
FIG_SIZE = (4, 3)    # default figure size (inches)\
"""))

# ── Load data ──────────────────────────────────────────────────────────────────
cells.append(md("## Load Data"))

cells.append(code("""\
fly_data = {}
for group, cfg in GROUPS.items():
    fly_data[group] = []
    for path, track_idx in cfg["files"]:
        labels = sio.load_slp(path)
        fly = get_bodypart_coords(labels, track_idx, smoothing=False)
        fly_data[group].append(fly)
    print(f"{group}: {len(fly_data[group])} flies loaded")\
"""))

# ── Feature extraction ─────────────────────────────────────────────────────────
cells.append(md("""\
## Feature Extraction

Run once — all downstream figure cells read from `group_data`.\
"""))

cells.append(code("""\
features = {}
for group, flies in fly_data.items():
    group_feats = []
    for fly in flies:
        pos_diffs    = get_position_diff(fly, smoothing=True, sigma=1)
        fwf          = get_frames_when_walking_forward(fly)
        binary_ranges = convert_ranges_to_binary(fly, fwf)

        fp_R, fp_L = get_footprint_alignment(fly, fwf, pos_diffs)

        group_feats.append({
            "fly":           fly,
            "fwf":           fwf,
            "pos_diffs":     pos_diffs,
            "binary_ranges": binary_ranges,
            "avg_speeds":    get_average_speed_for_ranges(fly, fwf, units="mm"),
            "inst_speeds":   get_raw_instantaneous_speeds(fly, units="mm"),
            "angles":        get_all_angles(fly, fwf),
            "gait_index":    get_average_gait_index_for_ranges(fly, fwf),
            "sal":           get_instantaneous_speeds_and_stance(fly, units="mm"),
            "swing_dur":     get_swing_durations(binary_ranges),
            "stance_dur":    get_stance_durations(binary_ranges),
            "step_len":      get_step_lengths(fly, fwf, pos_diffs, units="mm"),
            "periods":       get_period_durations(fwf, pos_diffs),
            "fp_R":          fp_R,
            "fp_L":          fp_L,
        })
    features[group] = group_feats
    print(f"{group}: features extracted")\
"""))

cells.append(code("""\
# Aggregate across flies within each group for convenient plotting
group_data = {}

for group, feats in features.items():
    avg_speeds  = np.concatenate([f["avg_speeds"]  for f in feats])
    inst_speeds = np.concatenate([f["inst_speeds"] for f in feats])
    angles      = np.array([a for f in feats for a in f["angles"]])
    gait_index  = np.concatenate([f["gait_index"] for f in feats])
    sal_list    = [f["sal"] for f in feats if len(f["sal"]) > 0]
    sal         = np.concatenate(sal_list) if sal_list else np.empty((0, 2))

    # Per-bout mean durations paired with per-bout average speeds
    def _paired(speed_list, dur_list):
        sp, mn = [], []
        for spd, durs in zip(speed_list, dur_list):
            if durs:
                sp.append(spd); mn.append(np.mean(durs))
        return np.array(sp), np.array(mn)

    sw_sp, sw_mn = zip(*[_paired(f["avg_speeds"], f["swing_dur"])  for f in feats]) if feats else ([], [])
    st_sp, st_mn = zip(*[_paired(f["avg_speeds"], f["stance_dur"]) for f in feats]) if feats else ([], [])

    swing_speeds  = np.concatenate([s for s in sw_sp if len(s)])
    swing_means   = np.concatenate([m for m in sw_mn if len(m)])
    stance_speeds = np.concatenate([s for s in st_sp if len(s)])
    stance_means  = np.concatenate([m for m in st_mn if len(m)])

    # Periods and step lengths (drop NaNs)
    per_sp, per_mn = [], []
    sl_sp,  sl_mn  = [], []
    fp_sp,  fp_mn  = [], []
    bout_lengths   = []

    for f in feats:
        for spd, per in zip(f["avg_speeds"], f["periods"]):
            if not np.isnan(per):
                per_sp.append(spd); per_mn.append(per)
        for spd, sl in zip(f["avg_speeds"], f["step_len"]):
            if not np.isnan(sl):
                sl_sp.append(spd); sl_mn.append(sl)
        for spd, r, l in zip(f["avg_speeds"], f["fp_R"], f["fp_L"]):
            if not (np.isnan(r) or np.isnan(l)):
                fp_sp.append(spd); fp_mn.append((r + l) / 2)
        for start, end in f["fwf"]:
            bout_lengths.append(end - start)

    group_data[group] = dict(
        avg_speeds    = avg_speeds,
        inst_speeds   = inst_speeds,
        angles        = angles,
        gait_index    = gait_index,
        sal           = sal,
        swing_speeds  = np.array(swing_speeds),
        swing_means   = np.array(swing_means),
        stance_speeds = np.array(stance_speeds),
        stance_means  = np.array(stance_means),
        period_speeds = np.array(per_sp),
        period_means  = np.array(per_mn),
        steplen_speeds= np.array(sl_sp),
        steplen_means = np.array(sl_mn),
        fp_speeds     = np.array(fp_sp),
        fp_means      = np.array(fp_mn),
        bout_lengths  = np.array(bout_lengths) / FPS,  # convert to seconds
    )

print("Aggregation complete.")\
"""))

# ── Speed ──────────────────────────────────────────────────────────────────────
cells.append(md("## Speed Analysis"))

cells.append(code("""\
# Average speed distribution
fig, ax = plt.subplots(figsize=FIG_SIZE)
for group, data in group_data.items():
    speeds = data["avg_speeds"]
    hist, edges = np.histogram(speeds, bins=25, density=True)
    hist = gaussian_filter1d(hist.astype(float), sigma=0.8)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, hist, color=GROUPS[group]["color"], label=group)

ax.set_xlabel("Average speed (mm/s)")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
fig.savefig("figures/avg_speed_distribution.pdf", bbox_inches="tight")
plt.show()\
"""))

cells.append(code("""\
# Instantaneous speed distributions (per group, truncated at speed_limit)
speed_limit = 35  # mm/s

fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharey=True)
for ax, (group, data) in zip(axes.flat, group_data.items()):
    speeds = data["inst_speeds"]
    speeds_trunc = speeds[speeds <= speed_limit]
    ax.hist(speeds_trunc, bins=50, color=GROUPS[group]["color"],
            alpha=ALPHA, density=True)
    ax.set_title(group)
    ax.set_xlabel("Speed (mm/s)")
    ax.set_ylabel("Density")

fig.tight_layout()
fig.savefig("figures/inst_speed_distribution.pdf", bbox_inches="tight")
plt.show()\
"""))

# ── Turning angle ──────────────────────────────────────────────────────────────
cells.append(md("## Turning Angle Analysis"))

cells.append(code("""\
angle_limit = 120  # degrees

fig, ax = plt.subplots(figsize=FIG_SIZE)
for group, data in group_data.items():
    angles = data["angles"]
    angles_trunc = angles[np.abs(angles) < angle_limit]
    hist, edges = np.histogram(angles_trunc, bins=30, density=True)
    hist = gaussian_filter1d(hist.astype(float), sigma=0.5)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, hist, color=GROUPS[group]["color"], label=group)

ax.set_xlabel("Turning angle (°)")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
fig.savefig("figures/turning_angles.pdf", bbox_inches="tight")
plt.show()\
"""))

# ── Gait ───────────────────────────────────────────────────────────────────────
cells.append(md("## Gait Analysis"))

cells.append(code("""\
# Gait index vs average speed
x_fit = np.linspace(0, 35, 200)

fig, ax = plt.subplots(figsize=FIG_SIZE)
for group, data in group_data.items():
    speeds = data["avg_speeds"]
    gi     = data["gait_index"]
    color  = GROUPS[group]["color"]
    ax.scatter(speeds, gi, color=color, alpha=ALPHA, s=8, label=group)
    try:
        params, _ = curve_fit(linear, speeds, gi)
        ax.plot(x_fit, linear(x_fit, *params), color=color)
    except RuntimeError:
        pass

ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("Average speed (mm/s)")
ax.set_ylabel("Gait index")
ax.legend()
fig.tight_layout()
fig.savefig("figures/gait_index_vs_speed.pdf", bbox_inches="tight")
plt.show()\
"""))

cells.append(code("""\
# P(tripod) and P(tetrapod) vs speed
n_bins = 35

fig, (ax_tetrapod, ax_tripod) = plt.subplots(1, 2, figsize=(8, 3.5))

for group, data in group_data.items():
    sal = data["sal"]
    if len(sal) < 2:
        continue
    color = GROUPS[group]["color"]
    bin_centers, pct = get_probability_distribution(sal, bins=n_bins, stance_type="gait")
    ax_tetrapod.plot(bin_centers, pct[2], color=color, label=group)  # index 2 = tetrapod
    ax_tripod.plot(bin_centers, pct[0], color=color, label=group)    # index 0 = tripod

ax_tetrapod.set_xlabel("Speed (mm/s)"); ax_tetrapod.set_ylabel("P(tetrapod)")
ax_tripod.set_xlabel("Speed (mm/s)");   ax_tripod.set_ylabel("P(tripod)")
ax_tetrapod.legend()
fig.tight_layout()
fig.savefig("figures/gait_probability_distributions.pdf", bbox_inches="tight")
plt.show()\
"""))

# ── Step timing ────────────────────────────────────────────────────────────────
cells.append(md("## Step Duration Analysis"))

cells.append(code("""\
# Swing and stance duration vs speed
x_fit = np.linspace(1, 35, 200)

fig, (ax_stance, ax_swing) = plt.subplots(1, 2, figsize=(8, 3.5))

for group, data in group_data.items():
    color = GROUPS[group]["color"]

    for ax, sp_key, mn_key, ylabel in [
        (ax_stance, "stance_speeds", "stance_means", "Stance duration (ms)"),
        (ax_swing,  "swing_speeds",  "swing_means",  "Swing duration (ms)"),
    ]:
        sp, mn = data[sp_key], data[mn_key]
        if len(sp) < 3:
            continue
        ax.scatter(sp, mn, color=color, alpha=ALPHA, s=5, label=group)
        try:
            params, _ = curve_fit(power_law, sp, mn, p0=[100, -0.5], maxfev=5000)
            ax.plot(x_fit, power_law(x_fit, *params), color=color)
        except RuntimeError:
            pass
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Average speed (mm/s)")

ax_stance.legend()
fig.tight_layout()
fig.savefig("figures/swing_stance_duration.pdf", bbox_inches="tight")
plt.show()\
"""))

cells.append(code("""\
# Stride period vs speed
x_fit = np.linspace(1, 35, 200)

fig, ax = plt.subplots(figsize=FIG_SIZE)
for group, data in group_data.items():
    sp, pm = data["period_speeds"], data["period_means"]
    if len(sp) < 3:
        continue
    # Remove extreme outliers before fitting
    mask = pm < np.percentile(pm, 99)
    sp, pm = sp[mask], pm[mask]
    color = GROUPS[group]["color"]
    ax.scatter(sp, pm, color=color, alpha=ALPHA, s=5, label=group)
    try:
        params, _ = curve_fit(power_law, sp, pm, p0=[500, -0.5], maxfev=5000)
        ax.plot(x_fit, power_law(x_fit, *params), color=color)
    except RuntimeError:
        pass

ax.set_xlabel("Average speed (mm/s)")
ax.set_ylabel("Stride period (ms)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/stride_period.pdf", bbox_inches="tight")
plt.show()\
"""))

cells.append(code("""\
# Step length vs speed
x_fit = np.linspace(1, 35, 200)

fig, ax = plt.subplots(figsize=FIG_SIZE)
for group, data in group_data.items():
    sp, sl = data["steplen_speeds"], data["steplen_means"]
    if len(sp) < 3:
        continue
    color = GROUPS[group]["color"]
    ax.scatter(sp, sl, color=color, alpha=ALPHA, s=5, label=group)
    try:
        params, _ = curve_fit(power_law, sp, sl, p0=[0.1, 0.5], maxfev=5000)
        ax.plot(x_fit, power_law(x_fit, *params), color=color)
    except RuntimeError:
        pass

ax.set_xlabel("Average speed (mm/s)")
ax.set_ylabel("Step length (mm)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/step_length.pdf", bbox_inches="tight")
plt.show()\
"""))

# ── Footprint ──────────────────────────────────────────────────────────────────
cells.append(md("## Footprint Alignment"))

cells.append(code("""\
fig, ax = plt.subplots(figsize=FIG_SIZE)
for group, data in group_data.items():
    color = GROUPS[group]["color"]
    ax.scatter(data["fp_speeds"], data["fp_means"],
               color=color, alpha=ALPHA, s=10, label=group)

ax.set_xlabel("Average speed (mm/s)")
ax.set_ylabel("Footprint alignment std dev (mm)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/footprint_alignment.pdf", bbox_inches="tight")
plt.show()\
"""))

# ── Movement duration ──────────────────────────────────────────────────────────
cells.append(md("## Walking Bout Duration"))

cells.append(code("""\
fig, ax = plt.subplots(figsize=FIG_SIZE)

all_lengths = [(group, data["bout_lengths"]) for group, data in group_data.items()]
max_len = max(l.max() for _, l in all_lengths if len(l))
bins = np.linspace(0, max_len, 30)

for group, lengths in all_lengths:
    hist, edges = np.histogram(lengths, bins=bins, density=True)
    hist = gaussian_filter1d(hist.astype(float), sigma=0.8)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, hist, color=GROUPS[group]["color"], label=group)

ax.set_xlabel("Walking bout duration (s)")
ax.set_ylabel("Density")
ax.legend()
fig.tight_layout()
fig.savefig("figures/walking_bout_duration.pdf", bbox_inches="tight")
plt.show()\
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Convert multiline source strings to lists of lines (nbformat convention)
for cell in nb["cells"]:
    src = cell["source"]
    if isinstance(src, str):
        lines = src.splitlines(keepends=True)
        # Last line should not have trailing newline
        cell["source"] = lines

with open("analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("analysis.ipynb written.")
