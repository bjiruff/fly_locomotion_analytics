"""Run this script once to generate umap_analysis.ipynb."""
import json, uuid

def code(src): return {"cell_type":"code","execution_count":None,"id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":src}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Fly Locomotion — UMAP Embedding

Embed short windows of egocentric leg trajectories into 3D using UMAP and
color by movement parameters (gait, stance duration, speed) to reveal locomotion
structure across groups.

**Quick start**
1. Edit the `GROUPS` dict and optionally `COMBINED` in the *Configuration* cell.
2. Run all cells top-to-bottom (`Kernel → Restart & Run All`).
3. Interactive figures are displayed inline and saved as HTML to `figures/`.\
"""))

# ── Imports ────────────────────────────────────────────────────────────────────
cells.append(code("""\
import os
import sys
import numpy as np
import sleap_io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src"))

from fla import (
    get_bodypart_coords,
    build_umap_dataset,
    normalize_window_vectors,
    compute_window_coloring,
    fit_umap,
    plot_umap_colored,
    plot_umap_by_group,
    plot_umap_by_individual,
)

os.makedirs("figures", exist_ok=True)\
"""))

# ── Configuration ──────────────────────────────────────────────────────────────
cells.append(md("""\
## Configuration

Edit `GROUPS` with your `.slp` file paths.  Each `"files"` entry is a
`(path, track_index)` tuple.

`COMBINED` defines multi-group embeddings (e.g. all males across ages).
Each entry lists which `GROUPS` keys to pool together.\
"""))

cells.append(code("""\
# ── EDIT THIS CELL ─────────────────────────────────────────────────────────────
GROUPS = {
    "Old Males": {
        "color": "#1750AC",
        "files": [
            ("path/to/IM062.slp", 0),
            ("path/to/IM062.slp", 1),
        ],
    },
    "2-Month Males": {
        "color": "#3373C4",
        "files": [
            ("path/to/m2_IM092.slp", 0),
        ],
    },
    "1-Month Males": {
        "color": "#5494DA",
        "files": [
            ("path/to/m1_IM058.slp", 0),
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
    "2-Month Females": {
        "color": "#D44C9C",
        "files": [
            ("path/to/f2_IM098.slp", 0),
        ],
    },
    "1-Month Females": {
        "color": "#E366B1",
        "files": [
            ("path/to/f1_IM057.slp", 0),
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

# Which groups to pool for combined embeddings
COMBINED = [
    {
        "label": "All Males",
        "groups": ["Old Males", "2-Month Males", "1-Month Males", "Young Males"],
        "camera": (2.1, -0.3, 0.2),
    },
    {
        "label": "All Females",
        "groups": ["Old Females", "2-Month Females", "1-Month Females", "Young Females"],
        "camera": (-1.7, 1.3, 0.5),
    },
]

# ── UMAP / window settings ──────────────────────────────────────────────────────
HALF_WINDOW  = 6      # frames on each side of trigger; window length = 2*HALF_WINDOW+1
FRAME_REQ    = 20     # min frames walking to count as a bout
N_NEIGHBORS  = 30     # UMAP: higher = more global structure
MIN_DIST     = 0.0    # UMAP: 0 = tight clusters
SIGMA        = 1.0    # Gaussian smoothing sigma\
"""))

# ── Load data ──────────────────────────────────────────────────────────────────
cells.append(md("## Load Data"))

cells.append(code("""\
fly_data = {}
for group, cfg in GROUPS.items():
    fly_data[group] = []
    for path, track_idx in cfg["files"]:
        labels = sio.load_slp(path)
        fly_data[group].append(get_bodypart_coords(labels, track_idx, smoothing=False))
    print(f"{group}: {len(fly_data[group])} flies loaded")\
"""))

# ── Build datasets ─────────────────────────────────────────────────────────────
cells.append(md("""\
## Build UMAP Datasets

For each group, extract all walking windows, normalize per leg coordinate, and
compute coloring values (gait, stance, speed).\
"""))

cells.append(code("""\
group_vectors   = {}   # group -> normalized (N, D) feature matrix
group_colorings = {}   # group -> {"gait": ..., "stance": ..., "speed": ...}
group_fly_slices = {}  # group -> (n_flies, 2) fly index ranges

for group, flies in fly_data.items():
    print(f"Building dataset for {group}...")
    vectors, window_ranges, fly_slices = build_umap_dataset(
        flies,
        half_window=HALF_WINDOW,
        sampling="all",
        allow_overlap=True,
        frame_req=FRAME_REQ,
        sigma=SIGMA,
    )
    vectors = normalize_window_vectors(vectors)
    group_vectors[group]    = vectors
    group_fly_slices[group] = fly_slices

    group_colorings[group] = {
        param: compute_window_coloring(flies, window_ranges, fly_slices,
                                       movement_parameter=param, sigma=SIGMA)
        for param in ("gait", "stance", "speed")
    }
    print(f"  {len(vectors)} windows extracted\\n")\
"""))

# ── Per-group embeddings ───────────────────────────────────────────────────────
cells.append(md("""\
## Per-Group Embeddings

Embed each group separately.  This shows the internal locomotion structure
within a group without being influenced by inter-group differences.\
"""))

cells.append(code("""\
group_embeddings = {}

for group, vectors in group_vectors.items():
    print(f"Fitting UMAP for {group}...")
    group_embeddings[group] = fit_umap(
        vectors, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST
    )
print("Done.")\
"""))

cells.append(code("""\
# ── Color by gait (average legs in swing) ──────────────────────────────────────
for group, embedding in group_embeddings.items():
    coloring = group_colorings[group]["gait"]
    fig = plot_umap_colored(
        embedding, coloring,
        colorscale="Viridis",
        colorbar_title="Mean legs in swing",
        mark_size=2, opacity=0.8,
        cmin=0, cmax=3,
    )
    fig.update_layout(title_text=group)
    fig.write_html(f"figures/umap_{group.lower().replace(' ', '_')}_gait.html")
    fig.show()\
"""))

cells.append(code("""\
# ── Color by stance duration ────────────────────────────────────────────────────
for group, embedding in group_embeddings.items():
    raw = group_colorings[group]["stance"]
    valid = ~np.isnan(raw)
    if valid.sum() < 10:
        print(f"Skipping {group}: too few valid stance windows.")
        continue
    fig = plot_umap_colored(
        embedding[valid], raw[valid],
        colorscale="matter",
        colorbar_title="Mean stance duration (ms)",
        mark_size=2, opacity=0.8,
        cmin=20, cmax=180,
    )
    fig.update_layout(title_text=group)
    fig.write_html(f"figures/umap_{group.lower().replace(' ', '_')}_stance.html")
    fig.show()\
"""))

cells.append(code("""\
# ── Color by speed ──────────────────────────────────────────────────────────────
for group, embedding in group_embeddings.items():
    coloring = group_colorings[group]["speed"]
    fig = plot_umap_colored(
        embedding, coloring,
        colorscale="plasma",
        colorbar_title="Mean speed (mm/s)",
        mark_size=2, opacity=0.8,
    )
    fig.update_layout(title_text=group)
    fig.write_html(f"figures/umap_{group.lower().replace(' ', '_')}_speed.html")
    fig.show()\
"""))

cells.append(code("""\
# ── Color by individual fly ─────────────────────────────────────────────────────
for group, embedding in group_embeddings.items():
    fly_slices = [tuple(s) for s in group_fly_slices[group]]
    fig = plot_umap_by_individual(embedding, fly_slices)
    fig.update_layout(title_text=group)
    fig.write_html(f"figures/umap_{group.lower().replace(' ', '_')}_individual.html")
    fig.show()\
"""))

# ── Combined embeddings ────────────────────────────────────────────────────────
cells.append(md("""\
## Combined Embeddings

Pool multiple groups into a single embedding to compare across ages or sexes.
Each group is shown as a separate trace so its position in the shared space
can be compared visually.\
"""))

cells.append(code("""\
combined_embeddings = {}   # label -> embedding
combined_slices     = {}   # label -> list of (start, end) per group
combined_colorings  = {}   # label -> {"gait": ..., "stance": ..., "speed": ...}

for cfg in COMBINED:
    label  = cfg["label"]
    groups = cfg["groups"]

    # Pool only the groups that were actually loaded
    present = [g for g in groups if g in group_vectors]
    if not present:
        print(f"Skipping '{label}': none of {groups} were loaded.")
        continue

    vec_list    = [group_vectors[g]    for g in present]
    color_lists = {p: [group_colorings[g][p] for g in present]
                   for p in ("gait", "stance", "speed")}

    # Build (start, end) slices for each group in the concatenated matrix
    slices, cursor = [], 0
    for v in vec_list:
        slices.append((cursor, cursor + len(v)))
        cursor += len(v)

    combined_X = np.concatenate(vec_list)
    print(f"Fitting UMAP for '{label}' ({len(combined_X)} windows)...")
    embedding = fit_umap(combined_X, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST)

    combined_embeddings[label] = embedding
    combined_slices[label]     = {"groups": present, "slices": slices}
    combined_colorings[label]  = {
        p: np.concatenate(color_lists[p]) for p in ("gait", "stance", "speed")
    }

print("Done.")\
"""))

cells.append(code("""\
# ── Plot combined: color by group ───────────────────────────────────────────────
for label, embedding in combined_embeddings.items():
    info   = combined_slices[label]
    groups = info["groups"]
    slices = info["slices"]
    colors   = [GROUPS[g]["color"] for g in groups]
    camera   = next(c["camera"] for c in COMBINED if c["label"] == label)

    fig = plot_umap_by_group(
        embedding, slices, groups, colors,
        mark_size=2, camera=camera,
    )
    fig.update_layout(title_text=label)
    fig.write_html(f"figures/umap_{label.lower().replace(' ', '_')}_by_group.html")
    fig.show()\
"""))

cells.append(code("""\
# ── Plot combined: color by gait ────────────────────────────────────────────────
for label, embedding in combined_embeddings.items():
    camera   = next(c["camera"] for c in COMBINED if c["label"] == label)
    coloring = combined_colorings[label]["gait"]

    fig = plot_umap_colored(
        embedding, coloring,
        colorscale="Viridis",
        colorbar_title="Mean legs in swing",
        mark_size=2, opacity=1.0,
        camera=camera, cmin=1, cmax=3,
    )
    fig.update_layout(title_text=f"{label} — gait")
    fig.write_html(f"figures/umap_{label.lower().replace(' ', '_')}_gait.html")
    fig.show()\
"""))

cells.append(code("""\
# ── Plot combined: color by stance duration ─────────────────────────────────────
for label, embedding in combined_embeddings.items():
    camera   = next(c["camera"] for c in COMBINED if c["label"] == label)
    raw      = combined_colorings[label]["stance"]
    valid    = ~np.isnan(raw)

    fig = plot_umap_colored(
        embedding[valid], raw[valid],
        colorscale="matter",
        colorbar_title="Mean stance duration (ms)",
        mark_size=2, opacity=1.0,
        camera=camera, cmin=20, cmax=180,
    )
    fig.update_layout(title_text=f"{label} — stance duration")
    fig.write_html(f"figures/umap_{label.lower().replace(' ', '_')}_stance.html")
    fig.show()\
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

for cell in nb["cells"]:
    src = cell["source"]
    if isinstance(src, str):
        cell["source"] = src.splitlines(keepends=True)

with open("umap_analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("umap_analysis.ipynb written.")
