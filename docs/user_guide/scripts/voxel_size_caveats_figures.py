# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate figures for docs/user_guide/voxel_size_caveats.md.

Run from repo root:

    python docs/user_guide/scripts/voxel_size_caveats_figures.py

Writes PNGs into ``docs/user_guide/img/voxel_caveats/``.
"""
from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), os.pardir, "img", "voxel_caveats")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

CLASS_A_COLOR = "#d62728"  # red
CLASS_B_COLOR = "#1f77b4"  # blue
GRID_COLOR = "#888888"
OCCUPIED_FACE = "#ffd966"
KERNEL_FACE_A = (1.0, 0.42, 0.21, 0.25)
KERNEL_FACE_B = (0.12, 0.47, 0.71, 0.25)
EDGE_COLOR = "#222222"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --------------------------------------------------------------------- #
# Data: two interlocking moons (no sklearn dependency).
# --------------------------------------------------------------------- #
def make_two_moons(n_per_class: int = 60, noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n_per_class)

    moon_a = np.stack([np.cos(t), np.sin(t)], axis=1)
    moon_b = np.stack([1.0 - np.cos(t), -np.sin(t) + 0.5], axis=1)

    moon_a += rng.normal(0.0, noise, moon_a.shape)
    moon_b += rng.normal(0.0, noise, moon_b.shape)

    pts = np.concatenate([moon_a, moon_b], axis=0)
    lab = np.concatenate(
        [np.zeros(n_per_class, dtype=np.int32), np.ones(n_per_class, dtype=np.int32)]
    )
    return pts, lab


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def voxelize(pts: np.ndarray, voxel_size: float):
    """Return (voxel_idx (N,2), unique_voxels (V,2), inverse (N,)) of int grid coords."""
    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    unique, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    return voxel_idx, unique, inverse


def voxel_majority_label(unique_idx, inverse, labels, num_classes=2):
    hist = np.zeros((unique_idx.shape[0], num_classes), dtype=np.int64)
    np.add.at(hist, (inverse, labels), 1)
    return hist.argmax(axis=1), hist


def draw_grid(ax, voxel_size, xlim, ylim, color=GRID_COLOR, alpha=0.35, lw=0.5):
    x0 = np.floor(xlim[0] / voxel_size) * voxel_size
    x1 = np.ceil(xlim[1] / voxel_size) * voxel_size
    y0 = np.floor(ylim[0] / voxel_size) * voxel_size
    y1 = np.ceil(ylim[1] / voxel_size) * voxel_size
    for x in np.arange(x0, x1 + 0.5 * voxel_size, voxel_size):
        ax.axvline(x, color=color, alpha=alpha, lw=lw, zorder=0)
    for y in np.arange(y0, y1 + 0.5 * voxel_size, voxel_size):
        ax.axhline(y, color=color, alpha=alpha, lw=lw, zorder=0)


def draw_occupied_voxels(
    ax, unique_idx, voxel_labels, voxel_size, edge=EDGE_COLOR, lw=0.6, alpha=0.85
):
    for v_idx, lab in zip(unique_idx, voxel_labels):
        x = v_idx[0] * voxel_size
        y = v_idx[1] * voxel_size
        face = CLASS_A_COLOR if lab == 0 else CLASS_B_COLOR
        ax.add_patch(
            mpatches.Rectangle(
                (x, y),
                voxel_size,
                voxel_size,
                facecolor=face,
                edgecolor=edge,
                lw=lw,
                alpha=alpha,
            )
        )


def draw_mixed_voxels(ax, unique_idx, hist, voxel_size, edge=EDGE_COLOR, lw=0.6):
    """Color voxel by red/blue mix proportional to per-class counts inside."""
    for v_idx, h in zip(unique_idx, hist):
        x = v_idx[0] * voxel_size
        y = v_idx[1] * voxel_size
        total = h.sum()
        if total == 0:
            face = "#dddddd"
        else:
            r, b = h[0] / total, h[1] / total
            face = (
                0.84 * r + 0.12 * b,
                0.16 * r + 0.47 * b,
                0.24 * r + 0.71 * b,
                0.85,
            )
        ax.add_patch(
            mpatches.Rectangle(
                (x, y),
                voxel_size,
                voxel_size,
                facecolor=face,
                edgecolor=edge,
                lw=lw,
            )
        )


def style_axes(ax, title, xlim, ylim, hide_ticks=False):
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(axis="both", labelsize=8, colors="#666666", length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#bbbbbb")


# --------------------------------------------------------------------- #
# Figure 1: raw two-moons dataset
# --------------------------------------------------------------------- #
def fig_raw_moons(pts, lab, xlim, ylim):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(
        pts[lab == 0, 0],
        pts[lab == 0, 1],
        s=36,
        c=CLASS_A_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        label="class A",
    )
    ax.scatter(
        pts[lab == 1, 0],
        pts[lab == 1, 1],
        s=36,
        c=CLASS_B_COLOR,
        marker="^",
        edgecolors="white",
        linewidths=0.4,
        label="class B",
    )
    style_axes(ax, f"Two moons — {pts.shape[0]} sparse points", xlim, ylim)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_raw_moons.png"))
    plt.close(fig)


# --------------------------------------------------------------------- #
# Figure 2: three voxel sizes — points + grid overlay
# --------------------------------------------------------------------- #
def fig_three_voxel_sizes(pts, lab, xlim, ylim):
    sizes = [(0.35, "Too coarse"), (0.12, "Just right"), (0.05, "Too fine")]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
    for ax, (vs, name) in zip(axes, sizes):
        draw_grid(ax, vs, xlim, ylim)
        ax.scatter(
            pts[lab == 0, 0],
            pts[lab == 0, 1],
            s=24,
            c=CLASS_A_COLOR,
            marker="o",
            edgecolors="white",
            linewidths=0.35,
        )
        ax.scatter(
            pts[lab == 1, 0],
            pts[lab == 1, 1],
            s=24,
            c=CLASS_B_COLOR,
            marker="^",
            edgecolors="white",
            linewidths=0.35,
        )
        _, unique, _ = voxelize(pts, vs)
        style_axes(
            ax,
            f"{name}\nvoxel = {vs:.2f}    ({unique.shape[0]} occupied voxels)",
            xlim,
            ylim,
        )
    fig.subplots_adjust(wspace=0.18, top=0.85)
    fig.savefig(os.path.join(OUT, "02_three_voxel_sizes.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Figure 3: voxelized occupancy (no raw points) — what conv actually sees
# --------------------------------------------------------------------- #
def fig_voxelized_occupancy(pts, lab, xlim, ylim):
    sizes = [
        (0.35, "Too coarse\nmoons fuse"),
        (0.12, "Just right\nmoons separable"),
        (0.05, "Too fine\nvoxels isolated"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
    for ax, (vs, name) in zip(axes, sizes):
        _, unique, inverse = voxelize(pts, vs)
        v_lab, v_hist = voxel_majority_label(unique, inverse, lab)
        if vs >= 0.3:
            draw_mixed_voxels(ax, unique, v_hist, vs, lw=0.4)
        else:
            draw_occupied_voxels(ax, unique, v_lab, vs, lw=0.6, alpha=1.0)
        # Draw grid darker for the too-fine case so empty cells are visible.
        grid_alpha = 0.55 if vs <= 0.06 else 0.20
        grid_color = "#888888" if vs <= 0.06 else GRID_COLOR
        draw_grid(ax, vs, xlim, ylim, color=grid_color, alpha=grid_alpha, lw=0.5)
        style_axes(ax, name, xlim, ylim)
    fig.subplots_adjust(wspace=0.18, top=0.85)
    fig.savefig(os.path.join(OUT, "03_voxelized_occupancy.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Figure 4: 3x3 sparse-conv kernel overlap — info exchange vs none
# --------------------------------------------------------------------- #
def fig_kernel_overlap():
    # Same axis limits in both panels so they read as equal-weight.
    panel_xlim = (-1.0, 9.0)
    panel_ylim = (-0.5, 5.5)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    def _draw_voxel_with_kernel(ax, cx, cy, kernel_face, voxel_face, label):
        ax.add_patch(
            mpatches.Rectangle(
                (cx - 1, cy - 1),
                3,
                3,
                facecolor=kernel_face,
                edgecolor=EDGE_COLOR,
                lw=1.0,
                linestyle="--",
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (cx, cy),
                1,
                1,
                facecolor=voxel_face,
                edgecolor=EDGE_COLOR,
                lw=1.2,
            )
        )
        ax.text(
            cx + 0.5,
            cy + 0.5,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    # Panel A: voxels adjacent → 3x3 kernels overlap → message passes.
    ax = axes[0]
    draw_grid(ax, 1.0, panel_xlim, panel_ylim, alpha=0.25, lw=0.4)
    _draw_voxel_with_kernel(ax, 3, 2, KERNEL_FACE_A, CLASS_A_COLOR, "u")
    _draw_voxel_with_kernel(ax, 4, 2, KERNEL_FACE_B, CLASS_B_COLOR, "v")
    # Overlap shading: cols x=3..4 lie in both 3x3 footprints (x∈[2,5] ∩ x∈[3,6]).
    ax.add_patch(
        mpatches.Rectangle(
            (3, 1),
            2,
            3,
            facecolor="#444444",
            alpha=0.20,
            edgecolor="none",
        )
    )
    ax.text(
        4.0,
        4.4,
        "kernel overlap → info exchange ✓",
        ha="center",
        fontsize=10,
        color="#1a7a1a",
        fontweight="bold",
    )
    style_axes(ax, "Voxels close: 3×3 kernels overlap", panel_xlim, panel_ylim)

    # Panel B: voxels far apart → 3x3 kernels disjoint → no message.
    ax = axes[1]
    draw_grid(ax, 1.0, panel_xlim, panel_ylim, alpha=0.25, lw=0.4)
    _draw_voxel_with_kernel(ax, 1, 2, KERNEL_FACE_A, CLASS_A_COLOR, "u")
    _draw_voxel_with_kernel(ax, 6, 2, KERNEL_FACE_B, CLASS_B_COLOR, "v")
    ax.annotate(
        "",
        xy=(6.0, 1.5),
        xytext=(2.0, 1.5),
        arrowprops=dict(arrowstyle="<->", color="#cc0000", lw=1.5),
    )
    ax.text(4.0, 1.0, "gap — kernels never meet", ha="center", color="#cc0000", fontsize=10)
    ax.text(
        4.0,
        4.4,
        "no overlap → no info exchange ✗",
        ha="center",
        fontsize=10,
        color="#cc0000",
        fontweight="bold",
    )
    style_axes(ax, "Voxels far: 3×3 kernels disjoint", panel_xlim, panel_ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_kernel_overlap.png"))
    plt.close(fig)


# --------------------------------------------------------------------- #
# Figure 5: connectivity graph — at each voxel size, draw edges between
# voxels reachable by a 3x3 kernel (Chebyshev distance ≤ 1).
# --------------------------------------------------------------------- #
def fig_connectivity(pts, lab, xlim, ylim):
    sizes = [
        (0.12, "Just right — connected backbone"),
        (0.05, "Too fine — graph fragmented"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (vs, name) in zip(axes, sizes):
        _, unique, inverse = voxelize(pts, vs)
        v_lab, _ = voxel_majority_label(unique, inverse, lab)
        centers = (unique.astype(np.float64) + 0.5) * vs
        # Edges: any pair with Chebyshev distance == 1 (== reachable by one 3x3 conv)
        diff = unique[:, None, :] - unique[None, :, :]
        cheby = np.max(np.abs(diff), axis=-1)
        i_idx, j_idx = np.where(cheby == 1)
        mask = i_idx < j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        # Build segments via LineCollection-style call for speed + uniform style.
        from matplotlib.collections import LineCollection

        segs = np.stack([centers[i_idx], centers[j_idx]], axis=1)
        lc = LineCollection(segs, colors="#333333", linewidths=0.9, alpha=0.7, zorder=1)
        ax.add_collection(lc)

        # Voxel centers — distinct markers per class for colorblind safety.
        for cls, marker in [(0, "o"), (1, "^")]:
            sel = v_lab == cls
            ax.scatter(
                centers[sel, 0],
                centers[sel, 1],
                c=(CLASS_A_COLOR if cls == 0 else CLASS_B_COLOR),
                marker=marker,
                s=40,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )
        n_edges = int(mask.sum())
        n_voxels = unique.shape[0]
        ax.set_xlabel(f"voxels: {n_voxels}    1-hop edges: {n_edges}", fontsize=9)
        style_axes(ax, name, xlim, ylim)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_connectivity_graph.png"))
    plt.close(fig)


# --------------------------------------------------------------------- #
# Figure 6: receptive-field growth — number of layers needed to bridge gap
# --------------------------------------------------------------------- #
def fig_receptive_field(pts, lab, xlim, ylim):
    """Visualize that smaller voxel → larger receptive-field need to span scene."""
    sizes = [0.35, 0.12, 0.05]
    L = 3  # number of stacked 3x3 convs
    rf_world_max = (2 * L + 1) * max(sizes)
    scene_extent = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    # Common axis limits across panels — wide enough to contain the largest RF box.
    cx, cy = (xlim[0] + xlim[1]) * 0.5, (ylim[0] + ylim[1]) * 0.5
    pad = max(rf_world_max - (xlim[1] - xlim[0]), 0.0) * 0.5 + 0.2
    common_xlim = (xlim[0] - pad, xlim[1] + pad)
    pad_y = max(rf_world_max - (ylim[1] - ylim[0]), 0.0) * 0.5 + 0.2
    common_ylim = (ylim[0] - pad_y, ylim[1] + pad_y)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, vs in zip(axes, sizes):
        _, unique, inverse = voxelize(pts, vs)
        v_lab, _ = voxel_majority_label(unique, inverse, lab)
        centers = (unique.astype(np.float64) + 0.5) * vs

        rf_voxels = 2 * L + 1
        rf_world = rf_voxels * vs
        rf_x0, rf_y0 = cx - rf_world / 2, cy - rf_world / 2

        for cls, marker in [(0, "o"), (1, "^")]:
            sel = v_lab == cls
            ax.scatter(
                centers[sel, 0],
                centers[sel, 1],
                c=(CLASS_A_COLOR if cls == 0 else CLASS_B_COLOR),
                marker=marker,
                s=28,
                edgecolors="white",
                linewidths=0.35,
            )
        ax.add_patch(
            mpatches.Rectangle(
                (rf_x0, rf_y0),
                rf_world,
                rf_world,
                facecolor="#cc6600",
                alpha=0.12,
                edgecolor="#cc6600",
                lw=1.6,
                linestyle="--",
            )
        )
        ratio = rf_world / scene_extent
        style_axes(
            ax,
            f"voxel = {vs:.2f}    RF (L={L}) = {rf_world:.2f}  ({ratio*100:.0f}% of scene)",
            common_xlim,
            common_ylim,
        )
    fig.suptitle(
        "Receptive field of 3 stacked 3×3 convs at different voxel sizes",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "06_receptive_field.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main() -> None:
    # Sparse capture (~120 points) — required for the small-voxel
    # disconnection story to be visible. With 800 dense points every
    # cell at vs=0.05 still hits ~6 points and the graph stays connected.
    pts, lab = make_two_moons(n_per_class=60, noise=0.05, seed=0)
    pad = 0.25
    xlim = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ylim = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)

    fig_raw_moons(pts, lab, xlim, ylim)
    fig_three_voxel_sizes(pts, lab, xlim, ylim)
    fig_voxelized_occupancy(pts, lab, xlim, ylim)
    fig_kernel_overlap()
    fig_connectivity(pts, lab, xlim, ylim)
    fig_receptive_field(pts, lab, xlim, ylim)

    print(f"wrote 6 figures to {OUT}")


if __name__ == "__main__":
    main()
