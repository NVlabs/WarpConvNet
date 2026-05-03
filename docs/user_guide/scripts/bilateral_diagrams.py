# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagrams illustrating bilateral filter as high-dimensional sparse convolution.

Designed for the WarpConvNet docs (mkdocs page width ~700-900 px). Uses the
figura skill's pubstyle / colors / export modules so output is print-ready
vector PDF/SVG with embedded fonts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Wire in the figura skill scripts.
SKILL_SCRIPTS = Path("/home/cchoy/.claude/plugins/cache/figura/figura/0.4.0/skills/figura/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle  # noqa: E402

import colors  # noqa: E402
import export  # noqa: E402
import pubstyle  # noqa: E402


OUT = Path(__file__).resolve().parent.parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

pubstyle.apply()
colors.apply_cycle()

# Locked palette: same hue for the same role across all three figures so a
# reader scanning the page builds one mental color->concept map.
C_DARK = colors.OKABE_ITO[0]  # blue            -> dark region / cube role
C_BRIGHT = colors.OKABE_ITO[1]  # vermilion      -> bright region / populated cells
C_QUERY = colors.OKABE_ITO[3]  # reddish purple  -> query points
C_BLUR = colors.OKABE_ITO[2]  # bluish green    -> blur kernel / simplex role
C_SLICE = colors.OKABE_ITO[5]  # orange          -> slice arrows
C_GUIDE = "#7d7d7d"  # neutral guides / labels
C_GRID = "#e0e0e0"


def _hide_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)


def _grid_axes(ax, lo: int = 0, hi: int = 4) -> None:
    ax.set_xlim(lo - 0.4, hi + 0.4)
    ax.set_ylim(lo - 0.4, hi + 0.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(lo, hi + 1))
    ax.set_yticks(range(lo, hi + 1))
    ax.grid(True, color=C_GRID, lw=0.5, zorder=0)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Figure 1: 1D signal -> 2D bilateral lift
# ---------------------------------------------------------------------------
def figure_lift() -> None:
    rng = np.random.default_rng(0)
    n_left, n_right = 22, 22
    x_left = np.linspace(0.05, 0.45, n_left)
    x_right = np.linspace(0.55, 0.95, n_right)
    i_left = 0.25 + rng.normal(0, 0.02, n_left)
    i_right = 0.78 + rng.normal(0, 0.02, n_right)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.6),
        gridspec_kw={"wspace": 0.32},
        constrained_layout=True,
    )

    # ---- left: 1D signal ----------------------------------------------------
    ax = axes[0]
    ax.set_title("1D signal: pixel position vs. intensity")
    ax.scatter(x_left, i_left, c=C_DARK, s=18, label="dark region", zorder=3)
    ax.scatter(x_right, i_right, c=C_BRIGHT, s=18, label="bright region", zorder=3)
    ax.axvline(0.5, color=C_GUIDE, ls="--", lw=0.8, zorder=1)
    ax.text(0.52, 1.18, "edge", ha="left", va="center", color=C_GUIDE, fontsize=8)
    ax.set_xlabel("pixel position $x$")
    ax.set_ylabel("intensity $I(x)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.25, 0.99),
        ncol=1,
        handletextpad=0.4,
        borderpad=0.2,
    )

    # ---- right: 2D bilateral lift ------------------------------------------
    ax = axes[1]
    ax.set_title(r"2D bilateral lift: $(x/\sigma_x,\; I/\sigma_I)$")
    sx, si = 0.05, 0.08
    lx_l, li_l = x_left / sx, i_left / si
    lx_r, li_r = x_right / sx, i_right / si

    # Background sparse-grid hint
    for xc in range(0, 22, 2):
        ax.axvline(xc, color=C_GRID, lw=0.4, zorder=0)
    for yc in range(0, 12, 2):
        ax.axhline(yc, color=C_GRID, lw=0.4, zorder=0)

    ax.scatter(lx_l, li_l, c=C_DARK, s=18, zorder=3)
    ax.scatter(lx_r, li_r, c=C_BRIGHT, s=18, zorder=3)

    # Indicate the intensity gap (clearly outside data clusters)
    y_mid = 0.5 * (li_l.mean() + li_r.mean())
    ax.axhline(y_mid, color=C_GUIDE, ls="--", lw=0.8, zorder=1)
    ax.text(
        21,
        y_mid,
        "intensity gap",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=C_GUIDE,
    )

    # Single short annotation anchored at the gap, placed below the data
    edge_x = 0.5 / sx
    ax.annotate(
        "spatially adjacent,\nseparated in intensity",
        xy=(edge_x, y_mid),
        xytext=(edge_x, -1.5),
        ha="center",
        va="top",
        fontsize=7.5,
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            lw=0.6,
            shrinkA=2,
            shrinkB=2,
        ),
    )

    ax.set_xlabel(r"$x / \sigma_x$")
    ax.set_ylabel(r"$I / \sigma_I$")
    ax.set_xlim(-0.5, 21.5)
    ax.set_ylim(-4.5, 12)

    fig.suptitle(
        "Bilateral lift: range axis separates spatially-overlapping regions",
        fontsize=10,
    )

    export.save(fig, "bilateral_lift_1d", formats=("pdf", "svg", "png"), outdir=OUT)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: splat-blur-slice on a 2D bilateral grid (bilinear)
# ---------------------------------------------------------------------------
def figure_bilinear() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.8, 2.6),
        gridspec_kw={"wspace": 0.18},
        constrained_layout=True,
    )

    # Query at (1.7, 1.35) inside cell (1,1)-(2,2)
    qx, qy = 1.7, 1.35
    fx, fy = qx - 1, qy - 1
    weights = {
        (1, 1): (1 - fx) * (1 - fy),
        (2, 1): fx * (1 - fy),
        (1, 2): (1 - fx) * fy,
        (2, 2): fx * fy,
    }
    corners = list(weights.keys())

    # ---- panel A: SPLAT --------------------------------------------------
    ax = axes[0]
    ax.set_title("(a) Splat")
    _grid_axes(ax)

    ax.add_patch(
        Rectangle(
            (1, 1),
            1,
            1,
            facecolor=C_DARK,
            alpha=0.06,
            edgecolor=C_DARK,
            lw=0.8,
            zorder=1,
        )
    )
    for i, j in corners:
        w = weights[(i, j)]
        ax.scatter([i], [j], c=C_DARK, s=20 + 220 * w, zorder=4, edgecolor="white", linewidth=0.8)
        dx = 0.18 if i == 2 else -0.18
        dy = 0.18 if j == 2 else -0.18
        ha = "left" if i == 2 else "right"
        va = "bottom" if j == 2 else "top"
        ax.text(i + dx, j + dy, f"$w={w:.2f}$", fontsize=7.5, ha=ha, va=va, color=C_DARK)
        ax.add_patch(
            FancyArrowPatch(
                (qx, qy),
                (i, j),
                arrowstyle="->",
                color=C_DARK,
                lw=0.8,
                mutation_scale=7,
                shrinkA=3,
                shrinkB=5,
                alpha=0.85,
            )
        )

    ax.scatter([qx], [qy], c=C_QUERY, s=42, zorder=6, edgecolor="white", linewidth=0.8)
    ax.annotate(
        r"query $\mathbf{p}$",
        xy=(qx, qy),
        xytext=(0.05, 3.8),
        fontsize=8,
        color=C_QUERY,
        arrowprops=dict(arrowstyle="->", color=C_QUERY, lw=0.6, shrinkA=2, shrinkB=4),
    )
    ax.text(
        2.0,
        -0.55,
        r"value $v \to w_{ij} \cdot v$ at 4 corners",
        ha="center",
        va="top",
        fontsize=7.5,
        color="black",
    )

    # ---- panel B: BLUR ---------------------------------------------------
    ax = axes[1]
    ax.set_title("(b) Blur")
    populated = {(0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)}
    tap_cells = {(1, 2), (2, 2), (3, 2)}
    _grid_axes(ax)

    for i, j in populated:
        ax.add_patch(
            Rectangle(
                (i - 0.42, j - 0.42),
                0.84,
                0.84,
                facecolor=C_DARK,
                alpha=0.08,
                edgecolor=C_DARK,
                lw=0.6,
                zorder=1,
            )
        )
        if (i, j) not in tap_cells:
            ax.scatter([i], [j], c=C_DARK, s=14, zorder=3)

    # Highlight 3-tap blur along x at row j0=2
    j0 = 2
    for i, w in zip([1, 2, 3], [0.25, 0.5, 0.25]):
        ax.add_patch(
            Rectangle(
                (i - 0.42, j0 - 0.42),
                0.84,
                0.84,
                facecolor=C_BLUR,
                alpha=0.18 + 0.5 * w,
                edgecolor=C_BLUR,
                lw=1.0,
                zorder=2,
            )
        )
        ax.text(
            i,
            j0,
            f"{w:g}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            zorder=5,
        )

    # Two-pass kernel sketch in clear right margin (small mini-diagram with
    # an x-arrow then a y-arrow, plus a one-line label). No leader lines
    # into the cell area, so nothing can overlap the green cells.
    ax.set_xlim(-0.4, 6.6)

    # x-pass mini arrow
    ax.annotate(
        "",
        xy=(5.7, 1.2),
        xytext=(4.7, 1.2),
        arrowprops=dict(arrowstyle="->", color=C_BLUR, lw=1.0),
    )
    ax.text(5.2, 0.85, r"$x$", ha="center", va="top", fontsize=8, color=C_BLUR)

    # y-pass mini arrow
    ax.annotate(
        "",
        xy=(5.2, 3.0),
        xytext=(5.2, 2.0),
        arrowprops=dict(arrowstyle="->", color=C_BLUR, lw=1.0),
    )
    ax.text(5.45, 2.5, r"$y$", ha="left", va="center", fontsize=8, color=C_BLUR)

    # Header label well above kernel cells, in clear right margin
    ax.text(5.2, 3.7, "two-pass\n3-tap blur", ha="center", va="bottom", fontsize=7.5, color=C_BLUR)

    ax.text(
        2.0,
        -0.55,
        "depthwise sparse conv on populated cells",
        ha="center",
        va="top",
        fontsize=7.5,
        color="black",
    )

    # ---- panel C: SLICE --------------------------------------------------
    ax = axes[2]
    ax.set_title("(c) Slice")
    _grid_axes(ax)

    ax.add_patch(
        Rectangle(
            (1, 1),
            1,
            1,
            facecolor=C_SLICE,
            alpha=0.06,
            edgecolor=C_SLICE,
            lw=0.8,
            zorder=1,
        )
    )
    for i, j in corners:
        w = weights[(i, j)]
        ax.scatter([i], [j], c=C_SLICE, s=20 + 220 * w, zorder=4, edgecolor="white", linewidth=0.8)
        dx = 0.18 if i == 2 else -0.18
        dy = 0.18 if j == 2 else -0.18
        ha = "left" if i == 2 else "right"
        va = "bottom" if j == 2 else "top"
        ax.text(i + dx, j + dy, f"$w={w:.2f}$", fontsize=7.5, ha=ha, va=va, color=C_SLICE)
        ax.add_patch(
            FancyArrowPatch(
                (i, j),
                (qx, qy),
                arrowstyle="->",
                color=C_SLICE,
                lw=0.8,
                mutation_scale=7,
                shrinkA=5,
                shrinkB=3,
                alpha=0.9,
            )
        )

    ax.scatter([qx], [qy], c=C_QUERY, s=42, zorder=6, edgecolor="white", linewidth=0.8)

    ax.text(
        2.0,
        -0.55,
        r"$\hat v = \sum_{ij} w_{ij}\,\tilde v_{ij}$",
        ha="center",
        va="top",
        fontsize=8,
        color="black",
    )

    fig.suptitle(
        "Bilateral filter on a 2D lifted grid = sparse depthwise convolution",
        fontsize=10,
    )

    export.save(fig, "bilateral_bilinear", formats=("pdf", "svg", "png"), outdir=OUT)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: permutohedral lattice in 2D
# ---------------------------------------------------------------------------
def figure_permutohedral_2d() -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.8, 3.2),
        gridspec_kw={"width_ratios": [1.0, 1.45], "wspace": 0.06},
        constrained_layout=True,
    )

    # ---- (a) cube vs simplex -----------------------------------------------
    ax = axes[0]
    ax.set_title("(a) Cube ($2^d$) vs simplex ($d{+}1$)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.0, 3.4)
    _hide_axes(ax)

    # Cube cell (left half)
    ax.add_patch(
        Rectangle(
            (0, 0),
            2,
            2,
            facecolor=C_DARK,
            alpha=0.08,
            edgecolor=C_DARK,
            lw=1.0,
        )
    )
    cube_corners = [(0, 0), (2, 0), (0, 2), (2, 2)]
    qx, qy = 1.4, 0.9
    for c in cube_corners:
        ax.add_patch(
            FancyArrowPatch(
                (qx, qy),
                c,
                arrowstyle="->",
                color=C_DARK,
                lw=0.7,
                mutation_scale=7,
                shrinkA=3,
                shrinkB=5,
                alpha=0.85,
            )
        )
    for c in cube_corners:
        ax.scatter(*c, c=C_DARK, s=42, zorder=4, edgecolor="white", linewidth=0.8)
    ax.scatter([qx], [qy], c=C_QUERY, s=42, zorder=5, edgecolor="white", linewidth=0.8)
    ax.text(1.0, -0.5, r"$2^d = 4$ corners", ha="center", va="top", color=C_DARK, fontsize=9)

    # Simplex (right half)
    tri = np.array([(3.2, 0.0), (5.2, 0.0), (4.2, 2.0)])
    ax.add_patch(Polygon(tri, closed=True, facecolor=C_BLUR, alpha=0.12, edgecolor=C_BLUR, lw=1.0))
    qx2, qy2 = 4.2, 0.7
    for c in tri:
        ax.add_patch(
            FancyArrowPatch(
                (qx2, qy2),
                tuple(c),
                arrowstyle="->",
                color=C_BLUR,
                lw=0.7,
                mutation_scale=7,
                shrinkA=3,
                shrinkB=5,
                alpha=0.9,
            )
        )
    for c in tri:
        ax.scatter(*c, c=C_BLUR, s=42, zorder=4, edgecolor="white", linewidth=0.8)
    ax.scatter([qx2], [qy2], c=C_QUERY, s=42, zorder=5, edgecolor="white", linewidth=0.8)
    ax.text(4.2, -0.5, r"$d{+}1 = 3$ vertices", ha="center", va="top", color=C_BLUR, fontsize=9)
    ax.text(2.6, 2.7, r"$d=2$", ha="center", va="bottom", fontsize=9, color="0.3")

    # ---- (b) triangular lattice tessellation -------------------------------
    ax = axes[1]
    ax.set_title("(b) Permutohedral lattice ($d=2$)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 9.5)
    ax.set_ylim(-0.5, 3.6)
    _hide_axes(ax)

    h = np.sqrt(3) / 2
    rows, cols = 5, 6
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = c + (0.5 if r % 2 else 0.0)
            y = r * h
            pts.append((x, y))

    # Draw alternating triangles
    for r in range(rows - 1):
        for c in range(cols - 1):
            if r % 2 == 0:
                t1 = [(c, r * h), (c + 1, r * h), (c + 0.5, (r + 1) * h)]
                t2 = [(c + 1, r * h), (c + 0.5, (r + 1) * h), (c + 1.5, (r + 1) * h)]
            else:
                t1 = [(c + 0.5, r * h), (c + 1.5, r * h), (c + 1, (r + 1) * h)]
                t2 = [(c, (r + 1) * h), (c + 0.5, r * h), (c + 1, (r + 1) * h)]
            for tri in (t1, t2):
                ax.add_patch(
                    Polygon(
                        tri,
                        closed=True,
                        fill=False,
                        edgecolor="0.78",
                        lw=0.5,
                        zorder=1,
                    )
                )

    pts_arr = np.array(pts)
    ax.scatter(pts_arr[:, 0], pts_arr[:, 1], c="0.78", s=8, zorder=2)

    # Populated vertices (vermilion) — distinct from cube blue in (a)
    populated_idx = [10, 11, 12, 16, 17, 18, 22, 23]
    pop_pts = pts_arr[populated_idx]
    ax.scatter(
        pop_pts[:, 0], pop_pts[:, 1], c=C_BRIGHT, s=28, zorder=4, edgecolor="white", linewidth=0.6
    )

    # Query inside one triangle
    tri_q = np.array([(2.0, 0.0), (3.0, 0.0), (2.5, h)])
    q = tri_q.mean(axis=0) + np.array([0.05, -0.05])
    ax.add_patch(
        Polygon(
            tri_q, closed=True, facecolor=C_BLUR, alpha=0.22, edgecolor=C_BLUR, lw=1.2, zorder=3
        )
    )
    for v in tri_q:
        ax.add_patch(
            FancyArrowPatch(
                tuple(q),
                tuple(v),
                arrowstyle="->",
                color=C_BLUR,
                lw=0.9,
                alpha=0.9,
                mutation_scale=7,
                shrinkA=3,
                shrinkB=4,
            )
        )
    for v in tri_q:
        ax.scatter(*v, c=C_BLUR, s=44, zorder=5, edgecolor="white", linewidth=0.7)
    ax.scatter([q[0]], [q[1]], c=C_QUERY, s=46, zorder=6, edgecolor="white", linewidth=0.7)

    # Callout placed in clear space right of lattice
    ax.annotate(
        "only populated\nvertices stored\n(PackedHashTable128)",
        xy=(pts_arr[17][0], pts_arr[17][1]),
        xytext=(7.0, 2.7),
        fontsize=8,
        color=C_BRIGHT,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color=C_BRIGHT, lw=0.7, shrinkA=4, shrinkB=4),
    )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=C_QUERY, markersize=7, label="query"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_BLUR,
            markersize=7,
            label="splat / slice",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_BRIGHT,
            markersize=7,
            label="populated",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="0.78", markersize=5.5, label="empty"
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        frameon=True,
        framealpha=0.97,
        edgecolor="0.85",
        fontsize=7.5,
        handlelength=1.0,
        borderpad=0.4,
    )

    fig.suptitle(
        "Permutohedral lattice: $d{+}1$ neighbors per cell instead of $2^d$",
        fontsize=10,
    )

    export.save(fig, "bilateral_permutohedral_2d", formats=("pdf", "svg", "png"), outdir=OUT)
    plt.close(fig)


def main() -> None:
    figure_lift()
    figure_bilinear()
    figure_permutohedral_2d()
    print(f"Wrote diagrams to {OUT}")


if __name__ == "__main__":
    main()
