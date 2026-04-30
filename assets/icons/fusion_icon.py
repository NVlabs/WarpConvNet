# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WarpConvNet fusion icon — clean isometric cube + warp threads.

Cube rendered as 3 big visible faces (top, right, left) with 3x3 grid
subdivisions and silhouette cutouts for sparsity. Top face sheared right
to imply spacetime warp. 24 CUDA threads bundle on left → through cube →
fan warped on right.

Renders fusion_icon.{svg,png} and fusion_icon_favicon.{svg,png} in ./figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


def save_figure(fig, name, outdir, formats=("svg", "png"), dpi=100, transparent=True):
    """Write fig to outdir/name.{ext} for each ext in formats."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = outdir / f"{name}.{ext}"
        kwargs = dict(format=ext, bbox_inches="tight", pad_inches=0.02, transparent=transparent)
        if ext in ("png", "jpg", "jpeg", "tiff"):
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


# ---- Palette ---------------------------------------------------------------
BG0 = "#0b1020"
BG1 = "#13203a"
GREEN_BRIGHT = "#c8ff7a"
GREEN_MID = "#a6f24a"
GREEN = "#76B900"  # NVIDIA green
GREEN_DARK = "#5a8f00"
GREEN_DEEP = "#3d6300"
EDGE = "#0a0d18"

# ---- Letters (5x5) ---------------------------------------------------------
# Row 0 = top, col 0 = left, in letter-reading orientation.
LETTER_W = [
    "X...X",
    "X...X",
    "X.X.X",
    "XX.XX",
    "X...X",
]
LETTER_C = [
    ".XXX.",
    "X....",
    "X....",
    "X....",
    ".XXX.",
]
LETTER_N = [
    "X...X",
    "XX..X",
    "X.X.X",
    "X..XX",
    "X...X",
]


def letter_cells(rows):
    """Set of (r, c) coords where letter is filled. r=row top→bottom, c=col left→right."""
    return {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "X"}


# ---- Iso geometry ----------------------------------------------------------
COS30 = np.sqrt(3) / 2
SIN30 = 0.5


def iso(x, y, z):
    """3D → 2D screen coords. Pure isometric, no warp."""
    return (x - y) * COS30, -(x + y) * SIN30 + z


def iso_warp(x, y, z, max_dx=0.0, z_apply=2.0):
    """Iso projection with optional rightward shear above z_apply.

    Below z=z_apply: no warp. Above: linear shear that scales with (z - z_apply).
    Bottom of top layer (z=z_apply) stays flush with mid layer top.
    """
    sx, sy = iso(x, y, z)
    if z > z_apply:
        sx = sx + max_dx * (z - z_apply)
    return sx, sy


# ---- Cube faces ------------------------------------------------------------


def cube_faces(N=3, size=1.0, ox=0.0, oy=0.0, warp_dx_per_z=0.0):
    """Return (top, left, right) face outlines of an NxNxN cube.

    Top face is sheared right by warp_dx_per_z * 1 (one unit of z above z=N-1).
    Left = y=N face. Right = x=N face. Cube spans (0..N, 0..N, 0..N).
    """

    def P(x, y, z):
        sx, sy = iso_warp(x, y, z, max_dx=warp_dx_per_z, z_apply=N)
        return (sx * size + ox, sy * size + oy)

    # Top face (z = N), outer outline
    top = [P(0, 0, N), P(N, 0, N), P(N, N, N), P(0, N, N)]
    # Right face (x = N) — visible east face
    right = [P(N, 0, 0), P(N, N, 0), P(N, N, N), P(N, 0, N)]
    # Left face (y = N) — visible north/back-left face
    left = [P(0, N, 0), P(N, N, 0), P(N, N, N), P(0, N, N)]
    return top, left, right


def grid_lines_top(N=3, size=1.0, ox=0.0, oy=0.0, warp_dx_per_z=0.0):
    """Return list of (start, end) line segments subdividing top face into NxN."""

    def P(x, y, z):
        sx, sy = iso_warp(x, y, z, max_dx=warp_dx_per_z, z_apply=N)
        return (sx * size + ox, sy * size + oy)

    segs = []
    for i in range(1, N):
        # lines along x at y=i (z = N)
        segs.append((P(0, i, N), P(N, i, N)))
        # lines along y at x=i (z = N)
        segs.append((P(i, 0, N), P(i, N, N)))
    return segs


def grid_lines_right(N=3, size=1.0, ox=0.0, oy=0.0):
    """Subdivide right face (x=N): horizontal at each z, vertical at each y."""

    def P(x, y, z):
        sx, sy = iso(x, y, z)
        return (sx * size + ox, sy * size + oy)

    segs = []
    for i in range(1, N):
        segs.append((P(N, 0, i), P(N, N, i)))  # z=i lines
        segs.append((P(N, i, 0), P(N, i, N)))  # y=i lines
    return segs


def grid_lines_left(N=3, size=1.0, ox=0.0, oy=0.0):
    """Subdivide left face (y=N)."""

    def P(x, y, z):
        sx, sy = iso(x, y, z)
        return (sx * size + ox, sy * size + oy)

    segs = []
    for i in range(1, N):
        segs.append((P(0, N, i), P(N, N, i)))  # z=i lines
        segs.append((P(i, N, 0), P(i, N, N)))  # x=i lines
    return segs


def cell_quad_top(gx, gy, N=3, size=1.0, ox=0.0, oy=0.0, warp_dx_per_z=0.0):
    """One 1x1 cell on top face at grid (gx, gy)."""

    def P(x, y, z):
        sx, sy = iso_warp(x, y, z, max_dx=warp_dx_per_z, z_apply=N)
        return (sx * size + ox, sy * size + oy)

    return [P(gx, gy, N), P(gx + 1, gy, N), P(gx + 1, gy + 1, N), P(gx, gy + 1, N)]


def cell_quad_right(gy, gz, N=3, size=1.0, ox=0.0, oy=0.0):
    def P(x, y, z):
        sx, sy = iso(x, y, z)
        return (sx * size + ox, sy * size + oy)

    return [P(N, gy, gz), P(N, gy + 1, gz), P(N, gy + 1, gz + 1), P(N, gy, gz + 1)]


def cell_quad_left(gx, gz, N=3, size=1.0, ox=0.0, oy=0.0):
    def P(x, y, z):
        sx, sy = iso(x, y, z)
        return (sx * size + ox, sy * size + oy)

    return [P(gx, N, gz), P(gx + 1, N, gz), P(gx + 1, N, gz + 1), P(gx, N, gz + 1)]


# ---- Threads ---------------------------------------------------------------


def cubic_bezier(p0, p1, p2, p3, n=140):
    t = np.linspace(0, 1, n)[:, None]
    return (
        ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t**2) * p2 + (t**3) * p3
    )


def add_thread(ax, p0, p1, p2, p3, *, color, lw, alpha, zorder, glow=True):
    pts = cubic_bezier(np.asarray(p0), np.asarray(p1), np.asarray(p2), np.asarray(p3))
    if glow:
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=lw * 3.4,
            alpha=alpha * 0.16,
            solid_capstyle="round",
            zorder=zorder - 0.2,
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=lw * 1.9,
            alpha=alpha * 0.32,
            solid_capstyle="round",
            zorder=zorder - 0.1,
        )
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=color,
        linewidth=lw,
        alpha=alpha,
        solid_capstyle="round",
        zorder=zorder,
    )


# ---- zorder layers ---------------------------------------------------------
Z_BG = -10
Z_BACK_THREADS = 1
Z_CUBE_FACE = 10
Z_CUBE_GRID = 12
Z_CUBE_HOLE = 15  # above cube faces, below front threads
Z_FRONT_THREADS = 60
Z_DUST = 70


# ---- Main render -----------------------------------------------------------


def draw_card(ax):
    ax.add_patch(
        FancyBboxPatch(
            (-1, -1),
            2,
            2,
            boxstyle="round,pad=0,rounding_size=0.18",
            facecolor=BG1,
            edgecolor="none",
            zorder=Z_BG,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (-1, -1),
            2,
            2,
            boxstyle="round,pad=0,rounding_size=0.18",
            facecolor=BG0,
            edgecolor="none",
            alpha=0.30,
            zorder=Z_BG + 1,
        )
    )


def draw_cube(
    ax,
    *,
    N=5,
    size,
    ox,
    oy,
    warp,
    letter_top=None,
    letter_right=None,
    letter_left=None,
    fg_top=GREEN_BRIGHT,
    fg_right=GREEN_MID,
    fg_left=GREEN,
    bg_cell=GREEN_DEEP,
    bg_top=None,
    bg_right=None,
    bg_left=None,
    carve_top=(),
    carve_right=(),
    carve_left=(),
    carve_color=BG1,
    line_w=1.0,
):
    """Draw 3-face isometric cube. Each face split into NxN cells.

    Cells in the letter pattern get fg color (per-face shade); other cells
    get bg color (per-face if set, else bg_cell). Cube outline drawn last.
    """
    bg_top = bg_top or bg_cell
    bg_right = bg_right or bg_cell
    bg_left = bg_left or bg_cell
    carve_top_set = set(carve_top)
    carve_right_set = set(carve_right)
    carve_left_set = set(carve_left)
    # --- top face cells: gx ∈ [0,N), gy ∈ [0,N) ---
    # Letter axes on top face:
    #   letter row 0 (top) = back of cube  → gy = N-1
    #   letter row N-1 (bot) = front       → gy = 0
    #   letter col 0 (left) = left of cube → gx = 0
    #   letter col N-1 (rt) = right        → gx = N-1
    top_letter = letter_cells(letter_top) if letter_top else set()
    for gx in range(N):
        for gy in range(N):
            r = (N - 1) - gy
            c = gx
            in_letter = (r, c) in top_letter
            carved = (r, c) in carve_top_set
            verts = cell_quad_top(gx, gy, N=N, size=size, ox=ox, oy=oy, warp_dx_per_z=warp)
            if carved:
                fc = carve_color
                ec = "none"
            else:
                fc = fg_top if in_letter else bg_top
                ec = EDGE
            ax.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=line_w * 0.6,
                    zorder=Z_CUBE_FACE + 1,
                    joinstyle="miter",
                )
            )

    # --- right face cells (x=N): gy ∈ [0,N), gz ∈ [0,N) ---
    # Letter axes on right face:
    #   row 0 (top) = top of cube → gz = N-1
    #   row N-1     = bottom      → gz = 0
    #   col 0 (left, deep into cube) = back → gy = N-1
    #   col N-1 (front, viewer side)        = gy = 0
    right_letter = letter_cells(letter_right) if letter_right else set()
    for gy in range(N):
        for gz in range(N):
            r = (N - 1) - gz
            c = (N - 1) - gy
            in_letter = (r, c) in right_letter
            carved = (r, c) in carve_right_set
            verts = cell_quad_right(gy, gz, N=N, size=size, ox=ox, oy=oy)
            if carved:
                fc = carve_color
                ec = "none"
            else:
                fc = fg_right if in_letter else bg_right
                ec = EDGE
            ax.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=line_w * 0.6,
                    zorder=Z_CUBE_FACE,
                    joinstyle="miter",
                )
            )

    # --- left face cells (y=N): gx ∈ [0,N), gz ∈ [0,N) ---
    # Letter axes on left face:
    #   row 0 (top) = top of cube → gz = N-1
    #   row N-1     = bottom      → gz = 0
    #   col 0 (left, viewer-side front) = gx = 0
    #   col N-1 (right, back)           = gx = N-1
    left_letter = letter_cells(letter_left) if letter_left else set()
    for gx in range(N):
        for gz in range(N):
            r = (N - 1) - gz
            c = gx
            in_letter = (r, c) in left_letter
            carved = (r, c) in carve_left_set
            verts = cell_quad_left(gx, gz, N=N, size=size, ox=ox, oy=oy)
            if carved:
                fc = carve_color
                ec = "none"
            else:
                fc = fg_left if in_letter else bg_left
                ec = EDGE
            ax.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=line_w * 0.6,
                    zorder=Z_CUBE_FACE,
                    joinstyle="miter",
                )
            )

    # --- cube silhouette outlines (drawn on top to crispen the cube) ---
    top, left, right = cube_faces(N=N, size=size, ox=ox, oy=oy, warp_dx_per_z=warp)
    for verts in (left, right, top):
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor="none",
                edgecolor=EDGE,
                linewidth=line_w * 2.0,
                zorder=Z_CUBE_GRID + 1,
                joinstyle="miter",
            )
        )


def draw_threads(
    ax,
    *,
    n=8,
    cube_left,
    cube_right,
    cube_mid_y,
    cube_cx=0.0,
    entry_x=-0.78,
    exit_x=0.86,
    bundle_h=0.42,
    fan_h=0.78,
    bundle_yc=None,
    lw_front=2.4,
    lw_back=1.8,
    alpha_front=1.0,
    alpha_back=0.55,
):
    """8 threads stacked vertically, each passes horizontally through cube,
    then curves out to fan exit.

    Path = line(entry → cube_right_edge_at_thread_y) + bezier(→ fan exit).
    Threads have vertical spacing both at entry (bundle_h) and through cube.
    """
    if bundle_yc is None:
        bundle_yc = cube_mid_y

    entry_ys = np.linspace(bundle_yc - bundle_h / 2, bundle_yc + bundle_h / 2, n)
    exit_ys = np.linspace(bundle_yc - fan_h, bundle_yc + fan_h, n)

    # All threads behind cube. Track inner/outer set only for thickness/alpha
    # variation (innermost half slightly bolder).
    abs_y = np.abs(exit_ys - cube_mid_y)
    order = np.argsort(abs_y)
    front_idx = set(order[: n // 2].tolist())  # bolder inner half

    for i in range(n):
        ey = entry_ys[i]
        oy_e = exit_ys[i]
        # straight horizontal section: entry → cube horizontal center
        x_curve_start = cube_cx
        straight_pts = np.linspace([entry_x, ey], [x_curve_start, ey], 80)
        # bezier from (cube_cx, ey) → (exit_x, fan_y); first control point
        # sits HORIZONTALLY ahead of b0 so the tangent matches the straight
        # section — no kink at the join.
        b0 = np.array([x_curve_start, ey])
        tangent_lead = (exit_x - x_curve_start) * 0.35
        b1 = np.array([x_curve_start + tangent_lead, ey])
        b2 = np.array([exit_x - tangent_lead, oy_e])
        b3 = np.array([exit_x, oy_e])
        curve_pts = cubic_bezier(b0, b1, b2, b3, n=140)
        pts = np.vstack([straight_pts, curve_pts[1:]])

        # All threads BEHIND cube — zorder lower than cube
        if i in front_idx:
            color, lw, alpha, z = GREEN_BRIGHT, lw_front, alpha_front, Z_BACK_THREADS + 1
        else:
            color, lw, alpha, z = GREEN, lw_back, alpha_back, Z_BACK_THREADS

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=lw * 3.4,
            alpha=alpha * 0.16,
            solid_capstyle="round",
            zorder=z - 0.2,
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=lw * 1.9,
            alpha=alpha * 0.32,
            solid_capstyle="round",
            zorder=z - 0.1,
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=z,
        )

        # Circuit-style terminal dots at entry + exit
        dot_r = 0.018 if i in front_idx else 0.014
        for px, py in ((entry_x, ey), (exit_x, oy_e)):
            ax.add_patch(
                Circle(
                    (px, py),
                    dot_r * 1.8,
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha * 0.25,
                    zorder=z - 0.05,
                )
            )
            ax.add_patch(
                Circle(
                    (px, py),
                    dot_r,
                    facecolor=color,
                    edgecolor=EDGE,
                    linewidth=1.2,
                    alpha=alpha,
                    zorder=z + 0.1,
                )
            )


def draw_bundle_and_dust(ax, *, entry_x, exit_x, bundle_yc, bundle_h):
    ax.add_patch(
        Rectangle(
            (entry_x - 0.05, bundle_yc - bundle_h / 2 - 0.015),
            0.06,
            bundle_h + 0.03,
            facecolor=GREEN_MID,
            edgecolor=GREEN_BRIGHT,
            linewidth=1.2,
            alpha=0.6,
            zorder=Z_FRONT_THREADS - 1,
        )
    )
    rng = np.random.default_rng(23)
    for _ in range(8):
        x = rng.uniform(entry_x - 0.10, entry_x + 0.04)
        y = rng.uniform(bundle_yc - 0.20, bundle_yc + 0.20)
        ax.add_patch(
            Circle(
                (x, y),
                rng.uniform(0.005, 0.013),
                facecolor=GREEN_MID,
                edgecolor="none",
                alpha=0.85,
                zorder=Z_DUST,
            )
        )
    for _ in range(10):
        x = rng.uniform(exit_x - 0.06, exit_x + 0.04)
        y = rng.uniform(bundle_yc - 0.82, bundle_yc + 0.82)
        ax.add_patch(
            Circle(
                (x, y),
                rng.uniform(0.004, 0.011),
                facecolor=GREEN_BRIGHT,
                edgecolor="none",
                alpha=0.92,
                zorder=Z_DUST,
            )
        )


def build():
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_card(ax)

    # Cube: 5x5x5 with letters W/C/N on left/top/right faces
    N = 5
    size = 0.135  # bigger — fills icon, letters readable
    warp = 0.07
    ox = -warp / 2
    oy = 0.0

    draw_cube(
        ax,
        N=N,
        size=size,
        ox=ox,
        oy=oy,
        warp=warp,
        letter_top=LETTER_C,
        letter_right=LETTER_N,
        letter_left=LETTER_W,
        fg_top=GREEN_BRIGHT,
        fg_right=GREEN_MID,
        fg_left=GREEN,
        bg_top="#2e4a00",
        bg_right="#1f3300",
        bg_left="#13200a",
        line_w=1.2,
    )

    # Cube screen bounds (post-warp)
    cube_left = -N * COS30 * size + ox - 0.02
    cube_right = N * COS30 * size + warp + ox + 0.02
    cube_top = N * size + oy
    cube_bot = -N * size + oy
    cube_mid_y = (cube_top + cube_bot) / 2

    cube_cx = (cube_left + cube_right) / 2
    entry_x = -0.92
    exit_x = 0.92
    bundle_h = 0.58  # span most of cube vertical for visible spacing
    draw_threads(
        ax,
        n=8,
        cube_left=cube_left,
        cube_right=cube_right,
        cube_mid_y=cube_mid_y,
        cube_cx=cube_cx,
        entry_x=entry_x,
        exit_x=exit_x,
        bundle_h=bundle_h,
        fan_h=0.88,
    )
    return fig


def build_favicon():
    """Stripped-down icon: solid 2x2 cube + single bright thread streak."""
    fig = plt.figure(figsize=(2.56, 2.56), dpi=100, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (-1, -1),
            2,
            2,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor=BG1,
            edgecolor="none",
            zorder=Z_BG,
        )
    )

    # Big chunky 2x2x2 cube — no letters at favicon size (illegible)
    N = 2
    size = 0.30
    warp = 0.06
    ox = -warp / 2
    oy = -0.05

    draw_cube(
        ax,
        N=N,
        size=size,
        ox=ox,
        oy=oy,
        warp=warp,
        letter_top=None,
        letter_right=None,
        letter_left=None,
        fg_top=GREEN_BRIGHT,
        fg_right=GREEN,
        fg_left=GREEN_DARK,
        bg_cell=GREEN_DARK,
        line_w=4.0,
    )

    # One bold horizontal thread cutting through cube center
    cube_mid_y = oy
    p0 = (-0.92, cube_mid_y)
    p1 = (-0.20, cube_mid_y)
    p2 = (0.20, cube_mid_y + 0.06)
    p3 = (0.92, cube_mid_y + 0.18)
    add_thread(ax, p0, p1, p2, p3, color=GREEN_BRIGHT, lw=5.0, alpha=1.0, zorder=Z_FRONT_THREADS)
    # Bundle dot
    ax.add_patch(
        Circle(
            (-0.85, cube_mid_y),
            0.11,
            facecolor=GREEN_BRIGHT,
            edgecolor="none",
            alpha=0.95,
            zorder=Z_FRONT_THREADS - 1,
        )
    )
    return fig


def build_banner():
    """Wide 4:1 README banner. Cube on left, wordmark + tagline on right.

    Threads stream from cube across the banner. Ratio 1920:480 → 12.8 x 3.2 in
    at 150 DPI. Saved as fusion_banner.{svg,png}.
    """
    W_IN, H_IN = 12.8, 3.2
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=150, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    # Use aspect ratio of the figure for x/y; map [0, ratio] x [0, 1]
    AR = W_IN / H_IN
    ax.set_xlim(0, AR)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background card with rounded corners
    pad = 0.0
    ax.add_patch(
        FancyBboxPatch(
            (pad, -0.5 + pad),
            AR - 2 * pad,
            1 - 2 * pad,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor=BG1,
            edgecolor="none",
            zorder=Z_BG,
        )
    )
    # Subtle radial-ish vignette via top overlay
    ax.add_patch(
        FancyBboxPatch(
            (pad, -0.5 + pad),
            AR - 2 * pad,
            1 - 2 * pad,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor=BG0,
            edgecolor="none",
            alpha=0.25,
            zorder=Z_BG + 1,
        )
    )

    # ---- Cube on left ------------------------------------------------------
    cube_cx = 0.85
    cube_cy = 0.0
    N = 5
    size = 0.07
    warp = 0.04
    cube_ox = cube_cx - 0  # cube origin shift in iso (computed in cell helpers)
    # Use cube helpers but translate via ox/oy
    draw_cube(
        ax,
        N=N,
        size=size,
        ox=cube_cx - warp / 2,
        oy=cube_cy,
        warp=warp,
        letter_top=LETTER_C,
        letter_right=LETTER_N,
        letter_left=LETTER_W,
        fg_top=GREEN_BRIGHT,
        fg_right=GREEN_MID,
        fg_left=GREEN,
        bg_top="#2e4a00",
        bg_right="#1f3300",
        bg_left="#13200a",
        line_w=0.7,
    )

    # Cube screen bounds (approx)
    cube_left = cube_cx - N * COS30 * size - 0.02
    cube_right = cube_cx + N * COS30 * size + warp + 0.02
    cube_mid_y = cube_cy

    # ---- Threads (compact, matched to icon proportions) -------------------
    n_threads = 8
    entry_x = cube_left - 0.22  # short straight section on left (matches icon)
    exit_x = cube_right + 0.20  # short fan exit on right
    bundle_h = 0.35
    fan_h = 0.40
    entry_ys = np.linspace(cube_mid_y - bundle_h / 2, cube_mid_y + bundle_h / 2, n_threads)
    exit_ys = np.linspace(cube_mid_y - fan_h, cube_mid_y + fan_h, n_threads)

    abs_y = np.abs(exit_ys - cube_mid_y)
    order = np.argsort(abs_y)
    front_idx = set(order[: n_threads // 2].tolist())

    for i in range(n_threads):
        ey = entry_ys[i]
        oy_e = exit_ys[i]
        x_curve_start = cube_cx
        straight_pts = np.linspace([entry_x, ey], [x_curve_start, ey], 80)
        b0 = np.array([x_curve_start, ey])
        tangent_lead = (exit_x - x_curve_start) * 0.35
        b1 = np.array([x_curve_start + tangent_lead, ey])
        b2 = np.array([exit_x - tangent_lead, oy_e])
        b3 = np.array([exit_x, oy_e])
        curve_pts = cubic_bezier(b0, b1, b2, b3, n=120)
        pts = np.vstack([straight_pts, curve_pts[1:]])

        if i in front_idx:
            color, lw, alpha = GREEN_BRIGHT, 1.6, 1.0
        else:
            color, lw, alpha = GREEN, 1.2, 0.55
        z = Z_BACK_THREADS + (1 if i in front_idx else 0)

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=lw * 3.4,
            alpha=alpha * 0.16,
            solid_capstyle="round",
            zorder=z - 0.2,
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=lw * 1.9,
            alpha=alpha * 0.32,
            solid_capstyle="round",
            zorder=z - 0.1,
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=z,
        )

        # circuit dots at ends
        for px, py in ((entry_x, ey), (exit_x, oy_e)):
            ax.add_patch(
                Circle(
                    (px, py),
                    0.012,
                    facecolor=color,
                    edgecolor=EDGE,
                    linewidth=0.8,
                    alpha=alpha,
                    zorder=z + 0.1,
                )
            )

    # ---- Wordmark + tagline -----------------------------------------------
    # "WarpConvNet" — large, mostly white with green accent
    title_x = exit_x + 0.45  # breathing room between threads and text
    ax.text(
        title_x,
        0.10,
        "WarpConvNet",
        fontsize=44,
        fontweight="bold",
        color="#e8eef9",
        family="sans-serif",
        ha="left",
        va="center",
        zorder=Z_DUST,
    )
    ax.text(
        title_x,
        -0.16,
        "High-Performance 3D Deep Learning",
        fontsize=18,
        fontweight="medium",
        color=GREEN_MID,
        family="sans-serif",
        ha="left",
        va="center",
        zorder=Z_DUST,
    )
    ax.text(
        title_x,
        -0.32,
        "Point Clouds  ·  Sparse Voxels  ·  CUDA",
        fontsize=12,
        fontweight="normal",
        color="#8aa0c0",
        family="sans-serif",
        ha="left",
        va="center",
        zorder=Z_DUST,
    )

    return fig


def main():
    outdir = Path(__file__).resolve().parent / "figures"

    fig = build()
    for p in save_figure(fig, "fusion_icon", outdir, dpi=100):
        print(f"wrote {p}")
    plt.close(fig)

    fig = build_favicon()
    for p in save_figure(fig, "fusion_icon_favicon", outdir, dpi=100):
        print(f"wrote {p}")
    plt.close(fig)

    fig = build_banner()
    for p in save_figure(fig, "fusion_banner", outdir, dpi=150):
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
