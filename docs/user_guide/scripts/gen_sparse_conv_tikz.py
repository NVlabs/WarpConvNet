# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate four sparse-convolution diagrams as static PDF + animated GIF.

Mirrors the visual idiom of NVlabs/MinkowskiEngine's sparse_tensor_network
docs (isometric grids, blue input below, teal output above, kernel "frustum"
lines connecting them) but uses WarpConvNet terminology (stride=1 instead
of submanifold). Each diagram renders multiple TikZ frames; PNG frames are
assembled into a looping GIF via PIL, and one representative frame is also
saved as PDF + PNG for static embedding.

Variants:
    sparse_conv_dense       : dense input  -> dense output, 3x3 kernel
    sparse_conv_sparse      : sparse input -> sparse output, 3x3 kernel
    sparse_conv_stride1     : sparse input -> output preserves input coords
    sparse_conv_generalized : arbitrary input/output sets, arbitrary offsets
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Common TikZ preamble. Includes a fixed bounding box so every animation
# frame compiles to the same canvas size (required for GIF assembly).
# ----------------------------------------------------------------------------
PREAMBLE = r"""
\documentclass[tikz,border=4pt]{standalone}
\usepackage{tikz}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usetikzlibrary{calc}

% --- MinkowskiEngine-flavored palette (teal output / blue input / gray) -----
\definecolor{outTeal}{HTML}{2FA89C}
\definecolor{outDark}{HTML}{1F7B72}
\definecolor{inBlue}{HTML}{2F8BC2}
\definecolor{inDark}{HTML}{1B5F8C}
\definecolor{inTint}{HTML}{D9EAF6}     % very light blue: receptive-field shading
\definecolor{cellGray}{HTML}{B5C2C8}
\definecolor{dashTeal}{HTML}{2C4A55}
\definecolor{linkInk}{HTML}{222222}

% --- Isometric projection ----------------------------------------------------
% Cell (i,j) on layer L (0=bottom input, 1=top output) projects to
%   (i*EX + j*EY, j*EZ + L*LH).
\newcommand{\EX}{1.00}
\newcommand{\EY}{0.55}
\newcommand{\EZ}{0.65}
\newcommand{\LH}{5.00}

% Filled cell (i, j, layer, fill, draw)
\newcommand{\cell}[5]{%
  \fill[#4,draw=#5,line width=0.55pt,line join=round]
    ({#1*\EX + #2*\EY}, {#2*\EZ + #3*\LH}) --
    ({(#1+1)*\EX + #2*\EY}, {#2*\EZ + #3*\LH}) --
    ({(#1+1)*\EX + (#2+1)*\EY}, {(#2+1)*\EZ + #3*\LH}) --
    ({#1*\EX + (#2+1)*\EY}, {(#2+1)*\EZ + #3*\LH}) -- cycle;
}

% Empty (dashed-outline-only) scaffold cell (i, j, layer)
\newcommand{\empcell}[3]{%
  \draw[dashTeal,line width=0.45pt,dash pattern=on 2pt off 1.6pt,line join=round]
    ({#1*\EX + #2*\EY}, {#2*\EZ + #3*\LH}) --
    ({(#1+1)*\EX + #2*\EY}, {#2*\EZ + #3*\LH}) --
    ({(#1+1)*\EX + (#2+1)*\EY}, {(#2+1)*\EZ + #3*\LH}) --
    ({#1*\EX + (#2+1)*\EY}, {(#2+1)*\EZ + #3*\LH}) -- cycle;
}

% Helpers for absolute corners on each layer.
\newcommand{\corner}[4]{({(#1+#3)*\EX + (#2+#4)*\EY}, {(#2+#4)*\EZ + 1*\LH})}
\newcommand{\incorner}[4]{({(#1+#3)*\EX + (#2+#4)*\EY}, {(#2+#4)*\EZ + 0*\LH})}

% Four corner-to-corner frustum lines from output cell (oi, oj) to the 3x3
% input neighborhood centered at the same coords (kernel size 3, dilation 1,
% same-padding stride=1).
\newcommand{\frustum}[2]{%
  \draw[linkInk,line width=0.55pt,line cap=round]
    \corner{#1}{#2}{0}{0} -- \incorner{#1-1}{#2-1}{0}{0};
  \draw[linkInk,line width=0.55pt,line cap=round]
    \corner{#1}{#2}{1}{0} -- \incorner{#1+1}{#2-1}{1}{0};
  \draw[linkInk,line width=0.55pt,line cap=round]
    \corner{#1}{#2}{1}{1} -- \incorner{#1+1}{#2+1}{1}{1};
  \draw[linkInk,line width=0.55pt,line cap=round]
    \corner{#1}{#2}{0}{1} -- \incorner{#1-1}{#2+1}{0}{1};
}

% Generic kernel link: undirected center-to-center line.
\newcommand{\link}[4]{%
  \draw[linkInk,line width=0.55pt,line cap=round]
    \corner{#1}{#2}{0.5}{0.5} -- \incorner{#3}{#4}{0.5}{0.5};
}

% Generative frustum: 4 corner lines from input cell (fi,fj) on layer 0
% to the corners of the 3x3 output neighborhood on layer 1.
% Mirror image of \frustum (direction reversed: input -> output).
\newcommand{\genfrustum}[2]{%
  \draw[linkInk,line width=0.55pt,line cap=round]
    \incorner{#1}{#2}{0}{0} -- \corner{#1-1}{#2-1}{0}{0};
  \draw[linkInk,line width=0.55pt,line cap=round]
    \incorner{#1}{#2}{1}{0} -- \corner{#1+1}{#2-1}{1}{0};
  \draw[linkInk,line width=0.55pt,line cap=round]
    \incorner{#1}{#2}{1}{1} -- \corner{#1+1}{#2+1}{1}{1};
  \draw[linkInk,line width=0.55pt,line cap=round]
    \incorner{#1}{#2}{0}{1} -- \corner{#1-1}{#2+1}{0}{1};
}
"""

# Fixed bounding box for animation frame consistency. Encompasses a 5x5 grid
# on both layers plus a small margin for frustum lines.
BBOX = r"\useasboundingbox (-0.6, -0.6) rectangle (8.6, 9.2);"


def _wrap(body: str) -> str:
    return (
        PREAMBLE
        + r"""
\begin{document}
\begin{tikzpicture}
"""
        + BBOX
        + "\n"
        + body
        + r"""
\end{tikzpicture}
\end{document}
"""
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _filled(cells: Iterable[tuple], layer: int, fill: str, draw: str) -> str:
    return "\n".join(f"  \\cell{{{i}}}{{{j}}}{{{layer}}}{{{fill}}}{{{draw}}}" for (i, j) in cells)


def _scaffold(layer: int, w: int, h: int) -> str:
    return "\n".join(f"  \\empcell{{{i}}}{{{j}}}{{{layer}}}" for j in range(h) for i in range(w))


def _reach_3x3(
    focus: tuple[int, int], layer: int = 0, grid_w: int = 5, grid_h: int = 5
) -> list[tuple[int, int]]:
    """3x3 receptive-field cells around `focus`, clipped to grid bounds."""
    fi, fj = focus
    return [
        (i, j)
        for j in (fj - 1, fj, fj + 1)
        for i in (fi - 1, fi, fi + 1)
        if 0 <= i < grid_w and 0 <= j < grid_h
    ]


# ----------------------------------------------------------------------------
# Diagram 1: dense input -> dense output, 3x3 kernel
# ----------------------------------------------------------------------------
def render_dense(focus: tuple[int, int]) -> str:
    """One animation frame for the dense diagram. `focus` = output cell."""
    W = H = 5
    out_focus = focus
    in_center = out_focus
    nbhd = {(in_center[0] + di, in_center[1] + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)}
    in_focused = [in_center]
    in_active = sorted(nbhd - {in_center})
    in_inactive = sorted({(i, j) for i in range(W) for j in range(H)} - nbhd)
    out_focused = [out_focus]
    out_other = sorted({(i, j) for i in range(1, 4) for j in range(1, 4)} - {out_focus})

    body = "\n".join(
        [
            "% input layer",
            _filled(in_inactive, 0, "cellGray", "dashTeal!90"),
            _filled(in_active, 0, "inBlue", "dashTeal!90"),
            _filled(in_focused, 0, "inDark", "dashTeal!90"),
            "% output layer",
            _filled(out_other, 1, "outTeal", "dashTeal!90"),
            _filled(out_focused, 1, "outDark", "dashTeal!90"),
            "% frustum",
            f"  \\frustum{{{out_focus[0]}}}{{{out_focus[1]}}}",
        ]
    )
    return _wrap(body)


def frames_dense() -> list[tuple[int, int]]:
    return [(oi, oj) for oj in (1, 2, 3) for oi in (1, 2, 3)]


# ----------------------------------------------------------------------------
# Diagram 2: sparse input -> sparse output, 3x3 kernel
# ----------------------------------------------------------------------------
SPARSE_IN = [(1, 1), (2, 2), (3, 3), (1, 3)]
SPARSE_OUT = [(1, 1), (2, 2), (1, 2), (2, 3)]


def render_sparse(focus: tuple[int, int]) -> str:
    W = H = 5
    out_focus = focus
    # Receptive field: 3x3 input cells the kernel reaches at this output.
    reach = _reach_3x3(out_focus, layer=0, grid_w=W, grid_h=H)
    # Input cell at the same coords as output focus is darker, others blue.
    in_focused = [out_focus] if out_focus in SPARSE_IN else []
    in_other = [c for c in SPARSE_IN if c not in in_focused]
    out_focused = [out_focus]
    out_other = [c for c in SPARSE_OUT if c != out_focus]

    body = "\n".join(
        [
            "% receptive-field tint first so scaffold lines render on top",
            _filled(reach, 0, "inTint", "inTint"),
            "% input layer scaffold (dashed) — on top of tint so grid is visible",
            _scaffold(0, W, H),
            "% sparse input cells (bright blue) — render on top of tint",
            _filled(in_other, 0, "inBlue", "dashTeal!90"),
            _filled(in_focused, 0, "inDark", "dashTeal!90"),
            "% output layer",
            _scaffold(1, W, H),
            _filled(out_other, 1, "outTeal", "dashTeal!90"),
            _filled(out_focused, 1, "outDark", "dashTeal!90"),
            "% frustum",
            f"  \\frustum{{{out_focus[0]}}}{{{out_focus[1]}}}",
        ]
    )
    return _wrap(body)


def frames_sparse() -> list[tuple[int, int]]:
    return list(SPARSE_OUT)


# ----------------------------------------------------------------------------
# Diagram 3: stride=1 (output preserves input coordinates)
# ----------------------------------------------------------------------------
STRIDE1_COORDS = [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1)]


def render_stride1(focus: tuple[int, int]) -> str:
    W = H = 5
    reach = _reach_3x3(focus, layer=0, grid_w=W, grid_h=H)
    in_focused = [focus]
    in_other = [c for c in STRIDE1_COORDS if c != focus]
    out_focused = [focus]
    out_other = [c for c in STRIDE1_COORDS if c != focus]

    body = "\n".join(
        [
            "% receptive-field tint first so scaffold lines render on top",
            _filled(reach, 0, "inTint", "inTint"),
            "% input layer scaffold (dashed) — on top of tint so grid is visible",
            _scaffold(0, W, H),
            "% sparse input cells (bright blue) — render on top of tint",
            _filled(in_other, 0, "inBlue", "dashTeal!90"),
            _filled(in_focused, 0, "inDark", "dashTeal!90"),
            "% output layer = input coords",
            _scaffold(1, W, H),
            _filled(out_other, 1, "outTeal", "dashTeal!90"),
            _filled(out_focused, 1, "outDark", "dashTeal!90"),
            "% frustum",
            f"  \\frustum{{{focus[0]}}}{{{focus[1]}}}",
        ]
    )
    return _wrap(body)


def frames_stride1() -> list[tuple[int, int]]:
    return list(STRIDE1_COORDS)


# ----------------------------------------------------------------------------
# Diagram 4: generalized convolution — + kernel with one extra top-right cell
# ----------------------------------------------------------------------------
# Offsets: plus shape {(0,0),(±1,0),(0,±1)} + top-right corner (1,1)
GEN_KERNEL: list[tuple[int, int]] = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)]

GEN_IN = list(SPARSE_IN)  # same occupied set as sparse diagram

_GEN_IN_SET = set(GEN_IN)


def _gen_active(out_cell: tuple[int, int]) -> list[tuple[int, int]]:
    oi, oj = out_cell
    return [(oi + di, oj + dj) for (di, dj) in GEN_KERNEL if (oi + di, oj + dj) in _GEN_IN_SET]


_GEN_OUT_CELLS: list[tuple[int, int]] = [(2, 3), (1, 2), (2, 2)]
GEN_PULLS: list[tuple[tuple[int, int], list[tuple[int, int]]]] = [
    (oc, _gen_active(oc)) for oc in _GEN_OUT_CELLS
]
GEN_OUT = [p[0] for p in GEN_PULLS]


def _kernel_perimeter_corners(kernel: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Polygon VERTICES of the union of unit cells in `kernel`. A corner is a
    vertex iff the perimeter turns there (i.e. not all 4 surrounding cells in
    kernel, not zero, and not 2 collinear)."""
    cells = set(kernel)
    candidates: set[tuple[int, int]] = set()
    for i, j in cells:
        for cx, cy in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]:
            candidates.add((cx, cy))

    vertices: list[tuple[int, int]] = []
    for cx, cy in candidates:
        around = [(cx - 1, cy - 1), (cx, cy - 1), (cx - 1, cy), (cx, cy)]
        in_kernel = [c for c in around if c in cells]
        n = len(in_kernel)
        if n == 0 or n == 4:
            continue
        if n == 2:
            (ax, ay), (bx, by) = in_kernel
            if ax == bx or ay == by:
                continue  # collinear: edge passes through, no turn
        vertices.append((cx, cy))
    return vertices


def _kernel_corner_links(oi: int, oj: int, kernel: list[tuple[int, int]]) -> list[str]:
    """Lines from each kernel-perimeter vertex on input layer to the nearest
    corner of the output cell. Visualizes the kernel SHAPE for the focused
    output cell, independent of which inputs are occupied."""
    lines: list[str] = []
    for px, py in _kernel_perimeter_corners(kernel):
        ci = 1 if px >= 1 else 0
        cj = 1 if py >= 1 else 0
        lines.append(
            f"  \\draw[linkInk,line width=0.55pt,line cap=round]"
            f" \\corner{{{oi}}}{{{oj}}}{{{ci}}}{{{cj}}}"
            f" -- \\incorner{{{oi}}}{{{oj}}}{{{px}}}{{{py}}};"
        )
    return lines


def render_generalized(state: tuple[tuple[int, int], list[tuple[int, int]]]) -> str:
    W = H = 5
    out_focus, links = state
    oi, oj = out_focus
    footprint = [
        (oi + di, oj + dj) for (di, dj) in GEN_KERNEL if 0 <= oi + di < W and 0 <= oj + dj < H
    ]
    in_other = [c for c in GEN_IN if c not in links]
    in_active = links
    out_focused = [out_focus]
    out_other = [c for c in GEN_OUT if c != out_focus]

    body = "\n".join(
        [
            "% kernel footprint tint first so scaffold renders on top",
            _filled(footprint, 0, "inTint", "inTint"),
            "% input layer",
            _scaffold(0, W, H),
            _filled(in_other, 0, "inBlue", "dashTeal!90"),
            _filled(in_active, 0, "inDark", "dashTeal!90"),
            "% output layer",
            _scaffold(1, W, H),
            _filled(out_other, 1, "outTeal", "dashTeal!90"),
            _filled(out_focused, 1, "outDark", "dashTeal!90"),
            "% kernel perimeter corner-to-corner lines (10 lines for + + corner shape)",
            *_kernel_corner_links(out_focus[0], out_focus[1], GEN_KERNEL),
        ]
    )
    return _wrap(body)


def frames_generalized():
    return list(GEN_PULLS)


# ----------------------------------------------------------------------------
# Diagram 5: generative convolution — each input expands C_out by kernel reach
# ----------------------------------------------------------------------------
_W_GEN = _H_GEN = 5

# Standard 3x3 kernel for generative diagram.
GENCONV_KERNEL: list[tuple[int, int]] = [(di, dj) for dj in (-1, 0, 1) for di in (-1, 0, 1)]


def _generated_by(in_cell: tuple[int, int]) -> list[tuple[int, int]]:
    ii, ij = in_cell
    return [
        (ii + di, ij + dj)
        for (di, dj) in GENCONV_KERNEL
        if 0 <= ii + di < _W_GEN and 0 <= ij + dj < _H_GEN
    ]


# Full output set: union of kernel expansions from every input cell.
GENCONV_OUT: list[tuple[int, int]] = sorted({c for inp in SPARSE_IN for c in _generated_by(inp)})

# One frame per input voxel: (focused_input, outputs_it_generates)
GENCONV_FRAMES: list[tuple[tuple[int, int], list[tuple[int, int]]]] = [
    (inp, _generated_by(inp)) for inp in SPARSE_IN
]


def render_generative(state: tuple[tuple[int, int], list[tuple[int, int]]]) -> str:
    W = H = _W_GEN
    in_focus, generated = state
    fi, fj = in_focus

    # Kernel footprint on input layer (visualizes the kernel shape).
    input_footprint = [
        (fi + di, fj + dj) for (di, dj) in GENCONV_KERNEL if 0 <= fi + di < W and 0 <= fj + dj < H
    ]
    in_focused = [in_focus]
    in_other = [c for c in SPARSE_IN if c != in_focus]

    gen_set = set(map(tuple, generated))
    out_active = generated  # outputs produced by this input (dark teal)
    out_other = [c for c in GENCONV_OUT if tuple(c) not in gen_set]  # from other inputs

    body = "\n".join(
        [
            "% kernel footprint tint on input layer",
            _filled(input_footprint, 0, "inTint", "inTint"),
            "% input layer scaffold on top of tint",
            _scaffold(0, W, H),
            _filled(in_other, 0, "inBlue", "dashTeal!90"),
            _filled(in_focused, 0, "inDark", "dashTeal!90"),
            "% output layer — all generated coords, focused ones highlighted",
            _scaffold(1, W, H),
            _filled(out_other, 1, "outTeal", "dashTeal!90"),
            _filled(out_active, 1, "outDark", "dashTeal!90"),
            "% genfrustum: 4 corner lines from input cell to 3x3 output neighborhood",
            f"  \\genfrustum{{{fi}}}{{{fj}}}",
        ]
    )
    return _wrap(body)


def frames_generative():
    return list(GENCONV_FRAMES)


# ----------------------------------------------------------------------------
# Build pipeline
# ----------------------------------------------------------------------------
def _compile(tex_src: str, name: str, build_dir: Path) -> Path:
    """Compile one TikZ source to PDF + PNG inside `build_dir`. Returns PDF path."""
    tex_path = build_dir / f"{name}.tex"
    tex_path.write_text(tex_src)
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log = (build_dir / f"{name}.log").read_text(errors="replace")
        sys.exit(f"pdflatex failed for {name}:\n--- log tail ---\n{log[-3000:]}")
    return build_dir / f"{name}.pdf"


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 200) -> None:
    """Rasterize PDF to a single PNG at `png_path`."""
    stem = png_path.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(stem)],
        check=True,
    )
    single = stem.parent / f"{stem.name}-1.png"
    if single.exists():
        single.replace(png_path)


def build_diagram(
    name: str,
    render_fn: Callable,
    frames_fn: Callable,
    duration_ms: int = 700,
) -> None:
    if not shutil.which("pdflatex"):
        sys.exit("error: pdflatex not on PATH")
    if not shutil.which("pdftoppm"):
        sys.exit("error: pdftoppm not on PATH (install poppler-utils)")

    states = list(frames_fn())

    with tempfile.TemporaryDirectory(prefix=f"tikz.{name}.") as bd_str:
        bd = Path(bd_str)
        png_paths: list[Path] = []
        for idx, state in enumerate(states):
            frame_name = f"{name}_f{idx:02d}"
            pdf_path = _compile(render_fn(state), frame_name, bd)
            png_path = bd / f"{frame_name}.png"
            _pdf_to_png(pdf_path, png_path, dpi=180)
            png_paths.append(png_path)

        # Assemble GIF. Use disposal=2 so each frame fully overwrites the
        # previous (no ghosting). Ensure all frames share the same canvas
        # size (the BBOX preamble guarantees this).
        images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in png_paths]
        gif_path = OUT / f"{name}.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=True,
        )

    print(f"wrote: {OUT/(name+'.gif')}  ({len(states)} frames @ {duration_ms}ms)")


def main() -> None:
    build_diagram("sparse_conv_dense", render_dense, frames_dense, duration_ms=600)
    build_diagram("sparse_conv_sparse", render_sparse, frames_sparse, duration_ms=900)
    build_diagram("sparse_conv_stride1", render_stride1, frames_stride1, duration_ms=900)
    build_diagram(
        "sparse_conv_generalized",
        render_generalized,
        frames_generalized,
        duration_ms=1100,
    )
    build_diagram("sparse_conv_generative", render_generative, frames_generative, duration_ms=1100)


if __name__ == "__main__":
    main()
