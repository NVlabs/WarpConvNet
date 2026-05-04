# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter sweep for the dense-CRF mean-field example. Internal tool.

Produces a single matrix figure with rows = (w_app, w_smooth) weight pairs
and columns = sigma_rgb (color bandwidth of the appearance kernel). Each
cell is the CRF-refined labeling under that configuration. The input
image and noisy annotation are shown in the top header strip for
reference.

Run:
    python scripts/crf_image_sweep.py --out /tmp/crf_sweep.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
from crf_image_example import _load_pair, run_crf  # noqa: E402


SIGMA_RGB_VALUES = [3.0, 8.0, 13.0, 30.0]
WEIGHT_PAIRS = [(10.0, 3.0), (5.0, 5.0), (3.0, 10.0), (10.0, 10.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/user_guide/img/crf_sweep.png")
    ap.add_argument("--cache-dir", default="examples/data/densecrf")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--sigma-xy-app", type=float, default=80.0)
    ap.add_argument("--sigma-xy-smooth", type=float, default=3.0)
    ap.add_argument("--gt-prob", type=float, default=0.7)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    im, anno = _load_pair(args.cache_dir)
    rows = len(WEIGHT_PAIRS)
    cols = len(SIGMA_RGB_VALUES)

    fig = plt.figure(figsize=(2.6 * (cols + 1), 2.6 * (rows + 1)))
    gs = fig.add_gridspec(rows + 1, cols + 1, hspace=0.18, wspace=0.05)

    # Header row: image + anno + blanks
    ax_im = fig.add_subplot(gs[0, 0])
    ax_im.imshow(im)
    ax_im.set_title("image", fontsize=11)
    ax_im.axis("off")
    ax_an = fig.add_subplot(gs[0, 1])
    ax_an.imshow(anno)
    ax_an.set_title("noisy anno (black = unknown)", fontsize=11)
    ax_an.axis("off")
    for c in range(2, cols + 1):
        ax = fig.add_subplot(gs[0, c])
        ax.axis("off")

    # Left header column (row labels)
    for r, (wa, ws) in enumerate(WEIGHT_PAIRS):
        ax = fig.add_subplot(gs[r + 1, 0])
        ax.text(0.5, 0.5, f"$w_a={wa:g}$\n$w_s={ws:g}$", ha="center", va="center", fontsize=13)
        ax.axis("off")

    # Sweep matrix
    for c, sr in enumerate(SIGMA_RGB_VALUES):
        for r, (wa, ws) in enumerate(WEIGHT_PAIRS):
            out, dt = run_crf(
                im,
                anno,
                iters=args.iters,
                sigma_xy_app=args.sigma_xy_app,
                sigma_rgb=sr,
                sigma_xy_smooth=args.sigma_xy_smooth,
                w_app=wa,
                w_smooth=ws,
                gt_prob=args.gt_prob,
                device=device,
            )
            ax = fig.add_subplot(gs[r + 1, c + 1])
            ax.imshow(out)
            if r == 0:
                ax.set_title(f"$\\sigma_{{rgb}}={sr:g}$", fontsize=12)
            ax.axis("off")
            print(f"σ_rgb={sr:5.1f} w_a={wa:5.1f} w_s={ws:5.1f}  {dt:6.1f} ms")

    fig.suptitle(
        f"CRF mean-field sweep — iters={args.iters}, "
        f"$\\sigma_{{xy}}^{{app}}={args.sigma_xy_app:g}$, "
        f"$\\sigma_{{xy}}^{{smooth}}={args.sigma_xy_smooth:g}$, "
        f"gt_prob={args.gt_prob:g}",
        fontsize=13,
        y=0.995,
    )
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
