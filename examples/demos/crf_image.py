# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense-CRF mean-field on a 2D image, expressed as sparse convolution.

Implements Krähenbühl & Koltun (NeurIPS 2011) "Efficient Inference in Fully
Connected CRFs with Gaussian Edge Potentials" and the trilateral-CRF
construction from MinkowskiNet (Choy et al., CVPR 2019, §5), reduced to 2D
for clarity.

Two pairwise terms:

  - Appearance kernel: 5D Gaussian on (x, y, r, g, b) — sparse convolution on
    the permutohedral lattice (Adams, Baek, Davis 2010). Edge-preserving.
    The 5D voxel grid is too large for a dense Conv5d (~10^11 cells); only
    the ~H*W occupied lattice points matter, so a sparse-conv lattice is the
    natural data structure. This is where sparse convolution earns its keep.
  - Smoothness kernel: 2D Gaussian on (x, y) only — implemented as a plain
    dense ``torch.nn.Conv2d`` with a fixed (non-learned) Gaussian weight.
    Pixels are dense in 2D; sparse machinery would buy nothing here.

The sample image and noisy annotation come from the densecrf reference
release (Krähenbühl, lucasb-eyer mirror): a single (image, anno) pair
fetched on demand from raw GitHub.

Run:
    python examples/demos/crf_image.py --out-dir docs/user_guide/img
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


DENSECRF_RAW = "https://raw.githubusercontent.com/lucasb-eyer/pydensecrf/master/examples"


def _fetch(url: str, dst: str) -> str:
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            f.write(r.read())
    return dst


def _load_pair(cache_dir: str):
    from PIL import Image

    im_path = _fetch(f"{DENSECRF_RAW}/im1.png", os.path.join(cache_dir, "im1.png"))
    anno_path = _fetch(f"{DENSECRF_RAW}/anno1.png", os.path.join(cache_dir, "anno1.png"))
    im = np.asarray(Image.open(im_path).convert("RGB"))
    anno = np.asarray(Image.open(anno_path).convert("RGB"))
    return im, anno


def _anno_to_labels(anno: np.ndarray):
    """Map RGB anno to integer labels, treating black as 'unknown'.

    The densecrf reference annotations use solid colors per class and
    pure black (0, 0, 0) for pixels where the class is not asserted. The
    unary for unknown pixels is uniform; labeled pixels get a peaked
    unary (see ``_unary_from_labels``).

    Returns
    -------
    labels : ndarray [H, W] int64 — class id, or -1 for unknown.
    palette : ndarray [K, 3] uint8 — palette[k] is the color for label k.
              Black is excluded; K is the number of *real* classes.
    """
    h, w, _ = anno.shape
    flat = anno.reshape(-1, 3)
    is_unk = (flat == 0).all(axis=-1)
    palette, inv = np.unique(flat[~is_unk], axis=0, return_inverse=True)
    labels = np.full(flat.shape[0], -1, dtype=np.int64)
    labels[~is_unk] = inv
    return labels.reshape(h, w), palette.astype(np.uint8)


def _unary_from_labels(labels: np.ndarray, num_classes: int, gt_prob: float) -> Tensor:
    """Standard densecrf unary with ``zero_unsure=True``.

      - labeled pixel (label k >= 0): P(k) = gt_prob, others share rest equally
      - unknown pixel (label = -1):   P uniform over all classes

    Returns U = -log P with shape [N, K].
    """
    n = labels.size
    flat = labels.reshape(-1)
    is_unk = flat < 0
    P = np.empty((n, num_classes), dtype=np.float32)
    P[is_unk] = 1.0 / num_classes
    if (~is_unk).any():
        idx = np.where(~is_unk)[0]
        P[idx] = (1.0 - gt_prob) / max(num_classes - 1, 1)
        P[idx, flat[idx]] = gt_prob
    return torch.from_numpy(-np.log(np.clip(P, 1e-12, 1.0)))


def _gaussian_2d_kernel(k: int, sigma: float) -> Tensor:
    """Symmetric 2D Gaussian, zero at center (excludes self-contribution)."""
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    g[(k - 1) // 2, (k - 1) // 2] = 0.0
    return g / g.sum().clamp_min(1e-12)


def _save_image(path: str, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr.astype(np.uint8)).save(path, optimize=True)


def run_crf(
    im: np.ndarray,
    anno: np.ndarray,
    *,
    iters: int = 5,
    sigma_xy_app: float = 80.0,
    sigma_rgb: float = 13.0,
    sigma_xy_smooth: float = 3.0,
    w_app: float = 10.0,
    w_smooth: float = 3.0,
    smooth_kernel: int = 7,
    gt_prob: float = 0.7,
    device: torch.device | None = None,
):
    """Run mean-field on a single (image, anno) pair. Returns (final_vis [H,W,3], dt_ms)."""
    import warpconvnet.nn as wn

    device = device or torch.device("cuda")
    h, w, _ = im.shape
    labels, palette = _anno_to_labels(anno)
    K = int(palette.shape[0])

    rgb = torch.from_numpy(im.astype(np.float32)).reshape(-1, 3).to(device)
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = torch.from_numpy(np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)).to(device)
    U = _unary_from_labels(labels, K, gt_prob=gt_prob).to(device)

    appearance = wn.BilateralPermutohedralFilterCached(
        sigma_xyz=sigma_xy_app,
        sigma_feat=sigma_rgb,
    ).to(device)
    appearance.build_lattice(xy, rgb)

    g2d = _gaussian_2d_kernel(smooth_kernel, sigma_xy_smooth).to(device)
    smooth_weight = g2d[None, None].expand(K, 1, smooth_kernel, smooth_kernel).contiguous()
    pad = (smooth_kernel - 1) // 2

    def smooth_filter(q_flat: Tensor) -> Tensor:
        q_img = q_flat.t().reshape(1, K, h, w)
        out = F.conv2d(q_img, smooth_weight, padding=pad, groups=K)
        return out.reshape(K, -1).t()

    Q = F.softmax(-U, dim=-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        Q_app = appearance(Q) - Q
        Q_sm = smooth_filter(Q)
        msg = w_app * Q_app + w_smooth * Q_sm
        compat = msg.sum(dim=-1, keepdim=True) - msg
        Q = F.softmax(-U - compat, dim=-1)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3

    final_lab = Q.argmax(dim=-1).reshape(h, w).cpu().numpy()
    final_vis = palette[final_lab].reshape(h, w, 3)
    return final_vis, dt_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/user_guide/img")
    ap.add_argument("--cache-dir", default="examples/data/densecrf")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--sigma-xy-app", type=float, default=80.0)
    ap.add_argument("--sigma-rgb", type=float, default=8.0)
    ap.add_argument("--sigma-xy-smooth", type=float, default=3.0)
    ap.add_argument("--w-app", type=float, default=10.0)
    ap.add_argument("--w-smooth", type=float, default=10.0)
    ap.add_argument("--smooth-kernel", type=int, default=7)
    ap.add_argument("--gt-prob", type=float, default=0.7)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    im, anno = _load_pair(args.cache_dir)
    h, w, _ = im.shape
    labels, palette = _anno_to_labels(anno)
    K = int(palette.shape[0])
    print(f"image {h}x{w}, {K} classes")

    final_vis, dt_ms = run_crf(
        im,
        anno,
        iters=args.iters,
        sigma_xy_app=args.sigma_xy_app,
        sigma_rgb=args.sigma_rgb,
        sigma_xy_smooth=args.sigma_xy_smooth,
        w_app=args.w_app,
        w_smooth=args.w_smooth,
        smooth_kernel=args.smooth_kernel,
        gt_prob=args.gt_prob,
        device=device,
    )
    print(f"{args.iters} mean-field iterations in {dt_ms:.1f} ms")

    _save_image(os.path.join(args.out_dir, "crf_input_image.png"), im)
    _save_image(os.path.join(args.out_dir, "crf_input_anno.png"), anno)
    _save_image(os.path.join(args.out_dir, "crf_refined_anno.png"), final_vis)
    print(f"saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
