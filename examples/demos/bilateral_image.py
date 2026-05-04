# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bilateral filtering on a 2D image — denoising the NASA `astronaut` test image.

Demonstrates the three bilateral families shipped in `warpconvnet.nn`:

  - BilateralFilter            (KNN / radius)
  - BilateralFilterGrid        (sparse d-cube lattice)
  - BilateralPermutohedralFilter (permutohedral lattice)

We treat each pixel as a point with 2D position (x, y) and 3D color (r, g, b).
The bilateral guide is concat(xy/sigma_xy, rgb/sigma_rgb); the value being
filtered is the noisy color. The astronaut image is in the public domain
(NASA), shipped with scikit-image.

Saves five PNGs to <out-dir>:
    astronaut_original.png, astronaut_noisy.png, astronaut_knn.png,
    astronaut_grid.png, astronaut_permutohedral.png

Run:
    python examples/demos/bilateral_image.py --out-dir docs/user_guide/img
"""

import argparse
import time

import numpy as np
import torch
from skimage import data, util


def _to_pixel_pointcloud(img: np.ndarray, device: torch.device):
    """Flatten an (H, W, 3) image into (N, 2) xy + (N, 3) rgb tensors."""
    h, w, _ = img.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    rgb = img.reshape(-1, 3).astype(np.float32)
    return (
        torch.from_numpy(xy).to(device),
        torch.from_numpy(rgb).to(device),
    )


def _from_pixel_pointcloud(values: torch.Tensor, h: int, w: int) -> np.ndarray:
    return values.detach().cpu().numpy().reshape(h, w, 3).clip(0, 1)


def _save_image(path: str, arr: np.ndarray) -> None:
    from PIL import Image

    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    img = Image.fromarray(arr)
    if path.lower().endswith((".jpg", ".jpeg")):
        img.save(path, quality=92, optimize=True, progressive=True)
    else:
        img.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="docs/user_guide/img")
    parser.add_argument("--noise-var", type=float, default=0.01)
    parser.add_argument("--sigma-xy", type=float, default=4.0)
    parser.add_argument("--sigma-rgb", type=float, default=0.1)
    parser.add_argument("--knn-k", type=int, default=24)
    args = parser.parse_args()
    import os

    os.makedirs(args.out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for bilateral filters")
    device = torch.device("cuda")

    import warpconvnet.nn as wn

    # ---- input image -------------------------------------------------------
    img = util.img_as_float(data.astronaut())  # (512, 512, 3) in [0, 1]
    noisy = util.random_noise(img, mode="gaussian", var=args.noise_var)
    h, w, _ = img.shape

    xy, rgb_clean = _to_pixel_pointcloud(img, device)
    _, rgb_noisy = _to_pixel_pointcloud(noisy.astype(np.float32), device)

    # ---- KNN bilateral -----------------------------------------------------
    knn_filter = wn.BilateralFilter(
        sigma_xyz=args.sigma_xy,
        sigma_feat=args.sigma_rgb,
        k=args.knn_k,
        mode="knn",
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_knn = knn_filter(xy, rgb_noisy, rgb_noisy)
    torch.cuda.synchronize()
    t_knn = time.perf_counter() - t0

    # ---- sparse d-cube grid -----------------------------------------------
    grid_filter = wn.BilateralFilterGrid(
        sigma_xyz=args.sigma_xy,
        sigma_feat=args.sigma_rgb,
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_grid = grid_filter(xy, rgb_noisy, rgb_noisy)
    torch.cuda.synchronize()
    t_grid = time.perf_counter() - t0

    # ---- permutohedral lattice --------------------------------------------
    perm_filter = wn.BilateralPermutohedralFilter(
        sigma_xyz=args.sigma_xy,
        sigma_feat=args.sigma_rgb,
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_perm = perm_filter(xy, rgb_noisy, rgb_noisy)
    torch.cuda.synchronize()
    t_perm = time.perf_counter() - t0

    # ---- save individual PNGs ---------------------------------------------
    knn_img = _from_pixel_pointcloud(out_knn, h, w)
    grid_img = _from_pixel_pointcloud(out_grid, h, w)
    perm_img = _from_pixel_pointcloud(out_perm, h, w)
    outputs = {
        "astronaut_original.jpg": img,
        "astronaut_noisy.jpg": noisy,
        "astronaut_knn.jpg": knn_img,
        "astronaut_grid.jpg": grid_img,
        "astronaut_permutohedral.jpg": perm_img,
    }
    for name, arr in outputs.items():
        path = os.path.join(args.out_dir, name)
        _save_image(path, arr)
        print(f"Saved {path}")

    # ---- PSNR vs original (data_range=1.0 since img is float in [0, 1]) ---
    from skimage.metrics import peak_signal_noise_ratio as psnr

    ref = img.astype(np.float32)
    psnr_noisy = psnr(ref, np.clip(noisy, 0, 1).astype(np.float32), data_range=1.0)
    psnr_knn = psnr(ref, np.clip(knn_img, 0, 1).astype(np.float32), data_range=1.0)
    psnr_grid = psnr(ref, np.clip(grid_img, 0, 1).astype(np.float32), data_range=1.0)
    psnr_perm = psnr(ref, np.clip(perm_img, 0, 1).astype(np.float32), data_range=1.0)

    print()
    print(f"  {'Stage':<22}{'Time':>10}   PSNR (dB)")
    print(f"  {'-' * 46}")
    print(f"  {'Noisy input':<22}{'-':>10}   {psnr_noisy:6.2f}")
    print(f"  {'KNN (k=' + str(args.knn_k) + ')':<22}{t_knn*1e3:>8.1f} ms   {psnr_knn:6.2f}")
    print(f"  {'Grid':<22}{t_grid*1e3:>8.1f} ms   {psnr_grid:6.2f}")
    print(f"  {'Permutohedral':<22}{t_perm*1e3:>8.1f} ms   {psnr_perm:6.2f}")


if __name__ == "__main__":
    main()
