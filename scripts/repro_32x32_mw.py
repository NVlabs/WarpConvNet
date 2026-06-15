# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate tile 28 (32x32 F16Accum) at K>32 (MW2/MW4) vs explicit_gemm.

Covers the K in [65,96] tier-vs-ceil band (K=80 -> ceil 3, dispatched MW4).
That band is a reference-stride trap: our explicit reference uses the kernel_map
directly (no pair_mask), so the stride mismatch does not apply, but we test K=80
explicitly to be sure.
"""

import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import _mask_gemm_forward_logic
from warpconvnet.nn.functional.sparse_conv.detail.explicit import _explicit_gemm_forward_logic


def _make_voxels(N=1500, coord_range=10, C_in=64, batch_size=2, seed=0):
    torch.manual_seed(seed)
    coords_list, feats_list = [], []
    for _ in range(batch_size):
        c = torch.unique(torch.randint(0, coord_range, (N, 3), dtype=torch.int32), dim=0)
        coords_list.append(c)
        feats_list.append(torch.randn(c.shape[0], C_in))
    return Voxels(coords_list, feats_list).to("cuda")


def _kernel_map(voxels, kernel_size):
    in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    return (
        generate_kernel_map(
            in_coords,
            in_coords,
            in_to_out_stride_ratio=(1,) * len(kernel_size),
            kernel_size=kernel_size,
        ),
        in_coords.shape[0],
    )


def _corr(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def _rdiff(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-8)).item()


def main():
    C = 64
    cases = [((4, 4, 4), 64, 2), ((4, 4, 5), 80, 4), ((5, 5, 5), 125, 4)]
    voxels = _make_voxels(C_in=C, seed=1)
    in_feats = voxels.feature_tensor.half()
    ok = True
    for ksize, K, mw_expected in cases:
        torch.manual_seed(7)
        weight = torch.randn(K, C, C, device="cuda", dtype=torch.float16)
        kmap, num_out = _kernel_map(voxels, ksize)
        assert len(kmap) == K, f"kernel_map K={len(kmap)} != {K}"
        out = _mask_gemm_forward_logic(in_feats, weight, kmap, num_out, {"tile_id": 28})
        ref = _explicit_gemm_forward_logic(in_feats, weight, kmap, num_out, torch.float16)
        r, c = _rdiff(out, ref), _corr(out, ref)
        band = " [65,96] tier-band" if 65 <= K <= 96 else ""
        status = "PASS" if (c > 0.99 and r < 0.15) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  K={K:>3} (MW{mw_expected}){band}: rdiff={r:.3e} corr={c:.5f} -> {status}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
