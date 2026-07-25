# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

sys.path.insert(0, "/home/cchoy/projects/warpconvnet")
import torch
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import _execute_backward

torch.manual_seed(0)
dev = "cuda"
N, C_IN, C_OUT = 500_000, 64, 32
coord = torch.rand(N, 3, device=dev) * 10.0
feat = torch.rand(N, C_IN, device=dev)
offs = torch.tensor([0, N], device=dev)
vox = Points(coord, feat, offsets=offs).to_voxels(voxel_size=0.02)
ic = batch_indexed_coordinates(vox.coordinate_tensor, vox.offsets)
kmap = generate_kernel_map(ic, ic, in_to_out_stride_ratio=(1, 1, 1), kernel_size=(3, 3, 3))
n = ic.shape[0]
K = len(kmap)

x = vox.feature_tensor.half()
# Adversarial cancellation: large uniformly-positive grad_output, weight sign
# flips halfway through the K offsets. True dX (fp32) is a small difference of
# two huge partial sums; an fp16 accumulator overflows mid-accumulation.
# Cancellation INSIDE one offset's channel reduction: alternating-sign
# grad_output across c_out, uniform positive weight. True per-offset sum ~0
# (representable); an fp16 accumulator peaks at ~16*3e4*0.5 = 2.4e5 mid-sum.
gy = torch.full((n, C_OUT), 3.0e4, device=dev, dtype=torch.float16)
gy[:, 1::2] *= -1.0
w = torch.full((K, C_IN, C_OUT), 0.5, device=dev, dtype=torch.float16)

print(
    f"n={n} K={K}  gy absmax={gy.abs().max().item():.3g}  finite gy={torch.isfinite(gy).all().item()}"
)

gi_ref, _ = _execute_backward(
    "explicit_gemm",
    {},
    gy,
    x,
    w,
    kmap,
    n,
    torch.float16,
    x.device,
    (True, False, False),
)
print(
    f"explicit_gemm            : absmax={gi_ref.float().abs().max().item():.4e} "
    f"finite={torch.isfinite(gi_ref).all().item()}"
)

for label, params, f16acc in [
    ("native f32acc (tile->0) ", {"tile_id": 3}, False),
    ("native f16acc (tile->22)", {"tile_id": 3}, True),
]:
    gi, _ = _execute_backward(
        "mask_gemm",
        params,
        gy,
        x,
        w,
        kmap,
        n,
        torch.float16,
        x.device,
        (True, False, False),
        None,
        1,
        f16acc,
    )
    rd = (
        (gi.float() - gi_ref.float()).abs().mean() / (gi_ref.float().abs().mean() + 1e-12)
    ).item()
    print(
        f"{label}: absmax={gi.float().abs().max().item():.4e} "
        f"finite={torch.isfinite(gi).all().item()}  rdiff_vs_explicit={rd:.3e}"
    )
