# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-candidate dgrad numeric diagnosis for the silent-zero-dgrad bug report.

Runs each mask_gemm dgrad candidate DIRECTLY (bypassing autotune) on a
synthetic problem matching the reported shape, and compares grad_in against
the explicit_gemm reference. Run on the affected machine (e.g. A100/sm_80):

    python scripts/diagnose_dgrad_candidates.py --c-in 64 --c-out 32 --n 500000

A candidate printing ZERO or a large rdiff is the kernel autotune must not
pick — please report the full output table back.
"""
import argparse

import torch

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import (
    _execute_backward,
    _execute_forward,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--c-in", type=int, default=64)
    ap.add_argument("--c-out", type=int, default=32)
    ap.add_argument("--voxel-size", type=float, default=0.02)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dev = "cuda"
    torch.manual_seed(args.seed)
    coord = torch.rand(args.n, 3, device=dev) * 10.0
    feat = torch.rand(args.n, args.c_in, device=dev)
    offs = torch.tensor([0, args.n], device=dev)
    vox = Points(coord, feat, offsets=offs).to_voxels(voxel_size=args.voxel_size)

    ic = batch_indexed_coordinates(vox.coordinate_tensor, vox.offsets)
    kmap = generate_kernel_map(ic, ic, in_to_out_stride_ratio=(1, 1, 1), kernel_size=(3, 3, 3))
    n = ic.shape[0]
    K = len(kmap)
    print(f"device={torch.cuda.get_device_name(0)} sm={torch.cuda.get_device_capability()}")
    print(f"n={n} K={K} C_in={args.c_in} C_out={args.c_out} dtype={args.dtype}")

    x = vox.feature_tensor.to(dt)
    w = (torch.randn(K, args.c_in, args.c_out, device=dev) * 0.05).to(dt)
    gy = torch.randn(n, args.c_out, device=dev).to(dt)

    out_mask = _execute_forward("mask_gemm", {"tile_id": 41}, x, w, kmap, n, dt, None, 1)
    out_ref = _execute_forward("explicit_gemm", {}, x, w, kmap, n, dt, None, 1)
    fr = ((out_mask.float() - out_ref.float()).abs().mean() / out_ref.float().abs().mean()).item()
    print(f"fwd sanity: mask41 vs explicit rdiff={fr:.3e}  (expect <1e-2)")

    gi_ref, _ = _execute_backward(
        "explicit_gemm", {}, gy, x, w, kmap, n, dt, x.device, (True, False, False)
    )
    ref_sum = gi_ref.float().abs().sum().item()
    print(f"reference explicit_gemm grad_in abs-sum={ref_sum:.4e}\n")

    candidates = [
        ("mask_gemm", {"tile_id": 41}),  # the auto winner in the report
        ("mask_gemm", {"tile_id": 0}),  # what the ladder resolves 41 to at C<=96
        ("mask_gemm", {"tile_id": 1}),
        ("mask_gemm", {"tile_id": 12}),
        ("mask_gemm_fwd_as_dgrad", {"tile_id": 900}),
        ("mask_gemm_fwd_as_dgrad", {"tile_id": 901}),
    ]
    print(f"{'algo':26s} {'params':22s} {'abs-sum':>12s} {'rdiff':>10s}  verdict")
    for algo, params in candidates:
        try:
            gi, _ = _execute_backward(
                algo, params, gy, x, w, kmap, n, dt, x.device, (True, False, False)
            )
        except RuntimeError as e:
            print(f"{algo:26s} {str(params):22s}  SKIPPED: {str(e).splitlines()[0][:60]}")
            continue
        s = gi.float().abs().sum().item()
        rd = (
            (gi.float() - gi_ref.float()).abs().mean() / (gi_ref.float().abs().mean() + 1e-12)
        ).item()
        if ref_sum > 0 and s < 1e-3 * ref_sum:
            verdict = "*** SILENT ZERO ***"
        elif rd > 0.25:
            verdict = "*** WRONG ***"
        else:
            verdict = "ok"
        print(f"{algo:26s} {str(params):22s} {s:12.4e} {rd:10.3e}  {verdict}")


if __name__ == "__main__":
    main()
