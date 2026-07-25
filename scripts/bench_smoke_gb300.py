# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test + timing breakdown for a ScanNet-shaped sparse conv stack.

Usage:
    python scripts/bench_smoke_gb300.py
"""

import argparse
import time

import torch

import warpconvnet  # noqa: F401
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SparseConv3d


def _report_env():
    import warpconvnet._C as _C

    major, minor = torch.cuda.get_device_capability(0)
    arch = major * 10 + minor
    print(f"device      : {torch.cuda.get_device_name(0)} (sm_{arch})")
    print(f"torch       : {torch.__version__}")
    print(f"warpconvnet : {warpconvnet.__version__}")

    from warpconvnet.nn.functional.sparse_conv.detail import algo_params as ap

    print(
        f"backends    : cutlass={ap._HAS_CUTLASS_BACKEND} cute={ap._HAS_CUTE_BACKEND} "
        f"cute_grouped={ap._HAS_CUTE_GROUPED} cute_sm90={ap._HAS_CUTE_SM90}"
    )

    from warpconvnet.nn.functional.sparse_conv.detail import tile_metadata as tm

    for op in ("forward", "dgrad", "wgrad"):
        all_tiles = tm._get_tiles(op, filter_arch=False)
        arch_ok = tm._get_tiles(op, filter_arch=True)
        print(f"mask_gemm {op:<7}: {len(arch_ok):>3}/{len(all_tiles):>3} tiles pass the arch gate")
    return arch


def make_scene(n_per_batch, batch_size, channels, device, dtype):
    """Voxelized-scene-shaped input: coordinates drawn from a 3D grid."""
    torch.manual_seed(0)
    coords = []
    for _ in range(batch_size):
        c = torch.randint(0, 256, (int(n_per_batch * 1.15), 3), device=device, dtype=torch.int32)
        c = torch.unique(c, dim=0)[:n_per_batch]
        coords.append(c)
    feats = [torch.randn(c.shape[0], channels, device=device, dtype=dtype) for c in coords]
    return Voxels(coords, feats).unique()


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000, help="voxels per batch element")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--cout", type=int, default=64)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument(
        "--algos",
        default="explicit_gemm,implicit_gemm,cutlass_implicit_gemm,"
        "cute_implicit_gemm,mask_gemm,auto",
    )
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    _report_env()
    print()

    t0 = time.perf_counter()
    vox = make_scene(args.n, args.batch, args.cin, device, dtype)
    torch.cuda.synchronize()
    print(
        f"scene: {vox.feature_tensor.shape[0]} voxels, C={args.cin}->{args.cout}, "
        f"dtype={args.dtype}, built in {time.perf_counter() - t0:.2f}s"
    )

    # --- kernel-map generation cost, isolated ---
    from warpconvnet.nn.functional.sparse_conv.helper import (
        generate_output_coords_and_kernel_map,
    )

    def kmap():
        return generate_output_coords_and_kernel_map(
            input_sparse_tensor=vox,
            kernel_size=(3, 3, 3),
            kernel_dilation=(1, 1, 1),
            stride=(1, 1, 1),
            generative=False,
            transposed=False,
            output_spatially_sparse_tensor=None,
        )

    try:
        t = timeit(kmap, warmup=2, iters=10)
        print(f"kernel-map generation (k=3, s=1): {t:.3f} ms")
    except Exception as e:  # signature drift is not fatal for the rest
        print(f"kernel-map isolated timing skipped: {type(e).__name__}: {e}")

    print()
    print(f"{'algo':<24} {'fwd (ms)':>10} {'fwd+bwd (ms)':>14}  status")
    print("-" * 62)
    for algo in args.algos.split(","):
        algo = algo.strip()
        try:
            conv = (
                SparseConv3d(
                    args.cin,
                    args.cout,
                    kernel_size=3,
                    bias=False,
                    fwd_algo=algo,
                    dgrad_algo=algo,
                    wgrad_algo=algo if algo != "mask_gemm" else algo,
                )
                .to(device)
                .to(dtype)
            )

            def fwd():
                with torch.no_grad():
                    conv(vox)

            def fwd_bwd():
                vox_g = vox.replace(
                    batched_features=vox.feature_tensor.detach().requires_grad_(True)
                )
                out = conv(vox_g)
                out.feature_tensor.sum().backward()

            tf = timeit(fwd)
            tb = timeit(fwd_bwd)
            print(f"{algo:<24} {tf:>10.3f} {tb:>14.3f}  ok")
        except Exception as e:
            msg = str(e).split("\n")[0][:70]
            print(f"{algo:<24} {'-':>10} {'-':>14}  {type(e).__name__}: {msg}")


if __name__ == "__main__":
    main()
