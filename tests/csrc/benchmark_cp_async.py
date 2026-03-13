#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark cp_async=True vs cp_async=False for SM90 grouped GEMM
in forward and backward passes.
"""

import torch
import sys

try:
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped_sm90 import (
        _cute_grouped_sm90_forward_logic,
        _cute_grouped_sm90_backward_logic,
    )
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped import (
        _cute_grouped_forward_logic,
        _cute_grouped_backward_logic,
    )
    from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
        _explicit_gemm_forward_logic,
    )
    from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
    from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
    from warpconvnet.geometry.types.voxels import Voxels
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def make_test_data(N, C_in, C_out, kernel_size=3, dtype=torch.float16, seed=42):
    torch.manual_seed(seed)
    coord_range = max(30, int(N ** (1.0 / 3.0) * 2))
    coords = torch.randint(0, coord_range, (N, 3), device="cuda", dtype=torch.int32)
    coords = torch.unique(coords, dim=0)
    N_actual = coords.shape[0]
    feats = torch.randn(N_actual, C_in, device="cuda", dtype=dtype) * 0.1
    offsets = torch.tensor([0, N_actual], dtype=torch.long, device="cuda")
    vox = Voxels(batched_coordinates=coords, batched_features=feats, offsets=offsets, voxel_size=1.0)
    weight = torch.randn(kernel_size**3, C_in, C_out, device="cuda", dtype=dtype) * 0.01
    batch_coords = batch_indexed_coordinates(vox.coordinate_tensor, offsets)
    kmap = generate_kernel_map(batch_coords, batch_coords, (1,1,1), (kernel_size,)*3)
    return feats, weight, kmap, N_actual


def benchmark_fn(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return min(times), sum(times) / len(times)


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM: {torch.cuda.get_device_capability()}")
    print()

    configs = [
        (100_000, 64, 64),
        (100_000, 128, 128),
        (100_000, 256, 256),
        (1_000_000, 64, 64),
        (1_000_000, 128, 128),
        (1_000_000, 256, 256),
    ]

    tiles = [(103, "SM90 256x128x64"), (100, "SM90 64x128x64"), (104, "SM90 64x64x64")]

    # ========== FORWARD ==========
    print("=" * 90)
    print("FORWARD: cp_async=True vs cp_async=False (+ SM80 baseline)")
    print("=" * 90)
    print()

    for N_target, C_in, C_out in configs:
        feats, weight, kmap, N_actual = make_test_data(N_target, C_in, C_out)
        total_pairs = sum(len(kmap[i][0]) for i in range(len(kmap)))
        print(f"  N={N_actual}, {C_in}->{C_out}, pairs={total_pairs}")

        results = {}

        # SM80 baseline
        def fn_sm80():
            _cute_grouped_forward_logic(feats, weight, kmap, N_actual, mma_tile=3)
        _, avg = benchmark_fn(fn_sm80, warmup=3, repeats=15)
        results["SM80 64x64x32"] = avg

        for tile, name in tiles:
            for cp in [True, False]:
                label = f"{name} cp={cp}"
                def fn(t=tile, c=cp):
                    _cute_grouped_sm90_forward_logic(feats, weight, kmap, N_actual, mma_tile=t, use_cp_async=c)
                try:
                    _, avg = benchmark_fn(fn, warmup=3, repeats=15)
                    results[label] = avg
                except Exception as e:
                    results[label] = f"ERR: {e}"

        # Print
        numeric = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        best = min(numeric, key=numeric.get)
        for name, t in sorted(numeric.items(), key=lambda x: x[1]):
            marker = " <-- BEST" if name == best else ""
            slowdown = f"(+{(t/numeric[best]-1)*100:.1f}%)" if name != best else ""
            print(f"    {name:<40s} {t:8.3f} ms  {slowdown}{marker}")
        print()

        del feats, weight, kmap
        torch.cuda.empty_cache()

    # ========== BACKWARD (grad_input only — isolates the SM90 grouped GEMM) ==========
    print("=" * 90)
    print("BACKWARD grad_input: cp_async=True vs cp_async=False (+ SM80 baseline)")
    print("=" * 90)
    print()

    for N_target, C_in, C_out in configs:
        feats, weight, kmap, N_actual = make_test_data(N_target, C_in, C_out)
        fwd_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
        grad_output = torch.randn_like(fwd_out)
        total_pairs = sum(len(kmap[i][0]) for i in range(len(kmap)))
        print(f"  N={N_actual}, {C_in}->{C_out}, pairs={total_pairs}")

        results = {}

        def fn_sm80():
            _cute_grouped_backward_logic(grad_output, feats, weight, kmap, (True, False), mma_tile=3)
        _, avg = benchmark_fn(fn_sm80, warmup=3, repeats=15)
        results["SM80 64x64x32"] = avg

        for tile, name in tiles:
            for cp in [True, False]:
                label = f"{name} cp={cp}"
                def fn(t=tile, c=cp):
                    _cute_grouped_sm90_backward_logic(grad_output, feats, weight, kmap, (True, False), mma_tile=t, use_cp_async=c)
                try:
                    _, avg = benchmark_fn(fn, warmup=3, repeats=15)
                    results[label] = avg
                except Exception as e:
                    results[label] = f"ERR: {e}"

        numeric = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        best = min(numeric, key=numeric.get)
        for name, t in sorted(numeric.items(), key=lambda x: x[1]):
            marker = " <-- BEST" if name == best else ""
            slowdown = f"(+{(t/numeric[best]-1)*100:.1f}%)" if name != best else ""
            print(f"    {name:<40s} {t:8.3f} ms  {slowdown}{marker}")
        print()

        del feats, weight, kmap, fwd_out, grad_output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
