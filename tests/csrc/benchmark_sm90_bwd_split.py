#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark backward pass split into grad_input and grad_weight separately
to identify which component is slower in SM90 vs SM80.
"""

import torch
import sys

try:
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped_sm90 import (
        _cute_grouped_sm90_backward_logic,
    )
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped import (
        _cute_grouped_backward_logic,
    )
    from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
        _explicit_gemm_forward_logic,
        _explicit_gemm_backward_logic,
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
    vox = Voxels(
        batched_coordinates=coords, batched_features=feats, offsets=offsets, voxel_size=1.0,
    )
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


def print_comparison(label, results):
    numeric = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    if not numeric:
        print(f"  {label}: no results")
        return
    best_name = min(numeric, key=numeric.get)
    best_time = numeric[best_name]
    print(f"  {label}:")
    for name, t in sorted(numeric.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if name == best_name else ""
        slowdown = f"(+{(t/best_time-1)*100:.1f}%)" if name != best_name else ""
        print(f"    {name:<40s} {t:8.3f} ms  {slowdown}{marker}")
    print()


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

    best_sm80 = 3   # SM80 64x64x32 — consistently best from forward benchmark
    best_sm90 = 103  # SM90 256x128x64 — consistently best SM90

    for N_target, C_in, C_out in configs:
        print(f"{'='*80}")
        print(f"  N~{N_target//1000}K, {C_in}->{C_out}, fp16")
        print(f"{'='*80}")
        feats, weight, kmap, N_actual = make_test_data(N_target, C_in, C_out)
        fwd_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
        grad_output = torch.randn_like(fwd_out)
        total_pairs = sum(len(kmap[i][0]) for i in range(len(kmap)))
        print(f"  N_actual={N_actual}, pairs={total_pairs}")
        print()

        # --- grad_input only ---
        results_gi = {}

        def fn_sm80_gi():
            _cute_grouped_backward_logic(grad_output, feats, weight, kmap, (True, False), mma_tile=best_sm80)
        _, avg = benchmark_fn(fn_sm80_gi, warmup=3, repeats=15)
        results_gi["SM80 64x64x32 grad_input"] = avg

        def fn_sm90_gi():
            _cute_grouped_sm90_backward_logic(grad_output, feats, weight, kmap, (True, False), mma_tile=best_sm90)
        _, avg = benchmark_fn(fn_sm90_gi, warmup=3, repeats=15)
        results_gi["SM90 256x128x64 grad_input"] = avg

        def fn_sm90_101_gi():
            _cute_grouped_sm90_backward_logic(grad_output, feats, weight, kmap, (True, False), mma_tile=101)
        _, avg = benchmark_fn(fn_sm90_101_gi, warmup=3, repeats=15)
        results_gi["SM90 128x128x64 grad_input"] = avg

        def fn_sm90_104_gi():
            _cute_grouped_sm90_backward_logic(grad_output, feats, weight, kmap, (True, False), mma_tile=104)
        _, avg = benchmark_fn(fn_sm90_104_gi, warmup=3, repeats=15)
        results_gi["SM90 64x64x64 grad_input"] = avg

        print_comparison("grad_input only", results_gi)

        # --- grad_weight only ---
        results_gw = {}

        def fn_sm80_gw():
            _cute_grouped_backward_logic(grad_output, feats, weight, kmap, (False, True), mma_tile=best_sm80)
        _, avg = benchmark_fn(fn_sm80_gw, warmup=3, repeats=15)
        results_gw["SM80 64x64x32 grad_weight"] = avg

        def fn_sm90_gw():
            _cute_grouped_sm90_backward_logic(grad_output, feats, weight, kmap, (False, True), mma_tile=best_sm90)
        _, avg = benchmark_fn(fn_sm90_gw, warmup=3, repeats=15)
        results_gw["SM90 256x128x64 grad_weight"] = avg

        print_comparison("grad_weight only (TrAB, both use SM80 tile)", results_gw)

        # --- both grads ---
        results_both = {}

        def fn_sm80_both():
            _cute_grouped_backward_logic(grad_output, feats, weight, kmap, (True, True), mma_tile=best_sm80)
        _, avg = benchmark_fn(fn_sm80_both, warmup=3, repeats=15)
        results_both["SM80 64x64x32 both"] = avg

        def fn_sm90_both():
            _cute_grouped_sm90_backward_logic(grad_output, feats, weight, kmap, (True, True), mma_tile=best_sm90)
        _, avg = benchmark_fn(fn_sm90_both, warmup=3, repeats=15)
        results_both["SM90 256x128x64 both"] = avg

        print_comparison("both grads (grad_input + grad_weight)", results_both)

        del feats, weight, kmap, fwd_out, grad_output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
