#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark SM90 vs SM80 grouped GEMM kernels in forward and backward passes
through the sparse convolution dispatch layer (cute_grouped_sm90 vs cute_grouped).

Tests: 100K/1M voxels × 64/128/256 channels × fp16, forward and backward separately.
"""

import torch
import sys
import time

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
        _explicit_gemm_backward_logic,
    )
    from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
    from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
    from warpconvnet.geometry.types.voxels import Voxels
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


SM80_TILES = [0, 1, 3]
SM80_TILE_NAMES = {0: "SM80 128x128x32", 1: "SM80 128x64x32", 3: "SM80 64x64x32"}
SM90_TILES = [100, 101, 103, 104]
SM90_TILE_NAMES = {
    100: "SM90 64x128x64",
    101: "SM90 128x128x64",
    103: "SM90 256x128x64",
    104: "SM90 64x64x64",
}


def make_test_data(N, C_in, C_out, kernel_size=3, dtype=torch.float16, seed=42):
    """Create test voxels, weights, and kernel map."""
    torch.manual_seed(seed)
    # Use larger coordinate space for bigger N to get realistic sparsity
    coord_range = max(30, int(N ** (1.0 / 3.0) * 2))
    coords = torch.randint(0, coord_range, (N, 3), device="cuda", dtype=torch.int32)
    coords = torch.unique(coords, dim=0)
    N_actual = coords.shape[0]
    feats = torch.randn(N_actual, C_in, device="cuda", dtype=dtype) * 0.1
    offsets = torch.tensor([0, N_actual], dtype=torch.long, device="cuda")
    vox = Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        voxel_size=1.0,
    )
    K = kernel_size ** 3
    weight = torch.randn(K, C_in, C_out, device="cuda", dtype=dtype) * 0.01
    batch_coords = batch_indexed_coordinates(vox.coordinate_tensor, offsets)
    stride = tuple([1] * 3)
    ksize = tuple([kernel_size] * 3)
    kmap = generate_kernel_map(batch_coords, batch_coords, stride, ksize)
    return feats, weight, kmap, N_actual


def benchmark_fn(fn, warmup=5, repeats=20):
    """Benchmark with CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return min(times), sum(times) / len(times)


def bench_forward(feats, weight, kmap, N_actual):
    """Benchmark forward pass across all backends."""
    results = {}

    # Explicit GEMM (reference)
    def fn_explicit():
        _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    mn, avg = benchmark_fn(fn_explicit)
    results["explicit_gemm"] = avg

    # SM80 grouped tiles
    for tile in SM80_TILES:
        name = SM80_TILE_NAMES[tile]
        def fn(t=tile):
            out = _cute_grouped_forward_logic(feats, weight, kmap, N_actual, mma_tile=t)
            assert not isinstance(out, int), f"SM80 grouped tile {t} failed: {out}"
        try:
            mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
            results[name] = avg
        except Exception as e:
            results[name] = f"ERROR: {e}"

    # SM90 grouped tiles
    for tile in SM90_TILES:
        name = SM90_TILE_NAMES[tile]
        def fn(t=tile):
            out = _cute_grouped_sm90_forward_logic(feats, weight, kmap, N_actual, mma_tile=t)
            assert not isinstance(out, int), f"SM90 grouped tile {t} failed: {out}"
        try:
            mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
            results[name] = avg
        except Exception as e:
            results[name] = f"ERROR: {e}"

    return results


def bench_backward(feats, weight, kmap, N_actual, grad_output):
    """Benchmark backward pass across all backends."""
    results = {}

    # Explicit GEMM backward (reference)
    def fn_explicit():
        _explicit_gemm_backward_logic(grad_output, feats, weight, kmap)
    mn, avg = benchmark_fn(fn_explicit)
    results["explicit_gemm"] = avg

    # SM80 grouped backward tiles
    for tile in SM80_TILES:
        name = SM80_TILE_NAMES[tile]
        def fn(t=tile):
            out = _cute_grouped_backward_logic(
                grad_output, feats, weight, kmap, (True, True), mma_tile=t
            )
            assert not isinstance(out[0], int), f"SM80 bwd tile {t} failed: {out}"
        try:
            mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
            results[name] = avg
        except Exception as e:
            results[name] = f"ERROR: {e}"

    # SM90 grouped backward tiles
    for tile in SM90_TILES:
        name = SM90_TILE_NAMES[tile]
        def fn(t=tile):
            out = _cute_grouped_sm90_backward_logic(
                grad_output, feats, weight, kmap, (True, True), mma_tile=t
            )
            assert not isinstance(out[0], int), f"SM90 bwd tile {t} failed: {out}"
        try:
            mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
            results[name] = avg
        except Exception as e:
            results[name] = f"ERROR: {e}"

    return results


def print_results(label, results):
    """Print benchmark results with comparison."""
    if not results:
        print(f"  {label}: no results")
        return

    # Filter numeric results
    numeric = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    errors = {k: v for k, v in results.items() if isinstance(v, str)}

    if not numeric:
        print(f"  {label}: all errored")
        for name, err in errors.items():
            print(f"    {name:<35s} {err}")
        return

    best_name = min(numeric, key=numeric.get)
    best_time = numeric[best_name]

    # Find best SM80 and best SM90
    sm80_results = {k: v for k, v in numeric.items() if k.startswith("SM80")}
    sm90_results = {k: v for k, v in numeric.items() if k.startswith("SM90")}
    best_sm80 = min(sm80_results, key=sm80_results.get) if sm80_results else None
    best_sm90 = min(sm90_results, key=sm90_results.get) if sm90_results else None

    print(f"  {label}:")
    for name, t in sorted(numeric.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if name == best_name else ""
        slowdown = f"(+{(t / best_time - 1) * 100:.1f}%)" if name != best_name else ""
        print(f"    {name:<35s} {t:8.3f} ms  {slowdown}{marker}")

    # SM90 vs SM80 comparison
    if best_sm80 and best_sm90:
        sm80_t = sm80_results[best_sm80]
        sm90_t = sm90_results[best_sm90]
        delta_pct = (sm80_t - sm90_t) / sm80_t * 100
        if delta_pct > 0:
            print(f"    >>> SM90 wins by {delta_pct:.1f}% ({best_sm90} vs {best_sm80})")
        else:
            print(f"    >>> SM80 wins by {-delta_pct:.1f}% ({best_sm80} vs {best_sm90})")

    for name, err in errors.items():
        print(f"    {name:<35s} {err}")
    print()


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM: {torch.cuda.get_device_capability()}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    configs = [
        # (N_target, C_in, C_out)
        (100_000, 64, 64),
        (100_000, 128, 128),
        (100_000, 256, 256),
        (1_000_000, 64, 64),
        (1_000_000, 128, 128),
        (1_000_000, 256, 256),
    ]

    dtype = torch.float16

    # ========== FORWARD ==========
    print("=" * 90)
    print("FORWARD PASS: cute_grouped_sm90 vs cute_grouped (SM80) vs explicit_gemm")
    print("=" * 90)
    print()

    for N_target, C_in, C_out in configs:
        print(f"  Generating data: N~{N_target // 1000}K, {C_in}->{C_out}, fp16 ...")
        feats, weight, kmap, N_actual = make_test_data(N_target, C_in, C_out, dtype=dtype)
        total_pairs = sum(len(kmap[i][0]) for i in range(len(kmap)))
        label = f"N={N_actual}, {C_in}->{C_out}, pairs={total_pairs}"
        results = bench_forward(feats, weight, kmap, N_actual)
        print_results(label, results)
        # Free memory
        del feats, weight, kmap
        torch.cuda.empty_cache()

    # ========== BACKWARD ==========
    print("=" * 90)
    print("BACKWARD PASS: cute_grouped_sm90 vs cute_grouped (SM80) vs explicit_gemm")
    print("=" * 90)
    print()

    for N_target, C_in, C_out in configs:
        print(f"  Generating data: N~{N_target // 1000}K, {C_in}->{C_out}, fp16 ...")
        feats, weight, kmap, N_actual = make_test_data(N_target, C_in, C_out, dtype=dtype)
        # Compute forward output shape for grad
        fwd_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
        grad_output = torch.randn_like(fwd_out)
        del fwd_out

        total_pairs = sum(len(kmap[i][0]) for i in range(len(kmap)))
        label = f"N={N_actual}, {C_in}->{C_out}, pairs={total_pairs}"
        results = bench_backward(feats, weight, kmap, N_actual, grad_output)
        print_results(label, results)
        del feats, weight, kmap, grad_output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
