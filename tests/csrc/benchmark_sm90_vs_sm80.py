#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark SM90 WGMMA vs SM80 CuTe GEMM kernels on representative sparse conv configs.

Tests single GEMM (per-offset) and grouped GEMM (fused multi-offset) variants.
"""

import torch
import sys

try:
    import warpconvnet._C as _C
except ImportError:
    print("warpconvnet._C not available")
    sys.exit(1)

HAS_SM80 = hasattr(_C.gemm, "cute_gemm_AD_gather_scatter")
HAS_SM90 = hasattr(_C.gemm, "cute_gemm_sm90_AD_gather_scatter")
HAS_SM80_GROUPED = hasattr(_C.gemm, "cute_gemm_grouped_AD_gather_scatter")
HAS_SM90_GROUPED = hasattr(_C.gemm, "cute_gemm_sm90_grouped_AD_gather_scatter")

SM80_TILES = {0: "SM80 128x128x32", 1: "SM80 128x64x32", 3: "SM80 64x64x32"}
SM90_TILES = {100: "SM90 64x128x64", 101: "SM90 128x128x64", 103: "SM90 256x128x64", 104: "SM90 64x64x64"}

SM80_TILE_M = {0: 128, 1: 128, 3: 64}
SM90_TILE_M = {100: 64, 101: 128, 103: 256, 104: 64}


def benchmark_fn(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start, end)]
    return min(times), sum(times) / len(times)


def bench_single_gemm(M, K, N, gather_size, dtype=torch.float16):
    """Benchmark single (per-offset) GEMM: D[out_map] = A[in_map] @ B."""
    A = torch.randn(M, K, dtype=dtype, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=dtype, device="cuda") * 0.1
    in_map = torch.randperm(M, device="cuda")[:gather_size].int()
    out_map = torch.randperm(M, device="cuda")[:gather_size].int()

    results = {}

    # SM80 tiles
    if HAS_SM80:
        for tile, name in SM80_TILES.items():
            D = torch.zeros(M, N, dtype=torch.float32, device="cuda")
            status = [0]
            def fn(t=tile, d=D):
                d.zero_()
                status[0] = _C.gemm.cute_gemm_AD_gather_scatter(
                    A, B, d, d, in_map, out_map, mma_tile=t, alpha=1.0, beta=0.0)
            try:
                mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
                if status[0] == 0:
                    results[name] = avg
            except Exception:
                pass

    # SM90 tiles
    if HAS_SM90:
        for tile, name in SM90_TILES.items():
            D = torch.zeros(M, N, dtype=torch.float32, device="cuda")
            status = [0]
            def fn(t=tile, d=D):
                d.zero_()
                status[0] = _C.gemm.cute_gemm_sm90_AD_gather_scatter(
                    A, B, d, d, in_map, out_map, mma_tile=t, alpha=1.0, beta=0.0)
            try:
                mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
                if status[0] == 0:
                    results[name] = avg
            except Exception:
                pass

    # PyTorch reference
    def pt_fn():
        torch.zeros(M, N, dtype=torch.float32, device="cuda")
        (A[in_map.long()].float() @ B.float())
    mn, avg = benchmark_fn(pt_fn, warmup=3, repeats=15)
    results["PyTorch gather+mm"] = avg

    return results


def bench_grouped_gemm(M, K, N, num_groups, group_size, dtype=torch.float16):
    """Benchmark grouped (fused multi-offset) GEMM."""
    A = torch.randn(M, K, dtype=dtype, device="cuda") * 0.1
    B_list = [torch.randn(K, N, dtype=dtype, device="cuda") * 0.1 for _ in range(num_groups)]

    # Build maps (unique output rows for non-atomic forward test)
    total_gs = num_groups * group_size
    in_maps = [torch.randint(0, M, (group_size,), device="cuda", dtype=torch.int32) for _ in range(num_groups)]
    all_out = torch.randperm(M, device="cuda")[:total_gs].int()
    out_maps = [all_out[i*group_size:(i+1)*group_size] for i in range(num_groups)]

    in_map = torch.cat(in_maps)
    out_map = torch.cat(out_maps)

    weight_ptrs = torch.tensor([b.data_ptr() for b in B_list], device="cuda", dtype=torch.int64)

    results = {}

    def make_grouped_params(tile_m):
        tile_offsets_list = [0]
        map_offsets_list = [0]
        total_m_tiles = 0
        for g in range(num_groups):
            ntiles = (group_size + tile_m - 1) // tile_m
            total_m_tiles += ntiles
            tile_offsets_list.append(total_m_tiles)
            map_offsets_list.append(map_offsets_list[-1] + group_size)
        tile_offsets = torch.tensor(tile_offsets_list, device="cuda", dtype=torch.int32)
        group_sizes = torch.full((num_groups,), group_size, device="cuda", dtype=torch.int32)
        map_offsets = torch.tensor(map_offsets_list[:-1], device="cuda", dtype=torch.int32)
        return tile_offsets, group_sizes, map_offsets, total_m_tiles

    # SM80 grouped
    if HAS_SM80_GROUPED:
        for tile, name in SM80_TILES.items():
            tile_m = SM80_TILE_M[tile]
            tile_offsets, group_sizes, map_offsets, total_m_tiles = make_grouped_params(tile_m)
            D = torch.zeros(M, N, dtype=torch.float32, device="cuda")
            status = [0]
            def fn(t=tile, d=D, to=tile_offsets, gs=group_sizes, mo=map_offsets, tmt=total_m_tiles):
                d.zero_()
                status[0] = _C.gemm.cute_gemm_grouped_AD_gather_scatter(
                    A, d, in_map, out_map, weight_ptrs, to, gs, mo, tmt, t, 1.0)
            try:
                mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
                if status[0] == 0:
                    results[name] = avg
            except Exception:
                pass

    # SM90 grouped (non-atomic forward)
    if HAS_SM90_GROUPED:
        for tile, name in SM90_TILES.items():
            tile_m = SM90_TILE_M[tile]
            tile_offsets, group_sizes, map_offsets, total_m_tiles = make_grouped_params(tile_m)
            D = torch.zeros(M, N, dtype=torch.float32, device="cuda")
            status = [0]
            def fn(t=tile, d=D, to=tile_offsets, gs=group_sizes, mo=map_offsets, tmt=total_m_tiles):
                d.zero_()
                status[0] = _C.gemm.cute_gemm_sm90_grouped_AD_gather_scatter(
                    A, d, in_map, out_map, weight_ptrs, to, gs, mo, tmt, t, 1.0, False)
            try:
                mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
                if status[0] == 0:
                    results[name + " (noatomic)"] = avg
            except Exception:
                pass

    # SM90 grouped (atomic for comparison)
    if HAS_SM90_GROUPED:
        for tile, name in [(101, "SM90 128x128x64")]:
            tile_m = SM90_TILE_M[tile]
            tile_offsets, group_sizes, map_offsets, total_m_tiles = make_grouped_params(tile_m)
            D = torch.zeros(M, N, dtype=torch.float32, device="cuda")
            status = [0]
            def fn(t=tile, d=D, to=tile_offsets, gs=group_sizes, mo=map_offsets, tmt=total_m_tiles):
                d.zero_()
                status[0] = _C.gemm.cute_gemm_sm90_grouped_AD_gather_scatter(
                    A, d, in_map, out_map, weight_ptrs, to, gs, mo, tmt, t, 1.0, True)
            try:
                mn, avg = benchmark_fn(fn, warmup=3, repeats=15)
                if status[0] == 0:
                    results[name + " (atomic)"] = avg
            except Exception:
                pass

    return results


def print_results(label, results):
    if not results:
        print(f"  {label}: no results")
        return
    best_name = min(results, key=results.get)
    best_time = results[best_name]
    print(f"  {label}:")
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if name == best_name else ""
        speedup = f"({best_time/t:.2f}x)" if name != best_name else ""
        slowdown = f"(+{(t/best_time - 1)*100:.1f}%)" if name != best_name else ""
        print(f"    {name:<35s} {t:8.3f} ms  {slowdown}{marker}")
    print()


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM: {torch.cuda.get_device_capability()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"SM80 GEMM: {HAS_SM80}, SM90 GEMM: {HAS_SM90}")
    print(f"SM80 Grouped: {HAS_SM80_GROUPED}, SM90 Grouped: {HAS_SM90_GROUPED}")
    print()

    # ========== Single GEMM benchmarks ==========
    print("=" * 80)
    print("SINGLE GEMM (per-offset): D[out_map] = A[in_map] @ B")
    print("=" * 80)

    single_configs = [
        # (M, K, N, gather_size)
        (100_000, 64, 64, 50_000),
        (100_000, 128, 128, 50_000),
        (100_000, 256, 256, 50_000),
        (1_000_000, 64, 64, 500_000),
        (1_000_000, 128, 128, 500_000),
        (1_000_000, 256, 256, 500_000),
    ]

    for M, K, N, gs in single_configs:
        label = f"M={M//1000}K, K={K}, N={N}, gather={gs//1000}K, fp16"
        results = bench_single_gemm(M, K, N, gs, torch.float16)
        print_results(label, results)

    # ========== Grouped GEMM benchmarks ==========
    print("=" * 80)
    print("GROUPED GEMM (fused multi-offset, 27 groups = 3x3x3 kernel)")
    print("=" * 80)

    grouped_configs = [
        # (M, K, N, num_groups, group_size)
        (100_000, 64, 64, 26, 2000),    # 100K voxels, 64->64, ~52K total pairs
        (100_000, 128, 128, 26, 2000),   # 100K voxels, 128->128
        (100_000, 256, 256, 26, 2000),   # 100K voxels, 256->256
        (1_000_000, 64, 64, 26, 20000),  # 1M voxels, 64->64, ~520K total pairs
        (1_000_000, 128, 128, 26, 20000),# 1M voxels, 128->128
        (1_000_000, 256, 256, 26, 20000),# 1M voxels, 256->256
    ]

    for M, K, N, ng, gs in grouped_configs:
        label = f"M={M//1000}K, {K}->{N}, {ng} groups x {gs}, fp16"
        results = bench_grouped_gemm(M, K, N, ng, gs, torch.float16)
        print_results(label, results)


if __name__ == "__main__":
    main()
