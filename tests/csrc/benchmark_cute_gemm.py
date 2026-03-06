# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark: CuTe 3.x GEMM vs CUTLASS 2.x vs PyTorch for AD gather-scatter.

import torch
import time
import sys

try:
    import warpconvnet._C as _C
    HAS_CUTE_GEMM = hasattr(_C.gemm, "cute_gemm_AD_gather_scatter")
    HAS_2X_GEMM = hasattr(_C.gemm, "cutlass_gemm_AD_gather_scatter")
except ImportError:
    print("warpconvnet._C not available")
    sys.exit(1)

if not HAS_CUTE_GEMM:
    print("CuTe GEMM bindings not available")
    sys.exit(1)


def benchmark_fn(fn, warmup=5, repeats=20):
    """Benchmark a CUDA function using events for accurate timing."""
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


def pytorch_reference(A, B, idx_a, idx_d, M, N, alpha=1.0):
    """PyTorch gather + matmul reference."""
    D = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    D[idx_d] = alpha * (A[idx_a].float() @ B.float())
    return D


def run_benchmarks():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    # Tile names for display
    tile_names = {
        0: "128x128x32",
        1: "128x64x32",
        2: "64x128x32",
        3: "64x64x32",
    }

    # Benchmark configurations: (M, K, N, gather_size)
    # These simulate realistic sparse convolution scenarios
    configs = [
        # Small channels (early layers)
        (100_000, 32, 32, 50_000),
        (100_000, 32, 64, 50_000),
        # Medium channels
        (100_000, 64, 64, 50_000),
        (100_000, 64, 128, 50_000),
        # Larger channels (deeper layers)
        (50_000, 128, 128, 25_000),
        (50_000, 128, 256, 25_000),
        # Large gather size (dense-ish)
        (200_000, 64, 64, 100_000),
        (200_000, 128, 128, 100_000),
        # Small gather (very sparse)
        (100_000, 64, 64, 5_000),
        (100_000, 128, 128, 5_000),
    ]

    # Header
    print(f"{'Config (MxKxN, gather)':<30} | {'Tile':<14} | {'CuTe 3.x (ms)':<15} | {'CUTLASS 2.x (ms)':<18} | {'PyTorch (ms)':<14} | {'CuTe/2.x':<10} | {'CuTe/PyTorch':<12}")
    print("-" * 130)

    for M, K, N, gather_size in configs:
        dtype = torch.float16

        A = torch.randn(M, K, dtype=dtype, device=device) * 0.1
        B = torch.randn(K, N, dtype=dtype, device=device) * 0.1
        idx_a = torch.randperm(M, device=device)[:gather_size].int()
        idx_d = torch.randperm(M, device=device)[:gather_size].int()

        config_str = f"{M//1000}Kx{K}x{N}, g={gather_size//1000}K"

        # PyTorch reference
        D_pt = torch.zeros(M, N, dtype=torch.float32, device=device)
        pt_min, pt_avg = benchmark_fn(
            lambda: pytorch_reference(A, B, idx_a, idx_d, M, N),
            warmup=3, repeats=10
        )

        # Test each tile config
        for tile in [3, 1, 0]:  # 64x64, 128x64, 128x128
            tile_str = tile_names[tile]

            # CuTe 3.x
            D_cute = torch.zeros(M, N, dtype=torch.float32, device=device)
            cute_status = [0]

            def cute_fn():
                D_cute.zero_()
                cute_status[0] = _C.gemm.cute_gemm_AD_gather_scatter(
                    A, B, D_cute, D_cute, idx_a, idx_d,
                    mma_tile=tile, alpha=1.0, beta=0.0
                )

            cute_min, cute_avg = benchmark_fn(cute_fn, warmup=3, repeats=10)
            if cute_status[0] != 0:
                cute_str = "FAIL"
                ratio_2x = "N/A"
                ratio_pt = "N/A"
            else:
                cute_str = f"{cute_avg:.3f}"

            # CUTLASS 2.x
            cutlass2x_str = "N/A"
            ratio_2x = "N/A"
            if HAS_2X_GEMM:
                D_2x = torch.zeros(M, N, dtype=torch.float32, device=device)
                status_2x = [0]

                def cutlass2x_fn():
                    D_2x.zero_()
                    status_2x[0] = _C.gemm.cutlass_gemm_AD_gather_scatter(
                        A, B, D_2x, D_2x, idx_a, idx_d, mma_tile=tile
                    )

                c2x_min, c2x_avg = benchmark_fn(cutlass2x_fn, warmup=3, repeats=10)
                if status_2x[0] == 0:
                    cutlass2x_str = f"{c2x_avg:.3f}"
                    if cute_status[0] == 0:
                        ratio_2x = f"{cute_avg / c2x_avg:.2f}x"
                else:
                    cutlass2x_str = "FAIL"

            if cute_status[0] == 0:
                ratio_pt = f"{cute_avg / pt_avg:.2f}x"

            print(f"{config_str:<30} | {tile_str:<14} | {cute_str:<15} | {cutlass2x_str:<18} | {pt_avg:<14.3f} | {ratio_2x:<10} | {ratio_pt:<12}")

        print()  # Blank line between configs

    # BF16 comparison (just one config)
    print("\n=== BFloat16 Comparison ===")
    print(f"{'Config':<30} | {'Tile':<14} | {'CuTe BF16 (ms)':<16} | {'CuTe FP16 (ms)':<16}")
    print("-" * 85)

    for M, K, N, gather_size in [(100_000, 64, 64, 50_000), (100_000, 128, 128, 50_000)]:
        config_str = f"{M//1000}Kx{K}x{N}, g={gather_size//1000}K"

        A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
        B_fp16 = torch.randn(K, N, dtype=torch.float16, device=device) * 0.1
        A_bf16 = A_fp16.to(torch.bfloat16)
        B_bf16 = B_fp16.to(torch.bfloat16)
        idx_a = torch.randperm(M, device=device)[:gather_size].int()
        idx_d = torch.randperm(M, device=device)[:gather_size].int()

        for tile in [3, 0]:
            tile_str = tile_names[tile]

            D_bf16 = torch.zeros(M, N, dtype=torch.float32, device=device)
            def bf16_fn():
                D_bf16.zero_()
                _C.gemm.cute_gemm_AD_gather_scatter(
                    A_bf16, B_bf16, D_bf16, D_bf16, idx_a, idx_d,
                    mma_tile=tile, alpha=1.0, beta=0.0
                )

            D_fp16 = torch.zeros(M, N, dtype=torch.float32, device=device)
            def fp16_fn():
                D_fp16.zero_()
                _C.gemm.cute_gemm_AD_gather_scatter(
                    A_fp16, B_fp16, D_fp16, D_fp16, idx_a, idx_d,
                    mma_tile=tile, alpha=1.0, beta=0.0
                )

            bf16_min, bf16_avg = benchmark_fn(bf16_fn, warmup=3, repeats=10)
            fp16_min, fp16_avg = benchmark_fn(fp16_fn, warmup=3, repeats=10)

            print(f"{config_str:<30} | {tile_str:<14} | {bf16_avg:<16.3f} | {fp16_avg:<16.3f}")


if __name__ == "__main__":
    run_benchmarks()
