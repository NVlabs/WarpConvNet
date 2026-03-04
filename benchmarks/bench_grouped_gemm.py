"""Benchmark grouped vs ungrouped GEMM for sparse convolution.

Compares all three backends (explicit, implicit, CUTLASS hybrid) with and
without adaptive offset grouping.
"""

import time
import torch
import numpy as np

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
from warpconvnet.geometry.coords.search.torch_discrete import _kernel_map_from_size
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_forward_grouped,
)
from warpconvnet.nn.functional.sparse_conv.detail.implicit_direct import (
    _implicit_gemm_forward_logic,
    _implicit_gemm_forward_grouped,
)
from warpconvnet.nn.functional.sparse_conv.detail.cutlass import (
    _cutlass_implicit_gemm_forward_logic,
    _cutlass_implicit_gemm_forward_grouped,
)
from warpconvnet.nn.functional.sparse_conv.detail.grouping import (
    prepare_grouped_kernel_map,
)


def bench_cuda_events(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times), np.std(times)


def make_kernel_map(num_coords, kernel_size=(3, 3, 3), device="cuda"):
    coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
    coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)
    ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
    kernel_map = _kernel_map_from_size(ht, coords, kernel_size, return_type="offsets")
    return kernel_map, num_coords


def print_grouping_stats(kernel_map, saturation_m):
    grouped = prepare_grouped_kernel_map(kernel_map, saturation_m=saturation_m)
    n_large = len(grouped.large_offset_indices)
    n_buckets = len(grouped.buckets)
    n_small = sum(len(b) for b in grouped.buckets)
    total_offsets = n_large + n_small
    print(
        f"  Grouping (sat_m={saturation_m}): {total_offsets} offsets -> "
        f"{n_large} large + {n_buckets} buckets ({n_small} small offsets)"
    )
    for i, (b, c, m) in enumerate(
        zip(grouped.buckets, grouped.bucket_pair_counts, grouped.bucket_max_m)
    ):
        waste = (m * len(b) - sum(c)) / sum(c) * 100 if sum(c) > 0 else 0
        print(f"    Bucket {i}: {len(b)} offsets, M_max={m:,}, waste={waste:.1f}%")


def main():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    for num_coords in [10_000, 100_000, 500_000]:
        kernel_map, N = make_kernel_map(num_coords)
        num_out = N

        # Print pair count distribution
        iden = kernel_map.identity_map_index
        counts = []
        for k in range(len(kernel_map)):
            if k == iden:
                continue
            counts.append(kernel_map.numel(k))
        counts.sort()
        print(f"=== {num_coords:,} coords ===")
        print(
            f"  Pair counts: min={min(counts):,} max={max(counts):,} "
            f"median={int(np.median(counts)):,} total={sum(counts):,}"
        )
        print_grouping_stats(kernel_map, saturation_m=5000)
        print()

        for C_in, C_out in [(32, 32), (64, 64)]:
            in_features = torch.randn(N, C_in, device=device)
            weight = torch.randn(len(kernel_map), C_in, C_out, device=device)

            # --- Explicit GEMM ---
            t_ref, _ = bench_cuda_events(
                lambda: _explicit_gemm_forward_logic(in_features, weight, kernel_map, num_out)
            )
            t_grouped, _ = bench_cuda_events(
                lambda: _explicit_gemm_forward_grouped(
                    in_features, weight, kernel_map, num_out, saturation_m=5000
                )
            )
            speedup = t_ref / t_grouped if t_grouped > 0 else 0
            print(
                f"  Explicit {C_in}x{C_out}: ref={t_ref:.3f}ms grouped={t_grouped:.3f}ms "
                f"speedup={speedup:.2f}x"
            )

            # --- Implicit GEMM ---
            t_ref, _ = bench_cuda_events(
                lambda: _implicit_gemm_forward_logic(
                    in_features, weight, kernel_map, num_out, None, 16
                )
            )
            t_grouped, _ = bench_cuda_events(
                lambda: _implicit_gemm_forward_grouped(
                    in_features, weight, kernel_map, num_out, None, 16, saturation_m=5000
                )
            )
            speedup = t_ref / t_grouped if t_grouped > 0 else 0
            print(
                f"  Implicit {C_in}x{C_out}: ref={t_ref:.3f}ms grouped={t_grouped:.3f}ms "
                f"speedup={speedup:.2f}x"
            )

            # --- CUTLASS hybrid ---
            try:
                t_ref, _ = bench_cuda_events(
                    lambda: _cutlass_implicit_gemm_forward_logic(
                        in_features, weight, kernel_map, num_out
                    )
                )
                t_grouped, _ = bench_cuda_events(
                    lambda: _cutlass_implicit_gemm_forward_grouped(
                        in_features, weight, kernel_map, num_out, saturation_m=5000
                    )
                )
                speedup = t_ref / t_grouped if t_grouped > 0 else 0
                print(
                    f"  CUTLASS  {C_in}x{C_out}: ref={t_ref:.3f}ms grouped={t_grouped:.3f}ms "
                    f"speedup={speedup:.2f}x"
                )
            except Exception as e:
                print(f"  CUTLASS  {C_in}x{C_out}: SKIPPED ({e})")

            print()


if __name__ == "__main__":
    main()
