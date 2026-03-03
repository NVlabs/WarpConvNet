"""A/B benchmark comparing baseline vs optimized sparse conv pipeline."""

import time
import torch
import numpy as np

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
from warpconvnet.geometry.coords.search.torch_discrete import (
    generate_kernel_map,
    _kernel_map_from_size,
)


def bench(fn, warmup=10, iters=50):
    """Benchmark a function with CUDA sync and return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return np.median(times), np.std(times)


def main():
    device = "cuda"
    print("=== Sparse Conv Pipeline Benchmark ===\n")

    for num_coords in [10_000, 50_000, 100_000, 500_000]:
        coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
        coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)
        kernel_size = (3, 3, 3)

        # 1. Hash table build
        med, std = bench(
            lambda: TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device),
            warmup=10,
            iters=50,
        )
        print(f"[{num_coords:>7,}] hash_build:     {med:7.3f} ms (std={std:.3f})")

        # 2. Search
        ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
        med, std = bench(lambda: ht.search(coords), warmup=10, iters=50)
        print(f"[{num_coords:>7,}] search:         {med:7.3f} ms (std={std:.3f})")

        # 3. Kernel map from size (the dominant cost)
        med, std = bench(
            lambda: _kernel_map_from_size(
                ht, coords, kernel_size, identity_map_index=13, return_type="offsets"
            ),
            warmup=10,
            iters=50,
        )
        print(f"[{num_coords:>7,}] kernel_map:     {med:7.3f} ms (std={std:.3f})")

        # 4. Full generate_kernel_map (hash build + kernel map)
        med, std = bench(
            lambda: generate_kernel_map(
                coords,
                coords,
                in_to_out_stride_ratio=(1, 1, 1),
                kernel_size=kernel_size,
                method="size",
                hash_method=HashMethod.CITY,
            ),
            warmup=10,
            iters=50,
        )
        print(f"[{num_coords:>7,}] full_gen:       {med:7.3f} ms (std={std:.3f})")

        print()


if __name__ == "__main__":
    main()
