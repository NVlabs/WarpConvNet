"""A/B comparison: with syncs vs without syncs in hash table build."""

import time
import torch
import numpy as np

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.torch_hashmap import HashMethod, _next_power_of_2


def bench_wallclock(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def build_ht_no_sync(coords, device):
    """Build hash table without syncs (current code)."""
    num_keys = coords.shape[0]
    key_dim = coords.shape[1]
    capacity = _next_power_of_2(max(16, num_keys * 2))
    table_kvs = torch.empty((capacity, 2), dtype=torch.int32, device=device)
    vector_keys = coords.contiguous()

    _C.coords.hashmap_prepare(table_kvs, capacity)
    _C.coords.hashmap_insert(
        table_kvs, vector_keys, num_keys, key_dim, capacity, HashMethod.CITY.value
    )
    return table_kvs, vector_keys, capacity


def build_ht_with_sync(coords, device):
    """Build hash table with explicit syncs (old code)."""
    num_keys = coords.shape[0]
    key_dim = coords.shape[1]
    capacity = _next_power_of_2(max(16, num_keys * 2))
    table_kvs = torch.empty((capacity, 2), dtype=torch.int32, device=device)
    vector_keys = coords.contiguous()

    _C.coords.hashmap_prepare(table_kvs, capacity)
    torch.cuda.synchronize()  # OLD sync #1
    _C.coords.hashmap_insert(
        table_kvs, vector_keys, num_keys, key_dim, capacity, HashMethod.CITY.value
    )
    torch.cuda.synchronize()  # OLD sync #2
    return table_kvs, vector_keys, capacity


def main():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Interleave A/B to avoid order-dependent warming effects
    print(f"{'N':>10s}  {'No Sync':>10s}  {'With Sync':>10s}  {'Speedup':>8s}  {'Saved':>10s}")
    print("-" * 58)

    for num_coords in [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]:
        coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
        coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)

        # Warm both paths
        for _ in range(50):
            build_ht_no_sync(coords, device)
            build_ht_with_sync(coords, device)
        torch.cuda.synchronize()

        # Interleaved measurement
        times_no_sync = []
        times_with_sync = []
        for _ in range(500):
            t0 = time.perf_counter()
            build_ht_with_sync(coords, device)
            torch.cuda.synchronize()
            times_with_sync.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            build_ht_no_sync(coords, device)
            torch.cuda.synchronize()
            times_no_sync.append((time.perf_counter() - t0) * 1000)

        t_no = np.median(times_no_sync)
        t_with = np.median(times_with_sync)
        speedup = t_with / t_no
        saved = t_with - t_no
        print(
            f"{num_coords:>10,}  {t_no:>9.3f}ms  {t_with:>9.3f}ms  {speedup:>7.2f}x  {saved:>9.3f}ms"
        )


if __name__ == "__main__":
    main()
