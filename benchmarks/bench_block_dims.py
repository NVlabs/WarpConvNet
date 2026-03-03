"""Benchmark: block dimension tuning for kernel_map_size_4d.

Tests different (threads_x, threads_y) configurations for the search kernel
and full search-once pipeline. Uses CUDA events for precise GPU-side timing.
"""

import torch
import numpy as np

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
import warpconvnet._C as _C


def bench_cuda_events(fn, warmup=20, iters=100):
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


def main():
    device = "cuda"
    kernel_size = (3, 3, 3)
    num_kernels = 27
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {kernel_size} ({num_kernels} offsets)")

    # Block dimension configurations to test
    # (threads_x, threads_y) — total must be <= 1024
    configs = [
        (32, 8),  # 256 threads
        (64, 4),  # 256 threads
        (64, 8),  # 512 threads
        (128, 4),  # 512 threads
        (128, 8),  # 1024 threads (current default)
        (256, 4),  # 1024 threads
        (32, 16),  # 512 threads
        (32, 27),  # 864 threads (exact K=27 coverage)
        (64, 16),  # 1024 threads
    ]

    for num_coords in [100_000, 500_000]:
        print(f"\n=== {num_coords:,} points ===")
        coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
        coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)

        ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
        tbl = ht._table_kvs.contiguous()
        vk = ht.vector_keys.contiguous()
        qc = coords.contiguous()
        cap = ht.capacity
        hm = ht.hash_method.value
        kst = torch.tensor(list(kernel_size), dtype=torch.int32, device=device)

        # --- Search kernel only ---
        print("\n  Search kernel only:")
        print(f"  {'Config':>12s}  {'Threads':>8s}  {'Time':>10s}  {'vs 128x8':>10s}")
        print(f"  {'-'*48}")

        baseline_t = None
        results = []
        for tx, ty in configs:
            found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
            try:
                t, s = bench_cuda_events(
                    lambda tx=tx, ty=ty: _C.coords.kernel_map_size_4d(
                        tbl,
                        vk,
                        qc,
                        kst,
                        found,
                        num_coords,
                        cap,
                        num_kernels,
                        hm,
                        tx,
                        ty,
                    )
                )
                if tx == 128 and ty == 8:
                    baseline_t = t
                results.append((tx, ty, t, s))
            except Exception as e:
                results.append((tx, ty, None, None))

        for tx, ty, t, s in results:
            if t is None:
                print(f"  {tx:>3d}x{ty:<3d}       {tx*ty:>6d}  {'ERROR':>10s}")
            else:
                ratio = baseline_t / t if baseline_t else 0
                marker = " <-- default" if tx == 128 and ty == 8 else ""
                print(
                    f"  {tx:>3d}x{ty:<3d}       {tx*ty:>6d}  {t:>9.3f}ms  {ratio:>9.2f}x{marker}"
                )

        # --- Full pipeline ---
        print("\n  Full pipeline (search + postprocess):")
        print(f"  {'Config':>12s}  {'Time':>10s}  {'vs 128x8':>10s}")
        print(f"  {'-'*40}")

        baseline_t = None
        results = []
        for tx, ty in configs:

            def pipeline(tx=tx, ty=ty):
                found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
                _C.coords.kernel_map_size_4d(
                    tbl,
                    vk,
                    qc,
                    kst,
                    found,
                    num_coords,
                    cap,
                    num_kernels,
                    hm,
                    tx,
                    ty,
                )
                counts = torch.zeros(num_kernels, dtype=torch.int32, device=device)
                _C.coords.postprocess_count(found, counts, num_kernels, num_coords)
                offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
                torch.cumsum(counts, dim=0, out=offsets[1:])
                n_total = offsets[-1].item()
                im = torch.empty(max(n_total, 1), dtype=torch.int32, device=device)
                om = torch.empty(max(n_total, 1), dtype=torch.int32, device=device)
                if n_total > 0:
                    sc = torch.zeros(num_kernels, dtype=torch.int32, device=device)
                    _C.coords.postprocess_scatter(
                        found, offsets, sc, im, om, num_kernels, num_coords
                    )

            try:
                t, s = bench_cuda_events(pipeline)
                if tx == 128 and ty == 8:
                    baseline_t = t
                results.append((tx, ty, t, s))
            except Exception as e:
                results.append((tx, ty, None, None))

        for tx, ty, t, s in results:
            if t is None:
                print(f"  {tx:>3d}x{ty:<3d}       {'ERROR':>10s}")
            else:
                ratio = baseline_t / t if baseline_t else 0
                marker = " <-- default" if tx == 128 and ty == 8 else ""
                print(f"  {tx:>3d}x{ty:<3d}       {t:>9.3f}ms  {ratio:>9.2f}x{marker}")


if __name__ == "__main__":
    main()
