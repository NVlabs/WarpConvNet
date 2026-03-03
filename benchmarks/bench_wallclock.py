"""Measure wall-clock time to show sync removal benefit on CPU-side latency."""

import time
import torch
import numpy as np

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
import warpconvnet._C as _C


def bench_wallclock(fn, warmup=20, iters=200):
    """Measure wall-clock time (includes CPU overhead + sync)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return np.median(times), np.std(times)


def main():
    device = "cuda"
    kernel_size = (3, 3, 3)
    num_kernels = 27
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {kernel_size} ({num_kernels} offsets)")
    print()

    print("Wall-clock time (includes CPU overhead):")
    print(f"{'N':>10s}  {'HT Build':>10s}  {'Pipeline':>10s}  {'Full E2E':>10s}")
    print("-" * 50)

    for num_coords in [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]:
        coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
        coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)

        # Hash table build
        def build_ht():
            return TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)

        t_build, _ = bench_wallclock(build_ht)

        ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
        tbl = ht._table_kvs.contiguous()
        vk = ht.vector_keys.contiguous()
        qc = coords.contiguous()
        cap = ht.capacity
        hm = ht.hash_method.value
        kst = torch.tensor(list(kernel_size), dtype=torch.int32, device=device)

        # Kernel map pipeline
        def pipeline():
            f = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
            _C.coords.kernel_map_size_4d(
                tbl, vk, qc, kst, f, num_coords, cap, num_kernels, hm, 64, 8
            )
            c = torch.zeros(num_kernels, dtype=torch.int32, device=device)
            _C.coords.postprocess_count(f, c, num_kernels, num_coords)
            o = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
            torch.cumsum(c, dim=0, out=o[1:])
            n = o[-1].item()
            i = torch.empty(max(n, 1), dtype=torch.int32, device=device)
            m = torch.empty(max(n, 1), dtype=torch.int32, device=device)
            if n > 0:
                sc = torch.zeros(num_kernels, dtype=torch.int32, device=device)
                _C.coords.postprocess_scatter(f, o, sc, i, m, num_kernels, num_coords)

        t_pipe, _ = bench_wallclock(pipeline)

        # End-to-end: build + pipeline
        def full_e2e():
            ht2 = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
            tbl2 = ht2._table_kvs.contiguous()
            vk2 = ht2.vector_keys.contiguous()
            f = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
            _C.coords.kernel_map_size_4d(
                tbl2, vk2, qc, kst, f, num_coords, ht2.capacity, num_kernels, hm, 64, 8
            )
            c = torch.zeros(num_kernels, dtype=torch.int32, device=device)
            _C.coords.postprocess_count(f, c, num_kernels, num_coords)
            o = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
            torch.cumsum(c, dim=0, out=o[1:])
            n = o[-1].item()
            i = torch.empty(max(n, 1), dtype=torch.int32, device=device)
            m = torch.empty(max(n, 1), dtype=torch.int32, device=device)
            if n > 0:
                sc = torch.zeros(num_kernels, dtype=torch.int32, device=device)
                _C.coords.postprocess_scatter(f, o, sc, i, m, num_kernels, num_coords)

        t_e2e, _ = bench_wallclock(full_e2e)

        print(f"{num_coords:>10,}  {t_build:>9.3f}ms  {t_pipe:>9.3f}ms  {t_e2e:>9.3f}ms")


if __name__ == "__main__":
    main()
