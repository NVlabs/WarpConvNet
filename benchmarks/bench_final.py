"""Final benchmark: current optimized pipeline with new block dims."""

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
    print("Default block dims: 64x8 (512 threads)")
    print()

    print(
        f"{'N':>10s}  {'Search':>10s}  {'Count':>10s}  {'Cumsum':>10s}  {'Scatter':>10s}  {'Pipeline':>10s}"
    )
    print("-" * 72)

    for num_coords in [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]:
        coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
        coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)

        ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
        tbl = ht._table_kvs.contiguous()
        vk = ht.vector_keys.contiguous()
        qc = coords.contiguous()
        cap = ht.capacity
        hm = ht.hash_method.value
        kst = torch.tensor(list(kernel_size), dtype=torch.int32, device=device)

        # Pre-run to get found buffer
        found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
        _C.coords.kernel_map_size_4d(
            tbl, vk, qc, kst, found, num_coords, cap, num_kernels, hm, 64, 8
        )

        # Individual stages
        t_search, _ = bench_cuda_events(
            lambda: _C.coords.kernel_map_size_4d(
                tbl, vk, qc, kst, found, num_coords, cap, num_kernels, hm, 64, 8
            )
        )

        t_count, _ = bench_cuda_events(
            lambda: _C.coords.postprocess_count(
                found,
                torch.zeros(num_kernels, dtype=torch.int32, device=device),
                num_kernels,
                num_coords,
            )
        )

        counts = torch.zeros(num_kernels, dtype=torch.int32, device=device)
        _C.coords.postprocess_count(found, counts, num_kernels, num_coords)
        offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        n_total = offsets[-1].item()

        def do_cumsum():
            o = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
            torch.cumsum(counts, dim=0, out=o[1:])
            _ = o[-1].item()

        t_cumsum, _ = bench_cuda_events(do_cumsum)

        im = torch.empty(max(n_total, 1), dtype=torch.int32, device=device)
        om = torch.empty(max(n_total, 1), dtype=torch.int32, device=device)
        t_scatter, _ = bench_cuda_events(
            lambda: _C.coords.postprocess_scatter(
                found,
                offsets,
                torch.zeros(num_kernels, dtype=torch.int32, device=device),
                im,
                om,
                num_kernels,
                num_coords,
            )
        )

        # Full pipeline
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

        t_pipe, _ = bench_cuda_events(pipeline)

        print(
            f"{num_coords:>10,}  {t_search:>9.3f}ms  {t_count:>9.3f}ms  {t_cumsum:>9.3f}ms  {t_scatter:>9.3f}ms  {t_pipe:>9.3f}ms"
        )


if __name__ == "__main__":
    main()
