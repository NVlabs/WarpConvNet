"""Fine-grained block dimension sweep around the sweet spot."""

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

    configs = [
        # 256 threads
        (32, 8),
        (64, 4),
        # 384 threads
        (32, 12),
        # 512 threads
        (32, 16),
        (64, 8),
        (128, 4),
        # 768 threads
        (32, 24),
        (64, 12),
        # 1024 threads (baseline)
        (128, 8),
    ]

    for num_coords in [100_000, 200_000, 500_000, 1_000_000]:
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

        print(
            f"  {'Config':>12s}  {'Threads':>8s}  {'Search':>10s}  {'Pipeline':>10s}  {'Srch/BL':>8s}  {'Pipe/BL':>8s}"
        )
        print(f"  {'-'*66}")

        baseline_search = None
        baseline_pipe = None

        for tx, ty in configs:
            found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
            t_search, _ = bench_cuda_events(
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

            def pipeline(tx=tx, ty=ty):
                f = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
                _C.coords.kernel_map_size_4d(
                    tbl, vk, qc, kst, f, num_coords, cap, num_kernels, hm, tx, ty
                )
                c = torch.zeros(num_kernels, dtype=torch.int32, device=device)
                _C.coords.postprocess_count(f, c, num_kernels, num_coords)
                o = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
                torch.cumsum(c, dim=0, out=o[1:])
                n = o[-1].item()
                im = torch.empty(max(n, 1), dtype=torch.int32, device=device)
                om = torch.empty(max(n, 1), dtype=torch.int32, device=device)
                if n > 0:
                    sc = torch.zeros(num_kernels, dtype=torch.int32, device=device)
                    _C.coords.postprocess_scatter(f, o, sc, im, om, num_kernels, num_coords)

            t_pipe, _ = bench_cuda_events(pipeline)

            if tx == 128 and ty == 8:
                baseline_search = t_search
                baseline_pipe = t_pipe

            marker = " <--" if tx == 128 and ty == 8 else ""
            print(
                f"  {tx:>3d}x{ty:<3d}       {tx*ty:>6d}  {t_search:>9.3f}ms  {t_pipe:>9.3f}ms  {baseline_search/t_search if baseline_search else 0:>7.2f}x  {baseline_pipe/t_pipe if baseline_pipe else 0:>7.2f}x{marker}"
            )


if __name__ == "__main__":
    main()
