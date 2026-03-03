"""Benchmark: Baseline (Tier 0) vs Tier 2b (fused count+scatter) vs Search-Once (Tier 3).

Measures kernel_map construction time for the "offsets" return path at various point counts.
"""

import time
import torch
import numpy as np

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
import warpconvnet._C as _C


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times), np.std(times)


def baseline_pipeline(
    table_kvs,
    vector_keys,
    query_coords,
    kernel_size_tensor,
    num_coords,
    capacity,
    num_kernels,
    hash_method_val,
):
    """Tier 0: search → PyTorch postprocess (cumsum, bool, map_found_indices)."""
    found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device="cuda")
    _C.coords.kernel_map_size_4d(
        table_kvs,
        vector_keys,
        query_coords,
        kernel_size_tensor,
        found,
        num_coords,
        capacity,
        num_kernels,
        hash_method_val,
        128,
        8,
    )
    found_bool = found >= 0
    mapped = torch.cumsum(found_bool.to(torch.int32), dim=1, dtype=torch.int32) - 1
    mapped = torch.clamp(mapped, min=-1)
    num_valid = mapped.max(dim=1).values + 1
    off = torch.cumsum(num_valid, dim=0, dtype=torch.int32)
    off = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), off], dim=0)
    n_total = off[-1].item()
    im = torch.empty(n_total, dtype=torch.int32, device="cuda")
    om = torch.empty(n_total, dtype=torch.int32, device="cuda")
    if n_total > 0:
        _C.coords.map_found_indices_to_maps(found, mapped, off, im, om, num_kernels, num_coords)
    return im, om, off


def tier2b_pipeline(
    table_kvs,
    vector_keys,
    query_coords,
    kernel_size_tensor,
    num_coords,
    capacity,
    num_kernels,
    hash_method_val,
):
    """Tier 2b: fused count (search+count) → cumsum → fused scatter (search+scatter)."""
    counts = torch.zeros(num_kernels, dtype=torch.int32, device="cuda")
    _C.coords.kernel_map_size_4d_count(
        table_kvs,
        vector_keys,
        query_coords,
        kernel_size_tensor,
        counts,
        num_coords,
        capacity,
        num_kernels,
        hash_method_val,
        128,
        8,
    )
    offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(counts, dim=0, out=offsets[1:])
    n_total = offsets[-1].item()
    im = torch.empty(n_total, dtype=torch.int32, device="cuda")
    om = torch.empty(n_total, dtype=torch.int32, device="cuda")
    if n_total > 0:
        sc = torch.zeros(num_kernels, dtype=torch.int32, device="cuda")
        _C.coords.kernel_map_size_4d_scatter(
            table_kvs,
            vector_keys,
            query_coords,
            kernel_size_tensor,
            offsets,
            sc,
            im,
            om,
            num_coords,
            capacity,
            num_kernels,
            hash_method_val,
            128,
            8,
        )
    return im, om, offsets


def search_once_pipeline(
    table_kvs,
    vector_keys,
    query_coords,
    kernel_size_tensor,
    num_coords,
    capacity,
    num_kernels,
    hash_method_val,
):
    """Search-Once (Tier 3): search once → postprocess_count → cumsum → postprocess_scatter."""
    found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device="cuda")
    _C.coords.kernel_map_size_4d(
        table_kvs,
        vector_keys,
        query_coords,
        kernel_size_tensor,
        found,
        num_coords,
        capacity,
        num_kernels,
        hash_method_val,
        128,
        8,
    )
    counts = torch.zeros(num_kernels, dtype=torch.int32, device="cuda")
    _C.coords.postprocess_count(found, counts, num_kernels, num_coords)
    offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(counts, dim=0, out=offsets[1:])
    n_total = offsets[-1].item()
    im = torch.empty(n_total, dtype=torch.int32, device="cuda")
    om = torch.empty(n_total, dtype=torch.int32, device="cuda")
    if n_total > 0:
        sc = torch.zeros(num_kernels, dtype=torch.int32, device="cuda")
        _C.coords.postprocess_scatter(found, offsets, sc, im, om, num_kernels, num_coords)
    return im, om, offsets


def main():
    device = "cuda"
    kernel_size = (3, 3, 3)
    num_kernels = 27
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {kernel_size} ({num_kernels} offsets)")
    print()
    print(
        f"{'N':>10s}  {'Baseline':>10s}  {'Tier2b':>10s}  {'SearchOnce':>10s}  {'T2b/BL':>8s}  {'SO/BL':>8s}  {'SO/T2b':>8s}"
    )
    print("-" * 82)

    for num_coords in [10_000, 50_000, 100_000, 200_000, 500_000]:
        coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
        coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)

        ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
        tbl = ht._table_kvs.contiguous()
        vk = ht.vector_keys.contiguous()
        qc = coords.contiguous()
        cap = ht.capacity
        hm = ht.hash_method.value
        kst = torch.tensor(list(kernel_size), dtype=torch.int32, device=device)

        args = (tbl, vk, qc, kst, num_coords, cap, num_kernels, hm)

        t_bl, _ = bench(lambda: baseline_pipeline(*args))
        t_2b, _ = bench(lambda: tier2b_pipeline(*args))
        t_so, _ = bench(lambda: search_once_pipeline(*args))

        print(
            f"{num_coords:>10,}  {t_bl:>9.3f}ms  {t_2b:>9.3f}ms  {t_so:>9.3f}ms  {t_bl/t_2b:>7.2f}x  {t_bl/t_so:>7.2f}x  {t_2b/t_so:>7.2f}x"
        )

    # Detailed breakdown at 500K
    print("\n=== Detailed breakdown at 500K ===\n")
    num_coords = 500_000
    coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
    coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)
    ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
    tbl = ht._table_kvs.contiguous()
    vk = ht.vector_keys.contiguous()
    qc = coords.contiguous()
    cap = ht.capacity
    hm = ht.hash_method.value
    kst = torch.tensor(list(kernel_size), dtype=torch.int32, device=device)

    # Search-only
    t, _ = bench(
        lambda: _C.coords.kernel_map_size_4d(
            tbl,
            vk,
            qc,
            kst,
            torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device),
            num_coords,
            cap,
            num_kernels,
            hm,
            128,
            8,
        )
    )
    print(f"  search kernel:          {t:.3f} ms")

    # Postprocess count
    found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
    _C.coords.kernel_map_size_4d(tbl, vk, qc, kst, found, num_coords, cap, num_kernels, hm, 128, 8)
    torch.cuda.synchronize()
    t, _ = bench(
        lambda: _C.coords.postprocess_count(
            found,
            torch.zeros(num_kernels, dtype=torch.int32, device=device),
            num_kernels,
            num_coords,
        )
    )
    print(f"  postprocess_count:      {t:.3f} ms")

    # Cumsum + .item()
    counts = torch.zeros(num_kernels, dtype=torch.int32, device=device)
    _C.coords.postprocess_count(found, counts, num_kernels, num_coords)
    torch.cuda.synchronize()

    def do_cumsum():
        offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        _ = offsets[-1].item()

    t, _ = bench(do_cumsum)
    print(f"  cumsum + .item():       {t:.3f} ms")

    # Postprocess scatter
    offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    n_total = offsets[-1].item()
    im = torch.empty(n_total, dtype=torch.int32, device=device)
    om = torch.empty(n_total, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    t, _ = bench(
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
    print(f"  postprocess_scatter:    {t:.3f} ms")

    print(f"\n  Intermediate: {num_kernels * num_coords * 4 / 1e6:.1f} MB")
    print(f"  Output maps:  {n_total * 4 * 2 / 1e6:.1f} MB ({n_total:,} entries)")


if __name__ == "__main__":
    main()
