"""Profile the fused count+scatter kernel map vs old approach."""

import time
import torch
import numpy as np

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
import warpconvnet._C as _C


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    device = "cuda"
    num_coords = 500_000
    kernel_size = (3, 3, 3)
    num_kernels = 27

    coords = torch.randint(-256, 256, (num_coords, 4), dtype=torch.int32, device=device)
    coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)

    ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
    kernel_size_tensor = torch.tensor(list(kernel_size), dtype=torch.int32, device=device)

    table_kvs = ht._table_kvs.contiguous()
    vector_keys = ht.vector_keys.contiguous()
    query_coords = coords.contiguous()
    capacity = ht.capacity
    hash_method_val = ht.hash_method.value

    # Warmup
    for _ in range(10):
        counts = torch.zeros(num_kernels, dtype=torch.int32, device=device)
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
    torch.cuda.synchronize()

    # Profile count kernel
    times = []
    for _ in range(50):
        counts = torch.zeros(num_kernels, dtype=torch.int32, device=device)
        t0 = sync_time()
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
        t1 = sync_time()
        times.append((t1 - t0) * 1000)
    print(f"count kernel:    {np.median(times):.3f} ms")

    # Profile cumsum + .item()
    times = []
    for _ in range(50):
        counts = torch.zeros(num_kernels, dtype=torch.int32, device=device)
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
        t0 = sync_time()
        offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        num_total = offsets[-1].item()
        t1 = sync_time()
        times.append((t1 - t0) * 1000)
    print(f"cumsum+.item():  {np.median(times):.3f} ms")
    print(f"  num_total_maps = {num_total}")

    # Profile scatter kernel
    offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
    counts = torch.zeros(num_kernels, dtype=torch.int32, device=device)
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
    torch.cumsum(counts, dim=0, out=offsets[1:])
    num_total = offsets[-1].item()
    in_maps = torch.empty(num_total, dtype=torch.int32, device=device)
    out_maps = torch.empty(num_total, dtype=torch.int32, device=device)

    # Warmup scatter
    for _ in range(10):
        scatter_counters = torch.zeros(num_kernels, dtype=torch.int32, device=device)
        _C.coords.kernel_map_size_4d_scatter(
            table_kvs,
            vector_keys,
            query_coords,
            kernel_size_tensor,
            offsets,
            scatter_counters,
            in_maps,
            out_maps,
            num_coords,
            capacity,
            num_kernels,
            hash_method_val,
            128,
            8,
        )
    torch.cuda.synchronize()

    times = []
    for _ in range(50):
        scatter_counters = torch.zeros(num_kernels, dtype=torch.int32, device=device)
        t0 = sync_time()
        _C.coords.kernel_map_size_4d_scatter(
            table_kvs,
            vector_keys,
            query_coords,
            kernel_size_tensor,
            offsets,
            scatter_counters,
            in_maps,
            out_maps,
            num_coords,
            capacity,
            num_kernels,
            hash_method_val,
            128,
            8,
        )
        t1 = sync_time()
        times.append((t1 - t0) * 1000)
    print(f"scatter kernel:  {np.median(times):.3f} ms")

    # Profile full fused pipeline
    times = []
    for _ in range(50):
        t0 = sync_time()
        counts = torch.zeros(num_kernels, dtype=torch.int32, device=device)
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
        offsets = torch.zeros(num_kernels + 1, dtype=torch.int32, device=device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        num_total = offsets[-1].item()
        in_maps = torch.empty(num_total, dtype=torch.int32, device=device)
        out_maps = torch.empty(num_total, dtype=torch.int32, device=device)
        scatter_counters = torch.zeros(num_kernels, dtype=torch.int32, device=device)
        _C.coords.kernel_map_size_4d_scatter(
            table_kvs,
            vector_keys,
            query_coords,
            kernel_size_tensor,
            offsets,
            scatter_counters,
            in_maps,
            out_maps,
            num_coords,
            capacity,
            num_kernels,
            hash_method_val,
            128,
            8,
        )
        t1 = sync_time()
        times.append((t1 - t0) * 1000)
    print(f"full fused:      {np.median(times):.3f} ms")

    # Profile old approach for comparison
    # Warmup
    for _ in range(10):
        found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
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
    torch.cuda.synchronize()

    # Search only
    times = []
    for _ in range(50):
        found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
        t0 = sync_time()
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
        t1 = sync_time()
        times.append((t1 - t0) * 1000)
    print(f"\nold search:      {np.median(times):.3f} ms")

    # Post-processing only
    found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
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
    torch.cuda.synchronize()

    times = []
    for _ in range(50):
        t0 = sync_time()
        found_bool = found >= 0
        mapped = torch.cumsum(found_bool.to(torch.int32), dim=1, dtype=torch.int32) - 1
        mapped = torch.clamp(mapped, min=-1)
        num_valid = mapped.max(dim=1).values + 1
        off = torch.cumsum(num_valid, dim=0, dtype=torch.int32)
        off = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), off], dim=0)
        n_total = off[-1].item()
        im = torch.empty(n_total, dtype=torch.int32, device=device)
        om = torch.empty(n_total, dtype=torch.int32, device=device)
        if n_total > 0:
            _C.coords.map_found_indices_to_maps(
                found, mapped, off, im, om, num_kernels, num_coords
            )
        t1 = sync_time()
        times.append((t1 - t0) * 1000)
    print(f"old postproc:    {np.median(times):.3f} ms")

    # Full old pipeline
    times = []
    for _ in range(50):
        t0 = sync_time()
        found = torch.empty((num_kernels, num_coords), dtype=torch.int32, device=device)
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
        off = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), off], dim=0)
        n_total = off[-1].item()
        im = torch.empty(n_total, dtype=torch.int32, device=device)
        om = torch.empty(n_total, dtype=torch.int32, device=device)
        if n_total > 0:
            _C.coords.map_found_indices_to_maps(
                found, mapped, off, im, om, num_kernels, num_coords
            )
        t1 = sync_time()
        times.append((t1 - t0) * 1000)
    print(f"full old:        {np.median(times):.3f} ms")

    print(f"\nIntermediate tensor: {num_kernels * num_coords * 4 / 1e6:.1f} MB")
    print(f"Output tensors:      {num_total * 4 * 2 / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
