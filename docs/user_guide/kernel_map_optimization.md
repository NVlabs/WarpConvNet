# Kernel Map Construction

WarpConvNet's sparse convolution relies on building a **kernel map** — the mapping from input to output coordinate indices for each kernel offset. This page describes how the kernel map is constructed and the CUDA kernels involved.

## Overview

Sparse convolution applies a 3D kernel (e.g., 3x3x3 = 27 offsets) to sparse voxel coordinates. For each of the K kernel offsets, the system must find which of the M query coordinates, when shifted by that offset, exist in the input coordinate hash table. The result is a pair of packed index arrays (`in_maps`, `out_maps`) with a per-offset `offsets` array.

The kernel map is constructed by `_kernel_map_from_size()` in `torch_discrete.py`, which calls into CUDA kernels defined in `discrete_kernels.cu`.

## Pipeline

The kernel map pipeline has four stages:

```
kernel_map_size_4d  →  postprocess_count  →  cumsum + .item()  →  postprocess_scatter
```

1. **Search** (`kernel_map_size_4d`): For each `(k, m)` pair, compute `query_coords[m] + offset[k]` and look it up in the hash table. Writes the found index (or -1) to a contiguous `(K, M)` intermediate.

2. **Count** (`postprocess_count`): Sequentially scan the intermediate. For each element >= 0, count it towards its kernel offset k. Output: `counts[K]`.

3. **Prefix sum**: `torch.cumsum` on counts to get output offsets, then `.item()` to read the total for output allocation.

4. **Scatter** (`postprocess_scatter`): Sequentially scan the intermediate again. For each match, write the input index and query index to the packed output arrays at the correct position.

The hash table search (step 1) dominates at ~87% of total time. Steps 2-4 operate on the contiguous intermediate via sequential scans, which run at near-peak memory bandwidth.

### Timing Breakdown (500K points, 3x3x3 kernel, RTX 6000 Ada)

| Stage                 | Time        | % of Total |
| --------------------- | ----------- | ---------- |
| `kernel_map_size_4d`  | 0.603ms     | 87.5%      |
| `postprocess_count`   | 0.044ms     | 6.4%       |
| `cumsum` + `.item()`  | 0.016ms     | 2.3%       |
| `postprocess_scatter` | 0.051ms     | 7.4%       |
| **Total**             | **0.689ms** |            |

### Memory

The `(K, M)` intermediate requires `K * M * 4` bytes of temporary GPU memory (54MB for K=27, M=500K). PyTorch's CUDA caching allocator reuses the allocation across calls.

## CUDA Kernels

### `kernel_map_size_4d`

Specialized search kernel for 4D coordinates (batch + 3 spatial dims) with kernel defined by size. Uses a 2D grid: x-dimension over query coordinates, y-dimension over kernel offsets. Each thread computes one `query_coord + offset`, searches the hash table via open addressing, and writes the result to the `(K, M)` intermediate. Kernel sizes and center offsets are cached in shared memory.

Templated on hash function (FNV1A, CityHash, MurmurHash) with `extern "C"` wrappers for each variant.

### `postprocess_count_kernel`

Counts matches per kernel offset from the `(K, M)` intermediate using shared memory privatized counters.

```cuda
extern "C" __global__ void postprocess_count_kernel(
    const int* __restrict__ found_in_coord_index,  // (K, M) flattened
    int* __restrict__ counts,                       // (K,) output
    int K, int M)
```

- **Grid:** 1D, `ceil(K*M / 256)` blocks, 256 threads
- Each block computes which k-values its elements belong to (`k_first` to `k_last`)
- Threads atomicAdd to shared memory counters `s_counts[k - k_first]`
- One global atomicAdd per k-value per block flushes to output
- For M >> 256 (the common case), each block touches exactly one k-value with zero shared memory contention
- Shared memory: 256 x 4B = 1KB

### `postprocess_scatter_kernel`

Scatters matches from the intermediate to packed `in_maps`/`out_maps` arrays.

```cuda
extern "C" __global__ void postprocess_scatter_kernel(
    const int* __restrict__ found_in_coord_index,  // (K, M) flattened
    const int* __restrict__ offsets,                // (K+1,) prefix sum
    int* __restrict__ scatter_counters,             // (K,) atomic counters
    int* __restrict__ in_maps,
    int* __restrict__ out_maps,
    int K, int M)
```

- **Grid:** Same 1D grid as count kernel
- Three phases per block to minimize global atomic contention:
  1. **Count** matches per k in shared memory
  2. **Reserve** a contiguous output range per k via one global atomicAdd per k per block
  3. **Write** each match to its unique position using a block-local atomicAdd counter
- `in_maps[pos]` receives the found input coordinate index; `out_maps[pos]` receives the query coordinate index
- Shared memory: 3 x 256 x 4B = 3KB

Both postprocess kernels use `extern "C"` (no hash function templating) since they only read the pre-computed intermediate.

## Benchmark

| Points | Kernel Map Time |
| ------ | --------------- |
| 10K    | 0.042ms         |
| 50K    | 0.094ms         |
| 100K   | 0.147ms         |
| 200K   | 0.263ms         |
| 500K   | 0.689ms         |

Measured on RTX 6000 Ada (SM 8.9), 3x3x3 kernel, CityHash.

```bash
python benchmarks/bench_search_once.py
```

## Source Files

| File                                                   | Contents                                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `warpconvnet/csrc/discrete_kernels.cu`                 | `kernel_map_size_4d`, `postprocess_count_kernel`, `postprocess_scatter_kernel` |
| `warpconvnet/csrc/coords_launch.cu`                    | Host wrappers for all kernels                                                  |
| `warpconvnet/csrc/bindings/coords_bindings.cpp`        | pybind11 bindings (`_C.coords.*`)                                              |
| `warpconvnet/geometry/coords/search/torch_discrete.py` | `_kernel_map_from_size()` Python entry point                                   |
