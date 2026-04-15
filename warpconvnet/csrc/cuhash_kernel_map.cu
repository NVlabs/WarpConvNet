// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// cuhash - Optimized CUDA Hash Table Library
// kernel_map.cu - Kernel map CUDA kernels and host launcher functions
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cuhash/kernel_map.cuh"

namespace cuhash {

// ============================================================================
// Packed Key Kernel Map Kernels (4D)
// ============================================================================

// Offset-based: for each (query, offset), search query+offset in hash table.
// Output: found_in_coord_index[kernel_idx * num_query + query_idx]
__global__ void packed_kernel_map_offset_kernel(const uint64_t *__restrict__ keys,
                                                const int *__restrict__ values,
                                                const int *__restrict__ query_coords,
                                                const int *__restrict__ kernel_offsets,
                                                int *__restrict__ found_in_coord_index,
                                                int num_query,
                                                int num_offsets,
                                                uint32_t capacity_mask) {
  int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int kernel_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (query_idx >= num_query || kernel_idx >= num_offsets) return;

  // Load query as int4 (128-bit coalesced)
  int4 q = *reinterpret_cast<const int4 *>(&query_coords[query_idx * 4]);
  uint64_t base = pack_key_4d(q.x, q.y, q.z, q.w);

  // Load offset
  const int *off = &kernel_offsets[kernel_idx * 4];
  // batch offset is always 0 for kernel offsets
  int found = packed_kernel_map_offset(keys, values, base, off[1], off[2], off[3], capacity_mask);

  found_in_coord_index[kernel_idx * num_query + query_idx] = found;
}

// Size-based: kernel defined by (kx, ky, kz), 4D coords (batch, x, y, z)
__global__ void packed_kernel_map_size_kernel(const uint64_t *__restrict__ keys,
                                              const int *__restrict__ values,
                                              const int *__restrict__ query_coords,
                                              const int *__restrict__ kernel_sizes,
                                              int *__restrict__ found_in_coord_index,
                                              int num_query,
                                              int num_kernels,
                                              uint32_t capacity_mask) {
  // Cache kernel sizes in shared memory
  __shared__ int s_ksz[3];
  __shared__ int s_center[3];
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    s_ksz[0] = kernel_sizes[0];
    s_ksz[1] = kernel_sizes[1];
    s_ksz[2] = kernel_sizes[2];
    s_center[0] = (s_ksz[0] % 2 != 0) ? s_ksz[0] / 2 : 0;
    s_center[1] = (s_ksz[1] % 2 != 0) ? s_ksz[1] / 2 : 0;
    s_center[2] = (s_ksz[2] % 2 != 0) ? s_ksz[2] / 2 : 0;
  }
  __syncthreads();

  int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int kernel_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (query_idx >= num_query || kernel_idx >= num_kernels) return;

  int4 q = *reinterpret_cast<const int4 *>(&query_coords[query_idx * 4]);
  uint64_t base = pack_key_4d(q.x, q.y, q.z, q.w);

  int found = packed_kernel_map_size(keys,
                                     values,
                                     base,
                                     kernel_idx,
                                     s_ksz[0],
                                     s_ksz[1],
                                     s_ksz[2],
                                     s_center[0],
                                     s_center[1],
                                     s_center[2],
                                     capacity_mask);

  found_in_coord_index[kernel_idx * num_query + query_idx] = found;
}

// ============================================================================
// Fused Kernel Map: Count + Scatter (eliminates K*M intermediate)
// ============================================================================

// Pass 1: Count matches per kernel offset
template <int BLOCK_DIM_Y = 8>
__global__ void packed_kernel_map_count_kernel(const uint64_t *__restrict__ keys,
                                               const int *__restrict__ values,
                                               const int *__restrict__ query_coords,
                                               const int *__restrict__ kernel_sizes,
                                               int *__restrict__ counts,
                                               int num_query,
                                               int num_kernels,
                                               uint32_t capacity_mask) {
  __shared__ int s_ksz[3], s_center[3];
  __shared__ int s_block_counts[BLOCK_DIM_Y];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (tid < BLOCK_DIM_Y) s_block_counts[tid] = 0;
  if (tid == 0) {
    s_ksz[0] = kernel_sizes[0];
    s_ksz[1] = kernel_sizes[1];
    s_ksz[2] = kernel_sizes[2];
    s_center[0] = (s_ksz[0] % 2 != 0) ? s_ksz[0] / 2 : 0;
    s_center[1] = (s_ksz[1] % 2 != 0) ? s_ksz[1] / 2 : 0;
    s_center[2] = (s_ksz[2] % 2 != 0) ? s_ksz[2] / 2 : 0;
  }
  __syncthreads();

  int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int kernel_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (query_idx < num_query && kernel_idx < num_kernels) {
    int4 q = *reinterpret_cast<const int4 *>(&query_coords[query_idx * 4]);
    uint64_t base = pack_key_4d(q.x, q.y, q.z, q.w);

    int found = packed_kernel_map_size(keys,
                                       values,
                                       base,
                                       kernel_idx,
                                       s_ksz[0],
                                       s_ksz[1],
                                       s_ksz[2],
                                       s_center[0],
                                       s_center[1],
                                       s_center[2],
                                       capacity_mask);
    if (found >= 0) {
      atomicAdd(&s_block_counts[threadIdx.y], 1);
    }
  }

  __syncthreads();

  // Flush to global
  if (threadIdx.x == 0 && threadIdx.y < BLOCK_DIM_Y) {
    int km_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (km_idx < num_kernels && s_block_counts[threadIdx.y] > 0) {
      atomicAdd(&counts[km_idx], s_block_counts[threadIdx.y]);
    }
  }
}

// Pass 2: Scatter results directly to in_maps/out_maps
template <int BLOCK_DIM_Y = 8>
__global__ void packed_kernel_map_scatter_kernel(const uint64_t *__restrict__ keys,
                                                 const int *__restrict__ values,
                                                 const int *__restrict__ query_coords,
                                                 const int *__restrict__ kernel_sizes,
                                                 const int *__restrict__ offsets,
                                                 int *__restrict__ scatter_counters,
                                                 int *__restrict__ in_maps,
                                                 int *__restrict__ out_maps,
                                                 int num_query,
                                                 int num_kernels,
                                                 uint32_t capacity_mask) {
  __shared__ int s_ksz[3], s_center[3];
  __shared__ int s_block_count[BLOCK_DIM_Y];
  __shared__ int s_block_base[BLOCK_DIM_Y];
  __shared__ int s_local_pos[BLOCK_DIM_Y];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (tid < BLOCK_DIM_Y) {
    s_block_count[tid] = 0;
    s_local_pos[tid] = 0;
  }
  if (tid == 0) {
    s_ksz[0] = kernel_sizes[0];
    s_ksz[1] = kernel_sizes[1];
    s_ksz[2] = kernel_sizes[2];
    s_center[0] = (s_ksz[0] % 2 != 0) ? s_ksz[0] / 2 : 0;
    s_center[1] = (s_ksz[1] % 2 != 0) ? s_ksz[1] / 2 : 0;
    s_center[2] = (s_ksz[2] % 2 != 0) ? s_ksz[2] / 2 : 0;
  }
  __syncthreads();

  int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int kernel_idx = blockIdx.y * blockDim.y + threadIdx.y;

  // Phase 1: Search and count
  int found = -1;
  if (query_idx < num_query && kernel_idx < num_kernels) {
    int4 q = *reinterpret_cast<const int4 *>(&query_coords[query_idx * 4]);
    uint64_t base = pack_key_4d(q.x, q.y, q.z, q.w);

    found = packed_kernel_map_size(keys,
                                   values,
                                   base,
                                   kernel_idx,
                                   s_ksz[0],
                                   s_ksz[1],
                                   s_ksz[2],
                                   s_center[0],
                                   s_center[1],
                                   s_center[2],
                                   capacity_mask);
    if (found >= 0) {
      atomicAdd(&s_block_count[threadIdx.y], 1);
    }
  }

  __syncthreads();

  // Phase 2: Reserve global range
  if (threadIdx.x == 0 && threadIdx.y < BLOCK_DIM_Y) {
    int km_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (km_idx < num_kernels && s_block_count[threadIdx.y] > 0) {
      s_block_base[threadIdx.y] =
          offsets[km_idx] + atomicAdd(&scatter_counters[km_idx], s_block_count[threadIdx.y]);
    }
  }

  __syncthreads();

  // Phase 3: Write results
  if (found >= 0) {
    int local_off = atomicAdd(&s_local_pos[threadIdx.y], 1);
    int pos = s_block_base[threadIdx.y] + local_off;
    in_maps[pos] = found;
    out_maps[pos] = query_idx;
  }
}

// ============================================================================
// Optimized: Per-query loop kernels
//
// Each thread handles ONE query and iterates over ALL K offsets.
// Advantages over the 2D-grid approach:
//   - Query coordinates loaded ONCE (vs K times)
//   - Base coords kept in registers — no repack per iteration
//   - Precomputed offsets in shared memory — no div/mod per iteration
//   - 1D grid = simpler scheduling, better occupancy
// ============================================================================

// Per-query loop: search all K offsets, write to K*M intermediate.
// spatial_offsets: (K, 3) int32, precomputed (ox, oy, oz) per kernel position.
__global__ void packed_kernel_map_loop_kernel(
    const uint64_t *__restrict__ keys,
    const int *__restrict__ values,
    const int *__restrict__ query_coords,
    const int *__restrict__ spatial_offsets,  // (K, 3) flattened
    int *__restrict__ found,                  // (K, M) output
    int num_query,
    int num_kernels,
    uint32_t capacity_mask) {
  // Load offsets into shared memory (K * 3 ints, e.g. 27*3 = 81 ints = 324B)
  extern __shared__ int s_off[];
  for (int i = threadIdx.x; i < num_kernels * 3; i += blockDim.x) {
    s_off[i] = spatial_offsets[i];
  }
  __syncthreads();

  int qidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (qidx >= num_query) return;

  // Load query coords ONCE — stays in registers for all K iterations
  int4 q = *reinterpret_cast<const int4 *>(&query_coords[qidx * 4]);
  int b = q.x, x = q.y, y = q.z, z = q.w;

  for (int k = 0; k < num_kernels; ++k) {
    uint64_t qk = pack_key_4d(b, x + s_off[k * 3], y + s_off[k * 3 + 1], z + s_off[k * 3 + 2]);
    found[k * num_query + qidx] = packed_search(keys, values, qk, capacity_mask);
  }
}

// Per-query loop: count matches per offset (no intermediate).
// 3-phase block reduction: thread-local count → shared → one global atomicAdd.
__global__ void packed_kernel_map_count_loop_kernel(const uint64_t *__restrict__ keys,
                                                    const int *__restrict__ values,
                                                    const int *__restrict__ query_coords,
                                                    const int *__restrict__ spatial_offsets,
                                                    int *__restrict__ counts,
                                                    int num_query,
                                                    int num_kernels,
                                                    uint32_t capacity_mask) {
  extern __shared__ int s_mem[];
  int *s_off = s_mem;                       // K*3 ints
  int *s_counts = &s_mem[num_kernels * 3];  // K ints

  for (int i = threadIdx.x; i < num_kernels * 3; i += blockDim.x) s_off[i] = spatial_offsets[i];
  for (int i = threadIdx.x; i < num_kernels; i += blockDim.x) s_counts[i] = 0;
  __syncthreads();

  int qidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (qidx < num_query) {
    int4 q = *reinterpret_cast<const int4 *>(&query_coords[qidx * 4]);
    int b = q.x, x = q.y, y = q.z, z = q.w;

    for (int k = 0; k < num_kernels; ++k) {
      uint64_t qk = pack_key_4d(b, x + s_off[k * 3], y + s_off[k * 3 + 1], z + s_off[k * 3 + 2]);
      if (packed_search(keys, values, qk, capacity_mask) >= 0) atomicAdd(&s_counts[k], 1);
    }
  }
  __syncthreads();

  // Flush to global — one atomicAdd per offset per block
  for (int k = threadIdx.x; k < num_kernels; k += blockDim.x) {
    if (s_counts[k] > 0) atomicAdd(&counts[k], s_counts[k]);
  }
}

// Per-query loop: scatter pass (no intermediate).
// Requires offsets[] from prefix-sum of counts.
__global__ void packed_kernel_map_scatter_loop_kernel(const uint64_t *__restrict__ keys,
                                                      const int *__restrict__ values,
                                                      const int *__restrict__ query_coords,
                                                      const int *__restrict__ spatial_offsets,
                                                      const int *__restrict__ offsets,
                                                      int *__restrict__ scatter_counters,
                                                      int *__restrict__ in_maps,
                                                      int *__restrict__ out_maps,
                                                      int num_query,
                                                      int num_kernels,
                                                      uint32_t capacity_mask) {
  extern __shared__ int s_mem[];
  int *s_off = s_mem;                            // K*3
  int *s_block_count = &s_mem[num_kernels * 3];  // K
  int *s_block_base = &s_mem[num_kernels * 4];   // K
  int *s_local_pos = &s_mem[num_kernels * 5];    // K

  for (int i = threadIdx.x; i < num_kernels * 3; i += blockDim.x) s_off[i] = spatial_offsets[i];
  for (int i = threadIdx.x; i < num_kernels; i += blockDim.x) {
    s_block_count[i] = 0;
    s_local_pos[i] = 0;
  }
  __syncthreads();

  int qidx = blockIdx.x * blockDim.x + threadIdx.x;

  // Phase 1: search + count per offset (block-local)
  int b = 0, x = 0, y = 0, z = 0;

  if (qidx < num_query) {
    int4 q = *reinterpret_cast<const int4 *>(&query_coords[qidx * 4]);
    b = q.x;
    x = q.y;
    y = q.z;
    z = q.w;

    for (int k = 0; k < num_kernels; ++k) {
      uint64_t qk = pack_key_4d(b, x + s_off[k * 3], y + s_off[k * 3 + 1], z + s_off[k * 3 + 2]);
      if (packed_search(keys, values, qk, capacity_mask) >= 0) atomicAdd(&s_block_count[k], 1);
    }
  }
  __syncthreads();

  // Phase 2: reserve global range
  for (int k = threadIdx.x; k < num_kernels; k += blockDim.x) {
    if (s_block_count[k] > 0) {
      s_block_base[k] = offsets[k] + atomicAdd(&scatter_counters[k], s_block_count[k]);
    }
  }
  __syncthreads();

  // Phase 3: scatter — re-search all offsets (misses terminate quickly)
  if (qidx < num_query) {
    for (int k = 0; k < num_kernels; ++k) {
      uint64_t qk = pack_key_4d(b, x + s_off[k * 3], y + s_off[k * 3 + 1], z + s_off[k * 3 + 2]);
      int val = packed_search(keys, values, qk, capacity_mask);
      if (val >= 0) {
        int pos = s_block_base[k] + atomicAdd(&s_local_pos[k], 1);
        in_maps[pos] = val;
        out_maps[pos] = qidx;
      }
    }
  }
}

// ============================================================================
// Single-pass fused kernel (K ≤ 32): search + count + scatter in ONE launch.
// Stores K found_indices in registers — no K*M intermediate, no re-search.
// ============================================================================

template <int MAX_K>
__global__ void packed_kernel_map_onepass_kernel(
    const uint64_t *__restrict__ keys,
    const int *__restrict__ values,
    const int *__restrict__ query_coords,
    const int *__restrict__ spatial_offsets,
    const int *__restrict__ offsets,     // (K+1,) from count-pass prefix sum
    int *__restrict__ scatter_counters,  // (K,)
    int *__restrict__ in_maps,
    int *__restrict__ out_maps,
    int num_query,
    int num_kernels,
    uint32_t capacity_mask) {
  extern __shared__ int s_mem[];
  int *s_off = s_mem;                            // K*3
  int *s_block_count = &s_mem[num_kernels * 3];  // K
  int *s_block_base = &s_mem[num_kernels * 4];   // K
  int *s_local_pos = &s_mem[num_kernels * 5];    // K

  for (int i = threadIdx.x; i < num_kernels * 3; i += blockDim.x) s_off[i] = spatial_offsets[i];
  for (int i = threadIdx.x; i < num_kernels; i += blockDim.x) {
    s_block_count[i] = 0;
    s_local_pos[i] = 0;
  }
  __syncthreads();

  int qidx = blockIdx.x * blockDim.x + threadIdx.x;

  // Thread-local storage for K found indices (in registers for K ≤ 32)
  int found_vals[MAX_K];
  uint32_t match_mask = 0;

  if (qidx < num_query) {
    int4 q = *reinterpret_cast<const int4 *>(&query_coords[qidx * 4]);
    int b = q.x, x = q.y, y = q.z, z = q.w;

#pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
      if (k >= num_kernels) break;
      uint64_t qk = pack_key_4d(b, x + s_off[k * 3], y + s_off[k * 3 + 1], z + s_off[k * 3 + 2]);
      found_vals[k] = packed_search(keys, values, qk, capacity_mask);
      if (found_vals[k] >= 0) {
        match_mask |= (1u << k);
        atomicAdd(&s_block_count[k], 1);
      }
    }
  }
  __syncthreads();

  // Reserve global range
  for (int k = threadIdx.x; k < num_kernels; k += blockDim.x) {
    if (s_block_count[k] > 0)
      s_block_base[k] = offsets[k] + atomicAdd(&scatter_counters[k], s_block_count[k]);
  }
  __syncthreads();

  // Scatter from registers — no re-search needed
  if (qidx < num_query && match_mask != 0) {
    uint32_t bits = match_mask;
    while (bits) {
      int k = __ffs(bits) - 1;
      bits &= bits - 1;  // clear lowest set bit
      int pos = s_block_base[k] + atomicAdd(&s_local_pos[k], 1);
      in_maps[pos] = found_vals[k];
      out_maps[pos] = qidx;
    }
  }
}

// ============================================================================
// Postprocess kernels (operate on K*M intermediate, no hash table access)
// ============================================================================

__global__ void postprocess_count_kernel(const int *__restrict__ found_in_coord_index,
                                         int *__restrict__ counts,
                                         int K,
                                         int M) {
  const int BLOCK_SIZE = 256;
  __shared__ int s_counts[BLOCK_SIZE];

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * M;

  int block_start = blockIdx.x * blockDim.x;
  int k_first = block_start / M;
  int block_end = min(block_start + BLOCK_SIZE - 1, total - 1);
  int k_last = (total > 0) ? block_end / M : 0;
  int num_k = k_last - k_first + 1;

  if (threadIdx.x < num_k) s_counts[threadIdx.x] = 0;
  __syncthreads();

  if (flat_idx < total) {
    if (found_in_coord_index[flat_idx] >= 0) {
      atomicAdd(&s_counts[flat_idx / M - k_first], 1);
    }
  }

  __syncthreads();

  if (threadIdx.x < num_k) {
    int k = k_first + threadIdx.x;
    if (k < K && s_counts[threadIdx.x] > 0) {
      atomicAdd(&counts[k], s_counts[threadIdx.x]);
    }
  }
}

__global__ void postprocess_scatter_kernel(const int *__restrict__ found_in_coord_index,
                                           const int *__restrict__ offsets,
                                           int *__restrict__ scatter_counters,
                                           int *__restrict__ in_maps,
                                           int *__restrict__ out_maps,
                                           int K,
                                           int M) {
  const int BLOCK_SIZE = 256;
  __shared__ int s_counts[BLOCK_SIZE];
  __shared__ int s_base[BLOCK_SIZE];
  __shared__ int s_local_pos[BLOCK_SIZE];

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * M;

  int block_start = blockIdx.x * blockDim.x;
  int k_first = block_start / M;
  int block_end = min(block_start + BLOCK_SIZE - 1, total - 1);
  int k_last = (total > 0) ? block_end / M : 0;
  int num_k = k_last - k_first + 1;

  if (threadIdx.x < num_k) {
    s_counts[threadIdx.x] = 0;
    s_local_pos[threadIdx.x] = 0;
  }
  __syncthreads();

  int val = -1, k = -1;
  if (flat_idx < total) {
    val = found_in_coord_index[flat_idx];
    k = flat_idx / M;
    if (val >= 0) {
      atomicAdd(&s_counts[k - k_first], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x < num_k) {
    int ki = k_first + threadIdx.x;
    if (ki < K && s_counts[threadIdx.x] > 0) {
      s_base[threadIdx.x] = offsets[ki] + atomicAdd(&scatter_counters[ki], s_counts[threadIdx.x]);
    }
  }
  __syncthreads();

  if (val >= 0) {
    int local_off = atomicAdd(&s_local_pos[k - k_first], 1);
    int pos = s_base[k - k_first] + local_off;
    in_maps[pos] = val;
    out_maps[pos] = flat_idx % M;
  }
}

// ============================================================================
// Host Launcher Functions for Kernel Map
// ============================================================================

void launch_packed_kernel_map_offset(torch::Tensor keys,
                                     torch::Tensor values,
                                     torch::Tensor query_coords,
                                     torch::Tensor kernel_offsets,
                                     torch::Tensor output,
                                     int num_query,
                                     int num_offsets,
                                     int capacity,
                                     int threads_x,
                                     int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_offsets + threads_y - 1) / threads_y);

  packed_kernel_map_offset_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      kernel_offsets.data_ptr<int>(),
      output.data_ptr<int>(),
      num_query,
      num_offsets,
      mask);
}

void launch_packed_kernel_map_size(torch::Tensor keys,
                                   torch::Tensor values,
                                   torch::Tensor query_coords,
                                   torch::Tensor kernel_sizes,
                                   torch::Tensor output,
                                   int num_query,
                                   int num_kernels,
                                   int capacity,
                                   int threads_x,
                                   int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_kernels + threads_y - 1) / threads_y);

  packed_kernel_map_size_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      kernel_sizes.data_ptr<int>(),
      output.data_ptr<int>(),
      num_query,
      num_kernels,
      mask);
}

void launch_packed_kernel_map_count(torch::Tensor keys,
                                    torch::Tensor values,
                                    torch::Tensor query_coords,
                                    torch::Tensor kernel_sizes,
                                    torch::Tensor counts,
                                    int num_query,
                                    int num_kernels,
                                    int capacity,
                                    int threads_x,
                                    int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_kernels + threads_y - 1) / threads_y);

  packed_kernel_map_count_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      kernel_sizes.data_ptr<int>(),
      counts.data_ptr<int>(),
      num_query,
      num_kernels,
      mask);
}

void launch_packed_kernel_map_scatter(torch::Tensor keys,
                                      torch::Tensor values,
                                      torch::Tensor query_coords,
                                      torch::Tensor kernel_sizes,
                                      torch::Tensor offsets,
                                      torch::Tensor scatter_counters,
                                      torch::Tensor in_maps,
                                      torch::Tensor out_maps,
                                      int num_query,
                                      int num_kernels,
                                      int capacity,
                                      int threads_x,
                                      int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_kernels + threads_y - 1) / threads_y);

  packed_kernel_map_scatter_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      kernel_sizes.data_ptr<int>(),
      offsets.data_ptr<int>(),
      scatter_counters.data_ptr<int>(),
      in_maps.data_ptr<int>(),
      out_maps.data_ptr<int>(),
      num_query,
      num_kernels,
      mask);
}

// --- Per-query loop launchers ---

void launch_packed_kernel_map_loop(torch::Tensor keys,
                                   torch::Tensor values,
                                   torch::Tensor query_coords,
                                   torch::Tensor spatial_offsets,
                                   torch::Tensor output,
                                   int num_query,
                                   int num_kernels,
                                   int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  int block_size = 128;
  int grid = (num_query + block_size - 1) / block_size;
  int smem = num_kernels * 3 * sizeof(int);

  packed_kernel_map_loop_kernel<<<grid, block_size, smem, stream>>>(
      reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      spatial_offsets.data_ptr<int>(),
      output.data_ptr<int>(),
      num_query,
      num_kernels,
      mask);
}

void launch_packed_kernel_map_count_loop(torch::Tensor keys,
                                         torch::Tensor values,
                                         torch::Tensor query_coords,
                                         torch::Tensor spatial_offsets,
                                         torch::Tensor counts,
                                         int num_query,
                                         int num_kernels,
                                         int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  int block_size = 128;
  int grid = (num_query + block_size - 1) / block_size;
  int smem = (num_kernels * 3 + num_kernels) * sizeof(int);  // offsets + counts

  packed_kernel_map_count_loop_kernel<<<grid, block_size, smem, stream>>>(
      reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      spatial_offsets.data_ptr<int>(),
      counts.data_ptr<int>(),
      num_query,
      num_kernels,
      mask);
}

void launch_packed_kernel_map_scatter_loop(torch::Tensor keys,
                                           torch::Tensor values,
                                           torch::Tensor query_coords,
                                           torch::Tensor spatial_offsets,
                                           torch::Tensor offsets,
                                           torch::Tensor scatter_counters,
                                           torch::Tensor in_maps,
                                           torch::Tensor out_maps,
                                           int num_query,
                                           int num_kernels,
                                           int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  int block_size = 128;
  int grid = (num_query + block_size - 1) / block_size;
  // offsets(K*3) + block_count(K) + block_base(K) + local_pos(K) = K*6
  int smem = num_kernels * 6 * sizeof(int);

  packed_kernel_map_scatter_loop_kernel<<<grid, block_size, smem, stream>>>(
      reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      spatial_offsets.data_ptr<int>(),
      offsets.data_ptr<int>(),
      scatter_counters.data_ptr<int>(),
      in_maps.data_ptr<int>(),
      out_maps.data_ptr<int>(),
      num_query,
      num_kernels,
      mask);
}

void launch_packed_kernel_map_onepass(torch::Tensor keys,
                                      torch::Tensor values,
                                      torch::Tensor query_coords,
                                      torch::Tensor spatial_offsets,
                                      torch::Tensor offsets,
                                      torch::Tensor scatter_counters,
                                      torch::Tensor in_maps,
                                      torch::Tensor out_maps,
                                      int num_query,
                                      int num_kernels,
                                      int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  int block_size = 128;
  int grid_size = (num_query + block_size - 1) / block_size;
  int smem = num_kernels * 6 * sizeof(int);

  // Dispatch to the right template instantiation
  if (num_kernels <= 27) {
    packed_kernel_map_onepass_kernel<27><<<grid_size, block_size, smem, stream>>>(
        reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
        values.data_ptr<int>(),
        query_coords.data_ptr<int>(),
        spatial_offsets.data_ptr<int>(),
        offsets.data_ptr<int>(),
        scatter_counters.data_ptr<int>(),
        in_maps.data_ptr<int>(),
        out_maps.data_ptr<int>(),
        num_query,
        num_kernels,
        mask);
  } else {
    // K > 27: use the loop scatter kernel (re-searches matching offsets)
    launch_packed_kernel_map_scatter_loop(keys,
                                          values,
                                          query_coords,
                                          spatial_offsets,
                                          offsets,
                                          scatter_counters,
                                          in_maps,
                                          out_maps,
                                          num_query,
                                          num_kernels,
                                          capacity);
  }
}

void launch_postprocess_count(torch::Tensor found, torch::Tensor counts, int K, int M) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int total = K * M;
  int blocks = (total + 255) / 256;
  postprocess_count_kernel<<<blocks, 256, 0, stream>>>(
      found.data_ptr<int>(), counts.data_ptr<int>(), K, M);
}

void launch_postprocess_scatter(torch::Tensor found,
                                torch::Tensor offsets,
                                torch::Tensor scatter_counters,
                                torch::Tensor in_maps,
                                torch::Tensor out_maps,
                                int K,
                                int M) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int total = K * M;
  int blocks = (total + 255) / 256;
  postprocess_scatter_kernel<<<blocks, 256, 0, stream>>>(found.data_ptr<int>(),
                                                         offsets.data_ptr<int>(),
                                                         scatter_counters.data_ptr<int>(),
                                                         in_maps.data_ptr<int>(),
                                                         out_maps.data_ptr<int>(),
                                                         K,
                                                         M);
}

// ============================================================================
// Mask data kernels for production masked GEMM
// ============================================================================

// Convert CSR (in_maps, out_maps, offsets) to pair_table [K * N_out].
// pair_table[k * N_out + out_idx] = in_idx, or -1 if no match.
__global__ void csr_to_pair_table_kernel(const int *__restrict__ in_maps,
                                         const int *__restrict__ out_maps,
                                         const int *__restrict__ offsets,
                                         int *__restrict__ pair_table,
                                         int N_out,
                                         int K,
                                         int L) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= L) return;
  // Binary search for which offset k this idx belongs to
  int lo = 0, hi = K;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (offsets[mid + 1] <= idx)
      lo = mid + 1;
    else
      hi = mid;
  }
  int k = lo;
  int out_idx = out_maps[idx];
  if (out_idx >= 0 && out_idx < N_out) pair_table[k * N_out + out_idx] = in_maps[idx];
}

// Build pair_mask from pair_table.
// pair_mask[i] = bitmask where bit k is set iff pair_table[k*N+i] >= 0.
__global__ void build_pair_mask_kernel(const int *__restrict__ pair_table,
                                       int *__restrict__ pair_mask,
                                       int N,
                                       int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  uint32_t mask = 0;
  for (int k = 0; k < K; ++k) {
    if (pair_table[k * N + i] >= 0) mask |= (1u << k);
  }
  pair_mask[i] = static_cast<int>(mask);
}

void launch_csr_to_pair_table(torch::Tensor in_maps,
                              torch::Tensor out_maps,
                              torch::Tensor offsets,
                              torch::Tensor pair_table,
                              int N_out,
                              int K,
                              int L) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (L + 255) / 256;
  csr_to_pair_table_kernel<<<blocks, 256, 0, stream>>>(in_maps.data_ptr<int>(),
                                                       out_maps.data_ptr<int>(),
                                                       offsets.data_ptr<int>(),
                                                       pair_table.data_ptr<int>(),
                                                       N_out,
                                                       K,
                                                       L);
}

void launch_build_pair_mask(torch::Tensor pair_table, torch::Tensor pair_mask, int N, int K) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (N + 255) / 256;
  build_pair_mask_kernel<<<blocks, 256, 0, stream>>>(
      pair_table.data_ptr<int>(), pair_mask.data_ptr<int>(), N, K);
}

// ============================================================================
// Hierarchical Kernel Map: coarse-probe + pruned fine-search
//
// Two-pass approach:
//   Pass 1 (coarse_probe): per-query loop over K_c coarse offsets.
//     Probes a stride-S coarse hash table. Stores a uint32 bitmask per query
//     indicating which coarse cells are occupied (K_c <= 27 for S=4, K=7).
//   Pass 2 (fine_search_pruned): 2D grid over (M, K_fine).
//     Each thread computes which coarse cell its fine offset maps to via
//     arithmetic right shift (1 instruction for power-of-2 stride).
//     Skips fine search if coarse cell is empty.
//
// Expected savings for 7x7x7 at ~20% coarse occupancy:
//   Flat: 343 probes.  Hierarchical: 27 coarse + 0.2*343 fine ≈ 96 probes.
// ============================================================================

// Pass 1: coarse probe (per-query loop)
__global__ void coarse_probe_kernel(const uint64_t *__restrict__ keys_c,
                                    const int *__restrict__ values_c,
                                    const int *__restrict__ query_coords,
                                    const int *__restrict__ coarse_spatial_offsets,  // (K_c, 3)
                                    int *__restrict__ coarse_masks,  // (M,) uint32 bitmask
                                    int num_query,
                                    int num_coarse_offsets,
                                    int stride_shift,
                                    uint32_t coarse_capacity_mask) {
  extern __shared__ int s_coff[];
  for (int i = threadIdx.x; i < num_coarse_offsets * 3; i += blockDim.x)
    s_coff[i] = coarse_spatial_offsets[i];
  __syncthreads();

  int qidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (qidx >= num_query) return;

  int4 q = *reinterpret_cast<const int4 *>(&query_coords[qidx * 4]);
  int cqx = q.y >> stride_shift;
  int cqy = q.z >> stride_shift;
  int cqz = q.w >> stride_shift;

  uint32_t mask = 0;
  for (int k = 0; k < num_coarse_offsets; ++k) {
    uint64_t ck =
        pack_key_4d(q.x, cqx + s_coff[k * 3], cqy + s_coff[k * 3 + 1], cqz + s_coff[k * 3 + 2]);
    if (packed_search(keys_c, values_c, ck, coarse_capacity_mask) >= 0) mask |= (1u << k);
  }
  coarse_masks[qidx] = static_cast<int>(mask);
}

// Pass 2: fine search with coarse pruning (2D grid)
__global__ void fine_search_pruned_kernel(const uint64_t *__restrict__ keys_f,
                                          const int *__restrict__ values_f,
                                          const int *__restrict__ query_coords,
                                          const int *__restrict__ fine_spatial_offsets,  // (K, 3)
                                          const int *__restrict__ coarse_masks,          // (M,)
                                          int *__restrict__ found,                       // (K, M)
                                          int num_query,
                                          int num_fine_offsets,
                                          int stride_shift,
                                          int coarse_min,
                                          int coarse_dy,
                                          int coarse_dz,
                                          uint32_t fine_capacity_mask) {
  // Load fine offsets into shared memory
  extern __shared__ int s_foff[];
  int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
  int block_size = blockDim.x * blockDim.y;
  for (int i = tid_flat; i < num_fine_offsets * 3; i += block_size)
    s_foff[i] = fine_spatial_offsets[i];
  __syncthreads();

  int qidx = blockIdx.x * blockDim.x + threadIdx.x;
  int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  if (qidx >= num_query || kidx >= num_fine_offsets) {
    return;
  }

  // Load coarse mask for this query (coalesced: consecutive threads read
  // consecutive qidx)
  uint32_t cmask = static_cast<uint32_t>(coarse_masks[qidx]);

  // Load query coords
  int4 q = *reinterpret_cast<const int4 *>(&query_coords[qidx * 4]);

  // Compute fine target coord
  int fx = q.y + s_foff[kidx * 3];
  int fy = q.z + s_foff[kidx * 3 + 1];
  int fz = q.w + s_foff[kidx * 3 + 2];

  // Coarse cell of fine target (arithmetic right shift = floor_div for pow2)
  int ccx = fx >> stride_shift;
  int ccy = fy >> stride_shift;
  int ccz = fz >> stride_shift;

  // Coarse cell of query (for relative offset)
  int cqx = q.y >> stride_shift;
  int cqy = q.z >> stride_shift;
  int cqz = q.w >> stride_shift;

  // Relative coarse offset → index into bitmask
  int ci = (ccx - cqx - coarse_min) * coarse_dy * coarse_dz + (ccy - cqy - coarse_min) * coarse_dz +
           (ccz - cqz - coarse_min);

  // Prune: skip fine search if coarse cell is empty
  if (ci < 0 || ci >= 27 || !(cmask & (1u << ci))) {
    found[kidx * num_query + qidx] = -1;
    return;
  }

  // Coarse cell occupied → do fine search
  uint64_t fkey = pack_key_4d(q.x, fx, fy, fz);
  found[kidx * num_query + qidx] = packed_search(keys_f, values_f, fkey, fine_capacity_mask);
}

// ============================================================================
// Host launchers for hierarchical search
// ============================================================================

void launch_coarse_probe(torch::Tensor keys_c,
                         torch::Tensor values_c,
                         torch::Tensor query_coords,
                         torch::Tensor coarse_spatial_offsets,
                         torch::Tensor coarse_masks,
                         int num_query,
                         int num_coarse_offsets,
                         int stride_shift,
                         int coarse_capacity) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(coarse_capacity - 1);
  int bs = 128;
  int grid = (num_query + bs - 1) / bs;
  int smem = num_coarse_offsets * 3 * sizeof(int);
  coarse_probe_kernel<<<grid, bs, smem, stream>>>(
      reinterpret_cast<const uint64_t *>(keys_c.data_ptr<int64_t>()),
      values_c.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      coarse_spatial_offsets.data_ptr<int>(),
      coarse_masks.data_ptr<int>(),
      num_query,
      num_coarse_offsets,
      stride_shift,
      mask);
}

void launch_fine_search_pruned(torch::Tensor keys_f,
                               torch::Tensor values_f,
                               torch::Tensor query_coords,
                               torch::Tensor fine_spatial_offsets,
                               torch::Tensor coarse_masks,
                               torch::Tensor found,
                               int num_query,
                               int num_fine_offsets,
                               int stride_shift,
                               int coarse_min,
                               int coarse_dy,
                               int coarse_dz,
                               int fine_capacity,
                               int threads_x,
                               int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(fine_capacity - 1);
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x,
            (num_fine_offsets + threads_y - 1) / threads_y);
  int smem = num_fine_offsets * 3 * sizeof(int);
  fine_search_pruned_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const uint64_t *>(keys_f.data_ptr<int64_t>()),
      values_f.data_ptr<int>(),
      query_coords.data_ptr<int>(),
      fine_spatial_offsets.data_ptr<int>(),
      coarse_masks.data_ptr<int>(),
      found.data_ptr<int>(),
      num_query,
      num_fine_offsets,
      stride_shift,
      coarse_min,
      coarse_dy,
      coarse_dz,
      mask);
}

// ============================================================================
// Fused hierarchical kernel map: single C++ call, no Python round-trips.
//
// Does all of: coarse table build → coarse probe → pruned fine search →
// postprocess (count + cumsum + scatter) → return (in_maps, out_maps,
// offsets, pair_table).
// ============================================================================

// Declared in cuhash_hash_table.cu — link-time resolution.
void launch_packed_prepare(torch::Tensor keys, torch::Tensor values, int capacity);
void launch_packed_build_coarse(torch::Tensor keys,
                                torch::Tensor values,
                                torch::Tensor fine_coords,
                                int num_fine,
                                int stride_shift,
                                int capacity,
                                torch::Tensor num_entries_tensor);

static int next_power_of_2(int n) {
  if (n <= 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_hierarchical_kernel_map(torch::Tensor fine_keys,       // int64 [fine_capacity]
                               torch::Tensor fine_values,     // int32 [fine_capacity]
                               torch::Tensor fine_coords,     // int32 [N, 4]
                               torch::Tensor query_coords,    // int32 [M, 4]
                               std::vector<int> kernel_size,  // [kx, ky, kz]
                               int stride,
                               int fine_capacity) {
  TORCH_CHECK(kernel_size.size() == 3);
  TORCH_CHECK(stride > 0 && (stride & (stride - 1)) == 0, "stride must be power of 2");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto dev = fine_keys.device();
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);

  int stride_shift = 0;
  {
    int s = stride;
    while (s > 1) {
      s >>= 1;
      stride_shift++;
    }
  }

  int N = fine_coords.size(0);
  int M = query_coords.size(0);
  int kx = kernel_size[0], ky = kernel_size[1], kz = kernel_size[2];
  int K = kx * ky * kz;

  // --- 1. Build coarse hash table ---
  int coarse_cap = next_power_of_2(std::max(16, N * 2));
  auto coarse_keys = torch::empty({coarse_cap}, opts_i64);
  auto coarse_vals = torch::empty({coarse_cap}, opts_i32);
  launch_packed_prepare(coarse_keys, coarse_vals, coarse_cap);
  auto num_entries_t = torch::zeros({1}, opts_i32);
  launch_packed_build_coarse(coarse_keys,
                             coarse_vals,
                             fine_coords.contiguous(),
                             N,
                             stride_shift,
                             coarse_cap,
                             num_entries_t);

  // --- 2. Compute spatial offsets (C++ side, no Python) ---
  int R[3] = {kx / 2, ky / 2, kz / 2};
  int c[3];
  for (int d = 0; d < 3; d++) c[d] = (R[d] + stride - 1) / stride;  // ceil(R/stride)

  int coarse_dim[3] = {2 * c[0] + 1, 2 * c[1] + 1, 2 * c[2] + 1};
  int K_c = coarse_dim[0] * coarse_dim[1] * coarse_dim[2];
  TORCH_CHECK(K_c <= 27, "Coarse grid too large (", K_c, " > 27). Use a larger stride.");

  // Coarse offsets: (K_c, 3)
  std::vector<int> coff_vec;
  coff_vec.reserve(K_c * 3);
  for (int i = -c[0]; i <= c[0]; i++)
    for (int j = -c[1]; j <= c[1]; j++)
      for (int k = -c[2]; k <= c[2]; k++) {
        coff_vec.push_back(i);
        coff_vec.push_back(j);
        coff_vec.push_back(k);
      }
  auto coarse_spatial_off = torch::from_blob(coff_vec.data(), {K_c, 3}, torch::kInt32).to(dev);

  // Fine offsets: (K, 3)
  int fcx = (kx % 2 == 1) ? kx / 2 : 0;
  int fcy = (ky % 2 == 1) ? ky / 2 : 0;
  int fcz = (kz % 2 == 1) ? kz / 2 : 0;
  std::vector<int> foff_vec;
  foff_vec.reserve(K * 3);
  for (int i = 0; i < kx; i++)
    for (int j = 0; j < ky; j++)
      for (int k = 0; k < kz; k++) {
        foff_vec.push_back(i - fcx);
        foff_vec.push_back(j - fcy);
        foff_vec.push_back(k - fcz);
      }
  auto fine_spatial_off = torch::from_blob(foff_vec.data(), {K, 3}, torch::kInt32).to(dev);

  int coarse_min = -c[0];  // same for all dims (symmetric kernel)

  // --- 3. Coarse probe ---
  auto coarse_masks = torch::zeros({M}, opts_i32);
  {
    uint32_t cmask = static_cast<uint32_t>(coarse_cap - 1);
    int bs = 128;
    int grid = (M + bs - 1) / bs;
    int smem = K_c * 3 * sizeof(int);
    coarse_probe_kernel<<<grid, bs, smem, stream>>>(
        reinterpret_cast<const uint64_t *>(coarse_keys.data_ptr<int64_t>()),
        coarse_vals.data_ptr<int>(),
        query_coords.data_ptr<int>(),
        coarse_spatial_off.data_ptr<int>(),
        coarse_masks.data_ptr<int>(),
        M,
        K_c,
        stride_shift,
        cmask);
  }

  // --- 4. Pruned fine search ---
  auto found = torch::empty({K, M}, opts_i32);
  {
    uint32_t fmask = static_cast<uint32_t>(fine_capacity - 1);
    int tx = 64, ty = 8;
    dim3 block(tx, ty);
    dim3 grid((M + tx - 1) / tx, (K + ty - 1) / ty);
    int smem = K * 3 * sizeof(int);
    fine_search_pruned_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const uint64_t *>(fine_keys.data_ptr<int64_t>()),
        fine_values.data_ptr<int>(),
        query_coords.data_ptr<int>(),
        fine_spatial_off.data_ptr<int>(),
        coarse_masks.data_ptr<int>(),
        found.data_ptr<int>(),
        M,
        K,
        stride_shift,
        coarse_min,
        coarse_dim[1],
        coarse_dim[2],
        fmask);
  }

  // --- 5. Postprocess: count → cumsum → scatter ---
  auto counts = torch::zeros({K}, opts_i32);
  {
    int total_elems = K * M;
    int blocks = (total_elems + 255) / 256;
    postprocess_count_kernel<<<blocks, 256, 0, stream>>>(
        found.data_ptr<int>(), counts.data_ptr<int>(), K, M);
  }

  auto offsets = torch::zeros({K + 1}, opts_i32);
  auto offsets_tail = offsets.slice(0, 1, K + 1);
  at::cumsum_out(offsets_tail, counts, 0);

  // Need total on CPU for allocation
  int total;
  {
    auto total_t = offsets[K].cpu();
    total = total_t.item<int>();
  }

  auto in_maps = torch::empty({total}, opts_i32);
  auto out_maps = torch::empty({total}, opts_i32);

  if (total > 0) {
    auto scatter_counters = torch::zeros({K}, opts_i32);
    int total_elems = K * M;
    int blocks = (total_elems + 255) / 256;
    postprocess_scatter_kernel<<<blocks, 256, 0, stream>>>(found.data_ptr<int>(),
                                                           offsets.data_ptr<int>(),
                                                           scatter_counters.data_ptr<int>(),
                                                           in_maps.data_ptr<int>(),
                                                           out_maps.data_ptr<int>(),
                                                           K,
                                                           M);
  }

  return std::make_tuple(in_maps, out_maps, offsets, found);
}

}  // namespace cuhash
