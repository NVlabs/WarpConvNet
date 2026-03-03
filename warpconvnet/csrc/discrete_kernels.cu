// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <vector_types.h>  // For int4

#include <cstdint>

__device__ inline uint32_t murmur_32_scramble(uint32_t k) {
  k *= 0xCC9E2D51;
  k = (k << 15) | (k >> 17);
  k *= 0x1B873593;
  return k;
}

__device__ inline uint32_t _hash_murmur_impl(uint32_t h, uint32_t k) {
  h ^= murmur_32_scramble(k);
  h = (h << 13) | (h >> 19);
  h = h * 5 + 0xE6546B64;
  return h;
}

__device__ inline uint32_t _hash_murmur_finalize(uint32_t h, int length_bytes) {
  h ^= length_bytes;
  h ^= h >> 16;
  h *= 0x85EBCA6B;
  h ^= h >> 13;
  h *= 0xC2B2AE35;
  h ^= h >> 16;
  return h;
}

__device__ inline uint32_t _hash_fnv1a_impl(uint32_t hash_val, uint32_t key) {
  hash_val ^= key;
  hash_val *= 16777619;  // FNV prime
  return hash_val;
}

__device__ inline uint32_t _hash_city_impl(uint32_t hash_val, uint32_t key) {
  hash_val += key * 0x9E3779B9;
  hash_val ^= hash_val >> 16;
  hash_val *= 0x85EBCA6B;
  hash_val ^= hash_val >> 13;
  hash_val *= 0xC2B2AE35;
  hash_val ^= hash_val >> 16;
  return hash_val;
}

// NOTE: capacity MUST be a power of 2 for correct behavior.

struct FNV1AHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t hash_val = 2166136261u;
    for (int i = 0; i < key_dim; ++i) {
      hash_val = _hash_fnv1a_impl(hash_val, (uint32_t)key[i]);
    }
    return (int)(hash_val & (uint32_t)(capacity - 1));
  }
};

struct CityHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t hash_val = 0;
    for (int i = 0; i < key_dim; ++i) {
      hash_val = _hash_city_impl(hash_val, (uint32_t)key[i]);
    }
    return (int)(hash_val & (uint32_t)(capacity - 1));
  }
};

struct MurmurHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t h = 0x9747B28Cu;
    for (int i = 0; i < key_dim; ++i) {
      h = _hash_murmur_impl(h, (uint32_t)key[i]);
    }
    h = _hash_murmur_finalize(h, key_dim * 4);
    return (int)(h & (uint32_t)(capacity - 1));
  }
};

// --- Vector Comparison ---
__device__ inline bool vec_equal(const int* a, const int* b, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// Vectorized comparison for 4D keys using 128-bit int4 loads
__device__ inline bool vec_equal_4d(const int* a, const int* b) {
  int4 va = *reinterpret_cast<const int4*>(a);
  int4 vb = *reinterpret_cast<const int4*>(b);
  return (va.x == vb.x) && (va.y == vb.y) && (va.z == vb.z) && (va.w == vb.w);
}

// --- Device Function for Hash Table Search ---
template <typename HashFuncT>
__device__ inline int search_hash_table(const int* __restrict__ table_kvs,
                                        const int* __restrict__ vector_keys,
                                        const int* __restrict__ query_key,
                                        int key_dim,
                                        int table_capacity) {
  int slot = HashFuncT::hash(query_key, key_dim, table_capacity);
  const int capacity_mask = table_capacity - 1;
  int initial_slot = slot;
  int attempts = 0;

  while (attempts < table_capacity) {
    // Single 64-bit load for both slot marker and vector index
    long long pair = *reinterpret_cast<const long long*>(&table_kvs[slot * 2]);
    int slot_marker = (int)(pair & 0xFFFFFFFF);
    int vector_index = (int)(pair >> 32);

    if (slot_marker == -1) {
      return -1;
    }

    if (vector_index >= 0) {
      bool keys_match;
      if (key_dim == 4) {
        keys_match = vec_equal_4d(&vector_keys[vector_index * 4], query_key);
      } else {
        keys_match = vec_equal(&vector_keys[vector_index * key_dim], query_key, key_dim);
      }
      if (keys_match) {
        return vector_index;
      }
    }

    slot = (slot + 1) & capacity_mask;
    if (slot == initial_slot) {
      return -1;
    }
    attempts++;
  }
  return -1;
}

// --- Kernel Implementations ---

// Equivalent of conv_kernel_map_arr / conv_kernel_map_vec4i (combined for array input)
template <typename HashFuncT>
__device__ void kernel_map_offset_templated(
    const int* __restrict__ table_kvs,       // Hash table key-value store (capacity, 2)
    const int* __restrict__ vector_keys,     // Original stored keys (num_in_keys, key_dim)
    const int* __restrict__ query_coords,    // Coordinates to query (num_query_coords, key_dim)
    const int* __restrict__ kernel_offsets,  // Offsets to apply (num_kernel_offsets, key_dim)
    int* __restrict__ found_in_coord_index,  // Output array (num_kernel_offsets, num_query_coords)
    int num_query_coords,
    int key_dim,
    int num_kernel_offsets,
    int table_capacity) {
  // Thread ID corresponds to the query coordinate index (x dimension)
  int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Thread ID corresponds to the kernel offset index (y dimension)
  int kernel_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (query_idx >= num_query_coords) {
    return;
  }

  if (kernel_idx >= num_kernel_offsets) {
    return;
  }

  // Temporary storage for the calculated query coordinate + offset
  // Using stack allocation assuming key_dim is small (e.g., 4)
  // For larger key_dim, consider shared memory or scratchpad global memory if needed.
  int temp_coord[16];  // Max key_dim assumed <= 16; adjust if necessary

  const int* base_query_coord = &query_coords[query_idx * key_dim];
  const int* offset = &kernel_offsets[kernel_idx * key_dim];

  // Calculate query_coord + offset
  for (int dim = 0; dim < key_dim; ++dim) {
    temp_coord[dim] = base_query_coord[dim] + offset[dim];
  }

  // Search for the calculated coordinate in the hash table
  int found_index =
      search_hash_table<HashFuncT>(table_kvs, vector_keys, temp_coord, key_dim, table_capacity);

  // Store the result in the output array [kernel_idx, query_idx]
  // Note: CuPy/PyTorch tensors are row-major by default.
  // Accessing found_in_coord_index[kernel_idx][query_idx] corresponds to index kernel_idx *
  // num_query_coords + query_idx
  found_in_coord_index[kernel_idx * num_query_coords + query_idx] = found_index;
}

// Kernel to map found indices to flattened in/out maps and offsets
__device__ void map_found_indices_to_maps_kernel(
    const int* __restrict__ found_in_coord_index,  // Input: Found indices (num_kernel_offsets,
                                                   // num_query_coords)
    const int* __restrict__ mapped_indices,        // Input: Cumulative sum (-1) per row
                                                   // (num_kernel_offsets, num_query_coords)
    const int* __restrict__ offsets,  // Input: Offsets per kernel (num_kernel_offsets + 1)
    int* __restrict__ out_in_maps,    // Output: Flattened input indices (num_total_maps)
    int* __restrict__ out_out_maps,   // Output: Flattened output indices (num_total_maps)
    int num_kernel_offsets,
    int num_query_coords) {
  // Global thread ID covering the entire found_in_coord_index matrix
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_kernel_offsets * num_query_coords;

  if (idx >= total_elements) {
    return;
  }

  // Decompose global index into kernel index (k) and query index (m)
  int k = idx / num_query_coords;
  int m = idx % num_query_coords;

  int found_index = found_in_coord_index[idx];  // Direct access using global index

  if (found_index >= 0) {
    // Calculate the output position in the flattened maps
    int output_map_idx = mapped_indices[idx] + offsets[k];

    out_in_maps[output_map_idx] = found_index;  // Input map index (from hash table search)
    out_out_maps[output_map_idx] = m;           // Output map index (query coordinate index)
  }
}

// --- Extern "C" Wrappers ---

// kernel_map_offset wrappers
extern "C" __global__ void kernel_map_offset_fnv1a(const int* table_kvs,
                                                   const int* vector_keys,
                                                   const int* query_coords,
                                                   const int* kernel_offsets,
                                                   int* found_in_coord_index,
                                                   int num_query_coords,
                                                   int key_dim,
                                                   int num_kernel_offsets,
                                                   int table_capacity) {
  kernel_map_offset_templated<FNV1AHash>(table_kvs,
                                         vector_keys,
                                         query_coords,
                                         kernel_offsets,
                                         found_in_coord_index,
                                         num_query_coords,
                                         key_dim,
                                         num_kernel_offsets,
                                         table_capacity);
}

extern "C" __global__ void kernel_map_offset_city(const int* table_kvs,
                                                  const int* vector_keys,
                                                  const int* query_coords,
                                                  const int* kernel_offsets,
                                                  int* found_in_coord_index,
                                                  int num_query_coords,
                                                  int key_dim,
                                                  int num_kernel_offsets,
                                                  int table_capacity) {
  kernel_map_offset_templated<CityHash>(table_kvs,
                                        vector_keys,
                                        query_coords,
                                        kernel_offsets,
                                        found_in_coord_index,
                                        num_query_coords,
                                        key_dim,
                                        num_kernel_offsets,
                                        table_capacity);
}

extern "C" __global__ void kernel_map_offset_murmur(const int* table_kvs,
                                                    const int* vector_keys,
                                                    const int* query_coords,
                                                    const int* kernel_offsets,
                                                    int* found_in_coord_index,
                                                    int num_query_coords,
                                                    int key_dim,
                                                    int num_kernel_offsets,
                                                    int table_capacity) {
  kernel_map_offset_templated<MurmurHash>(table_kvs,
                                          vector_keys,
                                          query_coords,
                                          kernel_offsets,
                                          found_in_coord_index,
                                          num_query_coords,
                                          key_dim,
                                          num_kernel_offsets,
                                          table_capacity);
}

// map_found_indices_to_maps wrapper (no template needed)
extern "C" __global__ void map_found_indices_to_maps_cuda(const int* found_in_coord_index,
                                                          const int* mapped_indices,
                                                          const int* offsets,
                                                          int* out_in_maps,
                                                          int* out_out_maps,
                                                          int num_kernel_offsets,
                                                          int num_query_coords) {
  map_found_indices_to_maps_kernel(found_in_coord_index,
                                   mapped_indices,
                                   offsets,
                                   out_in_maps,
                                   out_out_maps,
                                   num_kernel_offsets,
                                   num_query_coords);
}

// --- Specialized Kernel for 4D Coordinates and Kernel Size ---

// Optimized kernel for 4D coordinates (batch, x, y, z) when kernel is defined by size
template <typename HashFuncT>
__device__ void kernel_map_size_4d_templated(
    const int* __restrict__ table_kvs,       // Hash table key-value store (capacity, 2)
    const int* __restrict__ vector_keys,     // Original stored keys (num_in_keys, 4)
    const int* __restrict__ query_coords,    // Coordinates to query (num_query_coords, 4)
    const int* __restrict__ kernel_sizes,    // Kernel dimensions (kx, ky, kz)
    int* __restrict__ found_in_coord_index,  // Output array (kx*ky*kz, num_query_coords)
    int num_query_coords,
    int table_capacity,
    int num_kernels) {
  // Cache kernel sizes and derived values in shared memory (read once, used by all threads)
  __shared__ int s_ksz[3];     // kernel_sizes
  __shared__ int s_center[3];  // center offsets
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
  int kernel_map_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (query_idx >= num_query_coords || kernel_map_idx >= num_kernels) {
    return;
  }

  const int key_dim = 4;

  // Load query coordinate as int4 (single 128-bit load)
  int4 base = *reinterpret_cast<const int4*>(&query_coords[query_idx * key_dim]);

  // Convert linear kernel_map_idx back to 3D indices (i, j, k) using shared memory
  int kk = kernel_map_idx % s_ksz[2];
  int jj = (kernel_map_idx / s_ksz[2]) % s_ksz[1];
  int ii = kernel_map_idx / (s_ksz[2] * s_ksz[1]);

  int temp_coord[4];
  temp_coord[0] = base.x;  // batch index unchanged
  temp_coord[1] = base.y + ii - s_center[0];
  temp_coord[2] = base.z + jj - s_center[1];
  temp_coord[3] = base.w + kk - s_center[2];

  int found_index =
      search_hash_table<HashFuncT>(table_kvs, vector_keys, temp_coord, key_dim, table_capacity);

  found_in_coord_index[kernel_map_idx * num_query_coords + query_idx] = found_index;
}

// --- Extern "C" Wrappers ---

// kernel_map_size_4d wrappers with skip_symmetric_kernel_map
extern "C" __global__ void kernel_map_size_4d_fnv1a(const int* table_kvs,
                                                    const int* vector_keys,
                                                    const int* query_coords,
                                                    const int* kernel_sizes,
                                                    int* found_in_coord_index,
                                                    int num_query_coords,
                                                    int table_capacity,
                                                    int num_kernels) {
  kernel_map_size_4d_templated<FNV1AHash>(table_kvs,
                                          vector_keys,
                                          query_coords,
                                          kernel_sizes,
                                          found_in_coord_index,
                                          num_query_coords,
                                          table_capacity,
                                          num_kernels);
}

extern "C" __global__ void kernel_map_size_4d_city(const int* table_kvs,
                                                   const int* vector_keys,
                                                   const int* query_coords,
                                                   const int* kernel_sizes,
                                                   int* found_in_coord_index,
                                                   int num_query_coords,
                                                   int table_capacity,
                                                   int num_kernels) {
  kernel_map_size_4d_templated<CityHash>(table_kvs,
                                         vector_keys,
                                         query_coords,
                                         kernel_sizes,
                                         found_in_coord_index,
                                         num_query_coords,
                                         table_capacity,
                                         num_kernels);
}

extern "C" __global__ void kernel_map_size_4d_murmur(const int* table_kvs,
                                                     const int* vector_keys,
                                                     const int* query_coords,
                                                     const int* kernel_sizes,
                                                     int* found_in_coord_index,
                                                     int num_query_coords,
                                                     int table_capacity,
                                                     int num_kernels) {
  kernel_map_size_4d_templated<MurmurHash>(table_kvs,
                                           vector_keys,
                                           query_coords,
                                           kernel_sizes,
                                           found_in_coord_index,
                                           num_query_coords,
                                           table_capacity,
                                           num_kernels);
}

// =============================================================================
// Fused kernel_map: count pass + scatter pass (eliminates K×M intermediate)
// =============================================================================

// Pass 1: Search and count matches per kernel offset.
// Uses shared memory privatized counters to reduce global atomicAdd contention.
// Each block accumulates locally, then flushes to global once per offset.
template <typename HashFuncT, int BLOCK_DIM_Y = 8>
__device__ void kernel_map_size_4d_count_templated(
    const int* __restrict__ table_kvs,
    const int* __restrict__ vector_keys,
    const int* __restrict__ query_coords,
    const int* __restrict__ kernel_sizes,
    int* __restrict__ counts,  // (num_kernels,) atomic counters
    int num_query_coords,
    int table_capacity,
    int num_kernels) {
  __shared__ int s_ksz[3];
  __shared__ int s_center[3];
  __shared__ int s_block_counts[BLOCK_DIM_Y];  // per-offset local counters

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (tid < BLOCK_DIM_Y) {
    s_block_counts[tid] = 0;
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
  int kernel_map_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (query_idx < num_query_coords && kernel_map_idx < num_kernels) {
    const int key_dim = 4;
    int4 base = *reinterpret_cast<const int4*>(&query_coords[query_idx * key_dim]);

    int kk = kernel_map_idx % s_ksz[2];
    int jj = (kernel_map_idx / s_ksz[2]) % s_ksz[1];
    int ii = kernel_map_idx / (s_ksz[2] * s_ksz[1]);

    int temp_coord[4];
    temp_coord[0] = base.x;
    temp_coord[1] = base.y + ii - s_center[0];
    temp_coord[2] = base.z + jj - s_center[1];
    temp_coord[3] = base.w + kk - s_center[2];

    int found_index =
        search_hash_table<HashFuncT>(table_kvs, vector_keys, temp_coord, key_dim, table_capacity);

    if (found_index >= 0) {
      atomicAdd(&s_block_counts[threadIdx.y], 1);
    }
  }

  __syncthreads();

  // Flush block-local counts to global (one atomicAdd per offset per block)
  if (threadIdx.x == 0 && threadIdx.y < BLOCK_DIM_Y) {
    int km_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (km_idx < num_kernels && s_block_counts[threadIdx.y] > 0) {
      atomicAdd(&counts[km_idx], s_block_counts[threadIdx.y]);
    }
  }
}

// Pass 2: Search and scatter results directly to in_maps/out_maps.
// 3-phase approach to minimize global atomicAdd contention:
//   Phase 1: Search + count matches per offset in shared memory
//   Phase 2: One global atomicAdd per offset per block to reserve range
//   Phase 3: Write results using block-local positions
template <typename HashFuncT, int BLOCK_DIM_Y = 8>
__device__ void kernel_map_size_4d_scatter_templated(
    const int* __restrict__ table_kvs,
    const int* __restrict__ vector_keys,
    const int* __restrict__ query_coords,
    const int* __restrict__ kernel_sizes,
    const int* __restrict__ offsets,     // (num_kernels + 1,) prefix sum
    int* __restrict__ scatter_counters,  // (num_kernels,) atomic position counters
    int* __restrict__ in_maps,
    int* __restrict__ out_maps,
    int num_query_coords,
    int table_capacity,
    int num_kernels) {
  __shared__ int s_ksz[3];
  __shared__ int s_center[3];
  __shared__ int s_block_count[BLOCK_DIM_Y];  // matches per offset in this block
  __shared__ int s_block_base[BLOCK_DIM_Y];   // global base position per offset
  __shared__ int s_local_pos[BLOCK_DIM_Y];    // running counter for local write positions

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
  int kernel_map_idx = blockIdx.y * blockDim.y + threadIdx.y;

  // Phase 1: Search and count matches per offset in shared memory
  int found_index = -1;
  if (query_idx < num_query_coords && kernel_map_idx < num_kernels) {
    const int key_dim = 4;
    int4 base = *reinterpret_cast<const int4*>(&query_coords[query_idx * key_dim]);

    int kk = kernel_map_idx % s_ksz[2];
    int jj = (kernel_map_idx / s_ksz[2]) % s_ksz[1];
    int ii = kernel_map_idx / (s_ksz[2] * s_ksz[1]);

    int temp_coord[4];
    temp_coord[0] = base.x;
    temp_coord[1] = base.y + ii - s_center[0];
    temp_coord[2] = base.z + jj - s_center[1];
    temp_coord[3] = base.w + kk - s_center[2];

    found_index =
        search_hash_table<HashFuncT>(table_kvs, vector_keys, temp_coord, key_dim, table_capacity);

    if (found_index >= 0) {
      atomicAdd(&s_block_count[threadIdx.y], 1);
    }
  }

  __syncthreads();

  // Phase 2: Reserve global range (one atomicAdd per offset per block)
  if (threadIdx.x == 0 && threadIdx.y < BLOCK_DIM_Y) {
    int km_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (km_idx < num_kernels && s_block_count[threadIdx.y] > 0) {
      s_block_base[threadIdx.y] =
          offsets[km_idx] + atomicAdd(&scatter_counters[km_idx], s_block_count[threadIdx.y]);
    }
  }

  __syncthreads();

  // Phase 3: Each matching thread gets its unique position and writes
  if (found_index >= 0) {
    int local_offset = atomicAdd(&s_local_pos[threadIdx.y], 1);
    int pos = s_block_base[threadIdx.y] + local_offset;
    in_maps[pos] = found_index;
    out_maps[pos] = query_idx;
  }
}

// --- Extern "C" Wrappers for count pass ---
#define DEFINE_COUNT_WRAPPER(suffix, HashFunc)                                          \
  extern "C" __global__ void kernel_map_size_4d_count_##suffix(const int* table_kvs,    \
                                                               const int* vector_keys,  \
                                                               const int* query_coords, \
                                                               const int* kernel_sizes, \
                                                               int* counts,             \
                                                               int num_query_coords,    \
                                                               int table_capacity,      \
                                                               int num_kernels) {       \
    kernel_map_size_4d_count_templated<HashFunc>(table_kvs,                             \
                                                 vector_keys,                           \
                                                 query_coords,                          \
                                                 kernel_sizes,                          \
                                                 counts,                                \
                                                 num_query_coords,                      \
                                                 table_capacity,                        \
                                                 num_kernels);                          \
  }

DEFINE_COUNT_WRAPPER(fnv1a, FNV1AHash)
DEFINE_COUNT_WRAPPER(city, CityHash)
DEFINE_COUNT_WRAPPER(murmur, MurmurHash)

// --- Extern "C" Wrappers for scatter pass ---
#define DEFINE_SCATTER_WRAPPER(suffix, HashFunc)                                          \
  extern "C" __global__ void kernel_map_size_4d_scatter_##suffix(const int* table_kvs,    \
                                                                 const int* vector_keys,  \
                                                                 const int* query_coords, \
                                                                 const int* kernel_sizes, \
                                                                 const int* offsets,      \
                                                                 int* scatter_counters,   \
                                                                 int* in_maps,            \
                                                                 int* out_maps,           \
                                                                 int num_query_coords,    \
                                                                 int table_capacity,      \
                                                                 int num_kernels) {       \
    kernel_map_size_4d_scatter_templated<HashFunc>(table_kvs,                             \
                                                   vector_keys,                           \
                                                   query_coords,                          \
                                                   kernel_sizes,                          \
                                                   offsets,                               \
                                                   scatter_counters,                      \
                                                   in_maps,                               \
                                                   out_maps,                              \
                                                   num_query_coords,                      \
                                                   table_capacity,                        \
                                                   num_kernels);                          \
  }

DEFINE_SCATTER_WRAPPER(fnv1a, FNV1AHash)
DEFINE_SCATTER_WRAPPER(city, CityHash)
DEFINE_SCATTER_WRAPPER(murmur, MurmurHash)

// =============================================================================
// Postprocess kernels: operate on K×M intermediate, NO hash table access.
// Used by search-once pipeline to eliminate the second hash table search pass.
// =============================================================================

// Count matches per kernel offset k from the found_in_coord_index intermediate.
// Uses shared memory privatized counters to reduce global atomicAdd contention.
// Each block processes a contiguous chunk of the flattened (K, M) array.
extern "C" __global__ void postprocess_count_kernel(
    const int* __restrict__ found_in_coord_index,  // (K, M) flattened
    int* __restrict__ counts,                      // (K,) output counters
    int K,
    int M) {
  const int BLOCK_SIZE = 256;
  __shared__ int s_counts[BLOCK_SIZE];

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * M;

  // Compute k-range for this block
  int block_start = blockIdx.x * blockDim.x;
  int k_first = block_start / M;
  int block_end_idx = block_start + BLOCK_SIZE - 1;
  if (block_end_idx >= total) block_end_idx = total - 1;
  int k_last = (total > 0) ? block_end_idx / M : 0;
  int num_k = k_last - k_first + 1;

  // Zero shared memory for this block's k-range
  if (threadIdx.x < num_k) {
    s_counts[threadIdx.x] = 0;
  }
  __syncthreads();

  if (flat_idx < total) {
    int val = found_in_coord_index[flat_idx];
    if (val >= 0) {
      int k = flat_idx / M;
      atomicAdd(&s_counts[k - k_first], 1);
    }
  }

  __syncthreads();

  // Flush block-local counts to global (one atomicAdd per k per block)
  if (threadIdx.x < num_k) {
    int k = k_first + threadIdx.x;
    if (k < K && s_counts[threadIdx.x] > 0) {
      atomicAdd(&counts[k], s_counts[threadIdx.x]);
    }
  }
}

// Scatter matches from found_in_coord_index to in_maps/out_maps.
// 3-phase approach:
//   Phase 1: Count matches per k in shared memory
//   Phase 2: One global atomicAdd per k per block to reserve output range
//   Phase 3: Each matching thread writes its result
extern "C" __global__ void postprocess_scatter_kernel(
    const int* __restrict__ found_in_coord_index,  // (K, M) flattened
    const int* __restrict__ offsets,               // (K+1,) prefix sum
    int* __restrict__ scatter_counters,            // (K,) atomic position counters
    int* __restrict__ in_maps,
    int* __restrict__ out_maps,
    int K,
    int M) {
  const int BLOCK_SIZE = 256;
  __shared__ int s_counts[BLOCK_SIZE];
  __shared__ int s_base[BLOCK_SIZE];
  __shared__ int s_local_pos[BLOCK_SIZE];

  int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * M;

  // Compute k-range for this block
  int block_start = blockIdx.x * blockDim.x;
  int k_first = block_start / M;
  int block_end_idx = block_start + BLOCK_SIZE - 1;
  if (block_end_idx >= total) block_end_idx = total - 1;
  int k_last = (total > 0) ? block_end_idx / M : 0;
  int num_k = k_last - k_first + 1;

  // Zero shared memory
  if (threadIdx.x < num_k) {
    s_counts[threadIdx.x] = 0;
    s_local_pos[threadIdx.x] = 0;
  }
  __syncthreads();

  // Phase 1: Count matches per k in shared memory
  int val = -1;
  int k = -1;
  if (flat_idx < total) {
    val = found_in_coord_index[flat_idx];
    k = flat_idx / M;
    if (val >= 0) {
      atomicAdd(&s_counts[k - k_first], 1);
    }
  }

  __syncthreads();

  // Phase 2: Reserve global range (one atomicAdd per k per block)
  if (threadIdx.x < num_k) {
    int ki = k_first + threadIdx.x;
    if (ki < K && s_counts[threadIdx.x] > 0) {
      s_base[threadIdx.x] = offsets[ki] + atomicAdd(&scatter_counters[ki], s_counts[threadIdx.x]);
    }
  }

  __syncthreads();

  // Phase 3: Each matching thread writes its result
  if (val >= 0) {
    int local_offset = atomicAdd(&s_local_pos[k - k_first], 1);
    int pos = s_base[k - k_first] + local_offset;
    in_maps[pos] = val;            // found index (input coordinate index)
    out_maps[pos] = flat_idx % M;  // query coordinate index
  }
}
