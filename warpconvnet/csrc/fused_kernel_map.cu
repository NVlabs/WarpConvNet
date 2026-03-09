// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused kernel map generator: processes ALL kernel offsets in a single launch,
// eliminating the K*M intermediate matrix used by the original pipeline.
//
// This file provides a single-pass "direct-write" kernel variant that uses
// global atomic counters to write output pairs. It is suitable when the
// caller can provide a conservative upper bound on pairs per offset
// (e.g., max_pairs_per_offset = num_output_coords).
//
// The two-pass (count + scatter) variant reuses the existing kernels in
// discrete_kernels.cu and is orchestrated by launch_fused_kernel_map()
// in coords_launch.cu.

#include <vector_types.h>  // For int4

#include <cstdint>

// =============================================================================
// Hash function structs (must match discrete_kernels.cu / hashmap_kernels.cu)
// =============================================================================

__device__ inline uint32_t _hash_fnv1a_impl_fkm(uint32_t hash_val, uint32_t key) {
  hash_val ^= key;
  hash_val *= 16777619;
  return hash_val;
}

__device__ inline uint32_t _hash_city_impl_fkm(uint32_t hash_val, uint32_t key) {
  hash_val += key * 0x9E3779B9;
  hash_val ^= hash_val >> 16;
  hash_val *= 0x85EBCA6B;
  hash_val ^= hash_val >> 13;
  hash_val *= 0xC2B2AE35;
  hash_val ^= hash_val >> 16;
  return hash_val;
}

__device__ inline uint32_t murmur_32_scramble_fkm(uint32_t k) {
  k *= 0xCC9E2D51;
  k = (k << 15) | (k >> 17);
  k *= 0x1B873593;
  return k;
}

__device__ inline uint32_t _hash_murmur_impl_fkm(uint32_t h, uint32_t k) {
  h ^= murmur_32_scramble_fkm(k);
  h = (h << 13) | (h >> 19);
  h = h * 5 + 0xE6546B64;
  return h;
}

__device__ inline uint32_t _hash_murmur_finalize_fkm(uint32_t h, int length_bytes) {
  h ^= length_bytes;
  h ^= h >> 16;
  h *= 0x85EBCA6B;
  h ^= h >> 13;
  h *= 0xC2B2AE35;
  h ^= h >> 16;
  return h;
}

struct FNV1AHash_fkm {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t hash_val = 2166136261u;
    for (int i = 0; i < key_dim; ++i) {
      hash_val = _hash_fnv1a_impl_fkm(hash_val, (uint32_t)key[i]);
    }
    return (int)(hash_val & (uint32_t)(capacity - 1));
  }
};

struct CityHash_fkm {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t hash_val = 0;
    for (int i = 0; i < key_dim; ++i) {
      hash_val = _hash_city_impl_fkm(hash_val, (uint32_t)key[i]);
    }
    return (int)(hash_val & (uint32_t)(capacity - 1));
  }
};

struct MurmurHash_fkm {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t h = 0x9747B28Cu;
    for (int i = 0; i < key_dim; ++i) {
      h = _hash_murmur_impl_fkm(h, (uint32_t)key[i]);
    }
    h = _hash_murmur_finalize_fkm(h, key_dim * 4);
    return (int)(h & (uint32_t)(capacity - 1));
  }
};

// --- Vector Comparison ---
__device__ inline bool vec_equal_4d_fkm(const int* a, const int* b) {
  int4 va = *reinterpret_cast<const int4*>(a);
  int4 vb = *reinterpret_cast<const int4*>(b);
  return (va.x == vb.x) && (va.y == vb.y) && (va.z == vb.z) && (va.w == vb.w);
}

// --- Hash Table Search ---
template <typename HashFuncT>
__device__ inline int search_hash_table_fkm(const int* __restrict__ table_kvs,
                                            const int* __restrict__ vector_keys,
                                            const int* __restrict__ query_key,
                                            int key_dim,
                                            int table_capacity) {
  int slot = HashFuncT::hash(query_key, key_dim, table_capacity);
  const int capacity_mask = table_capacity - 1;
  int initial_slot = slot;
  int attempts = 0;

  while (attempts < table_capacity) {
    long long pair = *reinterpret_cast<const long long*>(&table_kvs[slot * 2]);
    int slot_marker = (int)(pair & 0xFFFFFFFF);
    int vector_index = (int)(pair >> 32);

    if (slot_marker == -1) {
      return -1;
    }

    if (vector_index >= 0) {
      bool keys_match;
      if (key_dim == 4) {
        keys_match = vec_equal_4d_fkm(&vector_keys[vector_index * 4], query_key);
      } else {
        keys_match = true;
        for (int i = 0; i < key_dim; ++i) {
          if (vector_keys[vector_index * key_dim + i] != query_key[i]) {
            keys_match = false;
            break;
          }
        }
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

// =============================================================================
// Fused kernel: single-pass direct-write variant
// Uses global atomic counters (one per offset) to allocate output positions.
// Requires pre-allocated output arrays sized at max_pairs_per_offset * K.
// =============================================================================

template <typename HashFuncT>
__device__ void fused_kernel_map_direct_impl(
    const int* __restrict__ output_coords,  // [M, 4] (batch, x, y, z)
    const int* __restrict__ table_kvs,      // Hash table slots
    const int* __restrict__ vector_keys,    // Hash table keys
    int table_capacity,
    const int* __restrict__ kernel_sizes,  // [3] (kx, ky, kz)
    int num_output_coords,                 // M
    int kernel_volume,                     // K = product(kernel_size)
    // Outputs:
    int* __restrict__ in_maps,      // [K * max_pairs_per_offset]
    int* __restrict__ out_maps,     // [K * max_pairs_per_offset]
    int* __restrict__ pair_counts,  // [K] atomic counters per offset
    int max_pairs_per_offset        // max allocation per offset
) {
  // Cache kernel sizes and center offsets in shared memory
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

  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int kernel_map_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (out_idx >= num_output_coords || kernel_map_idx >= kernel_volume) return;

  const int key_dim = 4;

  // Load output coordinate once (reused across all K offsets via the 2D grid)
  int4 base = *reinterpret_cast<const int4*>(&output_coords[out_idx * key_dim]);

  // Convert linear index to 3D offset
  int kz = kernel_map_idx % s_ksz[2];
  int ky = (kernel_map_idx / s_ksz[2]) % s_ksz[1];
  int kx = kernel_map_idx / (s_ksz[2] * s_ksz[1]);

  // Candidate coordinate = output + offset - center
  int candidate[4];
  candidate[0] = base.x;  // batch unchanged
  candidate[1] = base.y + kx - s_center[0];
  candidate[2] = base.z + ky - s_center[1];
  candidate[3] = base.w + kz - s_center[2];

  // Search hash table for this candidate
  int in_idx =
      search_hash_table_fkm<HashFuncT>(table_kvs, vector_keys, candidate, key_dim, table_capacity);

  if (in_idx >= 0) {
    // Found! Atomically allocate a position in this offset's output
    int pos = atomicAdd(&pair_counts[kernel_map_idx], 1);
    if (pos < max_pairs_per_offset) {
      in_maps[kernel_map_idx * max_pairs_per_offset + pos] = in_idx;
      out_maps[kernel_map_idx * max_pairs_per_offset + pos] = out_idx;
    }
  }
}

// --- Extern "C" Wrappers for direct-write variant ---

#define DEFINE_DIRECT_WRAPPER(suffix, HashFunc)                                           \
  extern "C" __global__ void fused_kernel_map_direct_##suffix(const int* output_coords,   \
                                                              const int* table_kvs,       \
                                                              const int* vector_keys,     \
                                                              int table_capacity,         \
                                                              const int* kernel_sizes,    \
                                                              int num_output_coords,      \
                                                              int kernel_volume,          \
                                                              int* in_maps,               \
                                                              int* out_maps,              \
                                                              int* pair_counts,           \
                                                              int max_pairs_per_offset) { \
    fused_kernel_map_direct_impl<HashFunc>(output_coords,                                 \
                                           table_kvs,                                     \
                                           vector_keys,                                   \
                                           table_capacity,                                \
                                           kernel_sizes,                                  \
                                           num_output_coords,                             \
                                           kernel_volume,                                 \
                                           in_maps,                                       \
                                           out_maps,                                      \
                                           pair_counts,                                   \
                                           max_pairs_per_offset);                         \
  }

DEFINE_DIRECT_WRAPPER(fnv1a, FNV1AHash_fkm)
DEFINE_DIRECT_WRAPPER(city, CityHash_fkm)
DEFINE_DIRECT_WRAPPER(murmur, MurmurHash_fkm)
