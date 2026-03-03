// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Cell-list radius search CUDA kernels.
// Two-pass approach: count neighbors per query, then write indices/distances.
// Uses the same hash table structure as hashmap_kernels.cu to look up cell IDs.

#include <cstdint>

// ============================================================================
// Hash function implementations (must match hashmap_kernels.cu exactly)
// ============================================================================

__device__ inline uint32_t _rs_hash_fnv1a_impl(uint32_t hash_val, uint32_t key) {
  hash_val ^= key;
  hash_val *= 16777619;
  return hash_val;
}

__device__ inline uint32_t _rs_hash_city_impl(uint32_t hash_val, uint32_t key) {
  hash_val += key * 0x9E3779B9;
  hash_val ^= hash_val >> 16;
  hash_val *= 0x85EBCA6B;
  hash_val ^= hash_val >> 13;
  hash_val *= 0xC2B2AE35;
  hash_val ^= hash_val >> 16;
  return hash_val;
}

__device__ inline uint32_t _rs_murmur_32_scramble(uint32_t k) {
  k *= 0xCC9E2D51;
  k = (k << 15) | (k >> 17);
  k *= 0x1B873593;
  return k;
}

__device__ inline uint32_t _rs_hash_murmur_impl(uint32_t h, uint32_t k) {
  h ^= _rs_murmur_32_scramble(k);
  h = (h << 13) | (h >> 19);
  h = h * 5 + 0xE6546B64;
  return h;
}

__device__ inline uint32_t _rs_hash_murmur_finalize(uint32_t h, int length_bytes) {
  h ^= length_bytes;
  h ^= h >> 16;
  h *= 0x85EBCA6B;
  h ^= h >> 13;
  h *= 0xC2B2AE35;
  h ^= h >> 16;
  return h;
}

// Hash function structs
struct RS_FNV1AHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t hash_val = 2166136261u;
    for (int i = 0; i < key_dim; ++i) {
      hash_val = _rs_hash_fnv1a_impl(hash_val, (uint32_t)key[i]);
    }
    int signed_hash = (int)hash_val;
    return ((signed_hash % capacity) + capacity) % capacity;
  }
};

struct RS_CityHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t hash_val = 0;
    for (int i = 0; i < key_dim; ++i) {
      hash_val = _rs_hash_city_impl(hash_val, (uint32_t)key[i]);
    }
    int signed_hash = (int)hash_val;
    return ((signed_hash % capacity) + capacity) % capacity;
  }
};

struct RS_MurmurHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t h = 0x9747B28Cu;
    for (int i = 0; i < key_dim; ++i) {
      h = _rs_hash_murmur_impl(h, (uint32_t)key[i]);
    }
    h = _rs_hash_murmur_finalize(h, key_dim * 4);
    int signed_hash = (int)h;
    return ((signed_hash % capacity) + capacity) % capacity;
  }
};

// Vector comparison
__device__ inline bool rs_vec_equal(const int* a, const int* b, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

// Hash table search (identical logic to hashmap_kernels.cu search_hash_table)
template <typename HashFuncT>
__device__ inline int rs_search_hash_table(
    const int* __restrict__ table_kvs,
    const int* __restrict__ vector_keys,
    const int* __restrict__ query_key,
    int key_dim,
    int table_capacity) {
  int slot = HashFuncT::hash(query_key, key_dim, table_capacity);
  int initial_slot = slot;
  int attempts = 0;

  while (attempts < table_capacity) {
    int slot_marker = table_kvs[slot * 2 + 0];
    if (slot_marker == -1) return -1;

    int vector_index = table_kvs[slot * 2 + 1];
    if (vector_index < 0) { continue; }

    const int* candidate_key = &vector_keys[vector_index * key_dim];
    if (rs_vec_equal(candidate_key, query_key, key_dim)) {
      return vector_index;
    }

    slot = (slot + 1) % table_capacity;
    if (slot == initial_slot) return -1;
    attempts++;
  }
  return -1;
}

// ============================================================================
// Templated radius search kernels
// ============================================================================

template <typename HashFuncT>
__device__ void radius_search_count_impl(
    const float* __restrict__ points,      // [N, 3]
    const float* __restrict__ queries,     // [M, 3]
    const int* __restrict__ sorted_indices,// [N] original point indices sorted by cell
    const int* __restrict__ cell_starts,   // [num_cells] start index in sorted_indices
    const int* __restrict__ cell_counts,   // [num_cells] number of points in cell
    const int* __restrict__ table_kvs,     // hash table key-value store
    const int* __restrict__ vector_keys,   // hash table vector keys
    int* __restrict__ result_count,        // [M] output: count per query
    int N, int M, int num_cells,
    float radius, float cell_size,
    int table_capacity) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M) return;

  float qx = queries[tid * 3 + 0];
  float qy = queries[tid * 3 + 1];
  float qz = queries[tid * 3 + 2];
  float radius_sq = radius * radius;

  // Compute the cell coordinate of this query
  int cx = (int)floorf(qx / cell_size);
  int cy = (int)floorf(qy / cell_size);
  int cz = (int)floorf(qz / cell_size);

  int count = 0;

  // Iterate over 27 neighbor cells
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        int ncx = cx + dx;
        int ncy = cy + dy;
        int ncz = cz + dz;

        // Look up this cell in the hash table
        int cell_key[3] = {ncx, ncy, ncz};
        int cell_id = rs_search_hash_table<HashFuncT>(
            table_kvs, vector_keys, cell_key, 3, table_capacity);

        if (cell_id < 0 || cell_id >= num_cells) continue;

        int start = cell_starts[cell_id];
        int cell_count = cell_counts[cell_id];

        // Check each point in this cell
        for (int j = 0; j < cell_count; j++) {
          int pt_idx = sorted_indices[start + j];
          float px = points[pt_idx * 3 + 0];
          float py = points[pt_idx * 3 + 1];
          float pz = points[pt_idx * 3 + 2];

          float dx2 = qx - px;
          float dy2 = qy - py;
          float dz2 = qz - pz;
          float dist_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

          if (dist_sq <= radius_sq) {
            count++;
          }
        }
      }
    }
  }

  result_count[tid] = count;
}

template <typename HashFuncT>
__device__ void radius_search_write_impl(
    const float* __restrict__ points,
    const float* __restrict__ queries,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_counts,
    const int* __restrict__ table_kvs,
    const int* __restrict__ vector_keys,
    const int* __restrict__ result_offsets,  // [M+1] cumulative sum of counts
    int* __restrict__ result_indices,        // [total] output indices
    float* __restrict__ result_distances,    // [total] output distances
    int N, int M, int num_cells,
    float radius, float cell_size,
    int table_capacity) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M) return;

  float qx = queries[tid * 3 + 0];
  float qy = queries[tid * 3 + 1];
  float qz = queries[tid * 3 + 2];
  float radius_sq = radius * radius;

  int cx = (int)floorf(qx / cell_size);
  int cy = (int)floorf(qy / cell_size);
  int cz = (int)floorf(qz / cell_size);

  int write_pos = result_offsets[tid];

  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        int ncx = cx + dx;
        int ncy = cy + dy;
        int ncz = cz + dz;

        int cell_key[3] = {ncx, ncy, ncz};
        int cell_id = rs_search_hash_table<HashFuncT>(
            table_kvs, vector_keys, cell_key, 3, table_capacity);

        if (cell_id < 0 || cell_id >= num_cells) continue;

        int start = cell_starts[cell_id];
        int cell_count = cell_counts[cell_id];

        for (int j = 0; j < cell_count; j++) {
          int pt_idx = sorted_indices[start + j];
          float px = points[pt_idx * 3 + 0];
          float py = points[pt_idx * 3 + 1];
          float pz = points[pt_idx * 3 + 2];

          float dx2 = qx - px;
          float dy2 = qy - py;
          float dz2 = qz - pz;
          float dist_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

          if (dist_sq <= radius_sq) {
            result_indices[write_pos] = pt_idx;
            result_distances[write_pos] = sqrtf(dist_sq);
            write_pos++;
          }
        }
      }
    }
  }
}

// ============================================================================
// Extern "C" wrappers for each hash method
// ============================================================================

// --- Count kernels ---
extern "C" __global__ void radius_search_count_kernel_fnv1a(
    const float* points, const float* queries,
    const int* sorted_indices, const int* cell_starts, const int* cell_counts,
    const int* table_kvs, const int* vector_keys,
    int* result_count,
    int N, int M, int num_cells,
    float radius, float cell_size, int table_capacity) {
  radius_search_count_impl<RS_FNV1AHash>(
      points, queries, sorted_indices, cell_starts, cell_counts,
      table_kvs, vector_keys, result_count,
      N, M, num_cells, radius, cell_size, table_capacity);
}

extern "C" __global__ void radius_search_count_kernel_city(
    const float* points, const float* queries,
    const int* sorted_indices, const int* cell_starts, const int* cell_counts,
    const int* table_kvs, const int* vector_keys,
    int* result_count,
    int N, int M, int num_cells,
    float radius, float cell_size, int table_capacity) {
  radius_search_count_impl<RS_CityHash>(
      points, queries, sorted_indices, cell_starts, cell_counts,
      table_kvs, vector_keys, result_count,
      N, M, num_cells, radius, cell_size, table_capacity);
}

extern "C" __global__ void radius_search_count_kernel_murmur(
    const float* points, const float* queries,
    const int* sorted_indices, const int* cell_starts, const int* cell_counts,
    const int* table_kvs, const int* vector_keys,
    int* result_count,
    int N, int M, int num_cells,
    float radius, float cell_size, int table_capacity) {
  radius_search_count_impl<RS_MurmurHash>(
      points, queries, sorted_indices, cell_starts, cell_counts,
      table_kvs, vector_keys, result_count,
      N, M, num_cells, radius, cell_size, table_capacity);
}

// --- Write kernels ---
extern "C" __global__ void radius_search_write_kernel_fnv1a(
    const float* points, const float* queries,
    const int* sorted_indices, const int* cell_starts, const int* cell_counts,
    const int* table_kvs, const int* vector_keys,
    const int* result_offsets, int* result_indices, float* result_distances,
    int N, int M, int num_cells,
    float radius, float cell_size, int table_capacity) {
  radius_search_write_impl<RS_FNV1AHash>(
      points, queries, sorted_indices, cell_starts, cell_counts,
      table_kvs, vector_keys, result_offsets, result_indices, result_distances,
      N, M, num_cells, radius, cell_size, table_capacity);
}

extern "C" __global__ void radius_search_write_kernel_city(
    const float* points, const float* queries,
    const int* sorted_indices, const int* cell_starts, const int* cell_counts,
    const int* table_kvs, const int* vector_keys,
    const int* result_offsets, int* result_indices, float* result_distances,
    int N, int M, int num_cells,
    float radius, float cell_size, int table_capacity) {
  radius_search_write_impl<RS_CityHash>(
      points, queries, sorted_indices, cell_starts, cell_counts,
      table_kvs, vector_keys, result_offsets, result_indices, result_distances,
      N, M, num_cells, radius, cell_size, table_capacity);
}

extern "C" __global__ void radius_search_write_kernel_murmur(
    const float* points, const float* queries,
    const int* sorted_indices, const int* cell_starts, const int* cell_counts,
    const int* table_kvs, const int* vector_keys,
    const int* result_offsets, int* result_indices, float* result_distances,
    int N, int M, int num_cells,
    float radius, float cell_size, int table_capacity) {
  radius_search_write_impl<RS_MurmurHash>(
      points, queries, sorted_indices, cell_starts, cell_counts,
      table_kvs, vector_keys, result_offsets, result_indices, result_distances,
      N, M, num_cells, radius, cell_size, table_capacity);
}
