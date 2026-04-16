// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Cell-list radius search CUDA kernels using PackedHashTable (cuhash).
// Two-pass approach: count neighbors per query, then write indices/distances.
// Uses packed uint64 keys and Splitmix64 hashing for fast neighbor cell lookup.

#include "cuhash/hash_functions.cuh"
#include "cuhash/hash_table.cuh"

using namespace cuhash;

// ============================================================================
// Radius search kernels
// ============================================================================

__global__ void radius_search_count_packed(const float *__restrict__ points,
                                           const float *__restrict__ queries,
                                           const int *__restrict__ sorted_indices,
                                           const int *__restrict__ cell_starts,
                                           const int *__restrict__ cell_counts,
                                           const uint64_t *__restrict__ keys,
                                           const int *__restrict__ values,
                                           int *__restrict__ result_count,
                                           int N,
                                           int M,
                                           int num_cells,
                                           float radius,
                                           float cell_size,
                                           uint32_t capacity_mask) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M) return;

  float qx = queries[tid * 3 + 0];
  float qy = queries[tid * 3 + 1];
  float qz = queries[tid * 3 + 2];
  float radius_sq = radius * radius;

  int cx = (int)floorf(qx / cell_size);
  int cy = (int)floorf(qy / cell_size);
  int cz = (int)floorf(qz / cell_size);

  int count = 0;

  // Iterate over 27 neighbor cells
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        // Pack neighbor cell as 4D coord with batch=0
        uint64_t packed = pack_key_4d(0, cx + dx, cy + dy, cz + dz);
        int cell_id = packed_search(keys, values, packed, capacity_mask);

        if (cell_id < 0 || cell_id >= num_cells) continue;

        int start = cell_starts[cell_id];
        int cell_count = cell_counts[cell_id];

        for (int j = 0; j < cell_count; j++) {
          int pt_idx = sorted_indices[start + j];
          float dx2 = qx - points[pt_idx * 3 + 0];
          float dy2 = qy - points[pt_idx * 3 + 1];
          float dz2 = qz - points[pt_idx * 3 + 2];
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

__global__ void radius_search_write_packed(const float *__restrict__ points,
                                           const float *__restrict__ queries,
                                           const int *__restrict__ sorted_indices,
                                           const int *__restrict__ cell_starts,
                                           const int *__restrict__ cell_counts,
                                           const uint64_t *__restrict__ keys,
                                           const int *__restrict__ values,
                                           const int *__restrict__ result_offsets,
                                           int *__restrict__ result_indices,
                                           float *__restrict__ result_distances,
                                           int N,
                                           int M,
                                           int num_cells,
                                           float radius,
                                           float cell_size,
                                           uint32_t capacity_mask) {
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
        uint64_t packed = pack_key_4d(0, cx + dx, cy + dy, cz + dz);
        int cell_id = packed_search(keys, values, packed, capacity_mask);

        if (cell_id < 0 || cell_id >= num_cells) continue;

        int start = cell_starts[cell_id];
        int cell_count = cell_counts[cell_id];

        for (int j = 0; j < cell_count; j++) {
          int pt_idx = sorted_indices[start + j];
          float dx2 = qx - points[pt_idx * 3 + 0];
          float dy2 = qy - points[pt_idx * 3 + 1];
          float dz2 = qz - points[pt_idx * 3 + 2];
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
