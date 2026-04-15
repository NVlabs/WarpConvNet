// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// cuhash - Optimized CUDA Hash Table Library
// kernel_map.cuh - Kernel map search device functions
#pragma once

#include "cuhash/hash_table.cuh"

namespace cuhash {

// ============================================================================
// Packed Key Kernel Map Search (4D coordinates)
//
// For each (query_coord, kernel_offset) pair, compute query + offset,
// pack into uint64, and search the packed hash table.
//
// Key advantage: No vector_keys indirection during search.
// ============================================================================

// Offset-based kernel map: query_coords (M, 4) + kernel_offsets (K, 4) -> (K, M)
__device__ __forceinline__ int packed_kernel_map_offset(const uint64_t *__restrict__ keys,
                                                        const int *__restrict__ values,
                                                        uint64_t base_packed,
                                                        int ox,
                                                        int oy,
                                                        int oz,
                                                        uint32_t capacity_mask) {
  uint64_t query_key = offset_packed_key_4d(base_packed, ox, oy, oz);
  return packed_search(keys, values, query_key, capacity_mask);
}

// Size-based kernel map: query_coords (M, 4) + kernel_sizes (3,) -> (K, M)
// Converts linear kernel index to 3D offset, then searches.
__device__ __forceinline__ int packed_kernel_map_size(const uint64_t *__restrict__ keys,
                                                      const int *__restrict__ values,
                                                      uint64_t base_packed,
                                                      int kernel_idx,
                                                      int kx,
                                                      int ky,
                                                      int kz,
                                                      int cx,
                                                      int cy,
                                                      int cz,
                                                      uint32_t capacity_mask) {
  int kk = kernel_idx % kz;
  int jj = (kernel_idx / kz) % ky;
  int ii = kernel_idx / (kz * ky);

  int ox = ii - cx;
  int oy = jj - cy;
  int oz = kk - cz;

  return packed_kernel_map_offset(keys, values, base_packed, ox, oy, oz, capacity_mask);
}

}  // namespace cuhash
