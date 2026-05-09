// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host-visible parameter struct for grouped GEMM.
// Separated from the kernel header so .cpp files can include it without CUDA device code.

#pragma once

namespace warpconvnet {
namespace cute_gemm {

/// Parameters passed to the grouped GEMM kernel.
struct GroupedGemmParams {
  int num_groups;
  const int *tile_offsets;         // [num_groups + 1] prefix sum of per-group m_tiles
  const int *group_sizes;          // [num_groups] M_g for each group
  const int *map_offsets;          // [num_groups + 1] offsets into in_map/out_map
  const void *const *ptr_B_array;  // [num_groups] per-group weight pointers
};

/// Parameters for the grouped TrAB GEMM kernel (weight gradient).
/// Each group shares input A and B but has its own gather indices and output.
///
/// `splits` controls split-K reduction parallelism. When splits > 1, the
/// gathered reduction dim per group is sharded across `splits` thread
/// blocks, each writing partial results via atomicAdd to ptr_D_array[g].
/// Caller must pre-zero ptr_D buffers when splits > 1. splits == 1 uses
/// the original direct-store epilogue (no atomic).
struct GroupedTrABGemmParams {
  int num_groups;
  const int *gather_sizes;   // [num_groups] gather_size per group (reduction dim)
  const int *map_offsets;    // [num_groups] offsets into idx_a/idx_b
  void *const *ptr_D_array;  // [num_groups] per-group output pointers
  int splits;                // split-K shards (default 1, no split)
};

}  // namespace cute_gemm
}  // namespace warpconvnet
