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
  const int *tile_offsets;    // [num_groups + 1] prefix sum of per-group m_tiles
  const int *group_sizes;     // [num_groups] M_g for each group
  const int *map_offsets;     // [num_groups + 1] offsets into in_map/out_map
  const void *const *ptr_B_array;  // [num_groups] per-group weight pointers
};

}  // namespace cute_gemm
}  // namespace warpconvnet
