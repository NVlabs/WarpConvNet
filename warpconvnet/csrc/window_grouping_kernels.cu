// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for O(N) window grouping via counting sort.
// Two kernels:
//   1. window_group_histogram_kernel: compute per-voxel window codes + dense histogram
//   2. window_group_scatter_kernel: scatter voxels into window-sorted positions

#include <cstdint>

// Device binary search: find batch index for voxel idx.
// batch_offsets is (B+1,) with batch_offsets[0]=0, batch_offsets[B]=N.
// Returns b such that batch_offsets[b] <= idx < batch_offsets[b+1].
__device__ inline int find_batch_idx(const int* __restrict__ batch_offsets, int B, int idx) {
  int lo = 0, hi = B;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (idx < batch_offsets[mid + 1]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Kernel 1: Compute window codes and build dense histogram.
//
// Each thread processes one voxel. Computes the window code as:
//   voxel_coord = (grid_coord + coord_offset - min_coord) / window_size
//   window_code = batch_idx * W + ravel(voxel_coord, grid_shape)
// where W = prod(grid_shape) is the max number of windows per batch.
//
// Atomically increments histogram[window_code].
extern "C" __global__ void window_group_histogram_kernel(
    const int* __restrict__ grid_coord,    // (N, 3) row-major int32
    const int* __restrict__ batch_offsets,  // (B+1,) int32
    const int* __restrict__ coord_offset,   // (3,) int32 (rounded from float offset * window_size)
    const int* __restrict__ min_coord,      // (3,) int32 (global min coordinate)
    const int* __restrict__ window_size,    // (3,) int32
    const int* __restrict__ grid_shape,     // (3,) int32 = ceil((max-min+1) / window_size)
    int64_t* __restrict__ codes,            // (N,) output window codes
    int* __restrict__ histogram,            // (B * W,) output dense histogram, init to 0
    int N,
    int B,
    int W  // max windows per batch = prod(grid_shape)
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  // Find batch index via binary search
  int batch_idx = find_batch_idx(batch_offsets, B, idx);

  // Load coordinates
  int cx = grid_coord[idx * 3 + 0];
  int cy = grid_coord[idx * 3 + 1];
  int cz = grid_coord[idx * 3 + 2];

  // Compute voxel coordinate: integer division (floor for positive values)
  int wx = (cx + coord_offset[0] - min_coord[0]) / window_size[0];
  int wy = (cy + coord_offset[1] - min_coord[1]) / window_size[1];
  int wz = (cz + coord_offset[2] - min_coord[2]) / window_size[2];

  // Ravel to linear index within the grid
  int gs1 = grid_shape[1];
  int gs2 = grid_shape[2];
  int local_code = wx * (gs1 * gs2) + wy * gs2 + wz;

  // Global code includes batch offset
  int64_t window_code = (int64_t)batch_idx * W + local_code;
  codes[idx] = window_code;

  // Atomically increment histogram
  atomicAdd(&histogram[window_code], 1);
}

// Kernel 2: Scatter voxels into window-sorted positions.
//
// Uses the prefix-summed histogram (window_offsets_dense) and per-window
// atomic counters to assign each voxel a unique position within its window.
// Builds perm and inverse_perm simultaneously.
extern "C" __global__ void window_group_scatter_kernel(
    const int64_t* __restrict__ codes,               // (N,) window codes from Kernel 1
    const int* __restrict__ window_offsets_dense,     // (B * W,) exclusive prefix sum of histogram
    int* __restrict__ scatter_counters,               // (B * W,) initialized to 0
    int64_t* __restrict__ perm,                       // (N,) output: sorted position -> original index
    int64_t* __restrict__ inverse_perm,               // (N,) output: original index -> sorted position
    int N
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  int64_t code = codes[idx];
  int pos_in_window = atomicAdd(&scatter_counters[code], 1);
  int global_pos = window_offsets_dense[code] + pos_in_window;

  perm[global_pos] = (int64_t)idx;
  inverse_perm[idx] = (int64_t)global_pos;
}
