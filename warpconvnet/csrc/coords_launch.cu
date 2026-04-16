// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host wrapper functions for coordinate search and utility kernels.
// Wraps kernels from morton_code.cu, find_first_gt_bsearch.cu,
// radius_search_kernels.cu, and window_grouping_kernels.cu.

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cub/cub.cuh>
#include <tuple>
#include <vector>

// ============================================================================
// Forward declarations of extern "C" __global__ kernels from other .cu files
// ============================================================================

// --- morton_code.cu ---
extern "C" __global__ void assign_order_discrete_16bit_kernel(const int* bcoords_data,
                                                              int num_points,
                                                              int64_t* result_order);
extern "C" __global__ void assign_order_discrete_20bit_kernel_4points(const int* coords_data,
                                                                      int num_points,
                                                                      int64_t* result_order);

// --- find_first_gt_bsearch.cu ---
extern "C" __global__ void find_first_gt_bsearch(
    const int* srcM, int M, const int* srcN, int N, int* out);

// --- radius_search_kernels.cu (packed cuhash) ---
__global__ void radius_search_count_packed(const float*,
                                           const float*,
                                           const int*,
                                           const int*,
                                           const int*,
                                           const uint64_t*,
                                           const int*,
                                           int*,
                                           int,
                                           int,
                                           int,
                                           float,
                                           float,
                                           uint32_t);
__global__ void radius_search_write_packed(const float*,
                                           const float*,
                                           const int*,
                                           const int*,
                                           const int*,
                                           const uint64_t*,
                                           const int*,
                                           const int*,
                                           int*,
                                           float*,
                                           int,
                                           int,
                                           int,
                                           float,
                                           float,
                                           uint32_t);

// ============================================================================
// coord_to_code kernel (moved from inline Python string in voxel_encode.py)
// ============================================================================

__device__ static int64_t part1by2_long_ctc(int64_t n) {
  n &= 0x1fffffLL;  // mask to 21 bits
  n = (n | (n << 32)) & 0x1f00000000ffffLL;
  n = (n | (n << 16)) & 0x1f0000ff0000ffLL;
  n = (n | (n << 8)) & 0x100f00f00f00f00fLL;
  n = (n | (n << 4)) & 0x10c30c30c30c30c3LL;
  n = (n | (n << 2)) & 0x1249249249249249LL;
  return n;
}

__device__ static int64_t morton_code_20bit_device_ctc(int64_t coord_x,
                                                       int64_t coord_y,
                                                       int64_t coord_z) {
  return (part1by2_long_ctc(coord_z) << 2) | (part1by2_long_ctc(coord_y) << 1) |
         part1by2_long_ctc(coord_x);
}

__global__ void coord_to_code_kernel_impl(const int* grid_coord,
                                          const int* coord_offset,
                                          const int* min_coord,
                                          const int* window_size,
                                          const int N,
                                          int64_t* codes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  __shared__ int s_coord_offset[3];
  __shared__ int s_min_coord[3];
  __shared__ int s_window_size[3];
  if (threadIdx.x < 3) {
    s_coord_offset[threadIdx.x] = coord_offset[threadIdx.x];
    s_min_coord[threadIdx.x] = min_coord[threadIdx.x];
    s_window_size[threadIdx.x] = window_size[threadIdx.x];
  }
  __syncthreads();

  int grid_coord_x = grid_coord[idx * 3 + 0];
  int grid_coord_y = grid_coord[idx * 3 + 1];
  int grid_coord_z = grid_coord[idx * 3 + 2];

  int64_t voxel_x = (grid_coord_x + s_coord_offset[0] - s_min_coord[0]) / s_window_size[0];
  int64_t voxel_y = (grid_coord_y + s_coord_offset[1] - s_min_coord[1]) / s_window_size[1];
  int64_t voxel_z = (grid_coord_z + s_coord_offset[2] - s_min_coord[2]) / s_window_size[2];

  codes[idx] = morton_code_20bit_device_ctc(voxel_x, voxel_y, voxel_z);
}

// ============================================================================
// Host wrapper functions (callable from coords_bindings.cpp)
// ============================================================================

void coords_morton_code_16bit(torch::Tensor bcoords, int num_points, torch::Tensor result) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (num_points + threads - 1) / threads;
  assign_order_discrete_16bit_kernel<<<blocks, threads, 0, stream>>>(
      bcoords.data_ptr<int>(), num_points, result.data_ptr<int64_t>());
}

void coords_morton_code_20bit(torch::Tensor coords, int num_points, torch::Tensor result) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  // The kernel processes 4 points per thread
  int blocks = (num_points + threads * 4 - 1) / (threads * 4);
  assign_order_discrete_20bit_kernel_4points<<<blocks, threads, 0, stream>>>(
      coords.data_ptr<int>(), num_points, result.data_ptr<int64_t>());
}

void coords_find_first_gt_bsearch(
    torch::Tensor offsets_tensor, int M, torch::Tensor indices, int N, torch::Tensor output) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  int shared_mem = M * sizeof(int);
  find_first_gt_bsearch<<<blocks, threads, shared_mem, stream>>>(
      offsets_tensor.data_ptr<int>(), M, indices.data_ptr<int>(), N, output.data_ptr<int>());
}

void coords_coord_to_code(torch::Tensor grid_coord,
                          torch::Tensor coord_offset,
                          torch::Tensor min_coord,
                          torch::Tensor window_size,
                          int N,
                          torch::Tensor codes) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  coord_to_code_kernel_impl<<<blocks, threads, 0, stream>>>(grid_coord.data_ptr<int>(),
                                                            coord_offset.data_ptr<int>(),
                                                            min_coord.data_ptr<int>(),
                                                            window_size.data_ptr<int>(),
                                                            N,
                                                            codes.data_ptr<int64_t>());
}

void coords_radius_search_count(torch::Tensor points,
                                torch::Tensor queries,
                                torch::Tensor sorted_indices,
                                torch::Tensor cell_starts,
                                torch::Tensor cell_counts,
                                torch::Tensor keys,
                                torch::Tensor values,
                                torch::Tensor result_count,
                                int N,
                                int M,
                                int num_cells,
                                float radius,
                                float cell_size,
                                int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (M + threads - 1) / threads;
  uint32_t capacity_mask = static_cast<uint32_t>(capacity - 1);

  radius_search_count_packed<<<blocks, threads, 0, stream>>>(
      points.data_ptr<float>(),
      queries.data_ptr<float>(),
      sorted_indices.data_ptr<int>(),
      cell_starts.data_ptr<int>(),
      cell_counts.data_ptr<int>(),
      reinterpret_cast<const uint64_t*>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      result_count.data_ptr<int>(),
      N,
      M,
      num_cells,
      radius,
      cell_size,
      capacity_mask);
}

void coords_radius_search_write(torch::Tensor points,
                                torch::Tensor queries,
                                torch::Tensor sorted_indices,
                                torch::Tensor cell_starts,
                                torch::Tensor cell_counts,
                                torch::Tensor keys,
                                torch::Tensor values,
                                torch::Tensor result_offsets,
                                torch::Tensor result_indices,
                                torch::Tensor result_distances,
                                int N,
                                int M,
                                int num_cells,
                                float radius,
                                float cell_size,
                                int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (M + threads - 1) / threads;
  uint32_t capacity_mask = static_cast<uint32_t>(capacity - 1);

  radius_search_write_packed<<<blocks, threads, 0, stream>>>(
      points.data_ptr<float>(),
      queries.data_ptr<float>(),
      sorted_indices.data_ptr<int>(),
      cell_starts.data_ptr<int>(),
      cell_counts.data_ptr<int>(),
      reinterpret_cast<const uint64_t*>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      result_offsets.data_ptr<int>(),
      result_indices.data_ptr<int>(),
      result_distances.data_ptr<float>(),
      N,
      M,
      num_cells,
      radius,
      cell_size,
      capacity_mask);
}

// ============================================================================
// Window grouping kernels (counting sort)
// ============================================================================

extern "C" __global__ void window_group_histogram_kernel(const int* __restrict__ grid_coord,
                                                         const int* __restrict__ batch_offsets,
                                                         const int* __restrict__ coord_offset,
                                                         const int* __restrict__ min_coord,
                                                         const int* __restrict__ window_size,
                                                         const int* __restrict__ grid_shape,
                                                         int64_t* __restrict__ codes,
                                                         int* __restrict__ histogram,
                                                         int N,
                                                         int B,
                                                         int W);

extern "C" __global__ void window_group_scatter_kernel(const int64_t* __restrict__ codes,
                                                       const int* __restrict__ window_offsets_dense,
                                                       int* __restrict__ scatter_counters,
                                                       int64_t* __restrict__ perm,
                                                       int64_t* __restrict__ inverse_perm,
                                                       int N);

void coords_window_group_histogram(torch::Tensor grid_coord,
                                   torch::Tensor batch_offsets,
                                   torch::Tensor coord_offset,
                                   torch::Tensor min_coord,
                                   torch::Tensor window_size,
                                   torch::Tensor grid_shape,
                                   torch::Tensor codes,
                                   torch::Tensor histogram,
                                   int N,
                                   int B,
                                   int W) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  window_group_histogram_kernel<<<blocks, threads, 0, stream>>>(grid_coord.data_ptr<int>(),
                                                                batch_offsets.data_ptr<int>(),
                                                                coord_offset.data_ptr<int>(),
                                                                min_coord.data_ptr<int>(),
                                                                window_size.data_ptr<int>(),
                                                                grid_shape.data_ptr<int>(),
                                                                codes.data_ptr<int64_t>(),
                                                                histogram.data_ptr<int>(),
                                                                N,
                                                                B,
                                                                W);
}

void coords_window_group_scatter(torch::Tensor codes,
                                 torch::Tensor window_offsets_dense,
                                 torch::Tensor scatter_counters,
                                 torch::Tensor perm,
                                 torch::Tensor inverse_perm,
                                 int N) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  window_group_scatter_kernel<<<blocks, threads, 0, stream>>>(codes.data_ptr<int64_t>(),
                                                              window_offsets_dense.data_ptr<int>(),
                                                              scatter_counters.data_ptr<int>(),
                                                              perm.data_ptr<int64_t>(),
                                                              inverse_perm.data_ptr<int64_t>(),
                                                              N);
}
