// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>

namespace warpconvnet {

// Simple kernel with shared memory optimization
template <typename scalar_t>
__global__ void points_to_closest_voxel_kernel_simple(
    const scalar_t* __restrict__ points,      // N x 3
    const int32_t* __restrict__ offsets,      // B + 1
    int32_t* __restrict__ voxel_indices,      // N
    const scalar_t* __restrict__ bounds_min,  // 3
    const scalar_t* __restrict__ bounds_max,  // 3
    const int* __restrict__ grid_shape,       // 3 (H, W, D)
    const int num_points,
    const int batch_size) {
  // Shared memory for frequently accessed data
  __shared__ scalar_t s_bounds_min[3];
  __shared__ scalar_t s_bounds_max[3];
  __shared__ int s_grid_shape[3];
  __shared__ scalar_t s_inv_grid_size[3];
  __shared__ int s_batch_voxel_size;
  __shared__ int32_t s_offsets[64];  // Support up to 63 batches in shared memory

  // Load shared data cooperatively
  if (threadIdx.x < 3) {
    s_bounds_min[threadIdx.x] = bounds_min[threadIdx.x];
    s_bounds_max[threadIdx.x] = bounds_max[threadIdx.x];
    s_grid_shape[threadIdx.x] = grid_shape[threadIdx.x];
    s_inv_grid_size[threadIdx.x] = scalar_t(s_grid_shape[threadIdx.x]) /
                                   (s_bounds_max[threadIdx.x] - s_bounds_min[threadIdx.x]);
  }

  if (threadIdx.x == 0) {
    s_batch_voxel_size = s_grid_shape[0] * s_grid_shape[1] * s_grid_shape[2];
  }

  // Load offsets cooperatively (up to 64 offsets)
  const int offsets_to_load = min(batch_size + 1, 64);
  for (int i = threadIdx.x; i < offsets_to_load; i += blockDim.x) {
    s_offsets[i] = offsets[i];
  }

  __syncthreads();

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  // Find which batch this point belongs to using binary search
  int batch_idx = 0;
  if (batch_size < 63) {
    // Use shared memory offsets
    int left = 0, right = batch_size;
    while (left < right) {
      int mid = (left + right) / 2;
      if (idx < s_offsets[mid]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    batch_idx = left - 1;
  } else {
    // Fall back to global memory for large batch sizes
    int left = 0, right = batch_size;
    while (left < right) {
      int mid = (left + right) / 2;
      if (idx < offsets[mid]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    batch_idx = left - 1;
  }

  // Load point coordinates
  scalar_t px = points[idx * 3 + 0];
  scalar_t py = points[idx * 3 + 1];
  scalar_t pz = points[idx * 3 + 2];

  // Find voxel indices using precomputed inverse grid sizes
  int vx = static_cast<int>((px - s_bounds_min[0]) * s_inv_grid_size[0]);
  int vy = static_cast<int>((py - s_bounds_min[1]) * s_inv_grid_size[1]);
  int vz = static_cast<int>((pz - s_bounds_min[2]) * s_inv_grid_size[2]);

  // Clamp to grid bounds
  vx = max(0, min(vx, s_grid_shape[0] - 1));
  vy = max(0, min(vy, s_grid_shape[1] - 1));
  vz = max(0, min(vz, s_grid_shape[2] - 1));

  // Compute flattened voxel index
  int voxel_idx = vx * s_grid_shape[1] * s_grid_shape[2] + vy * s_grid_shape[2] + vz;

  // Add batch offset
  voxel_indices[idx] = batch_idx * s_batch_voxel_size + voxel_idx;
}

}  // namespace warpconvnet

// PyTorch bindings
torch::Tensor points_to_closest_voxel_mapping(torch::Tensor points,      // N x 3
                                              torch::Tensor offsets,     // B + 1
                                              torch::Tensor grid_shape,  // 3
                                              torch::Tensor bounds_min,  // 3
                                              torch::Tensor bounds_max   // 3
) {
  // Input validation
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "Points must be N x 3");
  TORCH_CHECK(offsets.dim() == 1, "Offsets must be 1D");
  TORCH_CHECK(grid_shape.numel() == 3, "Grid shape must have 3 elements");
  TORCH_CHECK(bounds_min.numel() == 3, "Bounds min must have 3 elements");
  TORCH_CHECK(bounds_max.numel() == 3, "Bounds max must have 3 elements");

  TORCH_CHECK(points.is_cuda(), "Points must be on CUDA");
  TORCH_CHECK(offsets.is_cuda(), "Offsets must be on CUDA");

  const auto num_points = points.size(0);
  const auto batch_size = offsets.size(0) - 1;

  // Ensure tensors are contiguous
  points = points.contiguous();
  offsets = offsets.to(torch::kInt32).contiguous();
  grid_shape = grid_shape.to(torch::kInt32).cuda();
  bounds_min = bounds_min.to(points.dtype()).cuda();
  bounds_max = bounds_max.to(points.dtype()).cuda();

  // Allocate output
  auto voxel_indices = torch::empty(
      {num_points}, torch::TensorOptions().dtype(torch::kInt32).device(points.device()));

  // Launch simple kernel for debugging
  const int threads = 256;
  const int blocks = (num_points + threads - 1) / threads;

  // Use simple kernel to debug the issue
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "points_to_closest_voxel_simple", [&] {
    warpconvnet::points_to_closest_voxel_kernel_simple<scalar_t>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            points.data_ptr<scalar_t>(),
            offsets.data_ptr<int32_t>(),
            voxel_indices.data_ptr<int32_t>(),
            bounds_min.data_ptr<scalar_t>(),
            bounds_max.data_ptr<scalar_t>(),
            grid_shape.data_ptr<int32_t>(),
            num_points,
            batch_size);
  });

  // Check for CUDA errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(error));
  }

  return voxel_indices;
}
