// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdio>
#include <limits>

namespace warpconvnet {

namespace {

// Kernel 1: Step
// For a specific selected point 'old' (passed as scalar usually, or we read it from out_idxs[k-1]),
// update distances and find local max per block.
// We need one kernel call per 'step' of K? Yes.
// Launch: Grid dims (TotalBlocks), Block dims (BlockSize)
// We map blocks to batch items?
// User said "1 block per one scene" is inefficient.
// So we want multiple blocks per scene.
// We can use a flat grid of blocks covering all points, segment-aware?
// Or we can launch (B, BlocksPerScene) grid.
// Let's allow variable blocks per scene? Complicated with offsets.
// Simpler: Launch flat grid over N points.
// Each thread processes points[gid].
// We need to know which batch item `gid` belongs to.
// Upper bound binary search on offsets? Expensive per thread.
//
// Alternative: (B, BlocksPerScene) grid.
// Each block `BlockIdx.y` handles a chunk of points for batch `BlockIdx.x`.
// This assumes we distribute work evenly or just stride.
// blockIdx.x = batch index
// blockIdx.y = chunk index
//
// Let's stick to (B, BlocksPerScene) grid.
//
// Global state:
// temp: (N) min distances
// out_idxs: (B*K)
//
// Workspace:
// partial_max_dists: (B, BlocksPerScene)
// partial_max_idxs: (B, BlocksPerScene)
//
// But if N_i varies wildly, fixed BlocksPerScene might be inefficient.
// However, it's better than 1 block.
// We can set BlocksPerScene = 32 or something reasonable.

constexpr int BLOCK_SIZE = 512;
constexpr int BLOCKS_PER_SCENE = 64;  // Tunable max blocks per scene

__global__ void fps_step_kernel(
    const float* __restrict__ points,
    const int* __restrict__ offsets,
    float* __restrict__ temp,               // (N) min dists
    const int* __restrict__ out_idxs,       // (B * K) - we only need the LAST selected one
    int k_step,                             // current step 1..K-1
    int K,                                  // max K
    float* __restrict__ partial_max_dists,  // (B, BLOCKS_PER_SCENE)
    int* __restrict__ partial_max_idxs,     // (B, BLOCKS_PER_SCENE)
    int B) {
  int b = blockIdx.x;
  int block_idx_in_scene = blockIdx.y;

  if (b >= B) return;

  int start = offsets[b];
  int end = offsets[b + 1];
  int N_i = end - start;

  if (N_i <= 0) {
    if (threadIdx.x == 0) {
      partial_max_dists[b * BLOCKS_PER_SCENE + block_idx_in_scene] = -1.0f;
      partial_max_idxs[b * BLOCKS_PER_SCENE + block_idx_in_scene] = -1;
    }
    return;
  }

  // Get the index of the previously selected point
  // stored at out_idxs[b * K + (k_step - 1)]
  // Logic:
  // Step k=0: Init (done separately or implicit).
  // This kernel is for steps k=1..K-1.
  // So we read idx at k_step-1.

  int old_global_idx = out_idxs[b * K + (k_step - 1)];
  // Make sure it is valid?

  float x1 = points[old_global_idx * 3 + 0];
  float y1 = points[old_global_idx * 3 + 1];
  float z1 = points[old_global_idx * 3 + 2];

  // Grid stride loop for this block within the scene
  // We strive to cover [start, end)
  // Threads in this block: threadIdx.x
  // Total threads for this scene: blockDim.x * gridDim.y
  // Global thread index within scene: block_idx_in_scene * blockDim.x + threadIdx.x

  int tid_in_scene = block_idx_in_scene * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.y;

  float local_max_val = -1.0f;
  int local_max_idx = -1;

  for (int i = tid_in_scene; i < N_i; i += stride) {
    int p_idx = start + i;
    float x2 = points[p_idx * 3 + 0];
    float y2 = points[p_idx * 3 + 1];
    float z2 = points[p_idx * 3 + 2];

    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

    float current_min = temp[p_idx];  // Read from global memory
    if (d < current_min) {
      current_min = d;
      temp[p_idx] = current_min;  // Write back
    }

    if (current_min > local_max_val) {
      local_max_val = current_min;
      local_max_idx = p_idx;
    }
  }

  // Block Reduction
  __shared__ float s_dists[BLOCK_SIZE];
  __shared__ int s_idxs[BLOCK_SIZE];

  s_dists[threadIdx.x] = local_max_val;
  s_idxs[threadIdx.x] = local_max_idx;

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (s_dists[threadIdx.x + s] > s_dists[threadIdx.x]) {
        s_dists[threadIdx.x] = s_dists[threadIdx.x + s];
        s_idxs[threadIdx.x] = s_idxs[threadIdx.x + s];
      }
    }
    __syncthreads();
  }

  // Write partial result
  if (threadIdx.x == 0) {
    partial_max_dists[b * BLOCKS_PER_SCENE + block_idx_in_scene] = s_dists[0];
    partial_max_idxs[b * BLOCKS_PER_SCENE + block_idx_in_scene] = s_idxs[0];
  }
}

// Kernel 0: Init
// Select 0-th point randomly or first? Current impl picks first (relative index 0).
// Sets out_idxs[b*K + 0] = start.
// Also we need to ensure temp is INF? Caller does it.
__global__ void fps_init_kernel(const int* __restrict__ offsets,
                                int* __restrict__ out_idxs,
                                int K,
                                int B) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < B) {
    int start = offsets[b];
    // Pick first point
    out_idxs[b * K + 0] = start;
  }
}

// Kernel 2: Select
// Reduces partial results for each batch item and writes the next index.
// One block per batch item? Or one thread per batch item?
// Since B might be small or large, and BLOCKS_PER_SCENE is 64.
// One block per batch item is good.
__global__ void fps_select_kernel(const float* __restrict__ partial_max_dists,
                                  const int* __restrict__ partial_max_idxs,
                                  int* __restrict__ out_idxs,
                                  int k_step,
                                  int K,
                                  int BlocksPerScene,  // 64
                                  int B) {
  int b = blockIdx.x;
  if (b >= B) return;

  // Thread reduction over BlocksPerScene
  int tid = threadIdx.x;

  float best_val = -1.0f;
  int best_idx = -1;

  for (int i = tid; i < BlocksPerScene; i += blockDim.x) {
    float val = partial_max_dists[b * BlocksPerScene + i];
    int idx = partial_max_idxs[b * BlocksPerScene + i];
    if (val > best_val) {
      best_val = val;
      best_idx = idx;
    }
  }

  // Warp/Block reduction
  // Using shared mem
  __shared__ float s_vals[256];  // Assuming blockDim.x suffices
  __shared__ int s_idxs[256];

  // Assuming blockDim.x >= BlocksPerScene or at least covering well.
  // If blockDim.x is larger than BlocksPerScene, fine.
  // If smaller, loop above handled it.

  if (tid < 256) {
    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
  }
  __syncthreads();

  // Dimensions
  int dim = (blockDim.x > 256) ? 256 : blockDim.x;

  for (int s = dim / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_vals[tid + s] > s_vals[tid]) {
        s_vals[tid] = s_vals[tid + s];
        s_idxs[tid] = s_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_idxs[b * K + k_step] = s_idxs[0];
  }
}

}  // namespace

void farthest_point_sampling_cuda(
    at::Tensor points, at::Tensor offsets, at::Tensor temp, at::Tensor idxs, int K) {
  int B = offsets.size(0) - 1;
  if (B <= 0) return;

  TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");
  TORCH_CHECK(temp.is_cuda(), "temp must be a CUDA tensor");
  TORCH_CHECK(idxs.is_cuda(), "idxs must be a CUDA tensor");

  auto stream = at::cuda::getCurrentCUDAStream();

  // 1. Init
  int grid_init = (B + 255) / 256;
  fps_init_kernel<<<grid_init, 256, 0, stream>>>(
      offsets.data_ptr<int>(), idxs.data_ptr<int>(), K, B);

  // Workspace allocation
  // We need (B, BLOCKS_PER_SCENE) for dists and idxs
  auto options = torch::TensorOptions().device(points.device());
  auto partial_dists = torch::empty({B, BLOCKS_PER_SCENE}, options.dtype(torch::kFloat32));
  auto partial_idxs = torch::empty({B, BLOCKS_PER_SCENE}, options.dtype(torch::kInt32));

  for (int k = 1; k < K; ++k) {
    // 2. Step: Update dists and partial reduce
    dim3 grid(B, BLOCKS_PER_SCENE);
    fps_step_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(points.data_ptr<float>(),
                                                     offsets.data_ptr<int>(),
                                                     temp.data_ptr<float>(),
                                                     idxs.data_ptr<int>(),
                                                     k,
                                                     K,
                                                     partial_dists.data_ptr<float>(),
                                                     partial_idxs.data_ptr<int>(),
                                                     B);

    // 3. Select: Final reduce
    // 1 block per batch item, say 128 threads per block (enough for 64 items)
    fps_select_kernel<<<B, 128, 0, stream>>>(partial_dists.data_ptr<float>(),
                                             partial_idxs.data_ptr<int>(),
                                             idxs.data_ptr<int>(),
                                             k,
                                             K,
                                             BLOCKS_PER_SCENE,
                                             B);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace warpconvnet
