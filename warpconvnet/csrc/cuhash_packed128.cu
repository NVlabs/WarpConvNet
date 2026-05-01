// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// cuhash - 128-bit packed key hash table kernels and host launchers.
//
// Skeleton scope: D=7, CoordBits=17 instantiation only (covers permutohedral
// d=6 hero workload). Add additional (D, CoordBits) launchers as needed by
// instantiating packed128_{insert,search}_kernel<D, CoordBits>.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cuhash/cuda_check.cuh"
#include "cuhash/nvtx_range.cuh"
#include "cuhash/packed128.cuh"

namespace cuhash {
namespace packed128 {

static constexpr int kBlockSize = 256;

__global__ void packed128_prepare_kernel(PackedKey128 *keys, int32_t *vals, int capacity) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < capacity) {
    keys[tid].lo = 0;
    keys[tid].hi = kEmptyHi;
    vals[tid] = -1;
  }
}

template <int D, int CoordBits>
__global__ void packed128_insert_kernel(PackedKey128 *__restrict__ keys,
                                        int32_t *__restrict__ vals,
                                        const int32_t *__restrict__ coords,
                                        int num_keys,
                                        uint32_t capacity_mask,
                                        int *__restrict__ status_ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_keys) return;
  PackedKey128 k = pack_keyN<D, CoordBits>(&coords[idx * D]);
  packed128_insert(keys, vals, k, idx, capacity_mask, status_ptr);
}

template <int D, int CoordBits>
__global__ void packed128_search_kernel(const PackedKey128 *__restrict__ keys,
                                        const int32_t *__restrict__ vals,
                                        const int32_t *__restrict__ search_coords,
                                        int32_t *__restrict__ results,
                                        int num_search,
                                        uint32_t capacity_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_search) return;
  PackedKey128 k = pack_keyN<D, CoordBits>(&search_coords[idx * D]);
  results[idx] = packed128_search(keys, vals, k, capacity_mask);
}

// ============================================================================
// Host launchers
//
// `keys` is exposed to PyTorch as int64 with shape [capacity, 2] (16 B/slot).
// We reinterpret-cast the data pointer as PackedKey128*. `vals` is int32
// with shape [capacity].
// ============================================================================

void launch_packed128_prepare(torch::Tensor keys, torch::Tensor vals, int capacity) {
  CUHASH_NVTX_SCOPE("launch_packed128_prepare");
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (capacity + kBlockSize - 1) / kBlockSize;
  packed128_prepare_kernel<<<blocks, kBlockSize, 0, stream>>>(
      reinterpret_cast<PackedKey128 *>(keys.data_ptr<int64_t>()),
      vals.data_ptr<int32_t>(),
      capacity);
  CUHASH_CHECK_CUDA_LAUNCH();
}

void launch_packed128_insert_d7c17(torch::Tensor keys,
                                   torch::Tensor vals,
                                   torch::Tensor coords,
                                   int num_keys,
                                   int capacity,
                                   torch::Tensor status_tensor) {
  CUHASH_NVTX_SCOPE("launch_packed128_insert_d7c17");
  if (num_keys == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (num_keys + kBlockSize - 1) / kBlockSize;
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  packed128_insert_kernel<7, 17><<<blocks, kBlockSize, 0, stream>>>(
      reinterpret_cast<PackedKey128 *>(keys.data_ptr<int64_t>()),
      vals.data_ptr<int32_t>(),
      coords.data_ptr<int32_t>(),
      num_keys,
      mask,
      status_tensor.data_ptr<int32_t>());
  CUHASH_CHECK_CUDA_LAUNCH();
}

void launch_packed128_search_d7c17(torch::Tensor keys,
                                   torch::Tensor vals,
                                   torch::Tensor search_coords,
                                   torch::Tensor results,
                                   int num_search,
                                   int capacity) {
  CUHASH_NVTX_SCOPE("launch_packed128_search_d7c17");
  if (num_search == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (num_search + kBlockSize - 1) / kBlockSize;
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  packed128_search_kernel<7, 17><<<blocks, kBlockSize, 0, stream>>>(
      reinterpret_cast<const PackedKey128 *>(keys.data_ptr<int64_t>()),
      vals.data_ptr<int32_t>(),
      search_coords.data_ptr<int32_t>(),
      results.data_ptr<int32_t>(),
      num_search,
      mask);
  CUHASH_CHECK_CUDA_LAUNCH();
}

// ============================================================================
// Batched search: one launch over (M queries, K offsets), output K-major.
//
// Each thread handles one query, loops K offsets, packs (query + offset_k)
// and searches the table. Query coords loaded once into registers. Offsets
// staged through shared memory so warps hit the smem cache instead of L1
// per probe. Output layout matches the existing kernel-map convention:
//   results[k * M + qidx] = vals[slot] on hit, -1 on miss.
// K-major fits permutohedral blur's per-axis [M] contiguous read pattern.
//
// Use case: permutohedral / bilateral lattice blur — K = 2*(d+1) <= 16 in
// practice, well under MAX_K=32. K is runtime-bounded by an in-kernel guard
// against MAX_K to keep the per-thread offset register footprint small.
// ============================================================================

template <int D, int CoordBits, int MAX_K>
__global__ void packed128_batched_search_kernel(const PackedKey128 *__restrict__ keys,
                                                const int32_t *__restrict__ vals,
                                                const int32_t *__restrict__ queries,  // [M, D]
                                                const int32_t *__restrict__ offsets,  // [K, D]
                                                int32_t *__restrict__ results,  // [K, M] K-major
                                                int M,
                                                int K,
                                                uint32_t capacity_mask) {
  // Stage offsets in shared memory: one block-wide load amortised across
  // all queries the block handles. Size bound = MAX_K * D ints (e.g.
  // 32 * 7 = 896 B at D=7 — trivial smem cost).
  __shared__ int32_t s_off[MAX_K * D];
  int total_off = K * D;
  int tid = threadIdx.x;
  for (int i = tid; i < total_off; i += blockDim.x) {
    s_off[i] = offsets[i];
  }
  __syncthreads();

  int qidx = blockIdx.x * blockDim.x + tid;
  if (qidx >= M) return;

  // Load query coords into registers (1 load per axis, used K times).
  int32_t q[D];
#pragma unroll
  for (int i = 0; i < D; ++i) {
    q[i] = queries[qidx * D + i];
  }

  // Probe each offset. K is bounded by MAX_K; the loop is unrollable up to
  // MAX_K but exits early on the runtime K bound.
  int32_t pkey_buf[D];
  for (int k = 0; k < K; ++k) {
#pragma unroll
    for (int i = 0; i < D; ++i) {
      pkey_buf[i] = q[i] + s_off[k * D + i];
    }
    PackedKey128 pk = pack_keyN<D, CoordBits>(pkey_buf);
    results[k * M + qidx] = packed128_search(keys, vals, pk, capacity_mask);
  }
}

void launch_packed128_batched_search_d7c17(torch::Tensor keys,
                                           torch::Tensor vals,
                                           torch::Tensor queries,
                                           torch::Tensor offsets,
                                           torch::Tensor results,
                                           int M,
                                           int K,
                                           int capacity) {
  CUHASH_NVTX_SCOPE("launch_packed128_batched_search_d7c17");
  TORCH_CHECK(K > 0 && K <= 32, "K must be in [1, 32]; got ", K);
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (M + kBlockSize - 1) / kBlockSize;
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  packed128_batched_search_kernel<7, 17, 32><<<blocks, kBlockSize, 0, stream>>>(
      reinterpret_cast<const PackedKey128 *>(keys.data_ptr<int64_t>()),
      vals.data_ptr<int32_t>(),
      queries.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>(),
      results.data_ptr<int32_t>(),
      M,
      K,
      mask);
  CUHASH_CHECK_CUDA_LAUNCH();
}

}  // namespace packed128
}  // namespace cuhash
