// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for generating mask data (pair_mask + mask_argsort) from
// the kernel map's found_in_coord_index intermediate or from CSR format.

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

/**
 * Build pair_mask from pair_table (found_in_coord_index).
 *
 * pair_table: [K, N_out] int32, entry [k*N_out + i] = input idx or -1
 * pair_mask:  [N_out] uint32, bit k set if pair_table[k*N_out + i] >= 0
 *
 * Grid: (ceil(N_out / 256), 1)
 * Block: (256, 1)
 */
__global__ void build_pair_mask_kernel(
    const int *__restrict__ pair_table,  // [K * N_out]
    uint32_t *__restrict__ pair_mask,    // [N_out]
    const int N_out,
    const int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_out) return;

  uint32_t mask = 0;
  for (int k = 0; k < K; k++) {
    if (pair_table[k * N_out + i] >= 0) {
      mask |= (1u << k);
    }
  }
  pair_mask[i] = mask;
}

/**
 * Build pair_table from CSR format (in_maps, out_maps, offsets).
 *
 * For each offset k, scatter in_maps[offsets[k]:offsets[k+1]] to
 * pair_table[k, out_maps[...]].
 *
 * Grid: (ceil(L / 256), 1) where L = total pairs
 * Block: (256, 1)
 */
__global__ void csr_to_pair_table_kernel(
    const int *__restrict__ in_maps,     // [L]
    const int *__restrict__ out_maps,    // [L]
    const int *__restrict__ offsets,     // [K+1]
    int *__restrict__ pair_table,        // [K * N_out], pre-filled with -1
    const int N_out,
    const int K,
    const int L) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= L) return;

  // Binary search to find which offset k this idx belongs to
  int lo = 0, hi = K;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (offsets[mid + 1] <= idx)
      lo = mid + 1;
    else
      hi = mid;
  }
  int k = lo;

  int out_idx = out_maps[idx];
  int in_idx = in_maps[idx];
  if (out_idx >= 0 && out_idx < N_out) {
    pair_table[k * N_out + out_idx] = in_idx;
  }
}

namespace warpconvnet {
namespace mask_data {

void build_pair_mask(
    const int *pair_table,
    uint32_t *pair_mask,
    int N_out,
    int K) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N_out + threads - 1) / threads;
  build_pair_mask_kernel<<<blocks, threads, 0, stream>>>(
      pair_table, pair_mask, N_out, K);
}

void csr_to_pair_table(
    const int *in_maps,
    const int *out_maps,
    const int *offsets,
    int *pair_table,
    int N_out,
    int K,
    int L) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (L + threads - 1) / threads;
  csr_to_pair_table_kernel<<<blocks, threads, 0, stream>>>(
      in_maps, out_maps, offsets, pair_table, N_out, K, L);
}

}  // namespace mask_data
}  // namespace warpconvnet
