// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for generating mask data (pair_mask + mask_argsort) from
// the kernel map's found_in_coord_index intermediate or from CSR format.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
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
__global__ void build_pair_mask_kernel(const int *__restrict__ pair_table,  // [K * N_out]
                                       uint32_t *__restrict__ pair_mask,    // [N_out * mask_words]
                                       const int N_out,
                                       const int K,
                                       const int mask_words) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_out) return;

  // Interleaved layout: pair_mask[i * mask_words + w]
  for (int w = 0; w < mask_words; w++) {
    uint32_t word = 0;
    int k_start = w * 32;
    int k_end = k_start + 32;
    if (k_end > K) k_end = K;
    for (int k = k_start; k < k_end; k++) {
      if (pair_table[k * N_out + i] >= 0) {
        word |= (1u << (k - k_start));
      }
    }
    pair_mask[i * mask_words + w] = word;
  }
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
    const int *__restrict__ in_maps,   // [L]
    const int *__restrict__ out_maps,  // [L]
    const int *__restrict__ offsets,   // [K+1]
    int *__restrict__ pair_table,      // [K * N_out], pre-filled with -1
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

/**
 * Build reverse_pair_table + reverse pair_mask in a single fused launch.
 *
 * Forward pair_table maps (k, out_row) -> in_row (or -1).
 * Reverse pair_table maps (k, in_row) -> out_row (or -1).
 *
 * Each thread handles one (k, out_row) entry of the forward table:
 *   if pair_table[k * N_out + out_row] >= 0:
 *     in_row = pair_table[k * N_out + out_row]
 *     reverse_pair_table[k * N_in + in_row] = out_row
 *     atomicOr(reverse_pair_mask[in_row * mask_words + k/32], 1u << (k%32))
 *
 * Caller pre-fills reverse_pair_table with -1 and reverse_pair_mask with 0.
 *
 * Grid: (ceil(K * N_out / 256), 1)
 * Block: (256, 1)
 */
__global__ void build_reverse_mask_data_kernel(
    const int *__restrict__ pair_table,        // [K * N_out]
    int *__restrict__ reverse_pair_table,      // [K * N_in], pre-filled with -1
    uint32_t *__restrict__ reverse_pair_mask,  // [N_in * mask_words], pre-zeroed
    const int N_in,
    const int N_out,
    const int K,
    const int mask_words) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * N_out;
  if (tid >= total) return;

  int k = tid / N_out;
  int out_row = tid - k * N_out;

  int in_row = pair_table[tid];
  if (in_row < 0) return;
  if (in_row >= N_in) return;  // safety

  reverse_pair_table[k * N_in + in_row] = out_row;
  int word = k >> 5;
  uint32_t bit = 1u << (k & 31);
  atomicOr(&reverse_pair_mask[in_row * mask_words + word], bit);
}

namespace warpconvnet {
namespace mask_data {

void build_pair_mask(const int *pair_table, uint32_t *pair_mask, int N_out, int K, int mask_words) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N_out + threads - 1) / threads;
  build_pair_mask_kernel<<<blocks, threads, 0, stream>>>(
      pair_table, pair_mask, N_out, K, mask_words);
}

void csr_to_pair_table(const int *in_maps,
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

void build_reverse_mask_data(const int *pair_table,
                             int *reverse_pair_table,
                             uint32_t *reverse_pair_mask,
                             int N_in,
                             int N_out,
                             int K,
                             int mask_words) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int total = K * N_out;
  int blocks = (total + threads - 1) / threads;
  build_reverse_mask_data_kernel<<<blocks, threads, 0, stream>>>(
      pair_table, reverse_pair_table, reverse_pair_mask, N_in, N_out, K, mask_words);
}

/**
 * Initialize an int range [0, 1, ..., N-1].
 */
__global__ void iota_kernel(int *__restrict__ out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = i;
}

/**
 * Argsort uint32 keys and return permutation int32 indices.
 *
 * Uses cub::DeviceRadixSort::SortPairs. Not stable — voxels with identical
 * keys may be reordered within their group. mask_argsort callers tolerate
 * this since the use is cache-coherence grouping (semantic-preserving).
 *
 * Caller pre-allocates `out_perm` of size N. `keys_inout` may be modified
 * (used as both input and intermediate); pass a copy if the original keys
 * must be preserved.
 *
 * Returns 0 on success, -1 on cub query failure.
 */
int mask_argsort_uint32(const uint32_t *keys_in, int *out_perm, int N) {
  if (N <= 0) return 0;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // Stage 1: build [0..N-1] iota into a temp value array.
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  auto *opts_int = c10::cuda::CUDACachingAllocator::raw_alloc(N * sizeof(int));
  int *iota_in = static_cast<int *>(opts_int);
  iota_kernel<<<blocks, threads, 0, stream>>>(iota_in, N);

  // Stage 2: cub::DeviceRadixSort::SortPairs.
  // Keys: uint32 mask values. Values: int permutation.
  size_t temp_storage_bytes = 0;
  // Allocate scratch keys output (we only need values_out = permutation).
  auto *keys_out_storage = c10::cuda::CUDACachingAllocator::raw_alloc(N * sizeof(uint32_t));
  uint32_t *keys_out = static_cast<uint32_t *>(keys_out_storage);

  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, keys_in, keys_out, iota_in, out_perm, N, 0, 32, stream);
  if (temp_storage_bytes == 0) {
    c10::cuda::CUDACachingAllocator::raw_delete(opts_int);
    c10::cuda::CUDACachingAllocator::raw_delete(keys_out_storage);
    return -1;
  }
  auto *d_temp_storage = c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, keys_in, keys_out, iota_in, out_perm, N, 0, 32, stream);

  c10::cuda::CUDACachingAllocator::raw_delete(opts_int);
  c10::cuda::CUDACachingAllocator::raw_delete(keys_out_storage);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
  return 0;
}

}  // namespace mask_data
}  // namespace warpconvnet
