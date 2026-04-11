// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared launch wrappers and dispatch templates for fwd/dgrad/wgrad.
#pragma once

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace warpgemm {

// Forward/Dgrad launch wrapper
template <class Kernel>
__global__ __launch_bounds__(
    Kernel::MaxThreadsPerBlock,
    Kernel::MinBlocksPerMultiprocessor) void launch_mask_gemm(const typename Kernel::ElementInput
                                                                  *ptr_A,
                                                              const typename Kernel::ElementInput
                                                                  *ptr_B,
                                                              typename Kernel::ElementOutput *ptr_D,
                                                              const int *pair_table,
                                                              const uint32_t *pair_mask,
                                                              const int *mask_argsort,
                                                              int N_in,
                                                              int N_out,
                                                              int C_in,
                                                              int C_out,
                                                              int K,
                                                              float alpha) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A,
           ptr_B,
           ptr_D,
           pair_table,
           pair_mask,
           mask_argsort,
           N_in,
           N_out,
           C_in,
           C_out,
           K,
           alpha,
           smem);
}

// Forward dispatch
template <class Kernel>
int run_mask_gemm(const void *a,
                  const void *b,
                  void *d,
                  const int *pair_table,
                  const uint32_t *pair_mask,
                  const int *mask_argsort,
                  int N_in,
                  int N_out,
                  int C_in,
                  int C_out,
                  int K,
                  float alpha,
                  cudaStream_t stream = 0) {
  using ElementInput = typename Kernel::ElementInput;
  constexpr int TileM = cute::size<0>(typename Kernel::TileShape{});
  constexpr int TileN = cute::size<1>(typename Kernel::TileShape{});
  constexpr size_t smem_size = Kernel::SharedStorageSize;
  if (N_out == 0 || C_in == 0 || C_out == 0) return 0;
  int m_tiles = (N_out + TileM - 1) / TileM;
  int n_tiles = (C_out + TileN - 1) / TileN;
  dim3 grid(m_tiles * n_tiles, 1, 1);
  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(
        launch_mask_gemm<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) return -1;
  }
  launch_mask_gemm<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
      reinterpret_cast<const ElementInput *>(a),
      reinterpret_cast<const ElementInput *>(b),
      reinterpret_cast<typename Kernel::ElementOutput *>(d),
      pair_table,
      pair_mask,
      mask_argsort,
      N_in,
      N_out,
      C_in,
      C_out,
      K,
      alpha);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Dgrad dispatch
template <class Kernel>
int run_mask_gemm_dgrad(const void *a,
                        const void *b,
                        void *d,
                        const int *pair_table,
                        const uint32_t *pair_mask,
                        const int *mask_argsort,
                        int N_in,
                        int N_out,
                        int C_in,
                        int C_out,
                        int K,
                        float alpha,
                        cudaStream_t stream = 0) {
  using ElementInput = typename Kernel::ElementInput;
  constexpr int TileM = cute::size<0>(typename Kernel::TileShape{});
  constexpr int TileN = cute::size<1>(typename Kernel::TileShape{});
  constexpr size_t smem_size = Kernel::SharedStorageSize;
  if (N_in == 0 || C_in == 0 || C_out == 0) return 0;
  int m_tiles = (N_in + TileM - 1) / TileM;
  int n_tiles = (C_in + TileN - 1) / TileN;
  dim3 grid(m_tiles * n_tiles, Kernel::SplitOffsets, 1);
  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(
        launch_mask_gemm<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) return -1;
  }
  launch_mask_gemm<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
      reinterpret_cast<const ElementInput *>(a),
      reinterpret_cast<const ElementInput *>(b),
      reinterpret_cast<typename Kernel::ElementOutput *>(d),
      pair_table,
      pair_mask,
      mask_argsort,
      N_in,
      N_out,
      C_in,
      C_out,
      K,
      alpha);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Wgrad launch wrapper
template <class Kernel>
__global__ __launch_bounds__(
    Kernel::MaxThreadsPerBlock,
    Kernel::
        MinBlocksPerMultiprocessor) void launch_mask_gemm_wgrad(const typename Kernel::ElementInput
                                                                    *ptr_A,
                                                                const typename Kernel::ElementInput
                                                                    *ptr_B,
                                                                typename Kernel::ElementOutput
                                                                    *ptr_D,
                                                                const int *pair_table,
                                                                const uint32_t *pair_mask,
                                                                const int *mask_argsort,
                                                                const uint32_t *reduced_mask,
                                                                int N_in,
                                                                int N_out,
                                                                int C_in,
                                                                int C_out,
                                                                int K,
                                                                float alpha) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A,
           ptr_B,
           ptr_D,
           pair_table,
           pair_mask,
           mask_argsort,
           reduced_mask,
           N_in,
           N_out,
           C_in,
           C_out,
           K,
           alpha,
           smem);
}

// Wgrad dispatch
template <class Kernel>
int run_mask_gemm_wgrad(const void *a,
                        const void *b,
                        void *d,
                        const int *pair_table,
                        const uint32_t *pair_mask,
                        const int *mask_argsort,
                        const uint32_t *reduced_mask,
                        int N_in,
                        int N_out,
                        int C_in,
                        int C_out,
                        int K,
                        float alpha,
                        int split_k,
                        cudaStream_t stream = 0) {
  using ElementInput = typename Kernel::ElementInput;
  constexpr int TileM = cute::size<0>(typename Kernel::TileShape{});
  constexpr int TileN = cute::size<1>(typename Kernel::TileShape{});
  constexpr size_t smem_size = Kernel::SharedStorageSize;
  if (C_in == 0 || C_out == 0 || N_out == 0) return 0;
  int m_tiles = (C_in + TileM - 1) / TileM;
  int n_tiles = (C_out + TileN - 1) / TileN;
  dim3 grid(m_tiles * n_tiles, K, split_k);
  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(
        launch_mask_gemm_wgrad<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) return -1;
  }
  launch_mask_gemm_wgrad<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
      reinterpret_cast<const ElementInput *>(a),
      reinterpret_cast<const ElementInput *>(b),
      reinterpret_cast<typename Kernel::ElementOutput *>(d),
      pair_table,
      pair_mask,
      mask_argsort,
      reduced_mask,
      N_in,
      N_out,
      C_in,
      C_out,
      K,
      alpha);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

}  // namespace warpgemm
