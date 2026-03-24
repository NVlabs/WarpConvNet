// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Host-side launcher for mask-based fused CuTe GEMM.

#pragma once

#include <cuda_runtime.h>

#include "cute_gemm_mask_kernel.h"
#include "gemm_error_codes.h"

namespace warpconvnet {
namespace cute_gemm {

template <typename ElementInput, class TileConfig, typename ElementOutput>
int launch_cute_gemm_mask_fwd(
    const void *ptr_A,
    const void *ptr_B,
    void *ptr_D,
    const int *pair_table,
    const uint32_t *pair_mask,
    const int *mask_argsort,
    int N_out, int C_in, int C_out, int K,
    float alpha,
    cudaStream_t stream = 0) {

  using Kernel = CuteGemmMaskKernel<TileConfig, ElementOutput>;
  constexpr int TileM = cute::size<0>(typename TileConfig::TileShape{});
  constexpr int TileN = cute::size<1>(typename TileConfig::TileShape{});
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  if (N_out == 0 || C_in == 0 || C_out == 0) {
    return static_cast<int>(gemm::GemmStatus::kSuccess);
  }

  int m_tiles = (N_out + TileM - 1) / TileM;
  int n_tiles = (C_out + TileN - 1) / TileN;
  dim3 grid(m_tiles * n_tiles);

  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(
        cute_gemm_mask_kernel_entry<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_mask_kernel_entry<Kernel>
      <<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const ElementInput *>(ptr_A),
          reinterpret_cast<const ElementInput *>(ptr_B),
          reinterpret_cast<ElementOutput *>(ptr_D),
          pair_table, pair_mask, mask_argsort,
          N_out, C_in, C_out, K, alpha);

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

template <typename ElementInput, class TileConfig, typename ElementOutput>
int launch_cute_gemm_mask_dgrad(
    const void *ptr_GO,
    const void *ptr_B,
    void *ptr_GI,
    const int *pair_table,
    const uint32_t *pair_mask,
    const int *mask_argsort,
    int N_in, int N_out, int C_in, int C_out, int K,
    float alpha,
    cudaStream_t stream = 0) {

  using Kernel = CuteGemmMaskDgradKernel<TileConfig, ElementOutput>;
  constexpr int TileM = cute::size<0>(typename TileConfig::TileShape{});
  constexpr int TileN = cute::size<1>(typename TileConfig::TileShape{});
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  if (N_out == 0 || C_in == 0 || C_out == 0) {
    return static_cast<int>(gemm::GemmStatus::kSuccess);
  }

  int m_tiles = (N_out + TileM - 1) / TileM;
  int n_tiles = (C_in + TileN - 1) / TileN;
  dim3 grid(m_tiles * n_tiles);

  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(
        cute_gemm_mask_dgrad_kernel_entry<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_mask_dgrad_kernel_entry<Kernel>
      <<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const ElementInput *>(ptr_GO),
          reinterpret_cast<const ElementInput *>(ptr_B),
          reinterpret_cast<ElementOutput *>(ptr_GI),
          pair_table, pair_mask, mask_argsort,
          N_in, N_out, C_in, C_out, K, alpha);

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

}  // namespace cute_gemm
}  // namespace warpconvnet
