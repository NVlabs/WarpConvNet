// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host-side launcher for CuTe GEMM with gather/scatter.
// Uses raw pointers + gather indices — the kernel handles gather internally.

#pragma once

#include <cuda_runtime.h>

#include "cute_gemm_kernel.h"
#include "gemm_error_codes.h"

namespace warpconvnet {
namespace cute_gemm {

/// Launch a CuTe GEMM with AD gather-scatter
///
/// Computes: D[out_map, :] = alpha * A[in_map, :] @ B + beta * C[out_map, :]
///
/// A is (M_A, K) row-major, gathered by in_map
/// B is (K, N) row-major, dense
/// C is (M_C, N) row-major, gathered by out_map
/// D is (M_C, N) row-major, scattered by out_map
template <typename ElementInput, typename TileConfig, typename ElementOutput = float>
int launch_cute_gemm_ad_gather_scatter(const void *ptr_A,
                                       const void *ptr_B,
                                       const void *ptr_C,
                                       void *ptr_D,
                                       const int *in_map,
                                       const int *out_map,
                                       int gather_size,
                                       int M_A,
                                       int K,
                                       int N,
                                       int M_C,
                                       float alpha,
                                       float beta,
                                       cudaStream_t stream = 0) {
  using Kernel = CuteGemmKernel<TileConfig, ElementOutput>;
  constexpr int TileM = cute::size<0>(typename TileConfig::TileShape{});
  constexpr int TileN = cute::size<1>(typename TileConfig::TileShape{});

  dim3 grid((gather_size + TileM - 1) / TileM, (N + TileN - 1) / TileN, 1);
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  // Set max dynamic shared memory if needed
  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(
        cute_gemm_kernel_entry<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_kernel_entry<Kernel>
      <<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const ElementInput *>(ptr_A),
          reinterpret_cast<const ElementInput *>(ptr_B),
          reinterpret_cast<const ElementOutput *>(ptr_C),
          reinterpret_cast<ElementOutput *>(ptr_D),
          in_map, out_map,
          gather_size, N, K, alpha, beta);

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

/// Launch a CuTe GEMM with TrAB gather
///
/// Computes: D[k, n] = alpha * A[idx_a]^T @ B[idx_b] + beta * C[k, n]
///
/// A is (M_A, K) row-major, gathered by indices_a
/// B is (M_B, N) row-major, gathered by indices_b
/// D is (K, N) dense output
template <typename ElementInput, typename TileConfig, typename ElementOutput = float>
int launch_cute_gemm_trAB_gather(const void *ptr_A,
                                 const void *ptr_B,
                                 const void *ptr_C,
                                 void *ptr_D,
                                 const int *indices_a,
                                 const int *indices_b,
                                 int gather_size,
                                 int M_A,
                                 int K,
                                 int M_B,
                                 int N,
                                 float alpha,
                                 float beta,
                                 cudaStream_t stream = 0) {
  using Kernel = CuteGemmTrABKernel<TileConfig, ElementOutput>;
  constexpr int TileM = cute::size<0>(typename TileConfig::TileShape{});  // tiles K
  constexpr int TileN = cute::size<1>(typename TileConfig::TileShape{});  // tiles N

  dim3 grid((K + TileM - 1) / TileM, (N + TileN - 1) / TileN, 1);
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(
        cute_gemm_trAB_kernel_entry<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_trAB_kernel_entry<Kernel>
      <<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const ElementInput *>(ptr_A),
          reinterpret_cast<const ElementInput *>(ptr_B),
          reinterpret_cast<const ElementOutput *>(ptr_C),
          reinterpret_cast<ElementOutput *>(ptr_D),
          indices_a, indices_b,
          K, N, gather_size, alpha, beta);

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

}  // namespace cute_gemm
}  // namespace warpconvnet
