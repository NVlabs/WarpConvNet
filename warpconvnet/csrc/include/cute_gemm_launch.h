// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host-side launcher for CuTe GEMM with gather/scatter.
// Uses raw pointers + gather indices — the kernel handles gather internally.

#pragma once

#include <cuda_runtime.h>

#include "cute_gemm_grouped_kernel.h"
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
        cute_gemm_kernel_entry<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_kernel_entry<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
      reinterpret_cast<const ElementInput *>(ptr_A),
      reinterpret_cast<const ElementInput *>(ptr_B),
      reinterpret_cast<const ElementOutput *>(ptr_C),
      reinterpret_cast<ElementOutput *>(ptr_D),
      in_map,
      out_map,
      gather_size,
      N,
      K,
      alpha,
      beta);

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
    auto err = cudaFuncSetAttribute(cute_gemm_trAB_kernel_entry<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_trAB_kernel_entry<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
      reinterpret_cast<const ElementInput *>(ptr_A),
      reinterpret_cast<const ElementInput *>(ptr_B),
      reinterpret_cast<const ElementOutput *>(ptr_C),
      reinterpret_cast<ElementOutput *>(ptr_D),
      indices_a,
      indices_b,
      K,
      N,
      gather_size,
      alpha,
      beta);

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

// ============================================================================
// Grouped GEMM launcher — fused multi-offset sparse convolution
// ============================================================================

/// Launch a fused grouped CuTe GEMM with AD gather-scatter.
///
/// All groups share ptr_A (input features) and ptr_D (output, zero-initialized).
/// Each group has its own weight pointer (ptr_B_array[g]), gather indices
/// (in_map + map_offsets[g]), and scatter indices (out_map + map_offsets[g]).
/// Output is accumulated via atomicAdd since multiple groups may write to
/// overlapping output rows.
template <typename ElementInput, typename TileConfig, typename ElementOutput = float>
int launch_cute_gemm_grouped_ad_gather_scatter(const void *ptr_A,
                                               void *ptr_D,
                                               const int *in_map,
                                               const int *out_map,
                                               const GroupedGemmParams &params,
                                               int total_m_tiles,
                                               int K,
                                               int N,
                                               float alpha,
                                               cudaStream_t stream = 0) {
  using Kernel = CuteGemmGroupedKernel<TileConfig, ElementOutput>;
  constexpr int TileN = cute::size<1>(typename TileConfig::TileShape{});
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  if (total_m_tiles == 0) {
    return static_cast<int>(gemm::GemmStatus::kSuccess);
  }

  dim3 grid(total_m_tiles, (N + TileN - 1) / TileN, 1);

  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(cute_gemm_grouped_kernel_entry<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_grouped_kernel_entry<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
      reinterpret_cast<const ElementInput *>(ptr_A),
      reinterpret_cast<ElementOutput *>(ptr_D),
      in_map,
      out_map,
      params,
      N,
      K,
      alpha);

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

// ============================================================================
// Grouped TrAB launcher — fused multi-offset weight gradient
// ============================================================================

/// Launch a fused grouped CuTe TrAB GEMM for weight gradient.
///
/// All groups share ptr_A (input features) and ptr_B (grad_output).
/// Each group has its own output pointer (ptr_D_array[g]), gather indices
/// (idx_a + map_offsets[g], idx_b + map_offsets[g]), and gather_size.
/// Grid: (K_tiles, N_tiles, num_groups).
template <typename ElementInput, typename TileConfig, typename ElementOutput = float>
int launch_cute_gemm_grouped_trAB_gather(const void *ptr_A,
                                         const void *ptr_B,
                                         const int *idx_a,
                                         const int *idx_b,
                                         const GroupedTrABGemmParams &params,
                                         int K_dim,
                                         int N,
                                         float alpha,
                                         cudaStream_t stream = 0) {
  using Kernel = CuteGemmGroupedTrABKernel<TileConfig, ElementOutput>;
  constexpr int TileM = cute::size<0>(typename TileConfig::TileShape{});  // tiles K_dim
  constexpr int TileN = cute::size<1>(typename TileConfig::TileShape{});  // tiles N
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  if (params.num_groups == 0) {
    return static_cast<int>(gemm::GemmStatus::kSuccess);
  }

  dim3 grid((K_dim + TileM - 1) / TileM, (N + TileN - 1) / TileN, params.num_groups);

  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(cute_gemm_grouped_trAB_kernel_entry<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_grouped_trAB_kernel_entry<Kernel>
      <<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const ElementInput *>(ptr_A),
          reinterpret_cast<const ElementInput *>(ptr_B),
          idx_a,
          idx_b,
          params,
          K_dim,
          N,
          alpha);

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

}  // namespace cute_gemm
}  // namespace warpconvnet

// ============================================================================
// SM90 Grouped GEMM launcher — fused multi-offset sparse convolution (WGMMA)
// ============================================================================

#if defined(WARPCONVNET_SM90_ENABLED)

// Include SM90 grouped kernel OUTSIDE namespace to avoid nested namespace issues
#include "cute_gemm_grouped_kernel_sm90.h"

namespace warpconvnet {
namespace cute_gemm {

/// Launch a fused grouped CuTe GEMM with AD gather-scatter using SM90 WGMMA.
///
/// Same interface as the SM80 grouped launcher, but uses the SM90 WGMMA kernel
/// with GMMA-compatible smem layouts and warp-group synchronization.
template <typename ElementInput, typename TileConfig, typename ElementOutput = float>
int launch_cute_gemm_grouped_ad_gather_scatter_sm90(const void *ptr_A,
                                                    void *ptr_D,
                                                    const int *in_map,
                                                    const int *out_map,
                                                    const GroupedGemmParams &params,
                                                    int total_m_tiles,
                                                    int K,
                                                    int N,
                                                    float alpha,
                                                    cudaStream_t stream = 0) {
  using Kernel = CuteGemmGroupedKernelSm90<TileConfig, ElementOutput>;
  constexpr int TileN = cute::size<1>(typename TileConfig::TileShape{});
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  if (total_m_tiles == 0) {
    return static_cast<int>(gemm::GemmStatus::kSuccess);
  }

  dim3 grid(total_m_tiles, (N + TileN - 1) / TileN, 1);

  if (smem_size > 48 * 1024) {
    auto err = cudaFuncSetAttribute(cute_gemm_grouped_kernel_sm90_entry<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);
    if (err != cudaSuccess) {
      return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
    }
  }

  cute_gemm_grouped_kernel_sm90_entry<Kernel>
      <<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
          reinterpret_cast<const ElementInput *>(ptr_A),
          reinterpret_cast<ElementOutput *>(ptr_D),
          in_map,
          out_map,
          params,
          N,
          K,
          alpha);

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
