// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SM90 (Hopper) WGMMA GEMM gather-scatter wrapper templates.
// Template body header for run_cute_gemm_sm90_ad_gather_scatter and its
// dependencies (launch_cute_gemm_sm90_ad_gather_scatter, create_tma_desc_B).
// Included by warpconvnet/csrc/cutlass_cute_gemm_sm90.cu (existing
// instantiations) and by warpgemm-generated offset_gemm TUs that invoke
// INSTANTIATE_SM90_AD_GS for the stable tier.

#pragma once

#if defined(WARPCONVNET_SM90_ENABLED)

#include <cuda.h>  // CUtensorMap, cuTensorMapEncodeTiled
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cute_gemm_config_sm90.h"
#include "cute_gemm_kernel_sm90.h"
#include "gemm_error_codes.h"

namespace warpconvnet {
namespace cute_gemm {

/// TMA box width in elements: 128-byte swizzle requires exactly 128 bytes per row.
template <typename ElementInput>
static constexpr int kTmaBoxN = 128 / sizeof(ElementInput);  // 64 for fp16/bf16

/// Create a CUtensorMap TMA descriptor for dense matrix B.
/// B is (K_dim, N) row-major: B[k, n] = ptr[k * N + n], N contiguous.
/// The TMA descriptor encodes a 2D tile of size (tN, tK) in (N, K) order,
/// matching the smem layout (tN, tK) with 128-byte swizzle.
template <typename ElementInput, int tN, int tK>
CUtensorMap create_tma_desc_B(const void *ptr_B, int K_dim, int N) {
  CUtensorMap desc{};

  // TMA (cuTensorMapEncodeTiled) requires a 16-byte-aligned global base and a
  // 16-byte-aligned row stride. Unlike the cp.async B loader, there is NO in-kernel
  // scalar fallback for TMA — the alignment is a hardware descriptor constraint. The
  // dispatch layer therefore guarantees an aligned weight before any SM90 TMA kernel
  // runs (SM90 algos are excluded from the GEMM self-handling set, so a misaligned
  // DeepSpeed/ZeRO weight view is cloned at the Python boundary). Fail loudly if that
  // invariant is ever violated, rather than encoding a bad descriptor that NaNs later.
  if ((reinterpret_cast<uintptr_t>(ptr_B) | (uintptr_t)(N * (int)sizeof(ElementInput))) & 15u) {
    fprintf(stderr,
            "create_tma_desc_B: B base/stride not 16B-aligned (ptr%%16=%d, N*sizeof=%d). "
            "SM90 TMA requires alignment; the dispatch layer must clone misaligned weights.\n",
            (int)(reinterpret_cast<uintptr_t>(ptr_B) & 15u),
            (int)(N * (int)sizeof(ElementInput)));
  }

  uint64_t globalDim[2] = {static_cast<uint64_t>(N), static_cast<uint64_t>(K_dim)};
  uint64_t globalStride[1] = {static_cast<uint64_t>(N) * sizeof(ElementInput)};

  constexpr int box_n = kTmaBoxN<ElementInput>;
  static_assert(tN % box_n == 0, "tN must be a multiple of TMA box width (128B / elem_size)");
  uint32_t boxDim[2] = {static_cast<uint32_t>(box_n), static_cast<uint32_t>(tK)};
  uint32_t elemStride[2] = {1, 1};

  CUtensorMapDataType dataType;
  if constexpr (sizeof(ElementInput) == 2) {
    dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (sizeof(ElementInput) == 4) {
    dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    dataType = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }

  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
  CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;

  CUresult result = cuTensorMapEncodeTiled(&desc,
                                           dataType,
                                           2,
                                           const_cast<void *>(ptr_B),
                                           globalDim,
                                           globalStride,
                                           boxDim,
                                           elemStride,
                                           interleave,
                                           swizzle,
                                           l2promo,
                                           oobFill);

  if (result != CUDA_SUCCESS) {
    fprintf(stderr,
            "cuTensorMapEncodeTiled failed: %d (N=%lu, K=%lu, tN=%d, tK=%d)\n",
            (int)result,
            globalDim[0],
            globalDim[1],
            tN,
            tK);
  }
  return desc;
}

template <typename ElementInput, typename TileConfig, typename ElementOutput = float>
int launch_cute_gemm_sm90_ad_gather_scatter(const void *ptr_A,
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
  using Kernel = CuteGemmKernelSm90<TileConfig, ElementOutput>;
  constexpr int TileM = cute::size<0>(typename TileConfig::TileShape{});
  constexpr int TileN = cute::size<1>(typename TileConfig::TileShape{});
  constexpr int TileK = cute::size<2>(typename TileConfig::TileShape{});

  dim3 grid((gather_size + TileM - 1) / TileM, (N + TileN - 1) / TileN, 1);
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  if constexpr (Kernel::UseTmaLoadB) {
    CUtensorMap tma_desc = create_tma_desc_B<ElementInput, TileN, TileK>(ptr_B, K, N);

    if (smem_size > 48 * 1024) {
      auto err = cudaFuncSetAttribute(cute_gemm_sm90_kernel_entry_tma<Kernel>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);
      if (err != cudaSuccess) {
        return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
      }
    }

    cute_gemm_sm90_kernel_entry_tma<Kernel>
        <<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
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
            beta,
            tma_desc);
  } else {
    if (smem_size > 48 * 1024) {
      auto err = cudaFuncSetAttribute(cute_gemm_sm90_kernel_entry<Kernel>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);
      if (err != cudaSuccess) {
        return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
      }
    }

    cute_gemm_sm90_kernel_entry<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
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
  }

  auto err = cudaGetLastError();
  return err == cudaSuccess ? static_cast<int>(gemm::GemmStatus::kSuccess)
                            : static_cast<int>(gemm::GemmStatus::kErrorKernelExecution);
}

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_sm90_ad_gather_scatter(const void *a,
                                         const void *b,
                                         const void *c,
                                         void *d,
                                         const int *idx_a,
                                         const int *idx_d,
                                         int M_A,
                                         int K,
                                         int N,
                                         int M_C,
                                         int idx_size,
                                         float alpha,
                                         float beta) {
  using Config = CuteTileConfig<ElementInput, TileTag>;
  return launch_cute_gemm_sm90_ad_gather_scatter<ElementInput, Config, ElementOutput>(
      a, b, c, d, idx_a, idx_d, idx_size, M_A, K, N, M_C, alpha, beta);
}

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
