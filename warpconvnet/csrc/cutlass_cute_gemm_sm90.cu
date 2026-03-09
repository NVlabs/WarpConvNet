// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SM90 (Hopper) WGMMA GEMM compilation unit.
// Explicit template instantiations for SM90 WGMMA-based gather/scatter GEMM
// kernels using tile configs from cute_gemm_config_sm90.h.

#if defined(WARPCONVNET_SM90_ENABLED)

#include "cute/tensor.hpp"  // Must come before other CuTe headers for CUDA 12.8+ compat
#include <cuda.h>           // CUtensorMap, cuTensorMapEncodeTiled
#include <cuda_runtime.h>

#include "include/cute_gemm_config_sm90.h"
#include "include/cute_gemm_kernel_sm90.h"
#include "include/gemm_error_codes.h"

namespace warpconvnet {
namespace cute_gemm {

// ============================================================================
// SM90 AD gather-scatter launcher
// ============================================================================

/// Create a CUtensorMap TMA descriptor for dense matrix B.
/// B is (K_dim, N) row-major: B[k, n] = ptr[k * N + n], N contiguous.
/// The TMA descriptor encodes a 2D tile of size (tN, tK) in (N, K) order,
/// matching the smem layout (tN, tK) with 128-byte swizzle.
/// TMA box width in elements: 128-byte swizzle requires exactly 128 bytes per row.
template <typename ElementInput>
static constexpr int kTmaBoxN = 128 / sizeof(ElementInput);  // 64 for fp16/bf16

template <typename ElementInput, int tN, int tK>
CUtensorMap create_tma_desc_B(const void *ptr_B, int K_dim, int N) {
  CUtensorMap desc{};

  // TMA 2D tensor: globalDim[0] = N (fast dim), globalDim[1] = K (slow dim)
  uint64_t globalDim[2] = {static_cast<uint64_t>(N), static_cast<uint64_t>(K_dim)};

  // Global strides in bytes (stride of the slow dimension; fast dim stride is implicit = elem size)
  uint64_t globalStride[1] = {static_cast<uint64_t>(N) * sizeof(ElementInput)};

  // Box dimensions: fast dim must be exactly 128 bytes for CU_TENSOR_MAP_SWIZZLE_128B.
  // For larger tN, we issue multiple TMA loads (tN / kTmaBoxN) in the kernel.
  constexpr int box_n = kTmaBoxN<ElementInput>;  // 64 for fp16/bf16
  static_assert(tN % box_n == 0, "tN must be a multiple of TMA box width (128B / elem_size)");
  uint32_t boxDim[2] = {static_cast<uint32_t>(box_n), static_cast<uint32_t>(tK)};

  // Element strides within the box (1 = contiguous)
  uint32_t elemStride[2] = {1, 1};

  // Determine CUtensorMapDataType
  CUtensorMapDataType dataType;
  if constexpr (sizeof(ElementInput) == 2) {
    dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (sizeof(ElementInput) == 4) {
    dataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    dataType = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }

  // 128-byte swizzle to match GMMA Layout_MN_SW128_Atom
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

  // L2 promotion: none
  CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE;

  // OOB fill: zero (matches our manual cp.async zero-fill behavior)
  CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;

  // Interleave: none
  CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;

  CUresult result = cuTensorMapEncodeTiled(
      &desc, dataType, 2, const_cast<void *>(ptr_B), globalDim, globalStride, boxDim, elemStride,
      interleave, swizzle, l2promo, oobFill);

  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %d (N=%lu, K=%lu, tN=%d, tK=%d)\n",
            (int)result, globalDim[0], globalDim[1], tN, tK);
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
    // TMA path: create descriptor and launch TMA kernel entry
    CUtensorMap tma_desc = create_tma_desc_B<ElementInput, TileN, TileK>(ptr_B, K, N);

    if (smem_size > 48 * 1024) {
      auto err = cudaFuncSetAttribute(cute_gemm_sm90_kernel_entry_tma<Kernel>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);
      if (err != cudaSuccess) {
        return static_cast<int>(gemm::GemmStatus::kErrorKernelInitialization);
      }
    }

    cute_gemm_sm90_kernel_entry_tma<Kernel><<<grid, Kernel::MaxThreadsPerBlock, smem_size, stream>>>(
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
    // cp.async path: original kernel entry
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

// ============================================================================
// Type-erased entry points called by pybind11 dispatch
// ============================================================================

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

// ============================================================================
// Explicit instantiations
// ============================================================================

// Macro instantiates AD gather-scatter for one (InputType, OutputType, TileTag).
#define INSTANTIATE_SM90_AD_GS(ElemIn, ElemOut, TileTag)                                          \
  template int run_cute_gemm_sm90_ad_gather_scatter<ElemIn, gemm::TileTag, ElemOut>(const void *, \
                                                                                    const void *, \
                                                                                    const void *, \
                                                                                    void *,       \
                                                                                    const int *,  \
                                                                                    const int *,  \
                                                                                    int,          \
                                                                                    int,          \
                                                                                    int,          \
                                                                                    int,          \
                                                                                    int,          \
                                                                                    float,        \
                                                                                    float);

// Instantiate for all dtype pairs for a given tile.
#define INSTANTIATE_SM90_ALL_DTYPES(TileTag)              \
  INSTANTIATE_SM90_AD_GS(cutlass::half_t, float, TileTag) \
  INSTANTIATE_SM90_AD_GS(cutlass::half_t, cutlass::half_t, TileTag)

#ifndef DISABLE_BFLOAT16
#define INSTANTIATE_SM90_ALL_DTYPES_BF16(TileTag)             \
  INSTANTIATE_SM90_AD_GS(cutlass::bfloat16_t, float, TileTag) \
  INSTANTIATE_SM90_AD_GS(cutlass::bfloat16_t, cutlass::bfloat16_t, TileTag)
#else
#define INSTANTIATE_SM90_ALL_DTYPES_BF16(TileTag)
#endif

// --- SM90 tiles ---
INSTANTIATE_SM90_ALL_DTYPES(SM90_Tile64x64x64)
INSTANTIATE_SM90_ALL_DTYPES(SM90_Tile64x128x64)
INSTANTIATE_SM90_ALL_DTYPES(SM90_Tile128x128x64)
INSTANTIATE_SM90_ALL_DTYPES(SM90_Tile128x256x64)
INSTANTIATE_SM90_ALL_DTYPES(SM90_Tile256x128x64)

INSTANTIATE_SM90_ALL_DTYPES_BF16(SM90_Tile64x64x64)
INSTANTIATE_SM90_ALL_DTYPES_BF16(SM90_Tile64x128x64)
INSTANTIATE_SM90_ALL_DTYPES_BF16(SM90_Tile128x128x64)
INSTANTIATE_SM90_ALL_DTYPES_BF16(SM90_Tile128x256x64)
INSTANTIATE_SM90_ALL_DTYPES_BF16(SM90_Tile256x128x64)

#undef INSTANTIATE_SM90_ALL_DTYPES
#undef INSTANTIATE_SM90_ALL_DTYPES_BF16
#undef INSTANTIATE_SM90_AD_GS

}  // namespace cute_gemm
}  // namespace warpconvnet

#else  // !WARPCONVNET_SM90_ENABLED

// Empty translation unit when SM90 MMA is not supported.
namespace warpconvnet {
namespace cute_gemm {
void sm90_gemm_stub() {}
}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
