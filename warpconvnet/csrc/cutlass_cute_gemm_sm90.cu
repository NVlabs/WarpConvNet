// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SM90 (Hopper) WGMMA GEMM compilation unit.
// Explicit template instantiations for SM90 WGMMA-based gather/scatter GEMM
// kernels using tile configs from cute_gemm_config_sm90.h.

#if defined(WARPCONVNET_SM90_ENABLED)

#include <cuda_runtime.h>

#include "include/cute_gemm_config_sm90.h"
#include "include/cute_gemm_kernel_sm90.h"
#include "include/gemm_error_codes.h"

namespace warpconvnet {
namespace cute_gemm {

// ============================================================================
// SM90 AD gather-scatter launcher
// ============================================================================

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

  dim3 grid((gather_size + TileM - 1) / TileM, (N + TileN - 1) / TileN, 1);
  constexpr size_t smem_size = Kernel::SharedStorageSize;

  // SM90 kernels with WGMMA typically require > 48KB smem
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
