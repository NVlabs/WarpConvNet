// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SM90 WGMMA-based grouped GEMM instantiations for sparse convolution.
// Instantiates the SM90 grouped kernel for common tile sizes and data types.

#include "cutlass/arch/config.h"

#if defined(WARPCONVNET_SM90_ENABLED)

#include "include/cute_gemm_launch.h"

namespace warpconvnet {
namespace cute_gemm {

// ============================================================================
// SM90 Staged grouped AD gather-scatter
// ============================================================================

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_grouped_ad_gather_scatter_sm90_staged(const void *a,
                                                        void *d,
                                                        const int *in_map,
                                                        const int *out_map,
                                                        const GroupedGemmParams &params,
                                                        int total_m_tiles,
                                                        int K,
                                                        int N,
                                                        float alpha,
                                                        int num_stages,
                                                        bool use_cp_async,
                                                        bool use_atomic) {
  using Base = CuteTileConfig<ElementInput, TileTag>;

#define DISPATCH_SM90_STAGED_GROUPED(S, CP)                                                      \
  {                                                                                              \
    using Config = CuteTileConfigOverride<Base, S, CP>;                                          \
    return launch_cute_gemm_grouped_ad_gather_scatter_sm90<ElementInput, Config, ElementOutput>( \
        a, d, in_map, out_map, params, total_m_tiles, K, N, alpha, use_atomic);                  \
  }

  if (num_stages == 2 && !use_cp_async) DISPATCH_SM90_STAGED_GROUPED(2, false)
  if (num_stages == 2 && use_cp_async) DISPATCH_SM90_STAGED_GROUPED(2, true)
  if (num_stages == 3 && !use_cp_async) DISPATCH_SM90_STAGED_GROUPED(3, false)
  if (num_stages == 3 && use_cp_async) DISPATCH_SM90_STAGED_GROUPED(3, true)
  if (num_stages == 4 && !use_cp_async) DISPATCH_SM90_STAGED_GROUPED(4, false)
  if (num_stages == 4 && use_cp_async) DISPATCH_SM90_STAGED_GROUPED(4, true)

#undef DISPATCH_SM90_STAGED_GROUPED
  return static_cast<int>(gemm::GemmStatus::kErrorUnsupportedConfig);
}

// ============================================================================
// Explicit instantiations
// ============================================================================

#define INSTANTIATE_SM90_STAGED_GROUPED(ElemIn, ElemOut, TileTag)                      \
  template int                                                                         \
  run_cute_gemm_grouped_ad_gather_scatter_sm90_staged<ElemIn, gemm::TileTag, ElemOut>( \
      const void *,                                                                    \
      void *,                                                                          \
      const int *,                                                                     \
      const int *,                                                                     \
      const GroupedGemmParams &,                                                       \
      int,                                                                             \
      int,                                                                             \
      int,                                                                             \
      float,                                                                           \
      int,                                                                             \
      bool,                                                                            \
      bool);

// Primary tiles: SM90_Tile128x128x64, SM90_Tile64x128x64, SM90_Tile256x128x64
// Grouped kernel always accumulates in float via atomicAdd

// FP16 -> float output
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, float, SM90_Tile64x64x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, float, SM90_Tile128x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, float, SM90_Tile64x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, float, SM90_Tile256x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, float, SM90_Tile128x256x64)
// FP16 -> FP16 output
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, cutlass::half_t, SM90_Tile64x64x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, cutlass::half_t, SM90_Tile128x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, cutlass::half_t, SM90_Tile64x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, cutlass::half_t, SM90_Tile256x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::half_t, cutlass::half_t, SM90_Tile128x256x64)

// BF16 -> float output
#ifndef DISABLE_BFLOAT16
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, float, SM90_Tile64x64x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, float, SM90_Tile128x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, float, SM90_Tile64x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, float, SM90_Tile256x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, float, SM90_Tile128x256x64)
// BF16 -> BF16 output
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile64x64x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile128x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile64x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile256x128x64)
INSTANTIATE_SM90_STAGED_GROUPED(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile128x256x64)
#endif  // DISABLE_BFLOAT16

#undef INSTANTIATE_SM90_STAGED_GROUPED

// ============================================================================
// Simple entry point (no num_stages/use_cp_async — uses defaults from tile config)
// Called by pybind11 dispatch in gemm_bindings.cpp
// ============================================================================

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_sm90_grouped_ad_gather_scatter(const void *a,
                                                 void *d,
                                                 const int *in_map,
                                                 const int *out_map,
                                                 const GroupedGemmParams &params,
                                                 int total_m_tiles,
                                                 int K,
                                                 int N,
                                                 float alpha,
                                                 bool use_atomic,
                                                 bool use_cp_async) {
  constexpr int default_stages = CuteTileConfig<ElementInput, TileTag>::NumStages;
  return run_cute_gemm_grouped_ad_gather_scatter_sm90_staged<ElementInput, TileTag, ElementOutput>(
      a, d, in_map, out_map, params, total_m_tiles, K, N, alpha, default_stages, use_cp_async, use_atomic);
}

#define INSTANTIATE_SM90_GROUPED_SIMPLE(ElemIn, ElemOut, TileTag)                            \
  template int run_cute_gemm_sm90_grouped_ad_gather_scatter<ElemIn, gemm::TileTag, ElemOut>( \
      const void *,                                                                          \
      void *,                                                                                \
      const int *,                                                                           \
      const int *,                                                                           \
      const GroupedGemmParams &,                                                             \
      int,                                                                                   \
      int,                                                                                   \
      int,                                                                                   \
      float,                                                                                 \
      bool,                                                                                  \
      bool);

INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, float, SM90_Tile64x64x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, float, SM90_Tile128x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, float, SM90_Tile64x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, float, SM90_Tile256x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, float, SM90_Tile128x256x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, cutlass::half_t, SM90_Tile64x64x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, cutlass::half_t, SM90_Tile128x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, cutlass::half_t, SM90_Tile64x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, cutlass::half_t, SM90_Tile256x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::half_t, cutlass::half_t, SM90_Tile128x256x64)

#ifndef DISABLE_BFLOAT16
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, float, SM90_Tile64x64x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, float, SM90_Tile128x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, float, SM90_Tile64x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, float, SM90_Tile256x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, float, SM90_Tile128x256x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile64x64x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile128x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile64x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile256x128x64)
INSTANTIATE_SM90_GROUPED_SIMPLE(cutlass::bfloat16_t, cutlass::bfloat16_t, SM90_Tile128x256x64)
#endif

#undef INSTANTIATE_SM90_GROUPED_SIMPLE

}  // namespace cute_gemm
}  // namespace warpconvnet

#else  // !WARPCONVNET_SM90_ENABLED

// Empty translation unit when SM90 MMA is not supported.
namespace warpconvnet {
namespace cute_gemm {
void sm90_grouped_gemm_stub() {}
}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
