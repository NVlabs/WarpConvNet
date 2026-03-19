// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Staged CuTe GEMM variants: autotunable NumStages and UseCpAsyncGatherA.
// Only the 4 common tK=32 tiles (mma_tile 0-3) are instantiated with staged overrides.

#if defined(WARPCONVNET_SM80_ENABLED)

#include "include/cute_gemm_launch.h"

namespace warpconvnet {
namespace cute_gemm {

// ============================================================================
// Staged AD gather-scatter
// ============================================================================

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_ad_gather_scatter_staged(const void *a,
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
                                           float beta,
                                           int num_stages,
                                           bool use_cp_async) {
  using Base = CuteTileConfig<ElementInput, TileTag>;

#define DISPATCH_STAGED_AD(S, CP)                                                   \
  {                                                                                 \
    using Config = CuteTileConfigOverride<Base, S, CP>;                             \
    return launch_cute_gemm_ad_gather_scatter<ElementInput, Config, ElementOutput>( \
        a, b, c, d, idx_a, idx_d, idx_size, M_A, K, N, M_C, alpha, beta);           \
  }

  if (num_stages == 2 && !use_cp_async) DISPATCH_STAGED_AD(2, false)
  if (num_stages == 2 && use_cp_async) DISPATCH_STAGED_AD(2, true)
  if (num_stages == 3 && !use_cp_async) DISPATCH_STAGED_AD(3, false)
  if (num_stages == 3 && use_cp_async) DISPATCH_STAGED_AD(3, true)
  if (num_stages == 4 && !use_cp_async) DISPATCH_STAGED_AD(4, false)
  if (num_stages == 4 && use_cp_async) DISPATCH_STAGED_AD(4, true)

#undef DISPATCH_STAGED_AD
  return static_cast<int>(gemm::GemmStatus::kErrorUnsupportedConfig);
}

// ============================================================================
// Staged TrAB gather
// ============================================================================

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_trAB_gather_staged(const void *a,
                                     const void *b,
                                     const void *c,
                                     void *d,
                                     const int *idx_a,
                                     const int *idx_b,
                                     int M_A,
                                     int K,
                                     int K_B,
                                     int N,
                                     int idx_size,
                                     float alpha,
                                     float beta,
                                     int num_stages,
                                     bool use_cp_async) {
  using Base = CuteTileConfig<ElementInput, TileTag>;

#define DISPATCH_STAGED_TRAB(S, CP)                                           \
  {                                                                           \
    using Config = CuteTileConfigOverride<Base, S, CP>;                       \
    return launch_cute_gemm_trAB_gather<ElementInput, Config, ElementOutput>( \
        a, b, c, d, idx_a, idx_b, idx_size, M_A, K, K_B, N, alpha, beta);     \
  }

  if (num_stages == 2 && !use_cp_async) DISPATCH_STAGED_TRAB(2, false)
  if (num_stages == 2 && use_cp_async) DISPATCH_STAGED_TRAB(2, true)
  if (num_stages == 3 && !use_cp_async) DISPATCH_STAGED_TRAB(3, false)
  if (num_stages == 3 && use_cp_async) DISPATCH_STAGED_TRAB(3, true)
  if (num_stages == 4 && !use_cp_async) DISPATCH_STAGED_TRAB(4, false)
  if (num_stages == 4 && use_cp_async) DISPATCH_STAGED_TRAB(4, true)

#undef DISPATCH_STAGED_TRAB
  return static_cast<int>(gemm::GemmStatus::kErrorUnsupportedConfig);
}

// ============================================================================
// Staged grouped AD gather-scatter
// ============================================================================

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_grouped_ad_gather_scatter_staged(const void *a,
                                                   void *d,
                                                   const int *in_map,
                                                   const int *out_map,
                                                   const GroupedGemmParams &params,
                                                   int total_m_tiles,
                                                   int K,
                                                   int N,
                                                   float alpha,
                                                   int num_stages,
                                                   bool use_cp_async) {
  using Base = CuteTileConfig<ElementInput, TileTag>;

#define DISPATCH_STAGED_GROUPED(S, CP)                                                      \
  {                                                                                         \
    using Config = CuteTileConfigOverride<Base, S, CP>;                                     \
    return launch_cute_gemm_grouped_ad_gather_scatter<ElementInput, Config, ElementOutput>( \
        a, d, in_map, out_map, params, total_m_tiles, K, N, alpha);                         \
  }

  if (num_stages == 2 && !use_cp_async) DISPATCH_STAGED_GROUPED(2, false)
  if (num_stages == 2 && use_cp_async) DISPATCH_STAGED_GROUPED(2, true)
  if (num_stages == 3 && !use_cp_async) DISPATCH_STAGED_GROUPED(3, false)
  if (num_stages == 3 && use_cp_async) DISPATCH_STAGED_GROUPED(3, true)
  if (num_stages == 4 && !use_cp_async) DISPATCH_STAGED_GROUPED(4, false)
  if (num_stages == 4 && use_cp_async) DISPATCH_STAGED_GROUPED(4, true)

#undef DISPATCH_STAGED_GROUPED
  return static_cast<int>(gemm::GemmStatus::kErrorUnsupportedConfig);
}

// ============================================================================
// Explicit instantiations — 4 common tK=32 tiles × 4 dtype combos
// ============================================================================

#define INSTANTIATE_STAGED_AD_GS(ElemIn, ElemOut, TileTag)                             \
  template int run_cute_gemm_ad_gather_scatter_staged<ElemIn, gemm::TileTag, ElemOut>( \
      const void *,                                                                    \
      const void *,                                                                    \
      const void *,                                                                    \
      void *,                                                                          \
      const int *,                                                                     \
      const int *,                                                                     \
      int,                                                                             \
      int,                                                                             \
      int,                                                                             \
      int,                                                                             \
      int,                                                                             \
      float,                                                                           \
      float,                                                                           \
      int,                                                                             \
      bool);

#define INSTANTIATE_STAGED_TRAB(ElemIn, ElemOut, TileTag)                                     \
  template int run_cute_gemm_trAB_gather_staged<ElemIn, gemm::TileTag, ElemOut>(const void *, \
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
                                                                                float,        \
                                                                                int,          \
                                                                                bool);

#define INSTANTIATE_STAGED_GROUPED(ElemIn, ElemOut, TileTag)                                   \
  template int run_cute_gemm_grouped_ad_gather_scatter_staged<ElemIn, gemm::TileTag, ElemOut>( \
      const void *,                                                                            \
      void *,                                                                                  \
      const int *,                                                                             \
      const int *,                                                                             \
      const GroupedGemmParams &,                                                               \
      int,                                                                                     \
      int,                                                                                     \
      int,                                                                                     \
      float,                                                                                   \
      int,                                                                                     \
      bool);

#define INSTANTIATE_STAGED_ALL_DTYPES(TileTag)                                \
  INSTANTIATE_STAGED_AD_GS(cutlass::half_t, float, TileTag)                   \
  INSTANTIATE_STAGED_AD_GS(cutlass::half_t, cutlass::half_t, TileTag)         \
  INSTANTIATE_STAGED_AD_GS(cutlass::bfloat16_t, float, TileTag)               \
  INSTANTIATE_STAGED_AD_GS(cutlass::bfloat16_t, cutlass::bfloat16_t, TileTag) \
  INSTANTIATE_STAGED_TRAB(cutlass::half_t, float, TileTag)                    \
  INSTANTIATE_STAGED_TRAB(cutlass::half_t, cutlass::half_t, TileTag)          \
  INSTANTIATE_STAGED_TRAB(cutlass::bfloat16_t, float, TileTag)                \
  INSTANTIATE_STAGED_TRAB(cutlass::bfloat16_t, cutlass::bfloat16_t, TileTag)  \
  INSTANTIATE_STAGED_GROUPED(cutlass::half_t, float, TileTag)                 \
  INSTANTIATE_STAGED_GROUPED(cutlass::half_t, cutlass::half_t, TileTag)       \
  INSTANTIATE_STAGED_GROUPED(cutlass::bfloat16_t, float, TileTag)             \
  INSTANTIATE_STAGED_GROUPED(cutlass::bfloat16_t, cutlass::bfloat16_t, TileTag)

INSTANTIATE_STAGED_ALL_DTYPES(Tile128x128x32)
INSTANTIATE_STAGED_ALL_DTYPES(Tile128x64x32)
INSTANTIATE_STAGED_ALL_DTYPES(Tile64x128x32)
INSTANTIATE_STAGED_ALL_DTYPES(Tile64x64x32)

#undef INSTANTIATE_STAGED_ALL_DTYPES
#undef INSTANTIATE_STAGED_AD_GS
#undef INSTANTIATE_STAGED_TRAB
#undef INSTANTIATE_STAGED_GROUPED

}  // namespace cute_gemm
}  // namespace warpconvnet

#else  // !WARPCONVNET_SM80_ENABLED

// Empty translation unit when SM80 CUTLASS is not available.
namespace warpconvnet {
namespace cute_gemm {
void cute_gemm_staged_stub() {}
}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM80_ENABLED
