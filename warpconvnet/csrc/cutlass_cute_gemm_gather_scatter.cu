// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CuTe 3.x GEMM with gather/scatter via layout composition.
// Explicit template instantiations for all (element type × tile) combinations.

#include "include/cute_gemm_launch.h"

namespace warpconvnet {
namespace cute_gemm {

// ============================================================================
// Type-erased entry points called by pybind11 dispatch
// ============================================================================

template <typename ElementInput, typename TileTag>
int run_cute_gemm_ad_gather_scatter(const void *a,
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
  return launch_cute_gemm_ad_gather_scatter<ElementInput, Config>(
      a, b, c, d, idx_a, idx_d, idx_size, M_A, K, N, M_C, alpha, beta);
}

template <typename ElementInput, typename TileTag>
int run_cute_gemm_trAB_gather(const void *a,
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
                               float beta) {
  using Config = CuteTileConfig<ElementInput, TileTag>;
  return launch_cute_gemm_trAB_gather<ElementInput, Config>(
      a, b, c, d, idx_a, idx_b, idx_size, M_A, K, K_B, N, alpha, beta);
}

// ============================================================================
// Explicit instantiations
// ============================================================================

// --- AD gather-scatter: half_t × 4 tiles ---
#define INSTANTIATE_CUTE_AD_GS(ElemType, TileTag)                          \
  template int run_cute_gemm_ad_gather_scatter<ElemType, gemm::TileTag>(   \
      const void *, const void *, const void *, void *,                    \
      const int *, const int *,                                            \
      int, int, int, int, int, float, float);

INSTANTIATE_CUTE_AD_GS(cutlass::half_t, Tile64x64x32)
INSTANTIATE_CUTE_AD_GS(cutlass::half_t, Tile128x64x32)
INSTANTIATE_CUTE_AD_GS(cutlass::half_t, Tile64x128x32)
INSTANTIATE_CUTE_AD_GS(cutlass::half_t, Tile128x128x32)

INSTANTIATE_CUTE_AD_GS(cutlass::bfloat16_t, Tile64x64x32)
INSTANTIATE_CUTE_AD_GS(cutlass::bfloat16_t, Tile128x64x32)
INSTANTIATE_CUTE_AD_GS(cutlass::bfloat16_t, Tile64x128x32)
INSTANTIATE_CUTE_AD_GS(cutlass::bfloat16_t, Tile128x128x32)

#undef INSTANTIATE_CUTE_AD_GS

// --- TrAB gather: half_t × 4 tiles + bfloat16_t × 4 tiles ---
#define INSTANTIATE_CUTE_TRAB(ElemType, TileTag)                             \
  template int run_cute_gemm_trAB_gather<ElemType, gemm::TileTag>(           \
      const void *, const void *, const void *, void *,                      \
      const int *, const int *,                                              \
      int, int, int, int, int, float, float);

INSTANTIATE_CUTE_TRAB(cutlass::half_t, Tile64x64x32)
INSTANTIATE_CUTE_TRAB(cutlass::half_t, Tile128x64x32)
INSTANTIATE_CUTE_TRAB(cutlass::half_t, Tile64x128x32)
INSTANTIATE_CUTE_TRAB(cutlass::half_t, Tile128x128x32)

INSTANTIATE_CUTE_TRAB(cutlass::bfloat16_t, Tile64x64x32)
INSTANTIATE_CUTE_TRAB(cutlass::bfloat16_t, Tile128x64x32)
INSTANTIATE_CUTE_TRAB(cutlass::bfloat16_t, Tile64x128x32)
INSTANTIATE_CUTE_TRAB(cutlass::bfloat16_t, Tile128x128x32)

#undef INSTANTIATE_CUTE_TRAB

}  // namespace cute_gemm
}  // namespace warpconvnet
