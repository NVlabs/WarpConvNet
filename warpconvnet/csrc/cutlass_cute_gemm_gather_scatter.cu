// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CuTe 3.x GEMM with gather/scatter via layout composition.
// Explicit template instantiations for all (element type × tile × output type) combinations.

#include "include/cute_gemm_launch.h"

namespace warpconvnet {
namespace cute_gemm {

// ============================================================================
// Type-erased entry points called by pybind11 dispatch
// ============================================================================

template <typename ElementInput, typename TileTag, typename ElementOutput>
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
  return launch_cute_gemm_ad_gather_scatter<ElementInput, Config, ElementOutput>(
      a, b, c, d, idx_a, idx_d, idx_size, M_A, K, N, M_C, alpha, beta);
}

template <typename ElementInput, typename TileTag, typename ElementOutput>
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
  return launch_cute_gemm_trAB_gather<ElementInput, Config, ElementOutput>(
      a, b, c, d, idx_a, idx_b, idx_size, M_A, K, K_B, N, alpha, beta);
}

// ============================================================================
// Explicit instantiations
// ============================================================================

// Macro instantiates AD gather-scatter for one (InputType, OutputType, TileTag).
#define INSTANTIATE_CUTE_AD_GS(ElemIn, ElemOut, TileTag)                       \
  template int run_cute_gemm_ad_gather_scatter<ElemIn, gemm::TileTag, ElemOut>(\
      const void *, const void *, const void *, void *,                        \
      const int *, const int *,                                                \
      int, int, int, int, int, float, float);

// Macro instantiates TrAB gather for one (InputType, OutputType, TileTag).
#define INSTANTIATE_CUTE_TRAB(ElemIn, ElemOut, TileTag)                        \
  template int run_cute_gemm_trAB_gather<ElemIn, gemm::TileTag, ElemOut>(      \
      const void *, const void *, const void *, void *,                        \
      const int *, const int *,                                                \
      int, int, int, int, int, float, float);

// Instantiate both AD and TrAB for all 4 (input, output) dtype pairs for a given tile.
#define INSTANTIATE_ALL_DTYPES(TileTag)                                        \
  INSTANTIATE_CUTE_AD_GS(cutlass::half_t,      float,              TileTag)    \
  INSTANTIATE_CUTE_AD_GS(cutlass::half_t,      cutlass::half_t,    TileTag)    \
  INSTANTIATE_CUTE_AD_GS(cutlass::bfloat16_t,  float,              TileTag)    \
  INSTANTIATE_CUTE_AD_GS(cutlass::bfloat16_t,  cutlass::bfloat16_t,TileTag)   \
  INSTANTIATE_CUTE_TRAB(cutlass::half_t,        float,              TileTag)   \
  INSTANTIATE_CUTE_TRAB(cutlass::half_t,        cutlass::half_t,    TileTag)   \
  INSTANTIATE_CUTE_TRAB(cutlass::bfloat16_t,    float,              TileTag)   \
  INSTANTIATE_CUTE_TRAB(cutlass::bfloat16_t,    cutlass::bfloat16_t,TileTag)

// --- tK=32 tiles (original) ---
INSTANTIATE_ALL_DTYPES(Tile64x64x32)
INSTANTIATE_ALL_DTYPES(Tile128x64x32)
INSTANTIATE_ALL_DTYPES(Tile64x128x32)
INSTANTIATE_ALL_DTYPES(Tile128x128x32)

// --- tK=64 tiles ---
INSTANTIATE_ALL_DTYPES(Tile64x64x64)
INSTANTIATE_ALL_DTYPES(Tile128x64x64)
INSTANTIATE_ALL_DTYPES(Tile64x128x64)
INSTANTIATE_ALL_DTYPES(Tile128x128x64)

// --- Asymmetric M/N tiles (tK=32) ---
INSTANTIATE_ALL_DTYPES(Tile256x64x32)
INSTANTIATE_ALL_DTYPES(Tile64x256x32)

#undef INSTANTIATE_ALL_DTYPES
#undef INSTANTIATE_CUTE_AD_GS
#undef INSTANTIATE_CUTE_TRAB

}  // namespace cute_gemm
}  // namespace warpconvnet
