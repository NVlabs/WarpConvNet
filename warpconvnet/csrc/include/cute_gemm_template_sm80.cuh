// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CuTe 3.x SM80 GEMM gather/scatter wrappers.
// Template body header for run_cute_gemm_{ad_gather_scatter,trAB_gather,
// grouped_ad_gather_scatter,grouped_trAB_gather}. Included by
// warpconvnet/csrc/cutlass_cute_gemm_gather_scatter.cu (existing
// instantiations) and by warpgemm-generated offset_gemm TUs that invoke
// INSTANTIATE_CUTE_AD_GS / INSTANTIATE_CUTE_TRAB / INSTANTIATE_CUTE_GROUPED /
// INSTANTIATE_CUTE_GROUPED_TRAB for the stable tier.

#pragma once

#if defined(WARPCONVNET_SM80_ENABLED)

#include "cute_gemm_launch.h"

namespace warpconvnet {
namespace cute_gemm {

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

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_grouped_ad_gather_scatter(const void *a,
                                            void *d,
                                            const int *in_map,
                                            const int *out_map,
                                            const GroupedGemmParams &params,
                                            int total_m_tiles,
                                            int K,
                                            int N,
                                            float alpha) {
  using Config = CuteTileConfig<ElementInput, TileTag>;
  return launch_cute_gemm_grouped_ad_gather_scatter<ElementInput, Config, ElementOutput>(
      a, d, in_map, out_map, params, total_m_tiles, K, N, alpha);
}

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_grouped_trAB_gather(const void *a,
                                      const void *b,
                                      const int *idx_a,
                                      const int *idx_b,
                                      const GroupedTrABGemmParams &params,
                                      int K_dim,
                                      int N,
                                      float alpha) {
  using Config = CuteTileConfig<ElementInput, TileTag>;
  return launch_cute_gemm_grouped_trAB_gather<ElementInput, Config, ElementOutput>(
      a, b, idx_a, idx_b, params, K_dim, N, alpha);
}

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM80_ENABLED
