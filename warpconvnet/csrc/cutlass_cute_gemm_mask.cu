// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Mask-based fused CuTe GEMM for sparse convolution.
// Explicit template instantiations for all (element type x tile) combinations.

#if defined(WARPCONVNET_SM80_ENABLED)

#include "include/cute_gemm_mask_launch.h"

namespace warpconvnet {
namespace cute_gemm {

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_mask_fwd(
    const void *a, const void *b, void *d,
    const int *pair_table, const uint32_t *pair_mask, const int *mask_argsort,
    int N_in, int N_out, int C_in, int C_out, int K, float alpha) {
  using Config = CuteTileConfig<ElementInput, TileTag>;
  return launch_cute_gemm_mask_fwd<ElementInput, Config, ElementOutput>(
      a, b, d, pair_table, pair_mask, mask_argsort,
      N_in, N_out, C_in, C_out, K, alpha);
}

// Instantiation macro
#define INSTANTIATE_MASK_FWD(ElemIn, ElemOut, TileTag)                    \
  template int run_cute_gemm_mask_fwd<ElemIn, TileTag, ElemOut>(          \
      const void *, const void *, void *,                                \
      const int *, const uint32_t *, const int *,                        \
      int, int, int, int, int, float);

#define INSTANTIATE_MASK_ALL_DTYPES(TileTag)                             \
  INSTANTIATE_MASK_FWD(cutlass::half_t, cutlass::half_t, TileTag)        \
  INSTANTIATE_MASK_FWD(cutlass::half_t, float, TileTag)                  \
  INSTANTIATE_MASK_FWD(cutlass::bfloat16_t, cutlass::bfloat16_t, TileTag)\
  INSTANTIATE_MASK_FWD(cutlass::bfloat16_t, float, TileTag)

// Start with commonly-used tile sizes
INSTANTIATE_MASK_ALL_DTYPES(gemm::Tile64x64x32)
INSTANTIATE_MASK_ALL_DTYPES(gemm::Tile128x64x32)
INSTANTIATE_MASK_ALL_DTYPES(gemm::Tile64x128x32)
INSTANTIATE_MASK_ALL_DTYPES(gemm::Tile128x128x32)

#undef INSTANTIATE_MASK_ALL_DTYPES
#undef INSTANTIATE_MASK_FWD

// ============================================================================
// Backward dgrad instantiations
// ============================================================================

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_mask_dgrad(
    const void *go, const void *b, void *gi,
    const int *pair_table, const uint32_t *pair_mask, const int *mask_argsort,
    int N_in, int N_out, int C_in, int C_out, int K, float alpha) {
  using Config = CuteTileConfig<ElementInput, TileTag>;
  return launch_cute_gemm_mask_dgrad<ElementInput, Config, ElementOutput>(
      go, b, gi, pair_table, pair_mask, mask_argsort,
      N_in, N_out, C_in, C_out, K, alpha);
}

#define INSTANTIATE_MASK_DGRAD(ElemIn, ElemOut, TileTag)                  \
  template int run_cute_gemm_mask_dgrad<ElemIn, TileTag, ElemOut>(         \
      const void *, const void *, void *,                                \
      const int *, const uint32_t *, const int *,                        \
      int, int, int, int, int, float);

#define INSTANTIATE_DGRAD_ALL_DTYPES(TileTag)                            \
  INSTANTIATE_MASK_DGRAD(cutlass::half_t, cutlass::half_t, TileTag)      \
  INSTANTIATE_MASK_DGRAD(cutlass::half_t, float, TileTag)                \
  INSTANTIATE_MASK_DGRAD(cutlass::bfloat16_t, cutlass::bfloat16_t, TileTag) \
  INSTANTIATE_MASK_DGRAD(cutlass::bfloat16_t, float, TileTag)

INSTANTIATE_DGRAD_ALL_DTYPES(gemm::Tile64x64x32)
INSTANTIATE_DGRAD_ALL_DTYPES(gemm::Tile128x64x32)
INSTANTIATE_DGRAD_ALL_DTYPES(gemm::Tile64x128x32)
INSTANTIATE_DGRAD_ALL_DTYPES(gemm::Tile128x128x32)

#undef INSTANTIATE_DGRAD_ALL_DTYPES
#undef INSTANTIATE_MASK_DGRAD

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM80_ENABLED
