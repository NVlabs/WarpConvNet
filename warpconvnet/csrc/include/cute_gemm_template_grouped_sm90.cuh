// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SM90 grouped GEMM gather-scatter wrapper templates.
// Template body header for run_cute_gemm_grouped_ad_gather_scatter_sm90_staged
// and run_cute_gemm_sm90_grouped_ad_gather_scatter. Included by
// warpconvnet/csrc/cutlass_cute_gemm_grouped_sm90.cu (existing instantiations)
// and by warpgemm-generated offset_gemm TUs that invoke
// INSTANTIATE_SM90_STAGED_GROUPED / INSTANTIATE_SM90_GROUPED_SIMPLE for the
// stable tier.

#pragma once

#if defined(WARPCONVNET_SM90_ENABLED)

#include <c10/cuda/CUDAStream.h>

#include "cute_gemm_launch.h"
#include "gemm_error_codes.h"

namespace warpconvnet {
namespace cute_gemm {

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
  // Launch on the caller's current PyTorch stream, not the default stream 0
  // (cross-stream race under non-default streams; see SM80 stream fix).
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

#define DISPATCH_SM90_STAGED_GROUPED(S, CP)                                                      \
  {                                                                                              \
    using Config = CuteTileConfigOverride<Base, S, CP>;                                          \
    return launch_cute_gemm_grouped_ad_gather_scatter_sm90<ElementInput, Config, ElementOutput>( \
        a, d, in_map, out_map, params, total_m_tiles, K, N, alpha, use_atomic, stream);          \
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
      a,
      d,
      in_map,
      out_map,
      params,
      total_m_tiles,
      K,
      N,
      alpha,
      default_stages,
      use_cp_async,
      use_atomic);
}

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
