// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Forward kernel instantiations split out of mask_gemm_kernels.cu so
// nvcc compiles each op (fwd / dgrad / wgrad) in its own translation unit.

#include "mask_gemm_kernels_common.h"

// Forward kernels
#include "mask_gemm/include/MaskGemm_forward_128x64x32_2s_fused.h"
#include "mask_gemm/include/MaskGemm_forward_32x32x32_1s_flat.h"
#include "mask_gemm/include/MaskGemm_forward_64x128x32_2s_fused.h"
#include "mask_gemm/include/MaskGemm_forward_64x128x32_3s.h"
#include "mask_gemm/include/MaskGemm_forward_64x64x32_2s_pipelined.h"
// Forward scalar variants (for unaligned C)
#include "mask_gemm/include/MaskGemm_forward_64x64x32_1s_flat_sa.h"
#include "mask_gemm/include/MaskGemm_forward_64x64x32_1s_flat_sab_se.h"
#include "mask_gemm/include/MaskGemm_forward_64x64x32_1s_flat_sb_se.h"
// Forward fp32 output kernels (fp16/bf16 input, f32 output)
#include "mask_gemm/include/MaskGemm_forward_64x64x32_1s_flat.h"
#include "mask_gemm/include/MaskGemm_forward_64x64x32_1s_flat_direpi_sb.h"
// Forward pcoff (E1 offset-precompute) variants — warpgemm tiles 54-63
#include "mask_gemm/include/MaskGemm_forward_64x128x32_1s_flat_pcoff.h"
#include "mask_gemm/include/MaskGemm_forward_64x128x32_2s_warp_spec_pcoff.h"
#include "mask_gemm/include/MaskGemm_forward_64x64x32_1s_flat_pcoff.h"
#include "mask_gemm/include/MaskGemm_forward_64x64x32_2s_warp_spec_pcoff.h"
#include "mask_gemm/include/MaskGemm_forward_64x64x32_3s_pcoff.h"

namespace warpconvnet {
namespace cute_gemm {

// =============================================================================
// Forward instantiations
// =============================================================================

// Forward: 32x32x32 flat (C<=48, fp16 only — uses F16 accum)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_32x32x32_1s_flat,
                         cutlass::half_t,
                         Tile32x32x32_F16Accum,
                         cutlass::half_t)

// Forward: 64x64x32 pipelined (C=64 or C<=48 bf16)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_2s_pipelined,
                         cutlass::half_t,
                         Tile64x64x32,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_2s_pipelined,
                         cutlass::half_t,
                         Tile64x64x32,
                         float)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_2s_pipelined,
                         cutlass::bfloat16_t,
                         Tile64x64x32,
                         cutlass::bfloat16_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_2s_pipelined,
                         cutlass::bfloat16_t,
                         Tile64x64x32,
                         float)

// Forward: 64x128x32 fused (C>=128, C_in>=C_out, fp16 — F16 accum)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_2s_fused,
                         cutlass::half_t,
                         Tile64x128x32_F16Accum,
                         cutlass::half_t)

// Forward: 64x128x32 3-stage (C>=128)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_3s,
                         cutlass::half_t,
                         Tile64x128x32,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_3s, cutlass::half_t, Tile64x128x32, float)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_3s,
                         cutlass::bfloat16_t,
                         Tile64x128x32,
                         cutlass::bfloat16_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_3s, cutlass::bfloat16_t, Tile64x128x32, float)

// Forward: 128x64x32 fused (C>=128, C_in<C_out)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_128x64x32_2s_fused,
                         cutlass::half_t,
                         Tile128x64x32,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_128x64x32_2s_fused, cutlass::half_t, Tile128x64x32, float)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_128x64x32_2s_fused,
                         cutlass::bfloat16_t,
                         Tile128x64x32,
                         cutlass::bfloat16_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_128x64x32_2s_fused,
                         cutlass::bfloat16_t,
                         Tile128x64x32,
                         float)

// Forward pcoff variants (E1 offset-precompute): warpgemm tile IDs 54-63.
// fp16->fp16 canonical flavor for all 7; bf16->bf16 for tiles 58/59/63 whose
// base configs use F32 accumulator (54-57 are F16Accum/F16K8, fp16-only).
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_1s_flat_pcoff,  // tile 54
                         cutlass::half_t,
                         Tile64x64x32_Pcoff,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_1s_flat_pcoff,  // tile 55
                         cutlass::half_t,
                         Tile64x64x32_Pcoff_K8,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_1s_flat_pcoff,  // tile 56
                         cutlass::half_t,
                         Tile64x128x32_Pcoff_K8,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_1s_flat_pcoff,  // tile 57
                         cutlass::half_t,
                         Tile64x128x32_Pcoff,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_3s_pcoff,  // tile 58 fp16
                         cutlass::half_t,
                         Tile64x64x32_Pcoff_3s,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_3s_pcoff,  // tile 58 bf16
                         cutlass::bfloat16_t,
                         Tile64x64x32_Pcoff_3s,
                         cutlass::bfloat16_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_2s_warp_spec_pcoff,  // tile 59 fp16
                         cutlass::half_t,
                         Tile64x64x32_Pcoff_WS,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x64x32_2s_warp_spec_pcoff,  // tile 59 bf16
                         cutlass::bfloat16_t,
                         Tile64x64x32_Pcoff_WS,
                         cutlass::bfloat16_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_2s_warp_spec_pcoff,  // tile 63 fp16
                         cutlass::half_t,
                         Tile64x128x32_Pcoff_WS,
                         cutlass::half_t)
WCN_PROD_INSTANTIATE_FWD(MaskGemm_forward_64x128x32_2s_warp_spec_pcoff,  // tile 63 bf16
                         cutlass::bfloat16_t,
                         Tile64x128x32_Pcoff_WS,
                         cutlass::bfloat16_t)

// Forward: 64x64x32 flat with fp32 output — uses dedicated launch function
// to avoid conflict with existing (half_t, Tile64x64x32, float) instantiation.
#define INST_FWD_F32OUT(ElemIn)                                                 \
  template <>                                                                   \
  int launch_mask_gemm_fwd_f32out<ElemIn>(const void *a,                        \
                                          const void *b,                        \
                                          void *d,                              \
                                          const int *pt,                        \
                                          const uint32_t *pm,                   \
                                          const int *ms,                        \
                                          int N_in,                             \
                                          int N_out,                            \
                                          int C_in,                             \
                                          int C_out,                            \
                                          int K,                                \
                                          float alpha,                          \
                                          int groups,                           \
                                          int identity_offset,                  \
                                          cudaStream_t stream) {                \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                  \
    using Kernel = MaskGemm_forward_64x64x32_1s_flat<Config, float>;            \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});          \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});          \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                        \
    int m_tiles = (N_out + TileM - 1) / TileM;                                  \
    int n_tiles = (C_out + TileN - 1) / TileN;                                  \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                     \
    size_t smem = Kernel::SharedStorageSize;                                    \
    if (smem > 48 * 1024)                                                       \
      if (cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,                  \
                               cudaFuncAttributeMaxDynamicSharedMemorySize,     \
                               smem) != cudaSuccess)                            \
        return -1;                                                              \
    mask_gemm_kernel_entry<Kernel>                                              \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a, \
                                                             (const ElemIn *)b, \
                                                             (float *)d,        \
                                                             pt,                \
                                                             pm,                \
                                                             ms,                \
                                                             N_in,              \
                                                             N_out,             \
                                                             C_in,              \
                                                             C_out,             \
                                                             K,                 \
                                                             alpha,             \
                                                             C_in * groups,     \
                                                             C_out * groups,    \
                                                             identity_offset);  \
    return 0;                                                                   \
  }

INST_FWD_F32OUT(cutlass::half_t)
INST_FWD_F32OUT(cutlass::bfloat16_t)
#undef INST_FWD_F32OUT

// Forward: scalar B + fp32 output (unaligned C + non-AMP)
#define INST_FWD_F32OUT_SB(ElemIn)                                              \
  template <>                                                                   \
  int launch_mask_gemm_fwd_f32out_sb<ElemIn>(const void *a,                     \
                                             const void *b,                     \
                                             void *d,                           \
                                             const int *pt,                     \
                                             const uint32_t *pm,                \
                                             const int *ms,                     \
                                             int N_in,                          \
                                             int N_out,                         \
                                             int C_in,                          \
                                             int C_out,                         \
                                             int K,                             \
                                             float alpha,                       \
                                             int groups,                        \
                                             int identity_offset,               \
                                             cudaStream_t stream) {             \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                  \
    using Kernel = MaskGemm_forward_64x64x32_1s_flat_direpi_sb<Config, float>;  \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});          \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});          \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                        \
    int m_tiles = (N_out + TileM - 1) / TileM;                                  \
    int n_tiles = (C_out + TileN - 1) / TileN;                                  \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                     \
    size_t smem = Kernel::SharedStorageSize;                                    \
    if (smem > 48 * 1024)                                                       \
      if (cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,                  \
                               cudaFuncAttributeMaxDynamicSharedMemorySize,     \
                               smem) != cudaSuccess)                            \
        return -1;                                                              \
    mask_gemm_kernel_entry<Kernel>                                              \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a, \
                                                             (const ElemIn *)b, \
                                                             (float *)d,        \
                                                             pt,                \
                                                             pm,                \
                                                             ms,                \
                                                             N_in,              \
                                                             N_out,             \
                                                             C_in,              \
                                                             C_out,             \
                                                             K,                 \
                                                             alpha,             \
                                                             C_in * groups,     \
                                                             C_out * groups,    \
                                                             identity_offset);  \
    return 0;                                                                   \
  }

INST_FWD_F32OUT_SB(cutlass::half_t)
INST_FWD_F32OUT_SB(cutlass::bfloat16_t)
#undef INST_FWD_F32OUT_SB

// =============================================================================
// Forward MaskWords>1 launch functions (K>32 support)
// Uses 64x64x32_1s_flat kernel with MaskWords=2 (K<=64) or MaskWords=4 (K<=128)
// =============================================================================

#define INST_FWD_MW(ElemIn, MW)                                                 \
  template <>                                                                   \
  int launch_mask_gemm_fwd_mw<ElemIn, MW>(const void *a,                        \
                                          const void *b,                        \
                                          void *d,                              \
                                          const int *pt,                        \
                                          const uint32_t *pm,                   \
                                          const int *ms,                        \
                                          int N_in,                             \
                                          int N_out,                            \
                                          int C_in,                             \
                                          int C_out,                            \
                                          int K,                                \
                                          float alpha,                          \
                                          int groups,                           \
                                          int identity_offset,                  \
                                          cudaStream_t stream) {                \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                  \
    using Kernel = MaskGemm_forward_64x64x32_1s_flat<Config, ElemIn, MW>;       \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});          \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});          \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                        \
    int m_tiles = (N_out + TileM - 1) / TileM;                                  \
    int n_tiles = (C_out + TileN - 1) / TileN;                                  \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                     \
    size_t smem = Kernel::SharedStorageSize;                                    \
    if (smem > 48 * 1024)                                                       \
      if (cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,                  \
                               cudaFuncAttributeMaxDynamicSharedMemorySize,     \
                               smem) != cudaSuccess)                            \
        return -1;                                                              \
    mask_gemm_kernel_entry<Kernel>                                              \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a, \
                                                             (const ElemIn *)b, \
                                                             (ElemIn *)d,       \
                                                             pt,                \
                                                             pm,                \
                                                             ms,                \
                                                             N_in,              \
                                                             N_out,             \
                                                             C_in,              \
                                                             C_out,             \
                                                             K,                 \
                                                             alpha,             \
                                                             C_in * groups,     \
                                                             C_out * groups,    \
                                                             identity_offset);  \
    return 0;                                                                   \
  }

INST_FWD_MW(cutlass::half_t, 2)
INST_FWD_MW(cutlass::half_t, 4)
INST_FWD_MW(cutlass::half_t, 8)
INST_FWD_MW(cutlass::half_t, 12)
INST_FWD_MW(cutlass::bfloat16_t, 2)
INST_FWD_MW(cutlass::bfloat16_t, 4)
INST_FWD_MW(cutlass::bfloat16_t, 8)
INST_FWD_MW(cutlass::bfloat16_t, 12)
#undef INST_FWD_MW

// =============================================================================
// Forward f32-output MaskWords>1 launch functions (tile 80: aligned, tile 82: scalar B)
// =============================================================================

#define INST_FWD_F32OUT_MW(ElemIn, MW)                                          \
  template <>                                                                   \
  int launch_mask_gemm_fwd_f32out_mw<ElemIn, MW>(const void *a,                 \
                                                 const void *b,                 \
                                                 void *d,                       \
                                                 const int *pt,                 \
                                                 const uint32_t *pm,            \
                                                 const int *ms,                 \
                                                 int N_in,                      \
                                                 int N_out,                     \
                                                 int C_in,                      \
                                                 int C_out,                     \
                                                 int K,                         \
                                                 float alpha,                   \
                                                 int groups,                    \
                                                 int identity_offset,           \
                                                 cudaStream_t stream) {         \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                  \
    using Kernel = MaskGemm_forward_64x64x32_1s_flat<Config, float, MW>;        \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});          \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});          \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                        \
    int m_tiles = (N_out + TileM - 1) / TileM;                                  \
    int n_tiles = (C_out + TileN - 1) / TileN;                                  \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                     \
    size_t smem = Kernel::SharedStorageSize;                                    \
    if (smem > 48 * 1024)                                                       \
      if (cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,                  \
                               cudaFuncAttributeMaxDynamicSharedMemorySize,     \
                               smem) != cudaSuccess)                            \
        return -1;                                                              \
    mask_gemm_kernel_entry<Kernel>                                              \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a, \
                                                             (const ElemIn *)b, \
                                                             (float *)d,        \
                                                             pt,                \
                                                             pm,                \
                                                             ms,                \
                                                             N_in,              \
                                                             N_out,             \
                                                             C_in,              \
                                                             C_out,             \
                                                             K,                 \
                                                             alpha,             \
                                                             C_in * groups,     \
                                                             C_out * groups,    \
                                                             identity_offset);  \
    return 0;                                                                   \
  }

INST_FWD_F32OUT_MW(cutlass::half_t, 2)
INST_FWD_F32OUT_MW(cutlass::half_t, 4)
INST_FWD_F32OUT_MW(cutlass::half_t, 8)
INST_FWD_F32OUT_MW(cutlass::half_t, 12)
INST_FWD_F32OUT_MW(cutlass::bfloat16_t, 2)
INST_FWD_F32OUT_MW(cutlass::bfloat16_t, 4)
INST_FWD_F32OUT_MW(cutlass::bfloat16_t, 8)
INST_FWD_F32OUT_MW(cutlass::bfloat16_t, 12)
#undef INST_FWD_F32OUT_MW

#define INST_FWD_F32OUT_SB_MW(ElemIn, MW)                                          \
  template <>                                                                      \
  int launch_mask_gemm_fwd_f32out_sb_mw<ElemIn, MW>(const void *a,                 \
                                                    const void *b,                 \
                                                    void *d,                       \
                                                    const int *pt,                 \
                                                    const uint32_t *pm,            \
                                                    const int *ms,                 \
                                                    int N_in,                      \
                                                    int N_out,                     \
                                                    int C_in,                      \
                                                    int C_out,                     \
                                                    int K,                         \
                                                    float alpha,                   \
                                                    int groups,                    \
                                                    int identity_offset,           \
                                                    cudaStream_t stream) {         \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                     \
    using Kernel = MaskGemm_forward_64x64x32_1s_flat_direpi_sb<Config, float, MW>; \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});             \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});             \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                           \
    int m_tiles = (N_out + TileM - 1) / TileM;                                     \
    int n_tiles = (C_out + TileN - 1) / TileN;                                     \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                        \
    size_t smem = Kernel::SharedStorageSize;                                       \
    if (smem > 48 * 1024)                                                          \
      if (cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,                     \
                               cudaFuncAttributeMaxDynamicSharedMemorySize,        \
                               smem) != cudaSuccess)                               \
        return -1;                                                                 \
    mask_gemm_kernel_entry<Kernel>                                                 \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,    \
                                                             (const ElemIn *)b,    \
                                                             (float *)d,           \
                                                             pt,                   \
                                                             pm,                   \
                                                             ms,                   \
                                                             N_in,                 \
                                                             N_out,                \
                                                             C_in,                 \
                                                             C_out,                \
                                                             K,                    \
                                                             alpha,                \
                                                             C_in * groups,        \
                                                             C_out * groups,       \
                                                             identity_offset);     \
    return 0;                                                                      \
  }

INST_FWD_F32OUT_SB_MW(cutlass::half_t, 2)
INST_FWD_F32OUT_SB_MW(cutlass::half_t, 4)
INST_FWD_F32OUT_SB_MW(cutlass::half_t, 8)
INST_FWD_F32OUT_SB_MW(cutlass::half_t, 12)
INST_FWD_F32OUT_SB_MW(cutlass::bfloat16_t, 2)
INST_FWD_F32OUT_SB_MW(cutlass::bfloat16_t, 4)
INST_FWD_F32OUT_SB_MW(cutlass::bfloat16_t, 8)
INST_FWD_F32OUT_SB_MW(cutlass::bfloat16_t, 12)
#undef INST_FWD_F32OUT_SB_MW

// =============================================================================
// Forward vectorized MW>1 launch functions — tiles 42 (F16Acc 64x128 fused),
// 43 (64x128 3s), 44 (128x64 fused). Separate launch functions to avoid
// template key clash with the MW=1 specializations.
// =============================================================================

// Tile 42: 64x128 F16Accum fused (half-only)
#define INST_FWD_64x128_F16ACC_MW(MW)                                                         \
  template <>                                                                                 \
  int launch_mask_gemm_fwd_64x128_f16acc_mw<MW>(const void *a,                                \
                                                const void *b,                                \
                                                void *d,                                      \
                                                const int *pt,                                \
                                                const uint32_t *pm,                           \
                                                const int *ms,                                \
                                                int N_in,                                     \
                                                int N_out,                                    \
                                                int C_in,                                     \
                                                int C_out,                                    \
                                                int K,                                        \
                                                float alpha,                                  \
                                                int groups,                                   \
                                                int identity_offset,                          \
                                                cudaStream_t stream) {                        \
    using ElemIn = cutlass::half_t;                                                           \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x128x32_F16Accum>;                      \
    using Kernel = MaskGemm_forward_64x128x32_2s_fused<Config, ElemIn, MW>;                   \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                        \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                        \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                                      \
    int m_tiles = (N_out + TileM - 1) / TileM;                                                \
    int n_tiles = (C_out + TileN - 1) / TileN;                                                \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                                   \
    size_t smem = Kernel::SharedStorageSize;                                                  \
    if (smem > 48 * 1024) {                                                                   \
      auto err = cudaFuncSetAttribute(                                                        \
          mask_gemm_kernel_entry<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
      if (err != cudaSuccess) return -1;                                                      \
    }                                                                                         \
    mask_gemm_kernel_entry<Kernel>                                                            \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,               \
                                                             (const ElemIn *)b,               \
                                                             (ElemIn *)d,                     \
                                                             pt,                              \
                                                             pm,                              \
                                                             ms,                              \
                                                             N_in,                            \
                                                             N_out,                           \
                                                             C_in,                            \
                                                             C_out,                           \
                                                             K,                               \
                                                             alpha,                           \
                                                             C_in * groups,                   \
                                                             C_out * groups,                  \
                                                             identity_offset);                \
    return 0;                                                                                 \
  }

INST_FWD_64x128_F16ACC_MW(2) INST_FWD_64x128_F16ACC_MW(4) INST_FWD_64x128_F16ACC_MW(8)
    INST_FWD_64x128_F16ACC_MW(12)
#undef INST_FWD_64x128_F16ACC_MW

// Tile 43: 64x128 3-stage (half + bfloat16)
#define INST_FWD_64x128_3S_MW(ElemIn, MW)                                                     \
  template <>                                                                                 \
  int launch_mask_gemm_fwd_64x128_3s_mw<ElemIn, MW>(const void *a,                            \
                                                    const void *b,                            \
                                                    void *d,                                  \
                                                    const int *pt,                            \
                                                    const uint32_t *pm,                       \
                                                    const int *ms,                            \
                                                    int N_in,                                 \
                                                    int N_out,                                \
                                                    int C_in,                                 \
                                                    int C_out,                                \
                                                    int K,                                    \
                                                    float alpha,                              \
                                                    int groups,                               \
                                                    int identity_offset,                      \
                                                    cudaStream_t stream) {                    \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x128x32>;                               \
    using Kernel = MaskGemm_forward_64x128x32_3s<Config, ElemIn, MW>;                         \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                        \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                        \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                                      \
    int m_tiles = (N_out + TileM - 1) / TileM;                                                \
    int n_tiles = (C_out + TileN - 1) / TileN;                                                \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                                   \
    size_t smem = Kernel::SharedStorageSize;                                                  \
    if (smem > 48 * 1024) {                                                                   \
      auto err = cudaFuncSetAttribute(                                                        \
          mask_gemm_kernel_entry<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
      if (err != cudaSuccess) return -1;                                                      \
    }                                                                                         \
    mask_gemm_kernel_entry<Kernel>                                                            \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,               \
                                                             (const ElemIn *)b,               \
                                                             (ElemIn *)d,                     \
                                                             pt,                              \
                                                             pm,                              \
                                                             ms,                              \
                                                             N_in,                            \
                                                             N_out,                           \
                                                             C_in,                            \
                                                             C_out,                           \
                                                             K,                               \
                                                             alpha,                           \
                                                             C_in * groups,                   \
                                                             C_out * groups,                  \
                                                             identity_offset);                \
    return 0;                                                                                 \
  }

        INST_FWD_64x128_3S_MW(cutlass::half_t, 2) INST_FWD_64x128_3S_MW(cutlass::half_t, 4)
            INST_FWD_64x128_3S_MW(cutlass::half_t, 8) INST_FWD_64x128_3S_MW(cutlass::half_t, 12)
                INST_FWD_64x128_3S_MW(cutlass::bfloat16_t,
                                      2) INST_FWD_64x128_3S_MW(cutlass::bfloat16_t, 4)
                    INST_FWD_64x128_3S_MW(cutlass::bfloat16_t, 8)
                        INST_FWD_64x128_3S_MW(cutlass::bfloat16_t, 12)
#undef INST_FWD_64x128_3S_MW

// Tile 44: 128x64 fused (half + bfloat16)
#define INST_FWD_128x64_MW(ElemIn, MW)                                                        \
  template <>                                                                                 \
  int launch_mask_gemm_fwd_128x64_mw<ElemIn, MW>(const void *a,                               \
                                                 const void *b,                               \
                                                 void *d,                                     \
                                                 const int *pt,                               \
                                                 const uint32_t *pm,                          \
                                                 const int *ms,                               \
                                                 int N_in,                                    \
                                                 int N_out,                                   \
                                                 int C_in,                                    \
                                                 int C_out,                                   \
                                                 int K,                                       \
                                                 float alpha,                                 \
                                                 int groups,                                  \
                                                 int identity_offset,                         \
                                                 cudaStream_t stream) {                       \
    using Config = CuteTileConfig<ElemIn, gemm::Tile128x64x32>;                               \
    using Kernel = MaskGemm_forward_128x64x32_2s_fused<Config, ElemIn, MW>;                   \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                        \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                        \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                                      \
    int m_tiles = (N_out + TileM - 1) / TileM;                                                \
    int n_tiles = (C_out + TileN - 1) / TileN;                                                \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                                   \
    size_t smem = Kernel::SharedStorageSize;                                                  \
    if (smem > 48 * 1024) {                                                                   \
      auto err = cudaFuncSetAttribute(                                                        \
          mask_gemm_kernel_entry<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
      if (err != cudaSuccess) return -1;                                                      \
    }                                                                                         \
    mask_gemm_kernel_entry<Kernel>                                                            \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,               \
                                                             (const ElemIn *)b,               \
                                                             (ElemIn *)d,                     \
                                                             pt,                              \
                                                             pm,                              \
                                                             ms,                              \
                                                             N_in,                            \
                                                             N_out,                           \
                                                             C_in,                            \
                                                             C_out,                           \
                                                             K,                               \
                                                             alpha,                           \
                                                             C_in * groups,                   \
                                                             C_out * groups,                  \
                                                             identity_offset);                \
    return 0;                                                                                 \
  }

                            INST_FWD_128x64_MW(cutlass::half_t, 2)
                                INST_FWD_128x64_MW(cutlass::half_t, 4)
                                    INST_FWD_128x64_MW(cutlass::half_t, 8)
                                        INST_FWD_128x64_MW(cutlass::half_t, 12)
                                            INST_FWD_128x64_MW(cutlass::bfloat16_t, 2)
                                                INST_FWD_128x64_MW(cutlass::bfloat16_t, 4)
                                                    INST_FWD_128x64_MW(cutlass::bfloat16_t, 8)
                                                        INST_FWD_128x64_MW(cutlass::bfloat16_t, 12)
#undef INST_FWD_128x64_MW

    // =============================================================================
    // Scalar fwd launch functions (separate from generic template to avoid
    // duplicate specializations — all use Tile64x64x32 config)
    // =============================================================================

    // Forward scalar instantiations (MW=1)
    WCN_DEFINE_SCALAR_LAUNCH(
        fwd, sab_se, MaskGemm_forward_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t)
        WCN_DEFINE_SCALAR_LAUNCH(fwd,
                                 sab_se,
                                 MaskGemm_forward_64x64x32_1s_flat_sab_se,
                                 cutlass::bfloat16_t,
                                 cutlass::bfloat16_t)
            WCN_DEFINE_SCALAR_LAUNCH(
                fwd, sa, MaskGemm_forward_64x64x32_1s_flat_sa, cutlass::half_t, cutlass::half_t)
                WCN_DEFINE_SCALAR_LAUNCH(fwd,
                                         sa,
                                         MaskGemm_forward_64x64x32_1s_flat_sa,
                                         cutlass::bfloat16_t,
                                         cutlass::bfloat16_t)
                    WCN_DEFINE_SCALAR_LAUNCH(fwd,
                                             sb_se,
                                             MaskGemm_forward_64x64x32_1s_flat_sb_se,
                                             cutlass::half_t,
                                             cutlass::half_t)
                        WCN_DEFINE_SCALAR_LAUNCH(fwd,
                                                 sb_se,
                                                 MaskGemm_forward_64x64x32_1s_flat_sb_se,
                                                 cutlass::bfloat16_t,
                                                 cutlass::bfloat16_t)

    // Forward scalar SAB_SE with MW=2,4,8,12
    WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
        fwd, MaskGemm_forward_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t, 2)
        WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
            fwd, MaskGemm_forward_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t, 4)
            WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
                fwd, MaskGemm_forward_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t, 8)
                WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
                    fwd,
                    MaskGemm_forward_64x64x32_1s_flat_sab_se,
                    cutlass::half_t,
                    cutlass::half_t,
                    12) WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(fwd,
                                                           MaskGemm_forward_64x64x32_1s_flat_sab_se,
                                                           cutlass::bfloat16_t,
                                                           cutlass::bfloat16_t,
                                                           2)
                    WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(fwd,
                                                       MaskGemm_forward_64x64x32_1s_flat_sab_se,
                                                       cutlass::bfloat16_t,
                                                       cutlass::bfloat16_t,
                                                       4)
                        WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(fwd,
                                                           MaskGemm_forward_64x64x32_1s_flat_sab_se,
                                                           cutlass::bfloat16_t,
                                                           cutlass::bfloat16_t,
                                                           8)
                            WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
                                fwd,
                                MaskGemm_forward_64x64x32_1s_flat_sab_se,
                                cutlass::bfloat16_t,
                                cutlass::bfloat16_t,
                                12)

    // Forward SA / SB_SE with MW=2,4,8,12
    WCN_INST_ALL_MW_SA_SB(fwd, sa, MaskGemm_forward_64x64x32_1s_flat_sa)
        WCN_INST_ALL_MW_SA_SB(fwd, sb_se, MaskGemm_forward_64x64x32_1s_flat_sb_se)

}  // namespace cute_gemm
}  // namespace warpconvnet
