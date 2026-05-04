// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Dgrad kernel instantiations split out of mask_gemm_kernels.cu.

#include "mask_gemm_kernels_common.h"

// Dgrad kernels
#include "mask_gemm/include/MaskGemm_dgrad_32x32x32_1s_flat.h"
#include "mask_gemm/include/MaskGemm_dgrad_64x128x32_1s_flat_direpi.h"
#include "mask_gemm/include/MaskGemm_dgrad_64x64x32_1s_flat.h"
// Dgrad fp32 output kernel
#include "mask_gemm/include/MaskGemm_dgrad_64x64x32_1s_flat_direpi_sb.h"
// Dgrad pipelined variants
#include "mask_gemm/include/MaskGemm_dgrad_128x64x32_2s_pipelined.h"
#include "mask_gemm/include/MaskGemm_dgrad_64x128x32_2s_pipelined.h"
#include "mask_gemm/include/MaskGemm_dgrad_64x64x32_2s_pipelined.h"
// Dgrad scalar variants
#include "mask_gemm/include/MaskGemm_dgrad_64x64x32_1s_flat_sa.h"
#include "mask_gemm/include/MaskGemm_dgrad_64x64x32_1s_flat_sab_se.h"
#include "mask_gemm/include/MaskGemm_dgrad_64x64x32_1s_flat_sb_se.h"

namespace warpconvnet {
namespace cute_gemm {

// =============================================================================
// Dgrad instantiations
// =============================================================================
// Dgrad: A = grad_output [N_out, C_out*G], D = grad_input [N_in, C_in*G]
// stride_A = C_out*groups (grad_output row stride), stride_D = C_in*groups (grad_input row stride)

// Dgrad: 32x32x32 flat (C<=48, fp16)
WCN_PROD_INSTANTIATE_DGRAD(MaskGemm_dgrad_32x32x32_1s_flat,
                           cutlass::half_t,
                           Tile32x32x32,
                           cutlass::half_t)

// Dgrad: 64x64x32 flat (C<=96)
WCN_PROD_INSTANTIATE_DGRAD(MaskGemm_dgrad_64x64x32_1s_flat,
                           cutlass::half_t,
                           Tile64x64x32,
                           cutlass::half_t)
WCN_PROD_INSTANTIATE_DGRAD(MaskGemm_dgrad_64x64x32_1s_flat,
                           cutlass::half_t,
                           Tile64x64x32_F16Accum,
                           cutlass::half_t)
WCN_PROD_INSTANTIATE_DGRAD(MaskGemm_dgrad_64x64x32_1s_flat,
                           cutlass::bfloat16_t,
                           Tile64x64x32,
                           cutlass::bfloat16_t)

// Dgrad: 64x128x32 flat direct epilogue (C>=128)
WCN_PROD_INSTANTIATE_DGRAD(MaskGemm_dgrad_64x128x32_1s_flat_direpi,
                           cutlass::half_t,
                           Tile64x128x32,
                           cutlass::half_t)
WCN_PROD_INSTANTIATE_DGRAD(MaskGemm_dgrad_64x128x32_1s_flat_direpi,
                           cutlass::half_t,
                           Tile64x128x32_F16Accum,
                           cutlass::half_t)
WCN_PROD_INSTANTIATE_DGRAD(MaskGemm_dgrad_64x128x32_1s_flat_direpi,
                           cutlass::bfloat16_t,
                           Tile64x128x32,
                           cutlass::bfloat16_t)

// Dgrad pipelined: stride_A = C_out*groups (grad_output), stride_D = C_in*groups (grad_input)
#define INST_DGRAD_PIPE(SUFFIX, KernelClass, ElemIn, TileTag)                                 \
  template <>                                                                                 \
  int launch_dgrad_pipelined_##SUFFIX<ElemIn, ElemIn>(const void *a,                          \
                                                      const void *b,                          \
                                                      void *d,                                \
                                                      const int *pt,                          \
                                                      const uint32_t *pm,                     \
                                                      const int *ms,                          \
                                                      int N_in,                               \
                                                      int N_out,                              \
                                                      int C_in,                               \
                                                      int C_out,                              \
                                                      int K,                                  \
                                                      float alpha,                            \
                                                      int groups,                             \
                                                      int identity_offset,                    \
                                                      cudaStream_t stream) {                  \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                                     \
    using Kernel = KernelClass<Config, ElemIn>;                                               \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                        \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                        \
    if (N_in == 0 || C_in == 0 || C_out == 0) return 0;                                       \
    int m_tiles = (N_in + TileM - 1) / TileM;                                                 \
    int n_tiles = (C_in + TileN - 1) / TileN;                                                 \
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
                                                             C_out * groups,                  \
                                                             C_in * groups,                   \
                                                             identity_offset);                \
    return 0;                                                                                 \
  }

INST_DGRAD_PIPE(64x64, MaskGemm_dgrad_64x64x32_2s_pipelined, cutlass::half_t, Tile64x64x32)
INST_DGRAD_PIPE(64x64, MaskGemm_dgrad_64x64x32_2s_pipelined, cutlass::bfloat16_t, Tile64x64x32)
INST_DGRAD_PIPE(64x128, MaskGemm_dgrad_64x128x32_2s_pipelined, cutlass::half_t, Tile64x128x32)
INST_DGRAD_PIPE(64x128, MaskGemm_dgrad_64x128x32_2s_pipelined, cutlass::bfloat16_t, Tile64x128x32)
INST_DGRAD_PIPE(128x64, MaskGemm_dgrad_128x64x32_2s_pipelined, cutlass::half_t, Tile128x64x32)
INST_DGRAD_PIPE(128x64, MaskGemm_dgrad_128x64x32_2s_pipelined, cutlass::bfloat16_t, Tile128x64x32)
#undef INST_DGRAD_PIPE

// Dgrad f32out: stride_A = C_out*groups (grad_output), stride_D = C_in*groups (grad_input)
#define INST_DGRAD_F32OUT(ElemIn)                                               \
  template <>                                                                   \
  int launch_mask_gemm_dgrad_f32out<ElemIn>(const void *a,                      \
                                            const void *b,                      \
                                            void *d,                            \
                                            const int *pt,                      \
                                            const uint32_t *pm,                 \
                                            const int *ms,                      \
                                            int N_in,                           \
                                            int N_out,                          \
                                            int C_in,                           \
                                            int C_out,                          \
                                            int K,                              \
                                            float alpha,                        \
                                            int groups,                         \
                                            int identity_offset,                \
                                            cudaStream_t stream) {              \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                  \
    using Kernel = MaskGemm_dgrad_64x64x32_1s_flat_direpi_sb<Config, float>;    \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});          \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});          \
    if (N_in == 0 || C_in == 0 || C_out == 0) return 0;                         \
    int m_tiles = (N_in + TileM - 1) / TileM;                                   \
    int n_tiles = (C_in + TileN - 1) / TileN;                                   \
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
                                                             C_out * groups,    \
                                                             C_in * groups,     \
                                                             identity_offset);  \
    return 0;                                                                   \
  }

INST_DGRAD_F32OUT(cutlass::half_t)
INST_DGRAD_F32OUT(cutlass::bfloat16_t)
#undef INST_DGRAD_F32OUT

// =============================================================================
// Dgrad f32-output MaskWords>1 launch function (tile 81)
// =============================================================================
#define INST_DGRAD_F32OUT_MW(ElemIn, MW)                                         \
  template <>                                                                    \
  int launch_mask_gemm_dgrad_f32out_mw<ElemIn, MW>(const void *a,                \
                                                   const void *b,                \
                                                   void *d,                      \
                                                   const int *pt,                \
                                                   const uint32_t *pm,           \
                                                   const int *ms,                \
                                                   int N_in,                     \
                                                   int N_out,                    \
                                                   int C_in,                     \
                                                   int C_out,                    \
                                                   int K,                        \
                                                   float alpha,                  \
                                                   int groups,                   \
                                                   int identity_offset,          \
                                                   cudaStream_t stream) {        \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                   \
    using Kernel = MaskGemm_dgrad_64x64x32_1s_flat_direpi_sb<Config, float, MW>; \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});           \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});           \
    if (N_in == 0 || C_in == 0 || C_out == 0) return 0;                          \
    int m_tiles = (N_in + TileM - 1) / TileM;                                    \
    int n_tiles = (C_in + TileN - 1) / TileN;                                    \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                      \
    size_t smem = Kernel::SharedStorageSize;                                     \
    if (smem > 48 * 1024)                                                        \
      if (cudaFuncSetAttribute(mask_gemm_kernel_entry<Kernel>,                   \
                               cudaFuncAttributeMaxDynamicSharedMemorySize,      \
                               smem) != cudaSuccess)                             \
        return -1;                                                               \
    mask_gemm_kernel_entry<Kernel>                                               \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,  \
                                                             (const ElemIn *)b,  \
                                                             (float *)d,         \
                                                             pt,                 \
                                                             pm,                 \
                                                             ms,                 \
                                                             N_in,               \
                                                             N_out,              \
                                                             C_in,               \
                                                             C_out,              \
                                                             K,                  \
                                                             alpha,              \
                                                             C_out * groups,     \
                                                             C_in * groups,      \
                                                             identity_offset);   \
    return 0;                                                                    \
  }

INST_DGRAD_F32OUT_MW(cutlass::half_t, 2)
INST_DGRAD_F32OUT_MW(cutlass::half_t, 4)
INST_DGRAD_F32OUT_MW(cutlass::half_t, 8)
INST_DGRAD_F32OUT_MW(cutlass::half_t, 12)
INST_DGRAD_F32OUT_MW(cutlass::bfloat16_t, 2)
INST_DGRAD_F32OUT_MW(cutlass::bfloat16_t, 4)
INST_DGRAD_F32OUT_MW(cutlass::bfloat16_t, 8)
INST_DGRAD_F32OUT_MW(cutlass::bfloat16_t, 12)
#undef INST_DGRAD_F32OUT_MW

// =============================================================================
// Dgrad MaskWords>1 launch functions (K>32 support)
// Dgrad multi-word: stride_A = C_out*groups (grad_output), stride_D = C_in*groups (grad_input)
// =============================================================================
#define INST_DGRAD_MW(ElemIn, MW)                                               \
  template <>                                                                   \
  int launch_mask_gemm_dgrad_mw<ElemIn, MW>(const void *a,                      \
                                            const void *b,                      \
                                            void *d,                            \
                                            const int *pt,                      \
                                            const uint32_t *pm,                 \
                                            const int *ms,                      \
                                            int N_in,                           \
                                            int N_out,                          \
                                            int C_in,                           \
                                            int C_out,                          \
                                            int K,                              \
                                            float alpha,                        \
                                            int groups,                         \
                                            int identity_offset,                \
                                            cudaStream_t stream) {              \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                  \
    using Kernel = MaskGemm_dgrad_64x64x32_1s_flat<Config, ElemIn, MW>;         \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});          \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});          \
    if (N_in == 0 || C_in == 0 || C_out == 0) return 0;                         \
    int m_tiles = (N_in + TileM - 1) / TileM;                                   \
    int n_tiles = (C_in + TileN - 1) / TileN;                                   \
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
                                                             C_out * groups,    \
                                                             C_in * groups,     \
                                                             identity_offset);  \
    return 0;                                                                   \
  }

INST_DGRAD_MW(cutlass::half_t, 2)
INST_DGRAD_MW(cutlass::half_t, 4)
INST_DGRAD_MW(cutlass::half_t, 8)
INST_DGRAD_MW(cutlass::half_t, 12)
INST_DGRAD_MW(cutlass::bfloat16_t, 2)
INST_DGRAD_MW(cutlass::bfloat16_t, 4)
INST_DGRAD_MW(cutlass::bfloat16_t, 8)
INST_DGRAD_MW(cutlass::bfloat16_t, 12)
#undef INST_DGRAD_MW

// =============================================================================
// Scalar dgrad launch functions (separate from generic template)
// =============================================================================

// Dgrad scalar instantiations (MW=1)
WCN_DEFINE_SCALAR_LAUNCH(
    dgrad, sab_se, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t)
WCN_DEFINE_SCALAR_LAUNCH(
    dgrad, sab_se, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::bfloat16_t, cutlass::bfloat16_t)
WCN_DEFINE_SCALAR_LAUNCH(
    dgrad, sa, MaskGemm_dgrad_64x64x32_1s_flat_sa, cutlass::half_t, cutlass::half_t)
WCN_DEFINE_SCALAR_LAUNCH(
    dgrad, sa, MaskGemm_dgrad_64x64x32_1s_flat_sa, cutlass::bfloat16_t, cutlass::bfloat16_t)
WCN_DEFINE_SCALAR_LAUNCH(
    dgrad, sb_se, MaskGemm_dgrad_64x64x32_1s_flat_sb_se, cutlass::half_t, cutlass::half_t)
WCN_DEFINE_SCALAR_LAUNCH(
    dgrad, sb_se, MaskGemm_dgrad_64x64x32_1s_flat_sb_se, cutlass::bfloat16_t, cutlass::bfloat16_t)

// Dgrad scalar SAB_SE with MW=2,4,8,12
WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
    dgrad, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t, 2)
WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
    dgrad, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t, 4)
WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
    dgrad, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t, 8)
WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
    dgrad, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t, 12)
WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
    dgrad, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::bfloat16_t, cutlass::bfloat16_t, 2)
WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
    dgrad, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::bfloat16_t, cutlass::bfloat16_t, 4)
WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
    dgrad, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::bfloat16_t, cutlass::bfloat16_t, 8)
WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(
    dgrad, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::bfloat16_t, cutlass::bfloat16_t, 12)

// Dgrad SA / SB_SE with MW=2,4,8,12
WCN_INST_ALL_MW_SA_SB(dgrad, sa, MaskGemm_dgrad_64x64x32_1s_flat_sa)
WCN_INST_ALL_MW_SA_SB(dgrad, sb_se, MaskGemm_dgrad_64x64x32_1s_flat_sb_se)

}  // namespace cute_gemm
}  // namespace warpconvnet
