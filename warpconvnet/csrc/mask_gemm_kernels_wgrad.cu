// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Wgrad kernel instantiations split out of mask_gemm_kernels.cu.

#include "mask_gemm_kernels_common.h"

// Wgrad kernels
#include "mask_gemm/include/MaskGemm_wgrad_64x64x32_2s_f32.h"
#include "mask_gemm/include/MaskGemm_wgrad_64x64x32_2s_f32_sab.h"
// Wgrad atomic kernels (split-K with atomicAdd accumulation)
#include "mask_gemm/include/MaskGemm_wgrad_64x128x32_2s_f32_atomic.h"
#include "mask_gemm/include/MaskGemm_wgrad_64x128x32_2s_f32_workspace.h"
#include "mask_gemm/include/MaskGemm_wgrad_64x64x32_2s_f32_atomic.h"
#include "mask_gemm/include/MaskGemm_wgrad_64x64x32_2s_f32_workspace.h"
#include "mask_gemm/include/MaskGemm_wgrad_64x64x32_3s_f32_atomic.h"
#include "mask_gemm/include/MaskGemm_wgrad_64x64x32_3s_f32_workspace.h"

namespace warpconvnet {
namespace cute_gemm {

// =============================================================================
// Wgrad instantiations
// =============================================================================

// Wgrad: 64x64x32 2-stage f32 output (direct store, aligned C)
WCN_PROD_INSTANTIATE_WGRAD(MaskGemm_wgrad_64x64x32_2s_f32, cutlass::half_t, Tile64x64x32, float)
WCN_PROD_INSTANTIATE_WGRAD(MaskGemm_wgrad_64x64x32_2s_f32, cutlass::bfloat16_t, Tile64x64x32, float)

// =============================================================================
// Wgrad atomic variant launch functions (split-K with atomicAdd)
// =============================================================================

#define INST_WGRAD_ATOMIC(SUFFIX, KernelClass, ElemIn, TileTag)                    \
  template <>                                                                      \
  int launch_wgrad_atomic_##SUFFIX<ElemIn, float>(const void *a,                   \
                                                  const void *b,                   \
                                                  void *d,                         \
                                                  const int *pt,                   \
                                                  const uint32_t *pm,              \
                                                  const int *ms,                   \
                                                  const uint32_t *rm,              \
                                                  int N_in,                        \
                                                  int N_out,                       \
                                                  int C_in,                        \
                                                  int C_out,                       \
                                                  int K,                           \
                                                  int split_k,                     \
                                                  float alpha,                     \
                                                  int groups,                      \
                                                  cudaStream_t stream) {           \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                          \
    using Kernel = KernelClass<Config, float>;                                     \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});             \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});             \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                           \
    int m_tiles = (C_in + TileM - 1) / TileM;                                      \
    int n_tiles = (C_out + TileN - 1) / TileN;                                     \
    dim3 grid(m_tiles *n_tiles, groups *K, split_k);                               \
    size_t smem = Kernel::SharedStorageSize;                                       \
    if (smem > 48 * 1024) {                                                        \
      auto err = cudaFuncSetAttribute(mask_gemm_wgrad_kernel_entry<Kernel>,        \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smem);                                       \
      if (err != cudaSuccess) return -1;                                           \
    }                                                                              \
    mask_gemm_wgrad_kernel_entry<Kernel>                                           \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,    \
                                                             (const ElemIn *)b,    \
                                                             (float *)d,           \
                                                             pt,                   \
                                                             pm,                   \
                                                             ms,                   \
                                                             rm,                   \
                                                             N_in,                 \
                                                             N_out,                \
                                                             C_in,                 \
                                                             C_out,                \
                                                             K,                    \
                                                             alpha,                \
                                                             C_in * groups,        \
                                                             C_out * groups);      \
    return 0;                                                                      \
  }

INST_WGRAD_ATOMIC(64x64, MaskGemm_wgrad_64x64x32_2s_f32_atomic, cutlass::half_t, Tile64x64x32)
INST_WGRAD_ATOMIC(64x64, MaskGemm_wgrad_64x64x32_2s_f32_atomic, cutlass::bfloat16_t, Tile64x64x32)
INST_WGRAD_ATOMIC(64x128, MaskGemm_wgrad_64x128x32_2s_f32_atomic, cutlass::half_t, Tile64x128x32)
INST_WGRAD_ATOMIC(64x128,
                  MaskGemm_wgrad_64x128x32_2s_f32_atomic,
                  cutlass::bfloat16_t,
                  Tile64x128x32)
#undef INST_WGRAD_ATOMIC

// 3-stage atomic wgrad (separate launch function to avoid template key clash with 2-stage)
#define INST_WGRAD_ATOMIC_3S(ElemIn)                                               \
  template <>                                                                      \
  int launch_wgrad_atomic_3s<ElemIn, float>(const void *a,                         \
                                            const void *b,                         \
                                            void *d,                               \
                                            const int *pt,                         \
                                            const uint32_t *pm,                    \
                                            const int *ms,                         \
                                            const uint32_t *rm,                    \
                                            int N_in,                              \
                                            int N_out,                             \
                                            int C_in,                              \
                                            int C_out,                             \
                                            int K,                                 \
                                            int split_k,                           \
                                            float alpha,                           \
                                            int groups,                            \
                                            cudaStream_t stream) {                 \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                     \
    using Kernel = MaskGemm_wgrad_64x64x32_3s_f32_atomic<Config, float>;           \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});             \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});             \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                           \
    int m_tiles = (C_in + TileM - 1) / TileM;                                      \
    int n_tiles = (C_out + TileN - 1) / TileN;                                     \
    dim3 grid(m_tiles *n_tiles, groups *K, split_k);                               \
    size_t smem = Kernel::SharedStorageSize;                                       \
    if (smem > 48 * 1024) {                                                        \
      auto err = cudaFuncSetAttribute(mask_gemm_wgrad_kernel_entry<Kernel>,        \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smem);                                       \
      if (err != cudaSuccess) return -1;                                           \
    }                                                                              \
    mask_gemm_wgrad_kernel_entry<Kernel>                                           \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,    \
                                                             (const ElemIn *)b,    \
                                                             (float *)d,           \
                                                             pt,                   \
                                                             pm,                   \
                                                             ms,                   \
                                                             rm,                   \
                                                             N_in,                 \
                                                             N_out,                \
                                                             C_in,                 \
                                                             C_out,                \
                                                             K,                    \
                                                             alpha,                \
                                                             C_in * groups,        \
                                                             C_out * groups);      \
    return 0;                                                                      \
  }

INST_WGRAD_ATOMIC_3S(cutlass::half_t)
INST_WGRAD_ATOMIC_3S(cutlass::bfloat16_t)
#undef INST_WGRAD_ATOMIC_3S

// =============================================================================
// Wgrad workspace variant launch functions
//
// Each split_k shard writes to its own slice of a [split_k, K, G, C_in, C_out]
// fp32 workspace buffer. Caller (binding layer) is responsible for allocating
// the workspace, calling the launcher, and reducing workspace.sum(0) into
// grad_weight. Workspace kernel uses the same grid layout as the atomic variant
// (m_tiles*n_tiles, groups*K, split_k) but writes via direct store per slice —
// no atomic contention.
// =============================================================================

#define INST_WGRAD_WORKSPACE(SUFFIX, KernelClass, ElemIn, TileTag)                 \
  template <>                                                                      \
  int launch_wgrad_workspace_##SUFFIX<ElemIn, float>(const void *a,                \
                                                     const void *b,                \
                                                     void *d,                      \
                                                     const int *pt,                \
                                                     const uint32_t *pm,           \
                                                     const int *ms,                \
                                                     const uint32_t *rm,           \
                                                     int N_in,                     \
                                                     int N_out,                    \
                                                     int C_in,                     \
                                                     int C_out,                    \
                                                     int K,                        \
                                                     int split_k,                  \
                                                     float alpha,                  \
                                                     int groups,                   \
                                                     cudaStream_t stream) {        \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                          \
    using Kernel = KernelClass<Config, float>;                                     \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});             \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});             \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                           \
    int m_tiles = (C_in + TileM - 1) / TileM;                                      \
    int n_tiles = (C_out + TileN - 1) / TileN;                                     \
    dim3 grid(m_tiles *n_tiles, groups *K, split_k);                               \
    size_t smem = Kernel::SharedStorageSize;                                       \
    if (smem > 48 * 1024) {                                                        \
      auto err = cudaFuncSetAttribute(mask_gemm_wgrad_kernel_entry<Kernel>,        \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smem);                                       \
      if (err != cudaSuccess) return -1;                                           \
    }                                                                              \
    mask_gemm_wgrad_kernel_entry<Kernel>                                           \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,    \
                                                             (const ElemIn *)b,    \
                                                             (float *)d,           \
                                                             pt,                   \
                                                             pm,                   \
                                                             ms,                   \
                                                             rm,                   \
                                                             N_in,                 \
                                                             N_out,                \
                                                             C_in,                 \
                                                             C_out,                \
                                                             K,                    \
                                                             alpha,                \
                                                             C_in * groups,        \
                                                             C_out * groups);      \
    return 0;                                                                      \
  }

INST_WGRAD_WORKSPACE(64x64, MaskGemm_wgrad_64x64x32_2s_f32_workspace, cutlass::half_t, Tile64x64x32)
INST_WGRAD_WORKSPACE(64x64,
                     MaskGemm_wgrad_64x64x32_2s_f32_workspace,
                     cutlass::bfloat16_t,
                     Tile64x64x32)
INST_WGRAD_WORKSPACE(64x64_3s,
                     MaskGemm_wgrad_64x64x32_3s_f32_workspace,
                     cutlass::half_t,
                     Tile64x64x32)
INST_WGRAD_WORKSPACE(64x64_3s,
                     MaskGemm_wgrad_64x64x32_3s_f32_workspace,
                     cutlass::bfloat16_t,
                     Tile64x64x32)
INST_WGRAD_WORKSPACE(64x128,
                     MaskGemm_wgrad_64x128x32_2s_f32_workspace,
                     cutlass::half_t,
                     Tile64x128x32)
INST_WGRAD_WORKSPACE(64x128,
                     MaskGemm_wgrad_64x128x32_2s_f32_workspace,
                     cutlass::bfloat16_t,
                     Tile64x128x32)
#undef INST_WGRAD_WORKSPACE

// =============================================================================
// Wgrad scalar variant (unaligned C — scalar A and B loads)
// =============================================================================
#define INSTANTIATE_SCALAR_WGRAD(ElemIn, ElemOut)                                  \
  template <>                                                                      \
  int launch_scalar_wgrad_sab<ElemIn, ElemOut>(const void *a,                      \
                                               const void *b,                      \
                                               void *d,                            \
                                               const int *pt,                      \
                                               const uint32_t *pm,                 \
                                               const int *ms,                      \
                                               const uint32_t *rm,                 \
                                               int N_in,                           \
                                               int N_out,                          \
                                               int C_in,                           \
                                               int C_out,                          \
                                               int K,                              \
                                               int split_k,                        \
                                               float alpha,                        \
                                               int groups,                         \
                                               cudaStream_t stream) {              \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                     \
    using Kernel = MaskGemm_wgrad_64x64x32_2s_f32_sab<Config, ElemOut>;            \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});             \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});             \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                           \
    int m_tiles = (C_in + TileM - 1) / TileM;                                      \
    int n_tiles = (C_out + TileN - 1) / TileN;                                     \
    dim3 grid(m_tiles *n_tiles, groups *K, split_k);                               \
    size_t smem = Kernel::SharedStorageSize;                                       \
    if (smem > 48 * 1024) {                                                        \
      auto err = cudaFuncSetAttribute(mask_gemm_wgrad_kernel_entry<Kernel>,        \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smem);                                       \
      if (err != cudaSuccess) return -1;                                           \
    }                                                                              \
    mask_gemm_wgrad_kernel_entry<Kernel>                                           \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,    \
                                                             (const ElemIn *)b,    \
                                                             (ElemOut *)d,         \
                                                             pt,                   \
                                                             pm,                   \
                                                             ms,                   \
                                                             rm,                   \
                                                             N_in,                 \
                                                             N_out,                \
                                                             C_in,                 \
                                                             C_out,                \
                                                             K,                    \
                                                             alpha,                \
                                                             C_in * groups,        \
                                                             C_out * groups);      \
    return 0;                                                                      \
  }

INSTANTIATE_SCALAR_WGRAD(cutlass::half_t, float)
INSTANTIATE_SCALAR_WGRAD(cutlass::bfloat16_t, float)
#undef INSTANTIATE_SCALAR_WGRAD

}  // namespace cute_gemm
}  // namespace warpconvnet
