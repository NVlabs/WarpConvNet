// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Shared utilities for mask_gemm_kernels_{fwd,dgrad,wgrad}.cu split TUs.
// Contains: kernel-entry __global__ templates, launch_* prototype declarations,
// and instantiation-helper macros. Each split TU includes this header plus the
// kernel-class headers it needs, then emits its specializations.

#pragma once

// Canonical mask GEMM headers (committed under csrc/include/, optionally
// regenerated from warpgemm via setup.py WARPGEMM_REGEN=1). wcn_pcoff_tiles.h
// adds project-local Pcoff_* tile tags and their CuteTileConfig
// specializations.
#include "include/cute_gemm_config.h"
#include "include/gemm_mma_tiles.h"
#include "include/kernel_dispatch.h"
#include "include/mma_macros.h"
#include "include/wcn_pcoff_tiles.h"

namespace warpconvnet {
namespace cute_gemm {

// =============================================================================
// Generic kernel entry templates — instantiated per-TU as needed.
// =============================================================================

template <typename Kernel>
__global__ __launch_bounds__(
    Kernel::MaxThreadsPerBlock,
    Kernel::
        MinBlocksPerMultiprocessor) void mask_gemm_kernel_entry(const typename Kernel::ElementInput
                                                                    *ptr_A,
                                                                const typename Kernel::ElementInput
                                                                    *ptr_B,
                                                                typename Kernel::ElementOutput
                                                                    *ptr_D,
                                                                const int *pair_table,
                                                                const uint32_t *pair_mask,
                                                                const int *mask_argsort,
                                                                int N_in,
                                                                int N_out,
                                                                int C_in,
                                                                int C_out,
                                                                int K,
                                                                float alpha,
                                                                int stride_A,
                                                                int stride_D,
                                                                int identity_offset) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A,
           ptr_B,
           ptr_D,
           pair_table,
           pair_mask,
           mask_argsort,
           N_in,
           N_out,
           C_in,
           C_out,
           K,
           alpha,
           stride_A,
           stride_D,
           identity_offset,
           smem);
}

template <typename Kernel>
__global__ __launch_bounds__(
    Kernel::MaxThreadsPerBlock,
    Kernel::MinBlocksPerMultiprocessor) void mask_gemm_kernel_entry_strided(const typename Kernel::
                                                                                ElementInput *ptr_A,
                                                                            const typename Kernel::
                                                                                ElementInput *ptr_B,
                                                                            typename Kernel::
                                                                                ElementOutput
                                                                                    *ptr_D,
                                                                            const int *neighbor_map,
                                                                            int N_in,
                                                                            int N_out,
                                                                            int C_in,
                                                                            int C_out,
                                                                            int K,
                                                                            float alpha,
                                                                            int stride_A,
                                                                            int stride_D) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A,
           ptr_B,
           ptr_D,
           neighbor_map,
           N_in,
           N_out,
           C_in,
           C_out,
           K,
           alpha,
           stride_A,
           stride_D,
           smem);
}

template <typename Kernel>
__global__ __launch_bounds__(
    Kernel::MaxThreadsPerBlock,
    Kernel::MinBlocksPerMultiprocessor) void mask_gemm_wgrad_kernel_entry(const typename Kernel::
                                                                              ElementInput *ptr_A,
                                                                          const typename Kernel::
                                                                              ElementInput *ptr_B,
                                                                          typename Kernel::
                                                                              ElementOutput *ptr_D,
                                                                          const int *pair_table,
                                                                          const uint32_t *pair_mask,
                                                                          const int *mask_argsort,
                                                                          const uint32_t
                                                                              *reduced_mask,
                                                                          int N_in,
                                                                          int N_out,
                                                                          int C_in,
                                                                          int C_out,
                                                                          int K,
                                                                          float alpha,
                                                                          int stride_A,
                                                                          int stride_B) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A,
           ptr_B,
           ptr_D,
           pair_table,
           pair_mask,
           mask_argsort,
           reduced_mask,
           N_in,
           N_out,
           C_in,
           C_out,
           K,
           alpha,
           stride_A,
           stride_B,
           smem);
}

// =============================================================================
// Generic launch function template prototypes (called from bindings). Each
// split TU specializes the subset relevant to its op.
// =============================================================================

template <typename ElementInput, class TileTag, typename ElementOutput>
int launch_mask_gemm_fwd(const void *a,
                         const void *b,
                         void *d,
                         const int *pair_table,
                         const uint32_t *pair_mask,
                         const int *mask_argsort,
                         int N_in,
                         int N_out,
                         int C_in,
                         int C_out,
                         int K,
                         float alpha,
                         int groups = 1,
                         int identity_offset = -1,
                         cudaStream_t stream = 0);

template <typename ElementInput, class TileTag, typename ElementOutput>
int launch_mask_gemm_fwd_strided(const void *a,
                                 const void *b,
                                 void *d,
                                 const int *neighbor_map,
                                 int N_in,
                                 int N_out,
                                 int C_in,
                                 int C_out,
                                 int K,
                                 float alpha,
                                 int groups = 1,
                                 cudaStream_t stream = 0);

#define WCN_DECLARE_FWD_STRIDED_LAUNCH(FuncName)           \
  template <typename ElementInput, typename ElementOutput> \
  int FuncName(const void *a,                              \
               const void *b,                              \
               void *d,                                    \
               const int *neighbor_map,                    \
               int N_in,                                   \
               int N_out,                                  \
               int C_in,                                   \
               int C_out,                                  \
               int K,                                      \
               float alpha,                                \
               int groups = 1,                             \
               cudaStream_t stream = 0);

WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x64_2s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x64_3s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x128_2s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x128_3s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_128x64_2s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x64_2s_fused)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x128_2s_fused)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_128x64_2s_fused)
#undef WCN_DECLARE_FWD_STRIDED_LAUNCH

template <typename ElementInput, class TileTag, typename ElementOutput>
int launch_mask_gemm_dgrad(const void *a,
                           const void *b,
                           void *d,
                           const int *pair_table,
                           const uint32_t *pair_mask,
                           const int *mask_argsort,
                           int N_in,
                           int N_out,
                           int C_in,
                           int C_out,
                           int K,
                           float alpha,
                           int groups = 1,
                           int identity_offset = -1,
                           cudaStream_t stream = 0);

template <typename ElementInput, class TileTag, typename ElementOutput>
int launch_mask_gemm_wgrad(const void *a,
                           const void *b,
                           void *d,
                           const int *pair_table,
                           const uint32_t *pair_mask,
                           const int *mask_argsort,
                           const uint32_t *reduced_mask,
                           int N_in,
                           int N_out,
                           int C_in,
                           int C_out,
                           int K,
                           int split_k,
                           float alpha,
                           int groups = 1,
                           cudaStream_t stream = 0);

// Forward f32-output variants (aligned + scalar-B).
template <typename ElemIn>
int launch_mask_gemm_fwd_f32out(const void *a,
                                const void *b,
                                void *d,
                                const int *pt,
                                const uint32_t *pm,
                                const int *ms,
                                int N_in,
                                int N_out,
                                int C_in,
                                int C_out,
                                int K,
                                float alpha,
                                int groups = 1,
                                int identity_offset = -1,
                                cudaStream_t stream = 0);
template <typename ElemIn>
int launch_mask_gemm_fwd_f32out_sb(const void *a,
                                   const void *b,
                                   void *d,
                                   const int *pt,
                                   const uint32_t *pm,
                                   const int *ms,
                                   int N_in,
                                   int N_out,
                                   int C_in,
                                   int C_out,
                                   int K,
                                   float alpha,
                                   int groups = 1,
                                   int identity_offset = -1,
                                   cudaStream_t stream = 0);

// Forward MaskWords>1 variants.
template <typename ElemIn, int MaskWords>
int launch_mask_gemm_fwd_mw(const void *a,
                            const void *b,
                            void *d,
                            const int *pt,
                            const uint32_t *pm,
                            const int *ms,
                            int N_in,
                            int N_out,
                            int C_in,
                            int C_out,
                            int K,
                            float alpha,
                            int groups = 1,
                            int identity_offset = -1,
                            cudaStream_t stream = 0);
template <typename ElemIn, int MaskWords>
int launch_mask_gemm_fwd_f32out_mw(const void *a,
                                   const void *b,
                                   void *d,
                                   const int *pt,
                                   const uint32_t *pm,
                                   const int *ms,
                                   int N_in,
                                   int N_out,
                                   int C_in,
                                   int C_out,
                                   int K,
                                   float alpha,
                                   int groups = 1,
                                   int identity_offset = -1,
                                   cudaStream_t stream = 0);
template <typename ElemIn, int MaskWords>
int launch_mask_gemm_fwd_f32out_sb_mw(const void *a,
                                      const void *b,
                                      void *d,
                                      const int *pt,
                                      const uint32_t *pm,
                                      const int *ms,
                                      int N_in,
                                      int N_out,
                                      int C_in,
                                      int C_out,
                                      int K,
                                      float alpha,
                                      int groups = 1,
                                      int identity_offset = -1,
                                      cudaStream_t stream = 0);

template <int MW>
int launch_mask_gemm_fwd_64x128_f16acc_mw(const void *a,
                                          const void *b,
                                          void *d,
                                          const int *pt,
                                          const uint32_t *pm,
                                          const int *ms,
                                          int N_in,
                                          int N_out,
                                          int C_in,
                                          int C_out,
                                          int K,
                                          float alpha,
                                          int groups = 1,
                                          int identity_offset = -1,
                                          cudaStream_t stream = 0);
// Tile 28: 32x32 F16Accum (half-only), MW2/4 only. See mask_gemm_kernels_fwd.cu.
template <int MW>
int launch_mask_gemm_fwd_32x32_f16acc_mw(const void *a,
                                         const void *b,
                                         void *d,
                                         const int *pt,
                                         const uint32_t *pm,
                                         const int *ms,
                                         int N_in,
                                         int N_out,
                                         int C_in,
                                         int C_out,
                                         int K,
                                         float alpha,
                                         int groups = 1,
                                         int identity_offset = -1,
                                         cudaStream_t stream = 0);
template <typename ElemIn, int MW>
int launch_mask_gemm_fwd_64x128_3s_mw(const void *a,
                                      const void *b,
                                      void *d,
                                      const int *pt,
                                      const uint32_t *pm,
                                      const int *ms,
                                      int N_in,
                                      int N_out,
                                      int C_in,
                                      int C_out,
                                      int K,
                                      float alpha,
                                      int groups = 1,
                                      int identity_offset = -1,
                                      cudaStream_t stream = 0);
template <typename ElemIn, int MW>
int launch_mask_gemm_fwd_128x64_mw(const void *a,
                                   const void *b,
                                   void *d,
                                   const int *pt,
                                   const uint32_t *pm,
                                   const int *ms,
                                   int N_in,
                                   int N_out,
                                   int C_in,
                                   int C_out,
                                   int K,
                                   float alpha,
                                   int groups = 1,
                                   int identity_offset = -1,
                                   cudaStream_t stream = 0);

// Scalar fwd/dgrad variants (MW=1).
template <typename ElemIn, typename ElemOut>
int launch_scalar_fwd_sab_se(const void *,
                             const void *,
                             void *,
                             const int *,
                             const uint32_t *,
                             const int *,
                             int,
                             int,
                             int,
                             int,
                             int,
                             float,
                             int groups = 1,
                             int identity_offset = -1,
                             cudaStream_t = 0);
template <typename ElemIn, typename ElemOut>
int launch_scalar_fwd_sa(const void *,
                         const void *,
                         void *,
                         const int *,
                         const uint32_t *,
                         const int *,
                         int,
                         int,
                         int,
                         int,
                         int,
                         float,
                         int groups = 1,
                         int identity_offset = -1,
                         cudaStream_t = 0);
template <typename ElemIn, typename ElemOut>
int launch_scalar_fwd_sb_se(const void *,
                            const void *,
                            void *,
                            const int *,
                            const uint32_t *,
                            const int *,
                            int,
                            int,
                            int,
                            int,
                            int,
                            float,
                            int groups = 1,
                            int identity_offset = -1,
                            cudaStream_t = 0);
template <typename ElemIn, typename ElemOut>
int launch_scalar_dgrad_sab_se(const void *,
                               const void *,
                               void *,
                               const int *,
                               const uint32_t *,
                               const int *,
                               int,
                               int,
                               int,
                               int,
                               int,
                               float,
                               int groups = 1,
                               int identity_offset = -1,
                               cudaStream_t = 0);
template <typename ElemIn, typename ElemOut>
int launch_scalar_dgrad_sa(const void *,
                           const void *,
                           void *,
                           const int *,
                           const uint32_t *,
                           const int *,
                           int,
                           int,
                           int,
                           int,
                           int,
                           float,
                           int groups = 1,
                           int identity_offset = -1,
                           cudaStream_t = 0);
template <typename ElemIn, typename ElemOut>
int launch_scalar_dgrad_sb_se(const void *,
                              const void *,
                              void *,
                              const int *,
                              const uint32_t *,
                              const int *,
                              int,
                              int,
                              int,
                              int,
                              int,
                              float,
                              int groups = 1,
                              int identity_offset = -1,
                              cudaStream_t = 0);

// Scalar fwd/dgrad MW>1 variants.
template <typename ElemIn, typename ElemOut, int MW>
int launch_scalar_fwd_sab_se_mw(const void *,
                                const void *,
                                void *,
                                const int *,
                                const uint32_t *,
                                const int *,
                                int,
                                int,
                                int,
                                int,
                                int,
                                float,
                                int groups = 1,
                                int identity_offset = -1,
                                cudaStream_t = 0);
template <typename ElemIn, typename ElemOut, int MW>
int launch_scalar_dgrad_sab_se_mw(const void *,
                                  const void *,
                                  void *,
                                  const int *,
                                  const uint32_t *,
                                  const int *,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  float,
                                  int groups = 1,
                                  int identity_offset = -1,
                                  cudaStream_t = 0);
template <typename ElemIn, typename ElemOut, int MW>
int launch_scalar_fwd_sa_mw(const void *,
                            const void *,
                            void *,
                            const int *,
                            const uint32_t *,
                            const int *,
                            int,
                            int,
                            int,
                            int,
                            int,
                            float,
                            int groups = 1,
                            int identity_offset = -1,
                            cudaStream_t = 0);
template <typename ElemIn, typename ElemOut, int MW>
int launch_scalar_fwd_sb_se_mw(const void *,
                               const void *,
                               void *,
                               const int *,
                               const uint32_t *,
                               const int *,
                               int,
                               int,
                               int,
                               int,
                               int,
                               float,
                               int groups = 1,
                               int identity_offset = -1,
                               cudaStream_t = 0);
template <typename ElemIn, typename ElemOut, int MW>
int launch_scalar_dgrad_sa_mw(const void *,
                              const void *,
                              void *,
                              const int *,
                              const uint32_t *,
                              const int *,
                              int,
                              int,
                              int,
                              int,
                              int,
                              float,
                              int groups = 1,
                              int identity_offset = -1,
                              cudaStream_t = 0);
template <typename ElemIn, typename ElemOut, int MW>
int launch_scalar_dgrad_sb_se_mw(const void *,
                                 const void *,
                                 void *,
                                 const int *,
                                 const uint32_t *,
                                 const int *,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 float,
                                 int groups = 1,
                                 int identity_offset = -1,
                                 cudaStream_t = 0);

// Dgrad helper variants.
template <typename ElemIn, typename ElemOut>
int launch_dgrad_pipelined_64x64(const void *,
                                 const void *,
                                 void *,
                                 const int *,
                                 const uint32_t *,
                                 const int *,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 float,
                                 int groups = 1,
                                 int identity_offset = -1,
                                 cudaStream_t = 0);
template <typename ElemIn, typename ElemOut>
int launch_dgrad_pipelined_64x128(const void *,
                                  const void *,
                                  void *,
                                  const int *,
                                  const uint32_t *,
                                  const int *,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  float,
                                  int groups = 1,
                                  int identity_offset = -1,
                                  cudaStream_t = 0);
template <typename ElemIn, typename ElemOut>
int launch_dgrad_pipelined_128x64(const void *,
                                  const void *,
                                  void *,
                                  const int *,
                                  const uint32_t *,
                                  const int *,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  float,
                                  int groups = 1,
                                  int identity_offset = -1,
                                  cudaStream_t = 0);
template <typename ElemIn>
int launch_mask_gemm_dgrad_f32out(const void *a,
                                  const void *b,
                                  void *d,
                                  const int *pt,
                                  const uint32_t *pm,
                                  const int *ms,
                                  int N_in,
                                  int N_out,
                                  int C_in,
                                  int C_out,
                                  int K,
                                  float alpha,
                                  int groups = 1,
                                  int identity_offset = -1,
                                  cudaStream_t stream = 0);
template <typename ElemIn, int MaskWords>
int launch_mask_gemm_dgrad_f32out_mw(const void *a,
                                     const void *b,
                                     void *d,
                                     const int *pt,
                                     const uint32_t *pm,
                                     const int *ms,
                                     int N_in,
                                     int N_out,
                                     int C_in,
                                     int C_out,
                                     int K,
                                     float alpha,
                                     int groups = 1,
                                     int identity_offset = -1,
                                     cudaStream_t stream = 0);
template <typename ElemIn, int MaskWords>
int launch_mask_gemm_dgrad_mw(const void *a,
                              const void *b,
                              void *d,
                              const int *pt,
                              const uint32_t *pm,
                              const int *ms,
                              int N_in,
                              int N_out,
                              int C_in,
                              int C_out,
                              int K,
                              float alpha,
                              int groups = 1,
                              int identity_offset = -1,
                              cudaStream_t stream = 0);

// Wgrad helper variants.
template <typename ElemIn, typename ElemOut>
int launch_wgrad_atomic_64x64(const void *,
                              const void *,
                              void *,
                              const int *,
                              const uint32_t *,
                              const int *,
                              const uint32_t *,
                              int,
                              int,
                              int,
                              int,
                              int,
                              int,
                              float,
                              int,
                              cudaStream_t);
template <typename ElemIn, typename ElemOut>
int launch_wgrad_atomic_64x128(const void *,
                               const void *,
                               void *,
                               const int *,
                               const uint32_t *,
                               const int *,
                               const uint32_t *,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               float,
                               int,
                               cudaStream_t);
template <typename ElemIn, typename ElemOut>
int launch_wgrad_atomic_3s(const void *,
                           const void *,
                           void *,
                           const int *,
                           const uint32_t *,
                           const int *,
                           const uint32_t *,
                           int,
                           int,
                           int,
                           int,
                           int,
                           int,
                           float,
                           int,
                           cudaStream_t);
template <typename ElemIn, typename ElemOut>
int launch_wgrad_workspace_64x64(const void *,
                                 const void *,
                                 void *,
                                 const int *,
                                 const uint32_t *,
                                 const int *,
                                 const uint32_t *,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 float,
                                 int,
                                 cudaStream_t);
template <typename ElemIn, typename ElemOut>
int launch_wgrad_workspace_64x64_3s(const void *,
                                    const void *,
                                    void *,
                                    const int *,
                                    const uint32_t *,
                                    const int *,
                                    const uint32_t *,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    float,
                                    int,
                                    cudaStream_t);
template <typename ElemIn, typename ElemOut>
int launch_wgrad_workspace_64x128(const void *,
                                  const void *,
                                  void *,
                                  const int *,
                                  const uint32_t *,
                                  const int *,
                                  const uint32_t *,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  float,
                                  int,
                                  cudaStream_t);
template <typename ElemIn, typename ElemOut>
int launch_scalar_wgrad_sab(const void *a,
                            const void *b,
                            void *d,
                            const int *pt,
                            const uint32_t *pm,
                            const int *ms,
                            const uint32_t *rm,
                            int N_in,
                            int N_out,
                            int C_in,
                            int C_out,
                            int K,
                            int split_k,
                            float alpha,
                            int groups,
                            cudaStream_t stream);

}  // namespace cute_gemm
}  // namespace warpconvnet

// =============================================================================
// Instantiation-helper macros — used inside `namespace warpconvnet::cute_gemm`
// blocks of the split TUs. Defined at file scope (post-namespace) so they are
// visible to subsequent .cu sources that include this header.
// =============================================================================

#define WCN_PROD_INSTANTIATE_FWD(KernelClass, ElemIn, TileTag, ElemOut)                       \
  template <>                                                                                 \
  int launch_mask_gemm_fwd<ElemIn, gemm::TileTag, ElemOut>(const void *a,                     \
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
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                                     \
    using Kernel = KernelClass<Config, ElemOut>;                                              \
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
                                                             (ElemOut *)d,                    \
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

#define WCN_PROD_INSTANTIATE_FWD_STRIDED(KernelClass, ElemIn, TileTag, ElemOut)             \
  template <>                                                                               \
  int launch_mask_gemm_fwd_strided<ElemIn, gemm::TileTag, ElemOut>(const void *a,           \
                                                                   const void *b,           \
                                                                   void *d,                 \
                                                                   const int *neighbor_map, \
                                                                   int N_in,                \
                                                                   int N_out,               \
                                                                   int C_in,                \
                                                                   int C_out,               \
                                                                   int K,                   \
                                                                   float alpha,             \
                                                                   int groups,              \
                                                                   cudaStream_t stream) {   \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                                   \
    using Kernel = KernelClass<Config, ElemOut>;                                            \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                      \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                      \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                                    \
    int m_tiles = (N_out + TileM - 1) / TileM;                                              \
    int n_tiles = (C_out + TileN - 1) / TileN;                                              \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                                 \
    size_t smem = Kernel::SharedStorageSize;                                                \
    if (smem > 48 * 1024) {                                                                 \
      auto err = cudaFuncSetAttribute(mask_gemm_kernel_entry_strided<Kernel>,               \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                                      smem);                                                \
      if (err != cudaSuccess) return -1;                                                    \
    }                                                                                       \
    mask_gemm_kernel_entry_strided<Kernel>                                                  \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,             \
                                                             (const ElemIn *)b,             \
                                                             (ElemOut *)d,                  \
                                                             neighbor_map,                  \
                                                             N_in,                          \
                                                             N_out,                         \
                                                             C_in,                          \
                                                             C_out,                         \
                                                             K,                             \
                                                             alpha,                         \
                                                             C_in * groups,                 \
                                                             C_out * groups);               \
    return 0;                                                                               \
  }

#define WCN_PROD_INSTANTIATE_FWD_STRIDED_NAMED(FuncName, KernelClass, ElemIn, TileTag, ElemOut) \
  template <>                                                                                   \
  int FuncName<ElemIn, ElemOut>(const void *a,                                                  \
                                const void *b,                                                  \
                                void *d,                                                        \
                                const int *neighbor_map,                                        \
                                int N_in,                                                       \
                                int N_out,                                                      \
                                int C_in,                                                       \
                                int C_out,                                                      \
                                int K,                                                          \
                                float alpha,                                                    \
                                int groups,                                                     \
                                cudaStream_t stream) {                                          \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                                       \
    using Kernel = KernelClass<Config, ElemOut>;                                                \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                          \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                          \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                                        \
    int m_tiles = (N_out + TileM - 1) / TileM;                                                  \
    int n_tiles = (C_out + TileN - 1) / TileN;                                                  \
    dim3 grid(m_tiles *n_tiles, 1, groups);                                                     \
    size_t smem = Kernel::SharedStorageSize;                                                    \
    if (smem > 48 * 1024) {                                                                     \
      auto err = cudaFuncSetAttribute(mask_gemm_kernel_entry_strided<Kernel>,                   \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,              \
                                      smem);                                                    \
      if (err != cudaSuccess) return -1;                                                        \
    }                                                                                           \
    mask_gemm_kernel_entry_strided<Kernel>                                                      \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,                 \
                                                             (const ElemIn *)b,                 \
                                                             (ElemOut *)d,                      \
                                                             neighbor_map,                      \
                                                             N_in,                              \
                                                             N_out,                             \
                                                             C_in,                              \
                                                             C_out,                             \
                                                             K,                                 \
                                                             alpha,                             \
                                                             C_in * groups,                     \
                                                             C_out * groups);                   \
    return 0;                                                                                   \
  }

#define WCN_PROD_INSTANTIATE_DGRAD(KernelClass, ElemIn, TileTag, ElemOut)                     \
  template <>                                                                                 \
  int launch_mask_gemm_dgrad<ElemIn, gemm::TileTag, ElemOut>(const void *a,                   \
                                                             const void *b,                   \
                                                             void *d,                         \
                                                             const int *pt,                   \
                                                             const uint32_t *pm,              \
                                                             const int *ms,                   \
                                                             int N_in,                        \
                                                             int N_out,                       \
                                                             int C_in,                        \
                                                             int C_out,                       \
                                                             int K,                           \
                                                             float alpha,                     \
                                                             int groups,                      \
                                                             int identity_offset,             \
                                                             cudaStream_t stream) {           \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                                     \
    using Kernel = KernelClass<Config, ElemOut>;                                              \
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
                                                             (ElemOut *)d,                    \
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

#define WCN_PROD_INSTANTIATE_WGRAD(KernelClass, ElemIn, TileTag, ElemOut)           \
  template <>                                                                       \
  int launch_mask_gemm_wgrad<ElemIn, gemm::TileTag, ElemOut>(const void *a,         \
                                                             const void *b,         \
                                                             void *d,               \
                                                             const int *pt,         \
                                                             const uint32_t *pm,    \
                                                             const int *ms,         \
                                                             const uint32_t *rm,    \
                                                             int N_in,              \
                                                             int N_out,             \
                                                             int C_in,              \
                                                             int C_out,             \
                                                             int K,                 \
                                                             int split_k,           \
                                                             float alpha,           \
                                                             int groups,            \
                                                             cudaStream_t stream) { \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                           \
    using Kernel = KernelClass<Config, ElemOut>;                                    \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});              \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});              \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                            \
    int m_tiles = (C_in + TileM - 1) / TileM;                                       \
    int n_tiles = (C_out + TileN - 1) / TileN;                                      \
    dim3 grid(m_tiles *n_tiles, groups *K, split_k);                                \
    size_t smem = Kernel::SharedStorageSize;                                        \
    if (smem > 48 * 1024) {                                                         \
      auto err = cudaFuncSetAttribute(mask_gemm_wgrad_kernel_entry<Kernel>,         \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,  \
                                      smem);                                        \
      if (err != cudaSuccess) return -1;                                            \
    }                                                                               \
    mask_gemm_wgrad_kernel_entry<Kernel>                                            \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,     \
                                                             (const ElemIn *)b,     \
                                                             (ElemOut *)d,          \
                                                             pt,                    \
                                                             pm,                    \
                                                             ms,                    \
                                                             rm,                    \
                                                             N_in,                  \
                                                             N_out,                 \
                                                             C_in,                  \
                                                             C_out,                 \
                                                             K,                     \
                                                             alpha,                 \
                                                             C_in * groups,         \
                                                             C_out * groups);       \
    return 0;                                                                       \
  }

// Scalar fwd/dgrad MW=1 launcher (suffix selects sa/sb_se/sab_se).
#define WCN_DEFINE_SCALAR_LAUNCH(OP, SUFFIX, KernelClass, ElemIn, ElemOut)                    \
  template <>                                                                                 \
  int launch_scalar_##OP##_##SUFFIX<ElemIn, ElemOut>(const void *a,                           \
                                                     const void *b,                           \
                                                     void *d,                                 \
                                                     const int *pt,                           \
                                                     const uint32_t *pm,                      \
                                                     const int *ms,                           \
                                                     int N_in,                                \
                                                     int N_out,                               \
                                                     int C_in,                                \
                                                     int C_out,                               \
                                                     int K,                                   \
                                                     float alpha,                             \
                                                     int groups,                              \
                                                     int identity_offset,                     \
                                                     cudaStream_t stream) {                   \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                                \
    using Kernel = KernelClass<Config, ElemOut>;                                              \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                        \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                        \
    bool is_fwd = (std::string(#OP) == "fwd");                                                \
    int out_rows = is_fwd ? N_out : N_in;                                                     \
    int out_cols = is_fwd ? C_out : C_in;                                                     \
    int stride_a = (is_fwd ? C_in : C_out) * groups;                                          \
    int stride_d = (is_fwd ? C_out : C_in) * groups;                                          \
    if (out_rows == 0 || C_in == 0 || C_out == 0) return 0;                                   \
    int m_tiles = (out_rows + TileM - 1) / TileM;                                             \
    int n_tiles = (out_cols + TileN - 1) / TileN;                                             \
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
                                                             (ElemOut *)d,                    \
                                                             pt,                              \
                                                             pm,                              \
                                                             ms,                              \
                                                             N_in,                            \
                                                             N_out,                           \
                                                             C_in,                            \
                                                             C_out,                           \
                                                             K,                               \
                                                             alpha,                           \
                                                             stride_a,                        \
                                                             stride_d,                        \
                                                             identity_offset);                \
    return 0;                                                                                 \
  }

// Scalar fwd/dgrad SAB_SE MW>1 launcher.
#define WCN_DEFINE_SCALAR_LAUNCH_SAB_SE_MW(OP, KernelClass, ElemIn, ElemOut, MW)              \
  template <>                                                                                 \
  int launch_scalar_##OP##_sab_se_mw<ElemIn, ElemOut, MW>(const void *a,                      \
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
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                                \
    using Kernel = KernelClass<Config, ElemOut, MW>;                                          \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                        \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                        \
    bool is_fwd = (std::string(#OP) == "fwd");                                                \
    int out_rows = is_fwd ? N_out : N_in;                                                     \
    int out_cols = is_fwd ? C_out : C_in;                                                     \
    int stride_a = (is_fwd ? C_in : C_out) * groups;                                          \
    int stride_d = (is_fwd ? C_out : C_in) * groups;                                          \
    if (out_rows == 0 || C_in == 0 || C_out == 0) return 0;                                   \
    int m_tiles = (out_rows + TileM - 1) / TileM;                                             \
    int n_tiles = (out_cols + TileN - 1) / TileN;                                             \
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
                                                             (ElemOut *)d,                    \
                                                             pt,                              \
                                                             pm,                              \
                                                             ms,                              \
                                                             N_in,                            \
                                                             N_out,                           \
                                                             C_in,                            \
                                                             C_out,                           \
                                                             K,                               \
                                                             alpha,                           \
                                                             stride_a,                        \
                                                             stride_d,                        \
                                                             identity_offset);                \
    return 0;                                                                                 \
  }

// Scalar fwd/dgrad SA / SB_SE MW>1 launcher.
#define WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(OP, SUFFIX, KernelClass, ElemIn, ElemOut, MW)       \
  template <>                                                                                 \
  int launch_scalar_##OP##_##SUFFIX##_mw<ElemIn, ElemOut, MW>(const void *a,                  \
                                                              const void *b,                  \
                                                              void *d,                        \
                                                              const int *pt,                  \
                                                              const uint32_t *pm,             \
                                                              const int *ms,                  \
                                                              int N_in,                       \
                                                              int N_out,                      \
                                                              int C_in,                       \
                                                              int C_out,                      \
                                                              int K,                          \
                                                              float alpha,                    \
                                                              int groups,                     \
                                                              int identity_offset,            \
                                                              cudaStream_t stream) {          \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                                \
    using Kernel = KernelClass<Config, ElemOut, MW>;                                          \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});                        \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});                        \
    bool is_fwd = (std::string(#OP) == "fwd");                                                \
    int out_rows = is_fwd ? N_out : N_in;                                                     \
    int out_cols = is_fwd ? C_out : C_in;                                                     \
    int stride_a = (is_fwd ? C_in : C_out) * groups;                                          \
    int stride_d = (is_fwd ? C_out : C_in) * groups;                                          \
    if (out_rows == 0 || C_in == 0 || C_out == 0) return 0;                                   \
    int m_tiles = (out_rows + TileM - 1) / TileM;                                             \
    int n_tiles = (out_cols + TileN - 1) / TileN;                                             \
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
                                                             (ElemOut *)d,                    \
                                                             pt,                              \
                                                             pm,                              \
                                                             ms,                              \
                                                             N_in,                            \
                                                             N_out,                           \
                                                             C_in,                            \
                                                             C_out,                           \
                                                             K,                               \
                                                             alpha,                           \
                                                             stride_a,                        \
                                                             stride_d,                        \
                                                             identity_offset);                \
    return 0;                                                                                 \
  }

#define WCN_INST_ALL_MW_SA_SB(OP, SUFFIX, KernelClass)                                             \
  WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(OP, SUFFIX, KernelClass, cutlass::half_t, cutlass::half_t, 2)  \
  WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(OP, SUFFIX, KernelClass, cutlass::half_t, cutlass::half_t, 4)  \
  WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(OP, SUFFIX, KernelClass, cutlass::half_t, cutlass::half_t, 8)  \
  WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(OP, SUFFIX, KernelClass, cutlass::half_t, cutlass::half_t, 12) \
  WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(                                                               \
      OP, SUFFIX, KernelClass, cutlass::bfloat16_t, cutlass::bfloat16_t, 2)                        \
  WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(                                                               \
      OP, SUFFIX, KernelClass, cutlass::bfloat16_t, cutlass::bfloat16_t, 4)                        \
  WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(                                                               \
      OP, SUFFIX, KernelClass, cutlass::bfloat16_t, cutlass::bfloat16_t, 8)                        \
  WCN_DEFINE_SCALAR_LAUNCH_SA_SB_MW(                                                               \
      OP, SUFFIX, KernelClass, cutlass::bfloat16_t, cutlass::bfloat16_t, 12)
