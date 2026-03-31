// Production mask GEMM kernel instantiations.
// Generated kernels with warp shuffle, precomputed rows,
// and double-buffered register MMA.

#include "include/cute_gemm_config.h"
#include "include/gemm_mma_tiles.h"
#include "include/mma_macros.h"

// Forward kernels
#include "include/MaskGemm_forward_128x64x32_2s_fused.h"
#include "include/MaskGemm_forward_32x32x32_1s_flat.h"
#include "include/MaskGemm_forward_64x128x32_2s_fused.h"
#include "include/MaskGemm_forward_64x128x32_3s.h"
#include "include/MaskGemm_forward_64x64x32_2s_pipelined.h"
// Forward scalar variants (for unaligned C)
#include "include/MaskGemm_forward_64x64x32_1s_flat_sa.h"
#include "include/MaskGemm_forward_64x64x32_1s_flat_sab_se.h"
#include "include/MaskGemm_forward_64x64x32_1s_flat_sb_se.h"

// Dgrad kernels
#include "include/MaskGemm_dgrad_32x32x32_1s_flat.h"
#include "include/MaskGemm_dgrad_64x128x32_1s_flat_direpi.h"
#include "include/MaskGemm_dgrad_64x64x32_1s_flat.h"
// Dgrad scalar variants
#include "include/MaskGemm_dgrad_64x64x32_1s_flat_sa.h"
#include "include/MaskGemm_dgrad_64x64x32_1s_flat_sab_se.h"
#include "include/MaskGemm_dgrad_64x64x32_1s_flat_sb_se.h"

// Wgrad kernels
#include "include/MaskGemm_wgrad_64x64x32_2s_f32.h"
#include "include/MaskGemm_wgrad_64x64x32_2s_f32_sab.h"

namespace warpconvnet {
namespace cute_gemm {

// =============================================================================
// Launch wrapper: forward / dgrad
// =============================================================================

template <typename Kernel>
__global__ __launch_bounds__(
    Kernel::MaxThreadsPerBlock,
    Kernel::MinBlocksPerMultiprocessor) void production_mask_kernel_entry(const typename Kernel::
                                                                              ElementInput *ptr_A,
                                                                          const typename Kernel::
                                                                              ElementInput *ptr_B,
                                                                          typename Kernel::
                                                                              ElementOutput *ptr_D,
                                                                          const int *pair_table,
                                                                          const uint32_t *pair_mask,
                                                                          const int *mask_argsort,
                                                                          int N_in,
                                                                          int N_out,
                                                                          int C_in,
                                                                          int C_out,
                                                                          int K,
                                                                          float alpha) {
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
           smem);
}

// =============================================================================
// Launch wrapper: wgrad (extra reduced_mask parameter)
// =============================================================================

template <typename Kernel>
__global__ __launch_bounds__(
    Kernel::MaxThreadsPerBlock,
    Kernel::MinBlocksPerMultiprocessor) void production_wgrad_kernel_entry(const typename Kernel::
                                                                               ElementInput *ptr_A,
                                                                           const typename Kernel::
                                                                               ElementInput *ptr_B,
                                                                           typename Kernel::
                                                                               ElementOutput *ptr_D,
                                                                           const int *pair_table,
                                                                           const uint32_t
                                                                               *pair_mask,
                                                                           const int *mask_argsort,
                                                                           const uint32_t
                                                                               *reduced_mask,
                                                                           int N_in,
                                                                           int N_out,
                                                                           int C_in,
                                                                           int C_out,
                                                                           int K,
                                                                           float alpha) {
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
           smem);
}

// =============================================================================
// Generic launch functions (called from bindings)
// =============================================================================

template <typename ElementInput, class TileTag, typename ElementOutput>
int launch_production_fwd(const void *a,
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
                          cudaStream_t stream = 0);

template <typename ElementInput, class TileTag, typename ElementOutput>
int launch_production_dgrad(const void *a,
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
                            cudaStream_t stream = 0);

template <typename ElementInput, class TileTag, typename ElementOutput>
int launch_production_wgrad(const void *a,
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
                            cudaStream_t stream = 0);

// =============================================================================
// Forward instantiations
// =============================================================================

#define INSTANTIATE_PROD_FWD(KernelClass, ElemIn, TileTag, ElemOut)                \
  template <>                                                                      \
  int launch_production_fwd<ElemIn, gemm::TileTag, ElemOut>(const void *a,         \
                                                            const void *b,         \
                                                            void *d,               \
                                                            const int *pt,         \
                                                            const uint32_t *pm,    \
                                                            const int *ms,         \
                                                            int N_in,              \
                                                            int N_out,             \
                                                            int C_in,              \
                                                            int C_out,             \
                                                            int K,                 \
                                                            float alpha,           \
                                                            cudaStream_t stream) { \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                          \
    using Kernel = KernelClass<Config, ElemOut>;                                   \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});             \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});             \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                           \
    int m_tiles = (N_out + TileM - 1) / TileM;                                     \
    int n_tiles = (C_out + TileN - 1) / TileN;                                     \
    dim3 grid(m_tiles *n_tiles);                                                   \
    size_t smem = Kernel::SharedStorageSize;                                       \
    if (smem > 48 * 1024) {                                                        \
      auto err = cudaFuncSetAttribute(production_mask_kernel_entry<Kernel>,        \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smem);                                       \
      if (err != cudaSuccess) return -1;                                           \
    }                                                                              \
    production_mask_kernel_entry<Kernel>                                           \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,    \
                                                             (const ElemIn *)b,    \
                                                             (ElemOut *)d,         \
                                                             pt,                   \
                                                             pm,                   \
                                                             ms,                   \
                                                             N_in,                 \
                                                             N_out,                \
                                                             C_in,                 \
                                                             C_out,                \
                                                             K,                    \
                                                             alpha);               \
    return 0;                                                                      \
  }

// Forward: 32x32x32 flat (C<=48, fp16 only — uses F16 accum)
INSTANTIATE_PROD_FWD(MaskGemm_forward_32x32x32_1s_flat,
                     cutlass::half_t,
                     Tile32x32x32_F16Accum,
                     cutlass::half_t)

// Forward: 64x64x32 pipelined (C=64 or C<=48 bf16)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x64x32_2s_pipelined,
                     cutlass::half_t,
                     Tile64x64x32,
                     cutlass::half_t)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x64x32_2s_pipelined, cutlass::half_t, Tile64x64x32, float)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x64x32_2s_pipelined,
                     cutlass::bfloat16_t,
                     Tile64x64x32,
                     cutlass::bfloat16_t)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x64x32_2s_pipelined,
                     cutlass::bfloat16_t,
                     Tile64x64x32,
                     float)

// Forward: 64x128x32 fused (C>=128, C_in>=C_out, fp16 — F16 accum)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x128x32_2s_fused,
                     cutlass::half_t,
                     Tile64x128x32_F16Accum,
                     cutlass::half_t)

// Forward: 64x128x32 3-stage (C>=128)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x128x32_3s, cutlass::half_t, Tile64x128x32, cutlass::half_t)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x128x32_3s, cutlass::half_t, Tile64x128x32, float)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x128x32_3s,
                     cutlass::bfloat16_t,
                     Tile64x128x32,
                     cutlass::bfloat16_t)
INSTANTIATE_PROD_FWD(MaskGemm_forward_64x128x32_3s, cutlass::bfloat16_t, Tile64x128x32, float)

// Forward: 128x64x32 fused (C>=128, C_in<C_out)
INSTANTIATE_PROD_FWD(MaskGemm_forward_128x64x32_2s_fused,
                     cutlass::half_t,
                     Tile128x64x32,
                     cutlass::half_t)
INSTANTIATE_PROD_FWD(MaskGemm_forward_128x64x32_2s_fused, cutlass::half_t, Tile128x64x32, float)
INSTANTIATE_PROD_FWD(MaskGemm_forward_128x64x32_2s_fused,
                     cutlass::bfloat16_t,
                     Tile128x64x32,
                     cutlass::bfloat16_t)
INSTANTIATE_PROD_FWD(MaskGemm_forward_128x64x32_2s_fused, cutlass::bfloat16_t, Tile128x64x32, float)

// =============================================================================
// Scalar variant launch functions (separate from generic template to avoid
// duplicate specializations — all use Tile64x64x32 config)
// =============================================================================

#define DEFINE_SCALAR_LAUNCH(OP, SUFFIX, KernelClass, ElemIn, ElemOut)             \
  template <>                                                                      \
  int launch_scalar_##OP##_##SUFFIX<ElemIn, ElemOut>(const void *a,                \
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
                                                     cudaStream_t stream) {        \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                     \
    using Kernel = KernelClass<Config, ElemOut>;                                   \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});             \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});             \
    int out_rows = (std::string(#OP) == "fwd") ? N_out : N_in;                     \
    int out_cols = (std::string(#OP) == "fwd") ? C_out : C_in;                     \
    if (out_rows == 0 || C_in == 0 || C_out == 0) return 0;                        \
    int m_tiles = (out_rows + TileM - 1) / TileM;                                  \
    int n_tiles = (out_cols + TileN - 1) / TileN;                                  \
    dim3 grid(m_tiles *n_tiles);                                                   \
    size_t smem = Kernel::SharedStorageSize;                                       \
    if (smem > 48 * 1024) {                                                        \
      auto err = cudaFuncSetAttribute(production_mask_kernel_entry<Kernel>,        \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smem);                                       \
      if (err != cudaSuccess) return -1;                                           \
    }                                                                              \
    production_mask_kernel_entry<Kernel>                                           \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,    \
                                                             (const ElemIn *)b,    \
                                                             (ElemOut *)d,         \
                                                             pt,                   \
                                                             pm,                   \
                                                             ms,                   \
                                                             N_in,                 \
                                                             N_out,                \
                                                             C_in,                 \
                                                             C_out,                \
                                                             K,                    \
                                                             alpha);               \
    return 0;                                                                      \
  }

// Declare scalar launch function templates
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
                             cudaStream_t);
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
                         cudaStream_t);
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
                            cudaStream_t);
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
                               cudaStream_t);
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
                           cudaStream_t);
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
                              cudaStream_t);

// Forward scalar instantiations
DEFINE_SCALAR_LAUNCH(
    fwd, sab_se, MaskGemm_forward_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t)
DEFINE_SCALAR_LAUNCH(
    fwd, sab_se, MaskGemm_forward_64x64x32_1s_flat_sab_se, cutlass::bfloat16_t, cutlass::bfloat16_t)
DEFINE_SCALAR_LAUNCH(
    fwd, sa, MaskGemm_forward_64x64x32_1s_flat_sa, cutlass::half_t, cutlass::half_t)
DEFINE_SCALAR_LAUNCH(
    fwd, sa, MaskGemm_forward_64x64x32_1s_flat_sa, cutlass::bfloat16_t, cutlass::bfloat16_t)
DEFINE_SCALAR_LAUNCH(
    fwd, sb_se, MaskGemm_forward_64x64x32_1s_flat_sb_se, cutlass::half_t, cutlass::half_t)
DEFINE_SCALAR_LAUNCH(
    fwd, sb_se, MaskGemm_forward_64x64x32_1s_flat_sb_se, cutlass::bfloat16_t, cutlass::bfloat16_t)

// Dgrad scalar instantiations
DEFINE_SCALAR_LAUNCH(
    dgrad, sab_se, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::half_t, cutlass::half_t)
DEFINE_SCALAR_LAUNCH(
    dgrad, sab_se, MaskGemm_dgrad_64x64x32_1s_flat_sab_se, cutlass::bfloat16_t, cutlass::bfloat16_t)
DEFINE_SCALAR_LAUNCH(
    dgrad, sa, MaskGemm_dgrad_64x64x32_1s_flat_sa, cutlass::half_t, cutlass::half_t)
DEFINE_SCALAR_LAUNCH(
    dgrad, sa, MaskGemm_dgrad_64x64x32_1s_flat_sa, cutlass::bfloat16_t, cutlass::bfloat16_t)
DEFINE_SCALAR_LAUNCH(
    dgrad, sb_se, MaskGemm_dgrad_64x64x32_1s_flat_sb_se, cutlass::half_t, cutlass::half_t)
DEFINE_SCALAR_LAUNCH(
    dgrad, sb_se, MaskGemm_dgrad_64x64x32_1s_flat_sb_se, cutlass::bfloat16_t, cutlass::bfloat16_t)

// =============================================================================
// Dgrad instantiations
// =============================================================================

#define INSTANTIATE_PROD_DGRAD(KernelClass, ElemIn, TileTag, ElemOut)                \
  template <>                                                                        \
  int launch_production_dgrad<ElemIn, gemm::TileTag, ElemOut>(const void *a,         \
                                                              const void *b,         \
                                                              void *d,               \
                                                              const int *pt,         \
                                                              const uint32_t *pm,    \
                                                              const int *ms,         \
                                                              int N_in,              \
                                                              int N_out,             \
                                                              int C_in,              \
                                                              int C_out,             \
                                                              int K,                 \
                                                              float alpha,           \
                                                              cudaStream_t stream) { \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                            \
    using Kernel = KernelClass<Config, ElemOut>;                                     \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});               \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});               \
    if (N_in == 0 || C_in == 0 || C_out == 0) return 0;                              \
    int m_tiles = (N_in + TileM - 1) / TileM;                                        \
    int n_tiles = (C_in + TileN - 1) / TileN;                                        \
    dim3 grid(m_tiles *n_tiles);                                                     \
    size_t smem = Kernel::SharedStorageSize;                                         \
    if (smem > 48 * 1024) {                                                          \
      auto err = cudaFuncSetAttribute(production_mask_kernel_entry<Kernel>,          \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                                      smem);                                         \
      if (err != cudaSuccess) return -1;                                             \
    }                                                                                \
    production_mask_kernel_entry<Kernel>                                             \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,      \
                                                             (const ElemIn *)b,      \
                                                             (ElemOut *)d,           \
                                                             pt,                     \
                                                             pm,                     \
                                                             ms,                     \
                                                             N_in,                   \
                                                             N_out,                  \
                                                             C_in,                   \
                                                             C_out,                  \
                                                             K,                      \
                                                             alpha);                 \
    return 0;                                                                        \
  }

// Dgrad: 32x32x32 flat (C<=48, fp16)
INSTANTIATE_PROD_DGRAD(MaskGemm_dgrad_32x32x32_1s_flat,
                       cutlass::half_t,
                       Tile32x32x32,
                       cutlass::half_t)

// Dgrad: 64x64x32 flat (C<=96)
INSTANTIATE_PROD_DGRAD(MaskGemm_dgrad_64x64x32_1s_flat,
                       cutlass::half_t,
                       Tile64x64x32,
                       cutlass::half_t)
INSTANTIATE_PROD_DGRAD(MaskGemm_dgrad_64x64x32_1s_flat,
                       cutlass::half_t,
                       Tile64x64x32_F16Accum,
                       cutlass::half_t)
INSTANTIATE_PROD_DGRAD(MaskGemm_dgrad_64x64x32_1s_flat,
                       cutlass::bfloat16_t,
                       Tile64x64x32,
                       cutlass::bfloat16_t)

// Dgrad: 64x128x32 flat direct epilogue (C>=128)
INSTANTIATE_PROD_DGRAD(MaskGemm_dgrad_64x128x32_1s_flat_direpi,
                       cutlass::half_t,
                       Tile64x128x32,
                       cutlass::half_t)
INSTANTIATE_PROD_DGRAD(MaskGemm_dgrad_64x128x32_1s_flat_direpi,
                       cutlass::half_t,
                       Tile64x128x32_F16Accum,
                       cutlass::half_t)
INSTANTIATE_PROD_DGRAD(MaskGemm_dgrad_64x128x32_1s_flat_direpi,
                       cutlass::bfloat16_t,
                       Tile64x128x32,
                       cutlass::bfloat16_t)

// =============================================================================
// Wgrad instantiations
// =============================================================================

#define INSTANTIATE_PROD_WGRAD(KernelClass, ElemIn, TileTag, ElemOut)                \
  template <>                                                                        \
  int launch_production_wgrad<ElemIn, gemm::TileTag, ElemOut>(const void *a,         \
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
                                                              cudaStream_t stream) { \
    using Config = CuteTileConfig<ElemIn, gemm::TileTag>;                            \
    using Kernel = KernelClass<Config, ElemOut>;                                     \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});               \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});               \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                             \
    int m_tiles = (C_in + TileM - 1) / TileM;                                        \
    int n_tiles = (C_out + TileN - 1) / TileN;                                       \
    dim3 grid(m_tiles *n_tiles, K, split_k);                                         \
    size_t smem = Kernel::SharedStorageSize;                                         \
    if (smem > 48 * 1024) {                                                          \
      auto err = cudaFuncSetAttribute(production_wgrad_kernel_entry<Kernel>,         \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                                      smem);                                         \
      if (err != cudaSuccess) return -1;                                             \
    }                                                                                \
    production_wgrad_kernel_entry<Kernel>                                            \
        <<<grid, Kernel::MaxThreadsPerBlock, smem, stream>>>((const ElemIn *)a,      \
                                                             (const ElemIn *)b,      \
                                                             (ElemOut *)d,           \
                                                             pt,                     \
                                                             pm,                     \
                                                             ms,                     \
                                                             rm,                     \
                                                             N_in,                   \
                                                             N_out,                  \
                                                             C_in,                   \
                                                             C_out,                  \
                                                             K,                      \
                                                             alpha);                 \
    return 0;                                                                        \
  }

// Wgrad: 64x64x32 2-stage f32 output (aligned C)
INSTANTIATE_PROD_WGRAD(MaskGemm_wgrad_64x64x32_2s_f32, cutlass::half_t, Tile64x64x32, float)
INSTANTIATE_PROD_WGRAD(MaskGemm_wgrad_64x64x32_2s_f32, cutlass::bfloat16_t, Tile64x64x32, float)

// =============================================================================
// Wgrad scalar variant (unaligned C — scalar A and B loads)
// =============================================================================

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
                            cudaStream_t stream);

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
                                               cudaStream_t stream) {              \
    using Config = CuteTileConfig<ElemIn, gemm::Tile64x64x32>;                     \
    using Kernel = MaskGemm_wgrad_64x64x32_2s_f32_sab<Config, ElemOut>;            \
    constexpr int TileM = cute::size<0>(typename Config::TileShape{});             \
    constexpr int TileN = cute::size<1>(typename Config::TileShape{});             \
    if (N_out == 0 || C_in == 0 || C_out == 0) return 0;                           \
    int m_tiles = (C_in + TileM - 1) / TileM;                                      \
    int n_tiles = (C_out + TileN - 1) / TileN;                                     \
    dim3 grid(m_tiles *n_tiles, K, split_k);                                       \
    size_t smem = Kernel::SharedStorageSize;                                       \
    if (smem > 48 * 1024) {                                                        \
      auto err = cudaFuncSetAttribute(production_wgrad_kernel_entry<Kernel>,       \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      smem);                                       \
      if (err != cudaSuccess) return -1;                                           \
    }                                                                              \
    production_wgrad_kernel_entry<Kernel>                                          \
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
                                                             alpha);               \
    return 0;                                                                      \
  }

INSTANTIATE_SCALAR_WGRAD(cutlass::half_t, float)
INSTANTIATE_SCALAR_WGRAD(cutlass::bfloat16_t, float)

#undef INSTANTIATE_SCALAR_WGRAD

}  // namespace cute_gemm
}  // namespace warpconvnet
