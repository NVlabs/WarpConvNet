// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Production mask kernel bindings — separate from gemm_bindings.cpp to handle
// dtype-specific dispatch (F16Accum tiles are fp16-only).

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "../include/gemm_error_codes.h"
#include "../include/gemm_mma_tiles.h"
#include "cutlass/numeric_types.h"

namespace warpconvnet {
namespace cute_gemm {

// Forward declarations (from production_mask_kernels.cu)
template <typename ElementInput, typename TileTag, typename ElementOutput>
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
                          int groups,
                          int identity_offset,
                          cudaStream_t stream);

template <typename ElementInput, typename TileTag, typename ElementOutput>
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
                            int groups,
                            int identity_offset,
                            cudaStream_t stream);

template <typename ElementInput, typename TileTag, typename ElementOutput>
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
                            int groups,
                            cudaStream_t stream);

// Scalar variant launch functions (separate template families)
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
                             int,
                             int,
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
                         int,
                         int,
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
                            int,
                            int,
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
                               int,
                               int,
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
                           int,
                           int,
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
                              int,
                              int,
                              cudaStream_t);

// MaskWords>1 forward/dgrad launch functions (K>32)
template <typename ElemIn, int MaskWords>
int launch_production_fwd_mw(const void *,
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
                             int,
                             int,
                             cudaStream_t);
template <typename ElemIn, int MaskWords>
int launch_production_dgrad_mw(const void *,
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
                               int,
                               int,
                               cudaStream_t);

// Pipelined dgrad launch functions
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
                                 int,
                                 int,
                                 cudaStream_t);
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
                                  int,
                                  int,
                                  cudaStream_t);
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
                                  int,
                                  int,
                                  cudaStream_t);

// Scalar MW>1 launch functions (K>32 with unaligned channels)
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
                                int,
                                int,
                                cudaStream_t);
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
                                  int,
                                  int,
                                  cudaStream_t);

// fp32 output launch functions
template <typename ElemIn>
int launch_production_fwd_f32out(const void *,
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
                                 int,
                                 int,
                                 cudaStream_t);
template <typename ElemIn>
int launch_production_fwd_f32out_sb(const void *,
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
                                    int,
                                    int,
                                    cudaStream_t);
template <typename ElemIn>
int launch_production_dgrad_f32out(const void *,
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
                                   int,
                                   int,
                                   cudaStream_t);

// Atomic wgrad launch functions
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

// Scalar wgrad launch function
template <typename ElemIn, typename ElemOut>
int launch_scalar_wgrad_sab(const void *,
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

}  // namespace cute_gemm
}  // namespace warpconvnet

using namespace warpconvnet;

// =============================================================================
// Dispatch helpers — call the right template based on dtype and tile_id
// =============================================================================

#define LAUNCH_FWD(ElemIn, TileTag, ElemOut, ...) \
  cute_gemm::launch_production_fwd<ElemIn, gemm::TileTag, ElemOut>(__VA_ARGS__)

#define LAUNCH_DGRAD(ElemIn, TileTag, ElemOut, ...) \
  cute_gemm::launch_production_dgrad<ElemIn, gemm::TileTag, ElemOut>(__VA_ARGS__)

#define LAUNCH_WGRAD(ElemIn, TileTag, ElemOut, ...) \
  cute_gemm::launch_production_wgrad<ElemIn, gemm::TileTag, ElemOut>(__VA_ARGS__)

#define LAUNCH_SCALAR_FWD(suffix, ElemIn, ElemOut, ...) \
  cute_gemm::launch_scalar_fwd_##suffix<ElemIn, ElemOut>(__VA_ARGS__)

#define LAUNCH_SCALAR_DGRAD(suffix, ElemIn, ElemOut, ...) \
  cute_gemm::launch_scalar_dgrad_##suffix<ElemIn, ElemOut>(__VA_ARGS__)

// =============================================================================
// Forward dispatch
// =============================================================================

int production_fwd(torch::Tensor input,
                   torch::Tensor weight,
                   torch::Tensor output,
                   torch::Tensor pair_table,
                   torch::Tensor pair_mask,
                   torch::Tensor mask_argsort,
                   int K,
                   int tile_id,
                   int mask_words,
                   int identity_offset,
                   float alpha,
                   int groups) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda());
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
              "production_fwd requires fp16 or bf16 input (cast in Python before calling)");
  input = input.contiguous();
  weight = weight.contiguous();
  output = output.contiguous();

  int N_in = input.size(0), N_out = output.size(0);
  // For group conv: input is [N, C_in_total], C_in/C_out are per-group
  int C_in_total = input.size(1), C_out_total = output.size(1);
  int C_in = C_in_total / groups, C_out = C_out_total / groups;

  // Alignment check — skip for scalar tiles which handle any C
  int elem_sz = input.element_size(), vec = 16 / elem_sz;
  bool is_scalar_tile = (tile_id >= 70 && tile_id <= 72) || tile_id == 82;
  if (!is_scalar_tile && (C_in % vec != 0 || C_out % vec != 0))
    return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);

  auto si = input.scalar_type();
  auto so = output.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto tile = static_cast<gemm::MMATile>(tile_id);

  auto args = std::make_tuple(input.data_ptr(),
                              weight.data_ptr(),
                              output.data_ptr(),
                              pair_table.data_ptr<int>(),
                              reinterpret_cast<const uint32_t *>(pair_mask.data_ptr<int>()),
                              mask_argsort.data_ptr<int>(),
                              N_in,
                              N_out,
                              C_in,
                              C_out,
                              K,
                              alpha,
                              groups,
                              identity_offset,
                              stream);

  // MW>1 dispatch helper macro
#define DISPATCH_MW(CALL_MW1, CALL_MW2, CALL_MW4, CALL_MW8, CALL_MW12) \
  do {                                                                 \
    if (mask_words <= 1)                                               \
      return CALL_MW1;                                                 \
    else if (mask_words <= 2)                                          \
      return CALL_MW2;                                                 \
    else if (mask_words <= 4)                                          \
      return CALL_MW4;                                                 \
    else if (mask_words <= 8)                                          \
      return CALL_MW8;                                                 \
    else                                                               \
      return CALL_MW12;                                                \
  } while (0)

  // fp32 output tiles (fp16/bf16 input, f32 output — for non-AMP)
  if (tile == gemm::MMATile::Prod_Fwd_64x64x32_f32out ||
      tile == gemm::MMATile::Prod_Fwd_64x64x32_f32out_sb) {
    bool use_sb = (tile == gemm::MMATile::Prod_Fwd_64x64x32_f32out_sb);
    if (si == torch::kFloat16) {
      if (use_sb)
        return std::apply(
            [](auto &&...a) {
              return cute_gemm::launch_production_fwd_f32out_sb<cutlass::half_t>(a...);
            },
            args);
      else
        return std::apply(
            [](auto &&...a) {
              return cute_gemm::launch_production_fwd_f32out<cutlass::half_t>(a...);
            },
            args);
    }
#ifndef DISABLE_BFLOAT16
    if (si == torch::kBFloat16) {
      if (use_sb)
        return std::apply(
            [](auto &&...a) {
              return cute_gemm::launch_production_fwd_f32out_sb<cutlass::bfloat16_t>(a...);
            },
            args);
      else
        return std::apply(
            [](auto &&...a) {
              return cute_gemm::launch_production_fwd_f32out<cutlass::bfloat16_t>(a...);
            },
            args);
    }
#endif
  }

  // Scalar tiles — work with any dtype and any C alignment.
  // SAB_SE supports MW>1 for K>32; SA/SB_SE are MW=1 only (K>32 not common with partial unalign).
#define SCALAR_FWD_SAB_SE_MW(In, Out)                                                           \
  DISPATCH_MW(                                                                                  \
      std::apply([](auto &&...a) { return LAUNCH_SCALAR_FWD(sab_se, In, Out, a...); }, args),   \
      std::apply(                                                                               \
          [](auto &&...a) { return cute_gemm::launch_scalar_fwd_sab_se_mw<In, Out, 2>(a...); }, \
          args),                                                                                \
      std::apply(                                                                               \
          [](auto &&...a) { return cute_gemm::launch_scalar_fwd_sab_se_mw<In, Out, 4>(a...); }, \
          args),                                                                                \
      -2 /* MW>4 unsupported for scalar path */,                                                \
      -2)

  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Scalar_SAB_SE:
        SCALAR_FWD_SAB_SE_MW(In, Out);
      case gemm::MMATile::Prod_Scalar_SA:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_FWD(sa, In, Out, a...); }, args);
      case gemm::MMATile::Prod_Scalar_SB_SE:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_FWD(sb_se, In, Out, a...); },
                          args);
      default:
        break;
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (tile) {
      case gemm::MMATile::Prod_Scalar_SAB_SE:
        SCALAR_FWD_SAB_SE_MW(In, Out);
      case gemm::MMATile::Prod_Scalar_SA:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_FWD(sa, In, Out, a...); }, args);
      case gemm::MMATile::Prod_Scalar_SB_SE:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_FWD(sb_se, In, Out, a...); },
                          args);
      default:
        break;
    }
  }
#endif
#undef SCALAR_FWD_SAB_SE_MW

  // Vectorized fp16 dispatch (includes F16Accum tiles)
  // For MW>1 (K>32) with aligned C, the 64x64 flat tile is used via launch_production_fwd_mw.
#define FWD_64x64_MW(ElemIn)                                                                       \
  DISPATCH_MW(                                                                                     \
      std::apply([](auto &&...a) { return LAUNCH_FWD(ElemIn, Tile64x64x32, ElemIn, a...); },       \
                 args),                                                                            \
      std::apply([](auto &&...a) { return cute_gemm::launch_production_fwd_mw<ElemIn, 2>(a...); }, \
                 args),                                                                            \
      std::apply([](auto &&...a) { return cute_gemm::launch_production_fwd_mw<ElemIn, 4>(a...); }, \
                 args),                                                                            \
      std::apply([](auto &&...a) { return cute_gemm::launch_production_fwd_mw<ElemIn, 8>(a...); }, \
                 args),                                                                            \
      std::apply(                                                                                  \
          [](auto &&...a) { return cute_gemm::launch_production_fwd_mw<ElemIn, 12>(a...); },       \
          args))

  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Fwd_32x32x32_F16Acc:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile32x32x32_F16Accum, Out, a...); }, args);
      case gemm::MMATile::Prod_Fwd_64x64x32:
        FWD_64x64_MW(In);
      case gemm::MMATile::Prod_Fwd_64x128x32_F16Acc:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_F16Accum, Out, a...); }, args);
      case gemm::MMATile::Prod_Fwd_64x128x32_3s:
        return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Fwd_128x64x32:
        return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile128x64x32, Out, a...); },
                          args);
      default:
        break;
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (tile) {
      case gemm::MMATile::Prod_Fwd_64x64x32:
        FWD_64x64_MW(In);
      case gemm::MMATile::Prod_Fwd_64x128x32_3s:
        return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Fwd_128x64x32:
        return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile128x64x32, Out, a...); },
                          args);
      default:
        break;
    }
  }
#endif
#undef FWD_64x64_MW
  TORCH_CHECK(false, "Unsupported tile_id/dtype for production_fwd: tile=", tile_id);
  return -1;
}

// =============================================================================
// Dgrad dispatch
// =============================================================================

int production_dgrad(torch::Tensor grad_output,
                     torch::Tensor weight_T,
                     torch::Tensor grad_input,
                     torch::Tensor pair_table,
                     torch::Tensor pair_mask,
                     torch::Tensor mask_argsort,
                     int K,
                     int tile_id,
                     int mask_words,
                     int identity_offset,
                     float alpha,
                     int groups) {
  TORCH_CHECK(grad_output.is_cuda() && weight_T.is_cuda() && grad_input.is_cuda());
  TORCH_CHECK(
      grad_output.scalar_type() == torch::kFloat16 || grad_output.scalar_type() == torch::kBFloat16,
      "production_dgrad requires fp16 or bf16 input (cast in Python before calling)");
  grad_output = grad_output.contiguous();
  weight_T = weight_T.contiguous();

  int N_in = grad_input.size(0), N_out = grad_output.size(0);
  int C_in_total = grad_input.size(1), C_out_total = grad_output.size(1);
  int C_in = C_in_total / groups, C_out = C_out_total / groups;

  auto si = grad_output.scalar_type();
  auto so = grad_input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto tile = static_cast<gemm::MMATile>(tile_id);

  auto args = std::make_tuple(grad_output.data_ptr(),
                              weight_T.data_ptr(),
                              grad_input.data_ptr(),
                              pair_table.data_ptr<int>(),
                              reinterpret_cast<const uint32_t *>(pair_mask.data_ptr<int>()),
                              mask_argsort.data_ptr<int>(),
                              N_in,
                              N_out,
                              C_in,
                              C_out,
                              K,
                              alpha,
                              groups,
                              identity_offset,
                              stream);

  // fp32 output dgrad tile
  if (tile == gemm::MMATile::Prod_Dgrad_64x64x32_f32out) {
    if (si == torch::kFloat16)
      return std::apply(
          [](auto &&...a) {
            return cute_gemm::launch_production_dgrad_f32out<cutlass::half_t>(a...);
          },
          args);
#ifndef DISABLE_BFLOAT16
    if (si == torch::kBFloat16)
      return std::apply(
          [](auto &&...a) {
            return cute_gemm::launch_production_dgrad_f32out<cutlass::bfloat16_t>(a...);
          },
          args);
#endif
  }

  // Scalar dgrad tiles — any dtype, SAB_SE supports MW>1
#define SCALAR_DGRAD_SAB_SE_MW(In, Out)                                                           \
  DISPATCH_MW(                                                                                    \
      std::apply([](auto &&...a) { return LAUNCH_SCALAR_DGRAD(sab_se, In, Out, a...); }, args),   \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_scalar_dgrad_sab_se_mw<In, Out, 2>(a...); }, \
          args),                                                                                  \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_scalar_dgrad_sab_se_mw<In, Out, 4>(a...); }, \
          args),                                                                                  \
      -2 /* MW>4 unsupported for scalar path */,                                                  \
      -2)

  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Scalar_SAB_SE:
        SCALAR_DGRAD_SAB_SE_MW(In, Out);
      case gemm::MMATile::Prod_Scalar_SA:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_DGRAD(sa, In, Out, a...); }, args);
      case gemm::MMATile::Prod_Scalar_SB_SE:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_DGRAD(sb_se, In, Out, a...); },
                          args);
      default:
        break;
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (tile) {
      case gemm::MMATile::Prod_Scalar_SAB_SE:
        SCALAR_DGRAD_SAB_SE_MW(In, Out);
      case gemm::MMATile::Prod_Scalar_SA:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_DGRAD(sa, In, Out, a...); }, args);
      case gemm::MMATile::Prod_Scalar_SB_SE:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_DGRAD(sb_se, In, Out, a...); },
                          args);
      default:
        break;
    }
  }
#endif
#undef SCALAR_DGRAD_SAB_SE_MW

  // Vectorized dgrad tiles — 64x64 supports MW>1
#define DGRAD_64x64_MW(ElemIn)                                                                 \
  DISPATCH_MW(                                                                                 \
      std::apply([](auto &&...a) { return LAUNCH_DGRAD(ElemIn, Tile64x64x32, ElemIn, a...); }, \
                 args),                                                                        \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_production_dgrad_mw<ElemIn, 2>(a...); },  \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_production_dgrad_mw<ElemIn, 4>(a...); },  \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_production_dgrad_mw<ElemIn, 8>(a...); },  \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_production_dgrad_mw<ElemIn, 12>(a...); }, \
          args))

  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Dgrad_32x32x32:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile32x32x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Dgrad_64x64x32:
        DGRAD_64x64_MW(In);
      case gemm::MMATile::Prod_Dgrad_64x64x32_F16Acc:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x64x32_F16Accum, Out, a...); }, args);
      case gemm::MMATile::Prod_Dgrad_64x128x32:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Dgrad_64x128x32_F16Acc:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32_F16Accum, Out, a...); }, args);
      case gemm::MMATile::Prod_Dgrad_64x64x32_Pipe:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_64x64<In, Out>(a...); },
            args);
      case gemm::MMATile::Prod_Dgrad_64x128x32_Pipe:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_64x128<In, Out>(a...); },
            args);
      case gemm::MMATile::Prod_Dgrad_128x64x32_Pipe:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_128x64<In, Out>(a...); },
            args);
      default:
        break;
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (tile) {
      case gemm::MMATile::Prod_Dgrad_64x64x32:
        DGRAD_64x64_MW(In);
      case gemm::MMATile::Prod_Dgrad_64x128x32:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Dgrad_64x64x32_Pipe:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_64x64<In, Out>(a...); },
            args);
      case gemm::MMATile::Prod_Dgrad_64x128x32_Pipe:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_64x128<In, Out>(a...); },
            args);
      case gemm::MMATile::Prod_Dgrad_128x64x32_Pipe:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_128x64<In, Out>(a...); },
            args);
      default:
        break;
    }
  }
#endif
#undef DGRAD_64x64_MW
  TORCH_CHECK(false, "Unsupported tile_id/dtype for production_dgrad: tile=", tile_id);
  return -1;
}

// =============================================================================
// Wgrad dispatch
// =============================================================================

int production_wgrad(torch::Tensor input,
                     torch::Tensor grad_output,
                     torch::Tensor grad_weight,
                     torch::Tensor pair_table,
                     torch::Tensor pair_mask,
                     torch::Tensor mask_argsort,
                     torch::Tensor reduced_mask,
                     int K,
                     int tile_id,
                     int split_k,
                     float alpha,
                     int groups) {
  TORCH_CHECK(input.is_cuda() && grad_output.is_cuda() && grad_weight.is_cuda());
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
              "production_wgrad requires fp16 or bf16 input (cast in Python before calling)");
  input = input.contiguous();
  grad_output = grad_output.contiguous();

  int N_in = input.size(0), N_out = grad_output.size(0);
  // For group conv: input is [N, C_in_total], C_in/C_out are per-group
  int C_in_total = input.size(1), C_out_total = grad_output.size(1);
  int C_in = C_in_total / groups, C_out = C_out_total / groups;

  int elem_sz = input.element_size(), vec = 16 / elem_sz;
  bool is_scalar_tile = (tile_id == 73);
  if (!is_scalar_tile && (C_in % vec != 0 || C_out % vec != 0))
    return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);

  auto si = input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto tile = static_cast<gemm::MMATile>(tile_id);
  auto pt_ptr = pair_table.data_ptr<int>();
  auto pm_ptr = reinterpret_cast<const uint32_t *>(pair_mask.data_ptr<int>());
  auto ms_ptr = mask_argsort.data_ptr<int>();
  auto rm_ptr = reinterpret_cast<const uint32_t *>(reduced_mask.data_ptr<int>());

  // Scalar wgrad (any C alignment)
  if (tile == gemm::MMATile::Prod_Wgrad_Scalar_SAB) {
    if (si == torch::kFloat16)
      return cute_gemm::launch_scalar_wgrad_sab<cutlass::half_t, float>(input.data_ptr(),
                                                                        grad_output.data_ptr(),
                                                                        grad_weight.data_ptr(),
                                                                        pt_ptr,
                                                                        pm_ptr,
                                                                        ms_ptr,
                                                                        rm_ptr,
                                                                        N_in,
                                                                        N_out,
                                                                        C_in,
                                                                        C_out,
                                                                        K,
                                                                        split_k,
                                                                        alpha,
                                                                        groups,
                                                                        stream);
#ifndef DISABLE_BFLOAT16
    if (si == torch::kBFloat16)
      return cute_gemm::launch_scalar_wgrad_sab<cutlass::bfloat16_t, float>(input.data_ptr(),
                                                                            grad_output.data_ptr(),
                                                                            grad_weight.data_ptr(),
                                                                            pt_ptr,
                                                                            pm_ptr,
                                                                            ms_ptr,
                                                                            rm_ptr,
                                                                            N_in,
                                                                            N_out,
                                                                            C_in,
                                                                            C_out,
                                                                            K,
                                                                            split_k,
                                                                            alpha,
                                                                            groups,
                                                                            stream);
#endif
  }

  // Vectorized wgrad dispatch — direct, atomic 64x64, or atomic 64x128
#define WGRAD_DISPATCH(ElemIn, TileTag)                                                    \
  cute_gemm::launch_production_wgrad<ElemIn, gemm::TileTag, float>(input.data_ptr(),       \
                                                                   grad_output.data_ptr(), \
                                                                   grad_weight.data_ptr(), \
                                                                   pt_ptr,                 \
                                                                   pm_ptr,                 \
                                                                   ms_ptr,                 \
                                                                   rm_ptr,                 \
                                                                   N_in,                   \
                                                                   N_out,                  \
                                                                   C_in,                   \
                                                                   C_out,                  \
                                                                   K,                      \
                                                                   split_k,                \
                                                                   alpha,                  \
                                                                   groups,                 \
                                                                   stream)

#define WGRAD_ATOMIC(ElemIn, suffix)                                             \
  cute_gemm::launch_wgrad_atomic_##suffix<ElemIn, float>(input.data_ptr(),       \
                                                         grad_output.data_ptr(), \
                                                         grad_weight.data_ptr(), \
                                                         pt_ptr,                 \
                                                         pm_ptr,                 \
                                                         ms_ptr,                 \
                                                         rm_ptr,                 \
                                                         N_in,                   \
                                                         N_out,                  \
                                                         C_in,                   \
                                                         C_out,                  \
                                                         K,                      \
                                                         split_k,                \
                                                         alpha,                  \
                                                         groups,                 \
                                                         stream)

  if (si == torch::kFloat16) {
    using In = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Wgrad_64x64x32_f32_atomic:
        return WGRAD_ATOMIC(In, 64x64);
      case gemm::MMATile::Prod_Wgrad_64x128x32_f32_atomic:
        return WGRAD_ATOMIC(In, 64x128);
      case gemm::MMATile::Prod_Wgrad_64x64x32_3s_f32_atomic:
        return WGRAD_ATOMIC(In, 3s);
      default:
        return WGRAD_DISPATCH(In, Tile64x64x32);
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    switch (tile) {
      case gemm::MMATile::Prod_Wgrad_64x64x32_f32_atomic:
        return WGRAD_ATOMIC(In, 64x64);
      case gemm::MMATile::Prod_Wgrad_64x128x32_f32_atomic:
        return WGRAD_ATOMIC(In, 64x128);
      case gemm::MMATile::Prod_Wgrad_64x64x32_3s_f32_atomic:
        return WGRAD_ATOMIC(In, 3s);
      default:
        return WGRAD_DISPATCH(In, Tile64x64x32);
    }
  }
#endif
#undef WGRAD_DISPATCH
#undef WGRAD_ATOMIC
  TORCH_CHECK(false, "Unsupported dtype for production_wgrad");
  return -1;
}

// =============================================================================
// Build reduced_mask: OR-reduce pair_mask values per tK-row block
// =============================================================================

__global__ void build_reduced_mask_kernel(const uint32_t *pair_mask,
                                          const int *mask_argsort,
                                          uint32_t *reduced_mask,
                                          int N,
                                          int tK,
                                          int mask_words) {
  int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (N + tK - 1) / tK;
  if (block_idx >= num_blocks) return;

  int start = block_idx * tK;
  int end = start + tK;
  if (end > N) end = N;

  // OR-reduce each mask word independently across the block
  for (int w = 0; w < mask_words; ++w) {
    uint32_t acc = 0;
    for (int i = start; i < end; ++i) {
      int real_row = mask_argsort[i];
      acc |= pair_mask[real_row * mask_words + w];
    }
    reduced_mask[block_idx * mask_words + w] = acc;
  }
}

torch::Tensor build_reduced_mask(torch::Tensor pair_mask,
                                 torch::Tensor mask_argsort,
                                 int tK,
                                 int mask_words) {
  TORCH_CHECK(pair_mask.is_cuda());
  int N = mask_argsort.size(0);
  int num_blocks = (N + tK - 1) / tK;
  auto reduced = torch::zeros({num_blocks * mask_words}, pair_mask.options());

  int threads = 256;
  int blocks = (num_blocks + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  build_reduced_mask_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint32_t *>(pair_mask.data_ptr<int>()),
      mask_argsort.data_ptr<int>(),
      reinterpret_cast<uint32_t *>(reduced.data_ptr<int>()),
      N,
      tK,
      mask_words);
  return reduced;
}

// =============================================================================
// Python binding registration
// =============================================================================

namespace warpconvnet {
namespace bindings {
void register_production(py::module &m) {
  auto prod = m.def_submodule("production", "Production mask GEMM kernels");

  prod.def("fwd",
           &production_fwd,
           py::arg("input"),
           py::arg("weight"),
           py::arg("output"),
           py::arg("pair_table"),
           py::arg("pair_mask"),
           py::arg("mask_argsort"),
           py::arg("K"),
           py::arg("tile_id"),
           py::arg("mask_words") = 1,
           py::arg("identity_offset") = -1,
           py::arg("alpha") = 1.0f,
           py::arg("groups") = 1);

  prod.def("dgrad",
           &production_dgrad,
           py::arg("grad_output"),
           py::arg("weight_T"),
           py::arg("grad_input"),
           py::arg("pair_table"),
           py::arg("pair_mask"),
           py::arg("mask_argsort"),
           py::arg("K"),
           py::arg("tile_id"),
           py::arg("mask_words") = 1,
           py::arg("identity_offset") = -1,
           py::arg("alpha") = 1.0f,
           py::arg("groups") = 1);

  prod.def("wgrad",
           &production_wgrad,
           py::arg("input"),
           py::arg("grad_output"),
           py::arg("grad_weight"),
           py::arg("pair_table"),
           py::arg("pair_mask"),
           py::arg("mask_argsort"),
           py::arg("reduced_mask"),
           py::arg("K"),
           py::arg("tile_id") = 60,
           py::arg("split_k") = 64,
           py::arg("alpha") = 1.0f,
           py::arg("groups") = 1);

  prod.def("build_reduced_mask",
           &build_reduced_mask,
           py::arg("pair_mask"),
           py::arg("mask_argsort"),
           py::arg("tK") = 32,
           py::arg("mask_words") = 1);
}
}  // namespace bindings
}  // namespace warpconvnet
