// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Production mask kernel bindings — separate from gemm_bindings.cpp to handle
// dtype-specific dispatch (F16Accum tiles are fp16-only).

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstring>

#include "../include/gemm_error_codes.h"
#include "../include/gemm_mma_tiles.h"        // canonical tile_tag struct decls
#include "../include/mask_gemm_tile_enums.h"  // FwdTile/DgradTile/WgradTile (warpgemm-emitted)
#include "../include/wcn_pcoff_tiles.h"       // wcn-only Pcoff_* tile tags + CuteTileConfig specs
#include "cutlass/numeric_types.h"

// =============================================================================
// kMaskGemmTable[] — compile-time metadata sidecar populated from
// mask_gemm/mask_gemm_dispatch_table.inc via X-macro expansion. Used for
// runtime introspection (warpconvnet._C.mask_gemm.list_kernels()); not a
// dispatch driver — switch arms below switch directly on tile_id.
// =============================================================================

namespace warpconvnet {
namespace cute_gemm {

struct MaskGemmKernelEntry {
  int tile_id;
  const char *op;
  const char *kernel_struct;
  const char *tile_tag;
  const char *config_alias;
  const char *input_dtype;
  const char *output_dtype;
  const char *acc_dtype;
  const char *mainloop;
  const char *epilogue;
  int mask_words;
  bool persistent;
  const char *scalar_flags;
  const char *compile_archs;
};

[[maybe_unused]] constexpr MaskGemmKernelEntry kMaskGemmTable[] = {
#define MASK_GEMM_KERNEL(tile_id,       \
                         op,            \
                         kernel_struct, \
                         tile_tag,      \
                         config_alias,  \
                         in_dt,         \
                         out_dt,        \
                         acc_dt,        \
                         mainloop,      \
                         epilogue,      \
                         mask_words,    \
                         persistent,    \
                         scalar_flags,  \
                         compile_archs) \
  {tile_id,                             \
   op,                                  \
   kernel_struct,                       \
   tile_tag,                            \
   config_alias,                        \
   in_dt,                               \
   out_dt,                              \
   acc_dt,                              \
   mainloop,                            \
   epilogue,                            \
   mask_words,                          \
   static_cast<bool>(persistent),       \
   scalar_flags,                        \
   compile_archs},
#include "../mask_gemm/mask_gemm_dispatch_table.inc"
#undef MASK_GEMM_KERNEL
};

}  // namespace cute_gemm
}  // namespace warpconvnet

namespace warpconvnet {
namespace cute_gemm {

// Forward declarations (from mask_gemm_kernels.cu)
template <typename ElementInput, typename TileTag, typename ElementOutput>
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
                         int groups,
                         int identity_offset,
                         cudaStream_t stream);

template <typename ElementInput, typename TileTag, typename ElementOutput>
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
                                 int groups,
                                 cudaStream_t stream);

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
               int groups,                                 \
               cudaStream_t stream);

WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x64_2s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x64_3s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x128_2s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x128_3s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_128x64_2s_pipelined)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x64_2s_fused)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_64x128_2s_fused)
WCN_DECLARE_FWD_STRIDED_LAUNCH(launch_fwd_strided_128x64_2s_fused)
#undef WCN_DECLARE_FWD_STRIDED_LAUNCH

template <typename ElementInput, typename TileTag, typename ElementOutput>
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
                           int groups,
                           int identity_offset,
                           cudaStream_t stream);

template <typename ElementInput, typename TileTag, typename ElementOutput>
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
                           int groups,
                           cudaStream_t stream);

// Workspace wgrad launchers: write to [split_k, K, G, C_in_g, C_out_g] fp32
// workspace buffer. Caller owns workspace allocation + post-launch reduction
// (workspace.sum(0) -> grad_weight). See WGRAD_WORKSPACE_CASE macro.
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
int launch_mask_gemm_fwd_mw(const void *,
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
int launch_mask_gemm_dgrad_mw(const void *,
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
                            int,
                            int,
                            cudaStream_t);
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
                              int,
                              int,
                              cudaStream_t);
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
                                 int,
                                 int,
                                 cudaStream_t);

// fp32 output launch functions
template <typename ElemIn>
int launch_mask_gemm_fwd_f32out(const void *,
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
int launch_mask_gemm_fwd_f32out_sb(const void *,
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
int launch_mask_gemm_dgrad_f32out(const void *,
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

// fp32 output MW>1 launch functions (tiles 80, 81, 82)
template <typename ElemIn, int MW>
int launch_mask_gemm_fwd_f32out_mw(const void *,
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
template <typename ElemIn, int MW>
int launch_mask_gemm_fwd_f32out_sb_mw(const void *,
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
template <typename ElemIn, int MW>
int launch_mask_gemm_dgrad_f32out_mw(const void *,
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

// Vectorized MW>1 forward launch functions (tiles 42/43/44)
template <int MW>
int launch_mask_gemm_fwd_64x128_f16acc_mw(const void *,
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
// Tile 28: 32x32 F16Accum (half-only), MW2/4 only.
template <int MW>
int launch_mask_gemm_fwd_32x32_f16acc_mw(const void *,
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
template <typename ElemIn, int MW>
int launch_mask_gemm_fwd_64x128_3s_mw(const void *,
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
template <typename ElemIn, int MW>
int launch_mask_gemm_fwd_128x64_mw(const void *,
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
  cute_gemm::launch_mask_gemm_fwd<ElemIn, gemm::TileTag, ElemOut>(__VA_ARGS__)

#define LAUNCH_FWD_STRIDED(ElemIn, TileTag, ElemOut, ...) \
  cute_gemm::launch_mask_gemm_fwd_strided<ElemIn, gemm::TileTag, ElemOut>(__VA_ARGS__)

#define LAUNCH_FWD_STRIDED_NAMED(FuncName, ElemIn, ElemOut, ...) \
  cute_gemm::FuncName<ElemIn, ElemOut>(__VA_ARGS__)

#define LAUNCH_DGRAD(ElemIn, TileTag, ElemOut, ...) \
  cute_gemm::launch_mask_gemm_dgrad<ElemIn, gemm::TileTag, ElemOut>(__VA_ARGS__)

#define LAUNCH_WGRAD(ElemIn, TileTag, ElemOut, ...) \
  cute_gemm::launch_mask_gemm_wgrad<ElemIn, gemm::TileTag, ElemOut>(__VA_ARGS__)

#define LAUNCH_SCALAR_FWD(suffix, ElemIn, ElemOut, ...) \
  cute_gemm::launch_scalar_fwd_##suffix<ElemIn, ElemOut>(__VA_ARGS__)

#define LAUNCH_SCALAR_DGRAD(suffix, ElemIn, ElemOut, ...) \
  cute_gemm::launch_scalar_dgrad_##suffix<ElemIn, ElemOut>(__VA_ARGS__)

// =============================================================================
// Forward dispatch
// =============================================================================

int mask_gemm_fwd(torch::Tensor input,
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
              "mask_gemm_fwd requires fp16 or bf16 input (cast in Python before calling)");
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
  // Dispatch keys directly on canonical warpgemm tile_id integers.
  // See mask_gemm/mask_gemm_dispatch_table.inc.
  int tile = tile_id;

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
  auto strided_args = std::make_tuple(input.data_ptr(),
                                      weight.data_ptr(),
                                      output.data_ptr(),
                                      pair_table.data_ptr<int>(),
                                      N_in,
                                      N_out,
                                      C_in,
                                      C_out,
                                      K,
                                      alpha,
                                      groups,
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
  // Tile 80 (f32out aligned) and 82 (f32out scalar B) support MW>1 via dispatch.
#define FWD_F32OUT_MW(In)                                                                      \
  DISPATCH_MW(                                                                                 \
      std::apply([](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out<In>(a...); }, \
                 args),                                                                        \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_mw<In, 2>(a...); },  \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_mw<In, 4>(a...); },  \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_mw<In, 8>(a...); },  \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_mw<In, 12>(a...); }, \
          args))

#define FWD_F32OUT_SB_MW(In)                                                                      \
  DISPATCH_MW(                                                                                    \
      std::apply([](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_sb<In>(a...); }, \
                 args),                                                                           \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_sb_mw<In, 2>(a...); },  \
          args),                                                                                  \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_sb_mw<In, 4>(a...); },  \
          args),                                                                                  \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_sb_mw<In, 8>(a...); },  \
          args),                                                                                  \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_f32out_sb_mw<In, 12>(a...); }, \
          args))

  // wcn-only fwd f32-output tiles (no canonical equivalent):
  //   80 = aligned f32-output, 82 = scalar-B f32-output
  if (tile == 80 || tile == 82) {
    bool use_sb = (tile == 82);
    if (si == torch::kFloat16) {
      if (use_sb)
        FWD_F32OUT_SB_MW(cutlass::half_t);
      else
        FWD_F32OUT_MW(cutlass::half_t);
    }
#ifndef DISABLE_BFLOAT16
    if (si == torch::kBFloat16) {
      if (use_sb)
        FWD_F32OUT_SB_MW(cutlass::bfloat16_t);
      else
        FWD_F32OUT_MW(cutlass::bfloat16_t);
    }
#endif
  }
#undef FWD_F32OUT_MW
#undef FWD_F32OUT_SB_MW

  // Scalar tiles — work with any dtype and any C alignment.
  // SAB_SE / SA / SB_SE all support MW=1,2,4,8,12 via dispatched launchers.
#define SCALAR_FWD_MW(SUFFIX, In, Out)                                                        \
  DISPATCH_MW(                                                                                \
      std::apply([](auto &&...a) { return LAUNCH_SCALAR_FWD(SUFFIX, In, Out, a...); }, args), \
      std::apply(                                                                             \
          [](auto &&...a) {                                                                   \
            return cute_gemm::launch_scalar_fwd_##SUFFIX##_mw<In, Out, 2>(a...);              \
          },                                                                                  \
          args),                                                                              \
      std::apply(                                                                             \
          [](auto &&...a) {                                                                   \
            return cute_gemm::launch_scalar_fwd_##SUFFIX##_mw<In, Out, 4>(a...);              \
          },                                                                                  \
          args),                                                                              \
      std::apply(                                                                             \
          [](auto &&...a) {                                                                   \
            return cute_gemm::launch_scalar_fwd_##SUFFIX##_mw<In, Out, 8>(a...);              \
          },                                                                                  \
          args),                                                                              \
      std::apply(                                                                             \
          [](auto &&...a) {                                                                   \
            return cute_gemm::launch_scalar_fwd_##SUFFIX##_mw<In, Out, 12>(a...);             \
          },                                                                                  \
          args))

  // wcn-only scalar fwd tiles (unaligned C); not in canonical registry.
  // 70=sab_se, 71=sa, 72=sb_se — kept as raw integers.
  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case 70:  // wcn-only scalar tile, not in canonical registry
        SCALAR_FWD_MW(sab_se, In, Out);
      case 71:  // wcn-only scalar tile, not in canonical registry
        SCALAR_FWD_MW(sa, In, Out);
      case 72:  // wcn-only scalar tile, not in canonical registry
        SCALAR_FWD_MW(sb_se, In, Out);
      default:
        break;
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (tile) {
      case 70:  // wcn-only scalar tile, not in canonical registry
        SCALAR_FWD_MW(sab_se, In, Out);
      case 71:  // wcn-only scalar tile, not in canonical registry
        SCALAR_FWD_MW(sa, In, Out);
      case 72:  // wcn-only scalar tile, not in canonical registry
        SCALAR_FWD_MW(sb_se, In, Out);
      default:
        break;
    }
  }
#endif
#undef SCALAR_FWD_MW

  // Vectorized fp16 dispatch (includes F16Accum tiles)
  // For MW>1 (K>32) with aligned C, each tile has MW-parameterized launchers.
#define FWD_64x64_MW(ElemIn)                                                                       \
  DISPATCH_MW(                                                                                     \
      std::apply([](auto &&...a) { return LAUNCH_FWD(ElemIn, Tile64x64x32, ElemIn, a...); },       \
                 args),                                                                            \
      std::apply([](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_mw<ElemIn, 2>(a...); },  \
                 args),                                                                            \
      std::apply([](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_mw<ElemIn, 4>(a...); },  \
                 args),                                                                            \
      std::apply([](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_mw<ElemIn, 8>(a...); },  \
                 args),                                                                            \
      std::apply([](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_mw<ElemIn, 12>(a...); }, \
                 args))

#define FWD_64x128_F16ACC_MW_DISP()                                                               \
  DISPATCH_MW(                                                                                    \
      std::apply(                                                                                 \
          [](auto &&...a) {                                                                       \
            return LAUNCH_FWD(cutlass::half_t, Tile64x128x32_F16Accum, cutlass::half_t, a...);    \
          },                                                                                      \
          args),                                                                                  \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_64x128_f16acc_mw<2>(a...); },  \
          args),                                                                                  \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_64x128_f16acc_mw<4>(a...); },  \
          args),                                                                                  \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_64x128_f16acc_mw<8>(a...); },  \
          args),                                                                                  \
      std::apply(                                                                                 \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_64x128_f16acc_mw<12>(a...); }, \
          args))

// Tile 28: 32x32 F16Accum, MW1/2/4 only (K<=128). 32x32 has no MW8/12 launcher,
// so cap here and return -1 above MW4 (the Python guard rejects mask_words>4 for
// tile 28, so this arm is defensive). Custom dispatch, not DISPATCH_MW, because
// the 5-arm macro would route MW8/12 to a nonexistent launcher.
#define FWD_32x32_F16ACC_MW_DISP()                                                              \
  do {                                                                                          \
    if (mask_words <= 1)                                                                        \
      return std::apply(                                                                        \
          [](auto &&...a) {                                                                     \
            return LAUNCH_FWD(cutlass::half_t, Tile32x32x32_F16Accum, cutlass::half_t, a...);   \
          },                                                                                    \
          args);                                                                                \
    else if (mask_words <= 2)                                                                   \
      return std::apply(                                                                        \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_32x32_f16acc_mw<2>(a...); }, \
          args);                                                                                \
    else if (mask_words <= 4)                                                                   \
      return std::apply(                                                                        \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_32x32_f16acc_mw<4>(a...); }, \
          args);                                                                                \
    else                                                                                        \
      return -1;                                                                                \
  } while (0)

#define FWD_64x128_3S_MW(ElemIn)                                                              \
  DISPATCH_MW(                                                                                \
      std::apply([](auto &&...a) { return LAUNCH_FWD(ElemIn, Tile64x128x32, ElemIn, a...); }, \
                 args),                                                                       \
      std::apply(                                                                             \
          [](auto &&...a) {                                                                   \
            return cute_gemm::launch_mask_gemm_fwd_64x128_3s_mw<ElemIn, 2>(a...);             \
          },                                                                                  \
          args),                                                                              \
      std::apply(                                                                             \
          [](auto &&...a) {                                                                   \
            return cute_gemm::launch_mask_gemm_fwd_64x128_3s_mw<ElemIn, 4>(a...);             \
          },                                                                                  \
          args),                                                                              \
      std::apply(                                                                             \
          [](auto &&...a) {                                                                   \
            return cute_gemm::launch_mask_gemm_fwd_64x128_3s_mw<ElemIn, 8>(a...);             \
          },                                                                                  \
          args),                                                                              \
      std::apply(                                                                             \
          [](auto &&...a) {                                                                   \
            return cute_gemm::launch_mask_gemm_fwd_64x128_3s_mw<ElemIn, 12>(a...);            \
          },                                                                                  \
          args))

#define FWD_128x64_MW(ElemIn)                                                                      \
  DISPATCH_MW(                                                                                     \
      std::apply([](auto &&...a) { return LAUNCH_FWD(ElemIn, Tile128x64x32, ElemIn, a...); },      \
                 args),                                                                            \
      std::apply(                                                                                  \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_128x64_mw<ElemIn, 2>(a...); },  \
          args),                                                                                   \
      std::apply(                                                                                  \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_128x64_mw<ElemIn, 4>(a...); },  \
          args),                                                                                   \
      std::apply(                                                                                  \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_128x64_mw<ElemIn, 8>(a...); },  \
          args),                                                                                   \
      std::apply(                                                                                  \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_fwd_128x64_mw<ElemIn, 12>(a...); }, \
          args))

  // -- Canonical warpgemm fwd tile_ids. Each case label below references a
  //    member of gemm::FwdTile (warpgemm-emitted, see mask_gemm_tile_enums.h).
  using gemm::FwdTile;
#define FWD_STRIDED_CASE(TileEnum, FuncName)                                           \
  case FwdTile::TileEnum:                                                              \
    return std::apply(                                                                 \
        [](auto &&...a) { return LAUNCH_FWD_STRIDED_NAMED(FuncName, In, Out, a...); }, \
        strided_args)

  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (static_cast<FwdTile>(tile)) {
      case FwdTile::_32x32x32_1s_flat_F16Accum:
        FWD_32x32_F16ACC_MW_DISP();
      case FwdTile::_64x64x32_1s_flat_sa:
        FWD_64x64_MW(In);
      case FwdTile::_64x128x32_2s_fused_F16Accum:
        FWD_64x128_F16ACC_MW_DISP();
      case FwdTile::_64x128x32_3s:
        FWD_64x128_3S_MW(In);
      case FwdTile::_128x64x32_2s:
        FWD_128x64_MW(In);
        FWD_STRIDED_CASE(_64x64x32_2s_pipelined_strided, launch_fwd_strided_64x64_2s_pipelined);
        FWD_STRIDED_CASE(_64x64x32_3s_pipelined_strided, launch_fwd_strided_64x64_3s_pipelined);
        FWD_STRIDED_CASE(_64x128x32_2s_pipelined_strided, launch_fwd_strided_64x128_2s_pipelined);
        FWD_STRIDED_CASE(_64x128x32_3s_pipelined_strided, launch_fwd_strided_64x128_3s_pipelined);
        FWD_STRIDED_CASE(_128x64x32_2s_pipelined_strided, launch_fwd_strided_128x64_2s_pipelined);
        FWD_STRIDED_CASE(_64x64x32_2s_fused_strided, launch_fwd_strided_64x64_2s_fused);
        FWD_STRIDED_CASE(_64x128x32_2s_fused_strided, launch_fwd_strided_64x128_2s_fused);
        FWD_STRIDED_CASE(_128x64x32_2s_fused_strided, launch_fwd_strided_128x64_2s_fused);
      // Pcoff (E1) variants, MW=1 only (MW>1 instantiations deferred).
      case FwdTile::_64x64x32_1s_flat_pcoff_F16Accum:
        return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff, Out, a...); },
                          args);
      case FwdTile::_64x64x32_1s_flat_pcoff_F16K8:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_K8, Out, a...); }, args);
      case FwdTile::_64x128x32_1s_flat_pcoff_F16K8:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_Pcoff_K8, Out, a...); }, args);
      case FwdTile::_64x128x32_1s_flat_pcoff_F16Accum:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_Pcoff, Out, a...); }, args);
      case FwdTile::_64x64x32_3s_pcoff:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_3s, Out, a...); }, args);
      case FwdTile::_64x64x32_2s_warp_spec_pcoff:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_WS, Out, a...); }, args);
      case FwdTile::_64x128x32_2s_warp_spec_pcoff:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_Pcoff_WS, Out, a...); }, args);
      default:
        break;
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (static_cast<FwdTile>(tile)) {
      case FwdTile::_64x64x32_1s_flat_sa:
        FWD_64x64_MW(In);
      case FwdTile::_64x128x32_3s:
        FWD_64x128_3S_MW(In);
      case FwdTile::_128x64x32_2s:
        FWD_128x64_MW(In);
        FWD_STRIDED_CASE(_64x64x32_2s_pipelined_strided, launch_fwd_strided_64x64_2s_pipelined);
        FWD_STRIDED_CASE(_64x64x32_3s_pipelined_strided, launch_fwd_strided_64x64_3s_pipelined);
        FWD_STRIDED_CASE(_64x128x32_2s_pipelined_strided, launch_fwd_strided_64x128_2s_pipelined);
        FWD_STRIDED_CASE(_64x128x32_3s_pipelined_strided, launch_fwd_strided_64x128_3s_pipelined);
        FWD_STRIDED_CASE(_128x64x32_2s_pipelined_strided, launch_fwd_strided_128x64_2s_pipelined);
        FWD_STRIDED_CASE(_64x64x32_2s_fused_strided, launch_fwd_strided_64x64_2s_fused);
        FWD_STRIDED_CASE(_64x128x32_2s_fused_strided, launch_fwd_strided_64x128_2s_fused);
        FWD_STRIDED_CASE(_128x64x32_2s_fused_strided, launch_fwd_strided_128x64_2s_fused);
      // Pcoff bf16 variants — F32-accum base supports bf16
      case FwdTile::_64x64x32_3s_pcoff:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_3s, Out, a...); }, args);
      case FwdTile::_64x64x32_2s_warp_spec_pcoff:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_WS, Out, a...); }, args);
      case FwdTile::_64x128x32_2s_warp_spec_pcoff:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_Pcoff_WS, Out, a...); }, args);
      default:
        break;
    }
  }
#endif
#undef FWD_STRIDED_CASE
#undef FWD_64x64_MW
#undef FWD_64x128_F16ACC_MW_DISP
#undef FWD_64x128_3S_MW
#undef FWD_128x64_MW
  TORCH_CHECK(false, "Unsupported tile_id/dtype for mask_gemm_fwd: tile=", tile_id);
  return -1;
}

// =============================================================================
// Dgrad dispatch
// =============================================================================

int mask_gemm_dgrad(torch::Tensor grad_output,
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
      "mask_gemm_dgrad requires fp16 or bf16 input (cast in Python before calling)");
  grad_output = grad_output.contiguous();
  weight_T = weight_T.contiguous();

  int N_in = grad_input.size(0), N_out = grad_output.size(0);
  int C_in_total = grad_input.size(1), C_out_total = grad_output.size(1);
  int C_in = C_in_total / groups, C_out = C_out_total / groups;

  auto si = grad_output.scalar_type();
  auto so = grad_input.scalar_type();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Dispatch keys directly on canonical warpgemm tile_id integers.
  // Canonical dgrad ids 0-32 (native dgrad kernels), 900-911
  // (dgrad_wt: fwd kernel reused with pre-transposed weight). wcn-only:
  // 70-72 (scalar), 81 (f32out).
  int tile = tile_id;

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

  // -- dgrad_wt branch: canonical tile_ids 900-911 (DgradTile::_*_wt members)
  //    are aliases that route to the corresponding fwd kernel. Caller must
  //    have pre-transposed weight_T to swap channel axes before calling here
  //    (see dispatch.py use_fwd_for_dgrad path). The args tuple is structurally
  //    identical to the fwd args (grad_output stands in as 'a', weight_T as
  //    'b', grad_input as 'd'), so we can route through LAUNCH_FWD directly.
  using gemm::DgradTile;
  if (tile >= 900 && tile <= 911) {
    // Fwd kernel grid/strides interpret args as (n_in, n_out, c_in, c_out).
    // Our `args` tuple was built with dgrad semantics (N_in=grad_input rows,
    // N_out=grad_output rows). For fwd-as-dgrad, swap so the fwd kernel sees:
    //   n_in  ← N_out (grad_output is the "input" tensor it gathers from)
    //   n_out ← N_in  (grad_input is the "output" tensor it scatters to)
    //   c_in  ← C_out (grad_output channel count = fwd input channels)
    //   c_out ← C_in  (grad_input channel count = fwd output channels)
    auto fwd_args = std::make_tuple(grad_output.data_ptr(),
                                    weight_T.data_ptr(),
                                    grad_input.data_ptr(),
                                    pair_table.data_ptr<int>(),
                                    reinterpret_cast<const uint32_t *>(pair_mask.data_ptr<int>()),
                                    mask_argsort.data_ptr<int>(),
                                    N_out,
                                    N_in,
                                    C_out,
                                    C_in,
                                    K,
                                    alpha,
                                    groups,
                                    identity_offset,
                                    stream);
    if (si == torch::kFloat16 && so == torch::kFloat16) {
      using In = cutlass::half_t;
      using Out = cutlass::half_t;
      switch (static_cast<DgradTile>(tile)) {
        case DgradTile::_64x64x32_1s_flat_sa_wt:
          return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32, Out, a...); },
                            fwd_args);
        case DgradTile::_64x128x32_3s_wt:
          return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32, Out, a...); },
                            fwd_args);
        case DgradTile::_128x64x32_2s_wt:
          return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile128x64x32, Out, a...); },
                            fwd_args);
        case DgradTile::_32x32x32_1s_flat_wt_F16Accum:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile32x32x32_F16Accum, Out, a...); },
              fwd_args);
        case DgradTile::_64x128x32_2s_fused_wt_F16Accum:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_F16Accum, Out, a...); },
              fwd_args);
        case DgradTile::_64x64x32_1s_flat_pcoff_wt_F16Accum:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff, Out, a...); }, fwd_args);
        case DgradTile::_64x64x32_1s_flat_pcoff_wt_F16K8:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_K8, Out, a...); },
              fwd_args);
        case DgradTile::_64x128x32_1s_flat_pcoff_wt_F16K8:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_Pcoff_K8, Out, a...); },
              fwd_args);
        case DgradTile::_64x128x32_1s_flat_pcoff_wt_F16Accum:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_Pcoff, Out, a...); }, fwd_args);
        case DgradTile::_64x64x32_3s_pcoff_wt:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_3s, Out, a...); },
              fwd_args);
        case DgradTile::_64x64x32_2s_warp_spec_pcoff_wt:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_WS, Out, a...); },
              fwd_args);
        case DgradTile::_64x128x32_2s_warp_spec_pcoff_wt:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_Pcoff_WS, Out, a...); },
              fwd_args);
        default:
          break;
      }
    }
#ifndef DISABLE_BFLOAT16
    if (si == torch::kBFloat16 && so == torch::kBFloat16) {
      using In = cutlass::bfloat16_t;
      using Out = cutlass::bfloat16_t;
      switch (static_cast<DgradTile>(tile)) {
        case DgradTile::_64x64x32_1s_flat_sa_wt:
          return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32, Out, a...); },
                            fwd_args);
        case DgradTile::_64x128x32_3s_wt:
          return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32, Out, a...); },
                            fwd_args);
        case DgradTile::_128x64x32_2s_wt:
          return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile128x64x32, Out, a...); },
                            fwd_args);
        // Pcoff bf16 — F32-accum base supports bf16 (3s_pcoff, WS variants)
        case DgradTile::_64x64x32_3s_pcoff_wt:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_3s, Out, a...); },
              fwd_args);
        case DgradTile::_64x64x32_2s_warp_spec_pcoff_wt:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32_Pcoff_WS, Out, a...); },
              fwd_args);
        case DgradTile::_64x128x32_2s_warp_spec_pcoff_wt:
          return std::apply(
              [](auto &&...a) { return LAUNCH_FWD(In, Tile64x128x32_Pcoff_WS, Out, a...); },
              fwd_args);
        default:
          break;
      }
    }
#endif
    TORCH_CHECK(false, "Unsupported dtype for dgrad_wt tile=", tile_id);
    return -1;
  }

  // fp32 output dgrad tile (supports MW=1,2,4,8,12)
#define DGRAD_F32OUT_MW(In)                                                                      \
  DISPATCH_MW(                                                                                   \
      std::apply([](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_f32out<In>(a...); }, \
                 args),                                                                          \
      std::apply(                                                                                \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_f32out_mw<In, 2>(a...); },  \
          args),                                                                                 \
      std::apply(                                                                                \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_f32out_mw<In, 4>(a...); },  \
          args),                                                                                 \
      std::apply(                                                                                \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_f32out_mw<In, 8>(a...); },  \
          args),                                                                                 \
      std::apply(                                                                                \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_f32out_mw<In, 12>(a...); }, \
          args))

  // wcn-only dgrad f32-output (id=81, no canonical equivalent).
  if (tile == 81) {
    if (si == torch::kFloat16) DGRAD_F32OUT_MW(cutlass::half_t);
#ifndef DISABLE_BFLOAT16
    if (si == torch::kBFloat16) DGRAD_F32OUT_MW(cutlass::bfloat16_t);
#endif
  }
#undef DGRAD_F32OUT_MW

  // Scalar dgrad tiles — any dtype; SAB_SE / SA / SB_SE all support MW=1,2,4,8,12
#define SCALAR_DGRAD_MW(SUFFIX, In, Out)                                                        \
  DISPATCH_MW(                                                                                  \
      std::apply([](auto &&...a) { return LAUNCH_SCALAR_DGRAD(SUFFIX, In, Out, a...); }, args), \
      std::apply(                                                                               \
          [](auto &&...a) {                                                                     \
            return cute_gemm::launch_scalar_dgrad_##SUFFIX##_mw<In, Out, 2>(a...);              \
          },                                                                                    \
          args),                                                                                \
      std::apply(                                                                               \
          [](auto &&...a) {                                                                     \
            return cute_gemm::launch_scalar_dgrad_##SUFFIX##_mw<In, Out, 4>(a...);              \
          },                                                                                    \
          args),                                                                                \
      std::apply(                                                                               \
          [](auto &&...a) {                                                                     \
            return cute_gemm::launch_scalar_dgrad_##SUFFIX##_mw<In, Out, 8>(a...);              \
          },                                                                                    \
          args),                                                                                \
      std::apply(                                                                               \
          [](auto &&...a) {                                                                     \
            return cute_gemm::launch_scalar_dgrad_##SUFFIX##_mw<In, Out, 12>(a...);             \
          },                                                                                    \
          args))

  // wcn-only scalar dgrad tiles (not in canonical registry).
  // 70=sab_se, 71=sa, 72=sb_se — kept as raw integers.
  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case 70:  // wcn-only scalar tile, not in canonical registry
        SCALAR_DGRAD_MW(sab_se, In, Out);
      case 71:  // wcn-only scalar tile, not in canonical registry
        SCALAR_DGRAD_MW(sa, In, Out);
      case 72:  // wcn-only scalar tile, not in canonical registry
        SCALAR_DGRAD_MW(sb_se, In, Out);
      default:
        break;
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (tile) {
      case 70:  // wcn-only scalar tile, not in canonical registry
        SCALAR_DGRAD_MW(sab_se, In, Out);
      case 71:  // wcn-only scalar tile, not in canonical registry
        SCALAR_DGRAD_MW(sa, In, Out);
      case 72:  // wcn-only scalar tile, not in canonical registry
        SCALAR_DGRAD_MW(sb_se, In, Out);
      default:
        break;
    }
  }
#endif
#undef SCALAR_DGRAD_MW

  // Vectorized dgrad tiles — 64x64 supports MW>1
#define DGRAD_64x64_MW(ElemIn)                                                                 \
  DISPATCH_MW(                                                                                 \
      std::apply([](auto &&...a) { return LAUNCH_DGRAD(ElemIn, Tile64x64x32, ElemIn, a...); }, \
                 args),                                                                        \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_mw<ElemIn, 2>(a...); },   \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_mw<ElemIn, 4>(a...); },   \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_mw<ElemIn, 8>(a...); },   \
          args),                                                                               \
      std::apply(                                                                              \
          [](auto &&...a) { return cute_gemm::launch_mask_gemm_dgrad_mw<ElemIn, 12>(a...); },  \
          args))

  // -- Canonical warpgemm dgrad tile_ids (gemm::DgradTile members).
  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (static_cast<DgradTile>(tile)) {
      case DgradTile::_32x32x32_1s_flat:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile32x32x32, Out, a...); },
                          args);
      case DgradTile::_64x64x32_2s:
        DGRAD_64x64_MW(In);
      case DgradTile::_64x64x32_1s_flat_F16Accum:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x64x32_F16Accum, Out, a...); }, args);
      case DgradTile::_64x128x32_2s:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32, Out, a...); },
                          args);
      case DgradTile::_64x128x32_2s_F16Accum:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32_F16Accum, Out, a...); }, args);
      case DgradTile::_64x64x32_2s_pipelined:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_64x64<In, Out>(a...); },
            args);
      case DgradTile::_64x128x32_2s_pipelined:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_64x128<In, Out>(a...); },
            args);
      case DgradTile::_128x64x32_2s_pipelined:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_128x64<In, Out>(a...); },
            args);
      // Pcoff (E1) native dgrad variants — bond #23. MW=1 only.
      case DgradTile::_64x64x32_1s_flat_pcoff_F16Accum:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x64x32_Pcoff, Out, a...); }, args);
      case DgradTile::_64x64x32_1s_flat_pcoff_F16K8:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x64x32_Pcoff_K8, Out, a...); }, args);
      case DgradTile::_64x128x32_1s_flat_pcoff_F16K8:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32_Pcoff_K8, Out, a...); }, args);
      case DgradTile::_64x128x32_1s_flat_pcoff_F16Accum:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32_Pcoff, Out, a...); }, args);
      case DgradTile::_64x64x32_3s_pcoff:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x64x32_Pcoff_3s, Out, a...); }, args);
      case DgradTile::_64x128x32_3s_pcoff:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32_Pcoff_3s, Out, a...); }, args);
      default:
        break;
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (static_cast<DgradTile>(tile)) {
      case DgradTile::_64x64x32_2s:
        DGRAD_64x64_MW(In);
      case DgradTile::_64x128x32_2s:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32, Out, a...); },
                          args);
      case DgradTile::_64x64x32_2s_pipelined:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_64x64<In, Out>(a...); },
            args);
      case DgradTile::_64x128x32_2s_pipelined:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_64x128<In, Out>(a...); },
            args);
      case DgradTile::_128x64x32_2s_pipelined:
        return std::apply(
            [](auto &&...a) { return cute_gemm::launch_dgrad_pipelined_128x64<In, Out>(a...); },
            args);
      default:
        break;
    }
  }
#endif
#undef DGRAD_64x64_MW
  TORCH_CHECK(false, "Unsupported tile_id/dtype for mask_gemm_dgrad: tile=", tile_id);
  return -1;
}

// =============================================================================
// Wgrad dispatch
// =============================================================================

int mask_gemm_wgrad(torch::Tensor input,
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
              "mask_gemm_wgrad requires fp16 or bf16 input (cast in Python before calling)");
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

  // Dispatch keys directly on canonical warpgemm wgrad tile_ids
  // (gemm::WgradTile members; see mask_gemm_tile_enums.h).
  // wcn-only: 73 (scalar SAB), kept as raw integer below.
  using gemm::WgradTile;
  int tile = tile_id;
  auto pt_ptr = pair_table.data_ptr<int>();
  auto pm_ptr = reinterpret_cast<const uint32_t *>(pair_mask.data_ptr<int>());
  auto ms_ptr = mask_argsort.data_ptr<int>();
  auto rm_ptr = reinterpret_cast<const uint32_t *>(reduced_mask.data_ptr<int>());

  // wcn-only scalar wgrad (any C alignment, tile_id=73, not in canonical registry).
  if (tile == 73) {
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
#define WGRAD_DISPATCH(ElemIn, TileTag)                                                   \
  cute_gemm::launch_mask_gemm_wgrad<ElemIn, gemm::TileTag, float>(input.data_ptr(),       \
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

  // Workspace variant: allocate [split_k, K, G, C_in, C_out] fp32, launch with
  // workspace as target, reduce sum(dim=0) into grad_weight. Caller pre-zeroed
  // grad_weight so copy_ is the right final op. No atomics — each split shard
  // writes to its own slice.
#define WGRAD_WORKSPACE_CASE(ElemIn, suffix)                                                       \
  do {                                                                                             \
    auto workspace = torch::zeros({split_k, K, groups, C_in, C_out},                               \
                                  grad_weight.options().dtype(torch::kFloat32));                   \
    int status = cute_gemm::launch_wgrad_workspace_##suffix<ElemIn, float>(input.data_ptr(),       \
                                                                           grad_output.data_ptr(), \
                                                                           workspace.data_ptr(),   \
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
                                                                           stream);                \
    if (status != 0) return status;                                                                \
    /* Reduce workspace along split_k dim into grad_weight (pre-zeroed by caller).                 \
       Workspace carries an explicit groups dim; grad_weight from the groups=1                     \
       caller path is [K, C_in, C_out] with no G dim, so squeeze when groups==1. */                \
    auto reduced = workspace.sum(0);                                                               \
    if (groups == 1) reduced = reduced.squeeze(1);                                                 \
    grad_weight.copy_(reduced);                                                                    \
    return 0;                                                                                      \
  } while (0)

  if (si == torch::kFloat16) {
    using In = cutlass::half_t;
    switch (static_cast<WgradTile>(tile)) {
      case WgradTile::_64x64x32_2s_f32_atomic:
        return WGRAD_ATOMIC(In, 64x64);
      case WgradTile::_64x128x32_2s_f32_atomic:
        return WGRAD_ATOMIC(In, 64x128);
      case WgradTile::_64x64x32_3s_f32_atomic:
        return WGRAD_ATOMIC(In, 3s);
      case WgradTile::_64x64x32_2s_f32_workspace:
        WGRAD_WORKSPACE_CASE(In, 64x64);
      case WgradTile::_64x64x32_3s_f32_workspace:
        WGRAD_WORKSPACE_CASE(In, 64x64_3s);
      case WgradTile::_64x128x32_2s_f32_workspace:
        WGRAD_WORKSPACE_CASE(In, 64x128);
      default:  // WgradTile::_64x64x32_2s_f32 (canonical 0) — also fallback for unmapped ids
        return WGRAD_DISPATCH(In, Tile64x64x32);
    }
  }
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    switch (static_cast<WgradTile>(tile)) {
      case WgradTile::_64x64x32_2s_f32_atomic:
        return WGRAD_ATOMIC(In, 64x64);
      case WgradTile::_64x128x32_2s_f32_atomic:
        return WGRAD_ATOMIC(In, 64x128);
      case WgradTile::_64x64x32_3s_f32_atomic:
        return WGRAD_ATOMIC(In, 3s);
      case WgradTile::_64x64x32_2s_f32_workspace:
        WGRAD_WORKSPACE_CASE(In, 64x64);
      case WgradTile::_64x64x32_3s_f32_workspace:
        WGRAD_WORKSPACE_CASE(In, 64x64_3s);
      case WgradTile::_64x128x32_2s_f32_workspace:
        WGRAD_WORKSPACE_CASE(In, 64x128);
      default:
        return WGRAD_DISPATCH(In, Tile64x64x32);
    }
  }
#endif
#undef WGRAD_DISPATCH
#undef WGRAD_ATOMIC
#undef WGRAD_WORKSPACE_CASE
  TORCH_CHECK(false, "Unsupported dtype for mask_gemm_wgrad");
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
void register_mask_gemm(py::module &m) {
  auto prod = m.def_submodule("mask_gemm", "Masked GEMM kernels (mask-based fused sparse conv)");

  prod.def("fwd",
           &mask_gemm_fwd,
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
           &mask_gemm_dgrad,
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
           &mask_gemm_wgrad,
           py::arg("input"),
           py::arg("grad_output"),
           py::arg("grad_weight"),
           py::arg("pair_table"),
           py::arg("pair_mask"),
           py::arg("mask_argsort"),
           py::arg("reduced_mask"),
           py::arg("K"),
           py::arg("tile_id") = 0,  // WgradTile::_64x64x32_2s_f32 (canonical 0)
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
