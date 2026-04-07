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
                   float alpha) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda());
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
              "production_fwd requires fp16 or bf16 input (cast in Python before calling)");
  input = input.contiguous();
  weight = weight.contiguous();
  output = output.contiguous();

  int N_in = input.size(0), N_out = output.size(0);
  int C_in = input.size(1), C_out = output.size(1);

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
                              stream);

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

  // Scalar tiles — work with any dtype (fp16 and bf16)
  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Scalar_SAB_SE:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_FWD(sab_se, In, Out, a...); },
                          args);
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
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_FWD(sab_se, In, Out, a...); },
                          args);
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

  // fp16 dispatch (includes F16Accum tiles)
  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Fwd_32x32x32_F16Acc:
        return std::apply(
            [](auto &&...a) { return LAUNCH_FWD(In, Tile32x32x32_F16Accum, Out, a...); }, args);
      case gemm::MMATile::Prod_Fwd_64x64x32:
        return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32, Out, a...); },
                          args);
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
  // bf16 dispatch (no F16Accum tiles)
  if (si == torch::kBFloat16 && so == torch::kBFloat16) {
    using In = cutlass::bfloat16_t;
    using Out = cutlass::bfloat16_t;
    switch (tile) {
      case gemm::MMATile::Prod_Fwd_64x64x32:
        return std::apply([](auto &&...a) { return LAUNCH_FWD(In, Tile64x64x32, Out, a...); },
                          args);
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
                     float alpha) {
  TORCH_CHECK(grad_output.is_cuda() && weight_T.is_cuda() && grad_input.is_cuda());
  TORCH_CHECK(
      grad_output.scalar_type() == torch::kFloat16 || grad_output.scalar_type() == torch::kBFloat16,
      "production_dgrad requires fp16 or bf16 input (cast in Python before calling)");
  grad_output = grad_output.contiguous();
  weight_T = weight_T.contiguous();

  int N_in = grad_input.size(0), N_out = grad_output.size(0);
  int C_in = grad_input.size(1), C_out = grad_output.size(1);

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

  // Scalar tiles — any dtype
  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Scalar_SAB_SE:
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_DGRAD(sab_se, In, Out, a...); },
                          args);
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
        return std::apply([](auto &&...a) { return LAUNCH_SCALAR_DGRAD(sab_se, In, Out, a...); },
                          args);
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

  // Vectorized tiles — dtype-specific
  if (si == torch::kFloat16 && so == torch::kFloat16) {
    using In = cutlass::half_t;
    using Out = cutlass::half_t;
    switch (tile) {
      case gemm::MMATile::Prod_Dgrad_32x32x32:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile32x32x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Dgrad_64x64x32:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x64x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Dgrad_64x64x32_F16Acc:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x64x32_F16Accum, Out, a...); }, args);
      case gemm::MMATile::Prod_Dgrad_64x128x32:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Dgrad_64x128x32_F16Acc:
        return std::apply(
            [](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32_F16Accum, Out, a...); }, args);
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
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x64x32, Out, a...); },
                          args);
      case gemm::MMATile::Prod_Dgrad_64x128x32:
        return std::apply([](auto &&...a) { return LAUNCH_DGRAD(In, Tile64x128x32, Out, a...); },
                          args);
      default:
        break;
    }
  }
#endif
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
                     float alpha) {
  TORCH_CHECK(input.is_cuda() && grad_output.is_cuda() && grad_weight.is_cuda());
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
              "production_wgrad requires fp16 or bf16 input (cast in Python before calling)");
  input = input.contiguous();
  grad_output = grad_output.contiguous();

  int N_in = input.size(0), N_out = grad_output.size(0);
  int C_in = input.size(1), C_out = grad_output.size(1);

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
                                                                            stream);
#endif
  }

  // Vectorized wgrad (aligned C)
  if (si == torch::kFloat16)
    return cute_gemm::launch_production_wgrad<cutlass::half_t, gemm::Tile64x64x32, float>(
        input.data_ptr(),
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
        stream);
#ifndef DISABLE_BFLOAT16
  if (si == torch::kBFloat16)
    return cute_gemm::launch_production_wgrad<cutlass::bfloat16_t, gemm::Tile64x64x32, float>(
        input.data_ptr(),
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
        stream);
#endif
  TORCH_CHECK(false, "Unsupported dtype for production_wgrad");
  return -1;
}

// =============================================================================
// Build reduced_mask: OR-reduce pair_mask values per tK-row block
// =============================================================================

__global__ void build_reduced_mask_kernel(
    const uint32_t *pair_mask, const int *mask_argsort, uint32_t *reduced_mask, int N, int tK) {
  int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (N + tK - 1) / tK;
  if (block_idx >= num_blocks) return;

  int start = block_idx * tK;
  int end = start + tK;
  if (end > N) end = N;

  uint32_t acc = 0;
  for (int i = start; i < end; ++i) {
    int real_row = mask_argsort[i];
    acc |= pair_mask[real_row];
  }
  reduced_mask[block_idx] = acc;
}

torch::Tensor build_reduced_mask(torch::Tensor pair_mask, torch::Tensor mask_argsort, int tK) {
  TORCH_CHECK(pair_mask.is_cuda());
  int N = pair_mask.size(0);
  int num_blocks = (N + tK - 1) / tK;
  auto reduced = torch::zeros({num_blocks}, pair_mask.options());

  int threads = 256;
  int blocks = (num_blocks + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  build_reduced_mask_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint32_t *>(pair_mask.data_ptr<int>()),
      mask_argsort.data_ptr<int>(),
      reinterpret_cast<uint32_t *>(reduced.data_ptr<int>()),
      N,
      tK);
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
           py::arg("alpha") = 1.0f);

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
           py::arg("alpha") = 1.0f);

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
           py::arg("alpha") = 1.0f);

  prod.def("build_reduced_mask",
           &build_reduced_mask,
           py::arg("pair_mask"),
           py::arg("mask_argsort"),
           py::arg("tK") = 32);
}
}  // namespace bindings
}  // namespace warpconvnet
