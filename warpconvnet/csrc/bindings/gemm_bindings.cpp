// Copyright 2025 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/arch.h>
#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <algorithm>
#include <sstream>

#include "../include/gemm_error_codes.h"
#include "../include/gemm_mma_tiles.h"

namespace py = pybind11;

// Forward declare WMMA SM80 run helper (templated) implemented in CUDA
namespace warpconvnet {
namespace wmma_implicit_gemm_sm80 {
template <typename ElementInput, typename ElementOutput>
int run(const void *A,
        const void *B,
        const void *C,
        void *D,
        const long *a_inds,
        const long *d_inds,
        int M,
        int K,
        int N,
        int P,
        int Q,
        float alpha,
        float beta,
        cudaStream_t stream);
}  // namespace wmma_implicit_gemm_sm80
}  // namespace warpconvnet

// Forward declare Segmented WMMA SM80 run helper (templated) implemented in CUDA
namespace warpconvnet {
namespace wmma_segmented_gemm_sm80 {
template <typename ElementInput, typename ElementOutput>
int run_segmented(const void *A,
                  const void *B_multi,
                  const void *C,
                  void *D,
                  const long *a_inds,
                  const long *d_inds,
                  const float *gates,
                  const int *seg_offsets,
                  int M,
                  int K,
                  int N,
                  int P,
                  int Q,
                  int E,
                  float alpha,
                  float beta,
                  cudaStream_t stream);
}  // namespace wmma_segmented_gemm_sm80
}  // namespace warpconvnet

// Forward declare Segmented tr(A)B gather-scatter WMMA SM80 run helper (templated)
namespace warpconvnet {
namespace wmma_segmented_trab_sm80 {
template <typename ElementInput, typename ElementOutput>
int run_segmented_trab(const void *X,
                       const void *DY,
                       void *D_all,
                       const long *a_inds,
                       const long *d_inds,
                       const float *gates,
                       const int *seg_offsets,
                       int N_rows,
                       int M_rows,
                       int C_in,
                       int C_out,
                       int P,
                       int num_segments,
                       float alpha,
                       float beta,
                       cudaStream_t stream);
}  // namespace wmma_segmented_trab_sm80
}  // namespace warpconvnet

// Forward declarations for CUDA kernels and GEMM launchers in their namespaces
namespace warpconvnet {
namespace wmma_split_k_sm80 {
template <typename ElementIn, typename ElementOut, typename Itype>
int run_split_k_wmma_templated(const void *tensor_a,
                               const void *tensor_b,
                               void *tensor_c,
                               const Itype *indices_a,
                               const Itype *indices_b,
                               int N,
                               int C_a,
                               int C_b,
                               int K,
                               int split_k_factor);
}  // namespace wmma_split_k_sm80
namespace implicit_gemm {
template <typename ElementA, typename ElementB, typename ElementC, typename Itype>
int run_implicit_gemm_templated(const void *tensor_a,
                                const void *tensor_b,
                                void *tensor_c,
                                const Itype *in_map,
                                const Itype *out_map,
                                int wA,
                                int hA,
                                int wB,
                                int hB,
                                int indices_size,
                                const std::string &kernel_type,
                                int block_size);
}  // namespace implicit_gemm

namespace split_k_implicit_gemm {
template <typename ElementA, typename ElementB, typename ElementC, typename Itype>
int run_split_k_implicit_gemm_templated(const void *tensor_a,
                                        const void *tensor_b,
                                        void *tensor_c,
                                        const Itype *indices_a,
                                        const Itype *indices_b,
                                        int N,
                                        int C_a,
                                        int C_b,
                                        int K,
                                        int split_k_factor,
                                        int block_threads);
}  // namespace split_k_implicit_gemm

namespace gemm {
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename TileTag,
          typename Arch>
int run_cutlass_gemm_ad_gather_scatter(const void *tensor_a,
                                       const void *tensor_b,
                                       const void *tensor_c,
                                       void *tensor_d,
                                       const int *indices_a,
                                       const int *indices_d,
                                       int split_k_slices,
                                       int M_A,
                                       int K,
                                       int N,
                                       int M_C,
                                       int gather_ad_size,
                                       float alpha,
                                       float beta);
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename TileTag,
          typename Arch>
int run_cutlass_gemm_trAB_gather(const void *tensor_a,
                                 const void *tensor_b,
                                 const void *tensor_c,
                                 void *tensor_d,
                                 const int *indices_a,
                                 const int *indices_b,
                                 int split_k_slices,
                                 int M_A,
                                 int K,
                                 int K_B,
                                 int N,
                                 int gather_ab_size,
                                 float alpha,
                                 float beta);
}  // namespace gemm
}  // namespace warpconvnet

namespace warpconvnet {
namespace bindings {

// ------------------- Internal helpers and kernels -------------------

// Type mapping from PyTorch scalar types to CUTLASS types
template <torch::ScalarType T>
struct torch_to_cutlass;
template <>
struct torch_to_cutlass<torch::kFloat16> {
  using type = cutlass::half_t;
};
template <>
struct torch_to_cutlass<torch::kFloat32> {
  using type = float;
};
template <>
struct torch_to_cutlass<torch::kFloat64> {
  using type = double;
};
template <>
struct torch_to_cutlass<torch::kBFloat16> {
  using type = cutlass::bfloat16_t;
};

namespace {
inline void _check_2d(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.dim() == 2, name, " must be 2D");
}
inline torch::Tensor _ensure_2d(torch::Tensor idx) {
  return idx.dim() == 1 ? idx.unsqueeze(1) : idx;
}
struct AdGatherScatterParams {
  int M_A;
  int K;
  int N;
  int M_C;
  int gather_ad_size;
};
inline AdGatherScatterParams validate_ad_gather_scatter_args(torch::Tensor &tensor_a,
                                                             torch::Tensor &tensor_b,
                                                             torch::Tensor &tensor_c,
                                                             torch::Tensor &tensor_d,
                                                             torch::Tensor &indices_a,
                                                             torch::Tensor &indices_d) {
  _check_2d(tensor_a, "tensor_a");
  _check_2d(tensor_b, "tensor_b");
  _check_2d(tensor_c, "tensor_c");
  _check_2d(tensor_d, "tensor_d");
  indices_a = _ensure_2d(indices_a);
  indices_d = _ensure_2d(indices_d);
  TORCH_CHECK(indices_a.dim() == 2 && indices_d.dim() == 2, "indices_a and indices_d must be 2D");
  TORCH_CHECK(indices_a.scalar_type() == torch::kInt32, "indices_a must be int32");
  TORCH_CHECK(indices_d.scalar_type() == torch::kInt32, "indices_d must be int32");
  int M_A = tensor_a.size(0);
  int K = tensor_a.size(1);
  int N = tensor_b.size(1);
  int M_C = tensor_c.size(0);
  int gather_ad_size = indices_a.size(0);
  TORCH_CHECK(tensor_b.size(0) == K, "tensor_b.size(0) must be match tensor_a.size(1)");
  TORCH_CHECK(tensor_c.size(1) == N, "tensor_c.size(1) must be match tensor_b.size(1)");
  TORCH_CHECK(tensor_d.size(0) == M_C && tensor_d.size(1) == N,
              "tensor_d.shape must be match tensor_c.shape");
  TORCH_CHECK(indices_a.size(1) == 1 && indices_d.size(1) == 1,
              "indices_a and indices_d must be 1D");
  TORCH_CHECK(indices_a.size(0) == indices_d.size(0),
              "indices_a and indices_d must have the same number of rows");
  return {M_A, K, N, M_C, gather_ad_size};
}
struct TrABGatherParams {
  int M_A;
  int K;
  int K_B;
  int N;
  int gather_ab_size;
};
inline TrABGatherParams validate_trAB_gather_args(torch::Tensor &tensor_a,
                                                  torch::Tensor &tensor_b,
                                                  torch::Tensor &tensor_c,
                                                  torch::Tensor &tensor_d,
                                                  torch::Tensor &indices_a,
                                                  torch::Tensor &indices_b) {
  _check_2d(tensor_a, "tensor_a");
  _check_2d(tensor_b, "tensor_b");
  _check_2d(tensor_c, "tensor_c");
  _check_2d(tensor_d, "tensor_d");
  indices_a = _ensure_2d(indices_a);
  indices_b = _ensure_2d(indices_b);
  TORCH_CHECK(indices_a.dim() == 2 && indices_b.dim() == 2, "indices_a and indices_b must be 2D");
  TORCH_CHECK(indices_a.scalar_type() == torch::kInt32, "indices_a must be int32");
  TORCH_CHECK(indices_b.scalar_type() == torch::kInt32, "indices_b must be int32");
  int M_A = tensor_a.size(0);
  int K = tensor_a.size(1);
  int K_B = tensor_b.size(0);
  int N = tensor_b.size(1);
  int gather_ab_size = indices_a.size(0);
  TORCH_CHECK(tensor_c.size(0) == K && tensor_c.size(1) == N,
              "tensor_c.shape must be match tensor_a.shape");
  TORCH_CHECK(tensor_d.size(0) == K && tensor_d.size(1) == N,
              "tensor_d.shape must be match tensor_b.shape");
  TORCH_CHECK(indices_a.size(1) == 1 && indices_b.size(1) == 1,
              "indices_a and indices_b must be 1D");
  TORCH_CHECK(indices_a.size(0) == indices_b.size(0),
              "indices_a and indices_b must have the same number of rows");
  return {M_A, K, K_B, N, gather_ab_size};
}
}  // namespace

// Dispatch helpers
template <torch::ScalarType SA, torch::ScalarType SB, torch::ScalarType SO, torch::ScalarType ACC>
static int dispatch_cutlass_gemm_ad_gather_scatter(const torch::Tensor &tensor_a,
                                                   const torch::Tensor &tensor_b,
                                                   const torch::Tensor &tensor_c,
                                                   torch::Tensor &tensor_d,
                                                   const torch::Tensor &indices_a,
                                                   const torch::Tensor &indices_d,
                                                   int split_k_slices,
                                                   int mma_tile,
                                                   int M_A,
                                                   int K,
                                                   int N,
                                                   int M_C,
                                                   int gather_ad_size,
                                                   float alpha,
                                                   float beta) {
  using ElementA = typename torch_to_cutlass<SA>::type;
  using ElementB = typename torch_to_cutlass<SB>::type;
  using ElementOutput = typename torch_to_cutlass<SO>::type;
  using ElementAccumulator = typename torch_to_cutlass<ACC>::type;
  TORCH_CHECK(tensor_a.scalar_type() == SA && tensor_b.scalar_type() == SB);
  TORCH_CHECK(tensor_c.scalar_type() == SO && tensor_d.scalar_type() == SO);
  auto tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    case warpconvnet::gemm::MMATile::Tile128x128x32:
      return ::warpconvnet::gemm::run_cutlass_gemm_ad_gather_scatter<
          ElementA,
          ElementB,
          ElementOutput,
          ElementAccumulator,
          warpconvnet::gemm::Tile128x128x32,
          cutlass::arch::Sm80>(tensor_a.data_ptr(),
                               tensor_b.data_ptr(),
                               tensor_c.data_ptr(),
                               tensor_d.data_ptr(),
                               indices_a.data_ptr<int>(),
                               indices_d.data_ptr<int>(),
                               split_k_slices,
                               M_A,
                               K,
                               N,
                               M_C,
                               gather_ad_size,
                               alpha,
                               beta);
    case warpconvnet::gemm::MMATile::Tile128x64x32:
      return ::warpconvnet::gemm::run_cutlass_gemm_ad_gather_scatter<
          ElementA,
          ElementB,
          ElementOutput,
          ElementAccumulator,
          warpconvnet::gemm::Tile128x64x32,
          cutlass::arch::Sm80>(tensor_a.data_ptr(),
                               tensor_b.data_ptr(),
                               tensor_c.data_ptr(),
                               tensor_d.data_ptr(),
                               indices_a.data_ptr<int>(),
                               indices_d.data_ptr<int>(),
                               split_k_slices,
                               M_A,
                               K,
                               N,
                               M_C,
                               gather_ad_size,
                               alpha,
                               beta);
    case warpconvnet::gemm::MMATile::Tile64x128x32:
      return ::warpconvnet::gemm::run_cutlass_gemm_ad_gather_scatter<
          ElementA,
          ElementB,
          ElementOutput,
          ElementAccumulator,
          warpconvnet::gemm::Tile64x128x32,
          cutlass::arch::Sm80>(tensor_a.data_ptr(),
                               tensor_b.data_ptr(),
                               tensor_c.data_ptr(),
                               tensor_d.data_ptr(),
                               indices_a.data_ptr<int>(),
                               indices_d.data_ptr<int>(),
                               split_k_slices,
                               M_A,
                               K,
                               N,
                               M_C,
                               gather_ad_size,
                               alpha,
                               beta);
    case warpconvnet::gemm::MMATile::Tile64x64x32:
      return ::warpconvnet::gemm::run_cutlass_gemm_ad_gather_scatter<
          ElementA,
          ElementB,
          ElementOutput,
          ElementAccumulator,
          warpconvnet::gemm::Tile64x64x32,
          cutlass::arch::Sm80>(tensor_a.data_ptr(),
                               tensor_b.data_ptr(),
                               tensor_c.data_ptr(),
                               tensor_d.data_ptr(),
                               indices_a.data_ptr<int>(),
                               indices_d.data_ptr<int>(),
                               split_k_slices,
                               M_A,
                               K,
                               N,
                               M_C,
                               gather_ad_size,
                               alpha,
                               beta);
    default:
      TORCH_CHECK(false, "Unsupported mma_tile value");
  }
}

template <torch::ScalarType SA, torch::ScalarType SB, torch::ScalarType SO, torch::ScalarType ACC>
static int dispatch_cutlass_gemm_trAB_gather(const torch::Tensor &tensor_a,
                                             const torch::Tensor &tensor_b,
                                             const torch::Tensor &tensor_c,
                                             torch::Tensor &tensor_d,
                                             const torch::Tensor &indices_a,
                                             const torch::Tensor &indices_b,
                                             int split_k_slices,
                                             int mma_tile,
                                             int M_A,
                                             int K,
                                             int K_B,
                                             int N,
                                             int gather_ab_size,
                                             float alpha,
                                             float beta) {
  using ElementA = typename torch_to_cutlass<SA>::type;
  using ElementB = typename torch_to_cutlass<SB>::type;
  using ElementOutput = typename torch_to_cutlass<SO>::type;
  using ElementAccumulator = typename torch_to_cutlass<ACC>::type;
  TORCH_CHECK(tensor_a.scalar_type() == SA && tensor_b.scalar_type() == SB);
  TORCH_CHECK(tensor_c.scalar_type() == SO && tensor_d.scalar_type() == SO);
  auto tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    case warpconvnet::gemm::MMATile::Tile128x128x32:
      return ::warpconvnet::gemm::run_cutlass_gemm_trAB_gather<ElementA,
                                                               ElementB,
                                                               ElementOutput,
                                                               ElementAccumulator,
                                                               warpconvnet::gemm::Tile128x128x32,
                                                               cutlass::arch::Sm80>(
          tensor_a.data_ptr(),
          tensor_b.data_ptr(),
          tensor_c.data_ptr(),
          tensor_d.data_ptr(),
          indices_a.data_ptr<int>(),
          indices_b.data_ptr<int>(),
          split_k_slices,
          M_A,
          K,
          K_B,
          N,
          gather_ab_size,
          alpha,
          beta);
    case warpconvnet::gemm::MMATile::Tile128x64x32:
      return ::warpconvnet::gemm::run_cutlass_gemm_trAB_gather<ElementA,
                                                               ElementB,
                                                               ElementOutput,
                                                               ElementAccumulator,
                                                               warpconvnet::gemm::Tile128x64x32,
                                                               cutlass::arch::Sm80>(
          tensor_a.data_ptr(),
          tensor_b.data_ptr(),
          tensor_c.data_ptr(),
          tensor_d.data_ptr(),
          indices_a.data_ptr<int>(),
          indices_b.data_ptr<int>(),
          split_k_slices,
          M_A,
          K,
          K_B,
          N,
          gather_ab_size,
          alpha,
          beta);
    case warpconvnet::gemm::MMATile::Tile64x128x32:
      return ::warpconvnet::gemm::run_cutlass_gemm_trAB_gather<ElementA,
                                                               ElementB,
                                                               ElementOutput,
                                                               ElementAccumulator,
                                                               warpconvnet::gemm::Tile64x128x32,
                                                               cutlass::arch::Sm80>(
          tensor_a.data_ptr(),
          tensor_b.data_ptr(),
          tensor_c.data_ptr(),
          tensor_d.data_ptr(),
          indices_a.data_ptr<int>(),
          indices_b.data_ptr<int>(),
          split_k_slices,
          M_A,
          K,
          K_B,
          N,
          gather_ab_size,
          alpha,
          beta);
    case warpconvnet::gemm::MMATile::Tile64x64x32:
      return ::warpconvnet::gemm::run_cutlass_gemm_trAB_gather<ElementA,
                                                               ElementB,
                                                               ElementOutput,
                                                               ElementAccumulator,
                                                               warpconvnet::gemm::Tile64x64x32,
                                                               cutlass::arch::Sm80>(
          tensor_a.data_ptr(),
          tensor_b.data_ptr(),
          tensor_c.data_ptr(),
          tensor_d.data_ptr(),
          indices_a.data_ptr<int>(),
          indices_b.data_ptr<int>(),
          split_k_slices,
          M_A,
          K,
          K_B,
          N,
          gather_ab_size,
          alpha,
          beta);
    default:
      TORCH_CHECK(false, "Unsupported mma_tile value");
  }
}

// ------------------- Implementations -------------------
int cutlass_gemm_AD_gather_scatter(torch::Tensor tensor_a,
                                   torch::Tensor tensor_b,
                                   torch::Tensor tensor_c,
                                   torch::Tensor tensor_d,
                                   torch::Tensor indices_a,
                                   torch::Tensor indices_d,
                                   torch::ScalarType accumulator_type,
                                   int split_k_slices,
                                   int mma_tile,
                                   float alpha,
                                   float beta) {
  // Validate dimensions and types
  const auto params =
      validate_ad_gather_scatter_args(tensor_a, tensor_b, tensor_c, tensor_d, indices_a, indices_d);
  TORCH_CHECK(accumulator_type == torch::kFloat16 || accumulator_type == torch::kFloat32,
              "accumulator_type must be float16 or float32");

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_b = tensor_b.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  auto scalar_d = tensor_d.scalar_type();

  int status = 0;
  bool handled = false;

#define DISPATCH_GEMM_HANDLE(SA, SB, SO, ALLOW_F16_ACC)                                   \
  if (!handled && scalar_a == SA && scalar_b == SB && scalar_c == SO && scalar_d == SO) { \
    handled = true;                                                                       \
    if (accumulator_type == torch::kFloat16 && ALLOW_F16_ACC) {                           \
      status = dispatch_cutlass_gemm_ad_gather_scatter<SA, SB, SO, torch::kFloat16>(      \
          tensor_a,                                                                       \
          tensor_b,                                                                       \
          tensor_c,                                                                       \
          tensor_d,                                                                       \
          indices_a,                                                                      \
          indices_d,                                                                      \
          split_k_slices,                                                                 \
          mma_tile,                                                                       \
          params.M_A,                                                                     \
          params.K,                                                                       \
          params.N,                                                                       \
          params.M_C,                                                                     \
          params.gather_ad_size,                                                          \
          alpha,                                                                          \
          beta);                                                                          \
    } else if (accumulator_type == torch::kFloat32) {                                     \
      status = dispatch_cutlass_gemm_ad_gather_scatter<SA, SB, SO, torch::kFloat32>(      \
          tensor_a,                                                                       \
          tensor_b,                                                                       \
          tensor_c,                                                                       \
          tensor_d,                                                                       \
          indices_a,                                                                      \
          indices_d,                                                                      \
          split_k_slices,                                                                 \
          mma_tile,                                                                       \
          params.M_A,                                                                     \
          params.K,                                                                       \
          params.N,                                                                       \
          params.M_C,                                                                     \
          params.gather_ad_size,                                                          \
          alpha,                                                                          \
          beta);                                                                          \
    } else {                                                                              \
      TORCH_CHECK(false, "Unsupported accumulator type");                                 \
    }                                                                                     \
  }

  // Dispatch based on tensor types and accumulator type
  DISPATCH_GEMM_HANDLE(torch::kFloat16, torch::kFloat16, torch::kFloat16, true);
  DISPATCH_GEMM_HANDLE(torch::kFloat16, torch::kFloat16, torch::kFloat32, true);
  DISPATCH_GEMM_HANDLE(torch::kBFloat16, torch::kBFloat16, torch::kBFloat16, false);
  DISPATCH_GEMM_HANDLE(torch::kBFloat16, torch::kBFloat16, torch::kFloat32, false);
#undef DISPATCH_GEMM_HANDLE

  // Special case for float32 tensors. Tensor cores only support half precision. Convert to half
  // precision.
  if (!handled && scalar_a == torch::kFloat32 && scalar_b == torch::kFloat32 &&
      scalar_c == torch::kFloat32 && scalar_d == torch::kFloat32) {
    handled = true;
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    status = dispatch_cutlass_gemm_ad_gather_scatter<torch::kFloat16,
                                                     torch::kFloat16,
                                                     torch::kFloat32,
                                                     torch::kFloat32>(tensor_a,
                                                                      tensor_b,
                                                                      tensor_c,
                                                                      tensor_d,
                                                                      indices_a,
                                                                      indices_d,
                                                                      split_k_slices,
                                                                      mma_tile,
                                                                      params.M_A,
                                                                      params.K,
                                                                      params.N,
                                                                      params.M_C,
                                                                      params.gather_ad_size,
                                                                      alpha,
                                                                      beta);
  }
  if (!handled) {
    std::stringstream ss;
    ss << "Unsupported tensor type combination." << "A: " << scalar_a << " B: " << scalar_b
       << " C: " << scalar_c << " D: " << scalar_d << " Acc: " << accumulator_type;
    TORCH_CHECK(false, ss.str());
  }
  return status;
}

int cutlass_gemm_trAB_gather(torch::Tensor tensor_a,
                             torch::Tensor tensor_b,
                             torch::Tensor tensor_c,
                             torch::Tensor tensor_d,
                             torch::Tensor indices_a,
                             torch::Tensor indices_b,
                             torch::ScalarType accumulator_type,
                             int split_k_slices,
                             int mma_tile,
                             float alpha,
                             float beta) {
  const auto params =
      validate_trAB_gather_args(tensor_a, tensor_b, tensor_c, tensor_d, indices_a, indices_b);
  TORCH_CHECK(accumulator_type == torch::kFloat16 || accumulator_type == torch::kFloat32,
              "accumulator_type must be float16 or float32");
  auto scalar_a = tensor_a.scalar_type();
  auto scalar_b = tensor_b.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  auto scalar_d = tensor_d.scalar_type();

  int status = 0;
  bool handled = false;

#define DISPATCH_GEMM_HANDLE(SA, SB, SO, ALLOW_F16_ACC)                                         \
  if (!handled && scalar_a == SA && scalar_b == SB && scalar_c == SO && scalar_d == SO) {       \
    handled = true;                                                                             \
    if (accumulator_type == torch::kFloat16 && ALLOW_F16_ACC) {                                 \
      status =                                                                                  \
          dispatch_cutlass_gemm_trAB_gather<SA, SB, SO, torch::kFloat16>(tensor_a,              \
                                                                         tensor_b,              \
                                                                         tensor_c,              \
                                                                         tensor_d,              \
                                                                         indices_a,             \
                                                                         indices_b,             \
                                                                         split_k_slices,        \
                                                                         mma_tile,              \
                                                                         params.M_A,            \
                                                                         params.K,              \
                                                                         params.K_B,            \
                                                                         params.N,              \
                                                                         params.gather_ab_size, \
                                                                         alpha,                 \
                                                                         beta);                 \
    } else if (accumulator_type == torch::kFloat32) {                                           \
      status =                                                                                  \
          dispatch_cutlass_gemm_trAB_gather<SA, SB, SO, torch::kFloat32>(tensor_a,              \
                                                                         tensor_b,              \
                                                                         tensor_c,              \
                                                                         tensor_d,              \
                                                                         indices_a,             \
                                                                         indices_b,             \
                                                                         split_k_slices,        \
                                                                         mma_tile,              \
                                                                         params.M_A,            \
                                                                         params.K,              \
                                                                         params.K_B,            \
                                                                         params.N,              \
                                                                         params.gather_ab_size, \
                                                                         alpha,                 \
                                                                         beta);                 \
    } else {                                                                                    \
      TORCH_CHECK(false, "Unsupported accumulator type");                                       \
    }                                                                                           \
  }
  DISPATCH_GEMM_HANDLE(torch::kFloat16, torch::kFloat16, torch::kFloat16, true);
  DISPATCH_GEMM_HANDLE(torch::kFloat16, torch::kFloat16, torch::kFloat32, true);
  DISPATCH_GEMM_HANDLE(torch::kBFloat16, torch::kBFloat16, torch::kBFloat16, false);
  DISPATCH_GEMM_HANDLE(torch::kBFloat16, torch::kBFloat16, torch::kFloat32, false);

#undef DISPATCH_GEMM_HANDLE

  // Special case for float32 tensors. Tensor cores only support half precision. Convert to half
  // precision.
  if (!handled && scalar_a == torch::kFloat32 && scalar_b == torch::kFloat32 &&
      scalar_c == torch::kFloat32 && scalar_d == torch::kFloat32) {
    handled = true;
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    status = dispatch_cutlass_gemm_trAB_gather<torch::kFloat16,
                                               torch::kFloat16,
                                               torch::kFloat32,
                                               torch::kFloat32>(tensor_a,
                                                                tensor_b,
                                                                tensor_c,
                                                                tensor_d,
                                                                indices_a,
                                                                indices_b,
                                                                split_k_slices,
                                                                mma_tile,
                                                                params.M_A,
                                                                params.K,
                                                                params.K_B,
                                                                params.N,
                                                                params.gather_ab_size,
                                                                alpha,
                                                                beta);
  }
  if (!handled) {
    std::stringstream ss;
    ss << "Unsupported tensor type combination." << "A: " << scalar_a << " B: " << scalar_b
       << " C: " << scalar_c << " D: " << scalar_d << " Acc: " << accumulator_type;
    TORCH_CHECK(false, ss.str());
  }
  return status;
}

// Non cutlass implicit GEMM
int implicit_gemm_cuda(torch::Tensor a,
                       torch::Tensor b,
                       torch::Tensor c,
                       torch::Tensor in_map,
                       torch::Tensor out_map,
                       const std::string &kernel_type,
                       int block_size) {
  // Validate dimensions and types
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "a, b, and c must be 2D");
  TORCH_CHECK(in_map.dim() == 1 && out_map.dim() == 1, "in_map and out_map must be 1D");
  TORCH_CHECK(in_map.scalar_type() == torch::kInt32 && out_map.scalar_type() == torch::kInt32,
              "in_map and out_map must be int32");
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(),
              "a, b, and c must have the same type");
  int hA = a.size(0);
  int wA = a.size(1);
  int hB = b.size(0);
  int wB = b.size(1);
  TORCH_CHECK(wA == hB,
              "Matrix dimensions must be compatible for multiplication. wA: " + std::to_string(wA) +
                  ", hB: " + std::to_string(hB));
  TORCH_CHECK(c.size(1) == wB, "c.size(1) must be match wB");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda() && in_map.is_cuda() && out_map.is_cuda(),
              "a, b, c, in_map, and out_map must be on GPU");

  a = a.contiguous();
  b = b.contiguous();
  c = c.contiguous();
  in_map = in_map.contiguous();
  out_map = out_map.contiguous();

  int indices_size = in_map.size(0);
  TORCH_CHECK(indices_size == out_map.size(0),
              "in_map and out_map must have the same number of rows");

  // Dispatch based on tensor types
  int status = 0;
  if (a.scalar_type() == torch::kFloat32) {
    status = ::warpconvnet::implicit_gemm::run_implicit_gemm_templated<float, float, float, int>(
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        wA,
        hA,
        wB,
        hB,
        indices_size,
        kernel_type,
        block_size);
  } else if (a.scalar_type() == torch::kFloat16) {
    status = ::warpconvnet::implicit_gemm::run_implicit_gemm_templated<__half, __half, __half, int>(
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        wA,
        hA,
        wB,
        hB,
        indices_size,
        kernel_type,
        block_size);
  } else if (a.scalar_type() == torch::kBFloat16) {
    status = ::warpconvnet::implicit_gemm::
        run_implicit_gemm_templated<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int>(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            in_map.data_ptr<int>(),
            out_map.data_ptr<int>(),
            wA,
            hA,
            wB,
            hB,
            indices_size,
            kernel_type,
            block_size);
  } else {
    TORCH_CHECK(false, "Unsupported data type for implicit GEMM");
  }
  if (status != 0) {
    TORCH_CHECK(false, "Implicit GEMM kernel failed with status: " + std::to_string(status));
  }
  return status;
}

// Non cutlass split-K implicit GEMM
int split_k_implicit_gemm_cuda(torch::Tensor a,
                               torch::Tensor b,
                               torch::Tensor c,
                               torch::Tensor indices_a,
                               torch::Tensor indices_b,
                               int split_k_factor,
                               int block_size) {
  // Validate dimensions and types
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "a, b, and c must be 2D");
  TORCH_CHECK(indices_a.dim() == 1 && indices_b.dim() == 1, "indices_a and indices_b must be 1D");
  TORCH_CHECK(indices_a.scalar_type() == torch::kInt32 && indices_b.scalar_type() == torch::kInt32,
              "indices_a and indices_b must be int32");
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(),
              "a, b, and c must have the same type");

  // Validate dimensions
  int N = std::max(a.size(0), b.size(0));
  int C_a = a.size(1);
  int C_b = b.size(1);
  int K = indices_a.size(0);
  TORCH_CHECK(c.size(0) == C_a && c.size(1) == C_b, "c.shape must be match a.shape and b.shape");
  TORCH_CHECK(indices_b.size(0) == K, "indices_b.size(0) must be match K");
  TORCH_CHECK(
      a.is_cuda() && b.is_cuda() && c.is_cuda() && indices_a.is_cuda() && indices_b.is_cuda(),
      "a, b, c, indices_a, and indices_b must be on GPU");

  // Contiguous tensors
  a = a.contiguous();
  b = b.contiguous();
  c = c.contiguous();
  indices_a = indices_a.contiguous();
  indices_b = indices_b.contiguous();

  // Dispatch based on tensor types
  int status = 0;
  if (a.scalar_type() == torch::kFloat32) {
    status = ::warpconvnet::split_k_implicit_gemm::
        run_split_k_implicit_gemm_templated<float, float, float, int>(a.data_ptr(),
                                                                      b.data_ptr(),
                                                                      c.data_ptr(),
                                                                      indices_a.data_ptr<int>(),
                                                                      indices_b.data_ptr<int>(),
                                                                      N,
                                                                      C_a,
                                                                      C_b,
                                                                      K,
                                                                      split_k_factor,
                                                                      block_size);
  } else if (a.scalar_type() == torch::kFloat16) {
    status = ::warpconvnet::split_k_implicit_gemm::
        run_split_k_implicit_gemm_templated<__half, __half, __half, int>(a.data_ptr(),
                                                                         b.data_ptr(),
                                                                         c.data_ptr(),
                                                                         indices_a.data_ptr<int>(),
                                                                         indices_b.data_ptr<int>(),
                                                                         N,
                                                                         C_a,
                                                                         C_b,
                                                                         K,
                                                                         split_k_factor,
                                                                         block_size);
  } else if (a.scalar_type() == torch::kBFloat16) {
    status = ::warpconvnet::split_k_implicit_gemm::
        run_split_k_implicit_gemm_templated<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int>(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            indices_a.data_ptr<int>(),
            indices_b.data_ptr<int>(),
            N,
            C_a,
            C_b,
            K,
            split_k_factor,
            block_size);
  } else {
    TORCH_CHECK(false, "Unsupported data type for split-K implicit GEMM");
  }
  if (status != 0) {
    TORCH_CHECK(false,
                "Split-K implicit GEMM kernel failed with status: " + std::to_string(status));
  }
  return status;
}

// WMMA SM80 split-K implicit GEMM
int wmma_split_k_implicit_gemm_sm80(torch::Tensor a,
                                    torch::Tensor b,
                                    torch::Tensor c,
                                    torch::Tensor indices_a,
                                    torch::Tensor indices_b,
                                    int split_k_factor) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "a, b, and c must be 2D");
  TORCH_CHECK(indices_a.dim() == 1 && indices_b.dim() == 1, "indices_a and indices_b must be 1D");
  TORCH_CHECK(indices_a.scalar_type() == torch::kInt32 && indices_b.scalar_type() == torch::kInt32,
              "indices_a and indices_b must be int32");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same type");
  TORCH_CHECK(
      a.is_cuda() && b.is_cuda() && c.is_cuda() && indices_a.is_cuda() && indices_b.is_cuda(),
      "a, b, c, indices_a, and indices_b must be on GPU");

  // Validate dimensions
  int N = std::max(a.size(0), b.size(0));
  int C_a = a.size(1);
  int C_b = b.size(1);
  int K = indices_a.size(0);
  TORCH_CHECK(c.size(0) == C_a && c.size(1) == C_b, "c.shape must be match a.shape and b.shape");
  TORCH_CHECK(indices_b.size(0) == K, "indices_b.size(0) must be match K");

  // Contiguous tensors
  a = a.contiguous();
  b = b.contiguous();
  c = c.contiguous();
  indices_a = indices_a.contiguous();
  indices_b = indices_b.contiguous();

  int status = 0;
  auto dt = a.scalar_type();
  if (dt == torch::kFloat16) {
    TORCH_CHECK(c.scalar_type() == torch::kFloat16, "For float16 inputs, c must be float16");
    status = ::warpconvnet::wmma_split_k_sm80::run_split_k_wmma_templated<half, half, int>(
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        indices_a.data_ptr<int>(),
        indices_b.data_ptr<int>(),
        N,
        C_a,
        C_b,
        K,
        split_k_factor);
#ifndef DISABLE_BFLOAT16
  } else if (dt == torch::kBFloat16) {
    TORCH_CHECK(c.scalar_type() == torch::kBFloat16, "For bfloat16 inputs, c must be bfloat16");
    status =
        ::warpconvnet::wmma_split_k_sm80::run_split_k_wmma_templated<nv_bfloat16, nv_bfloat16, int>(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            indices_a.data_ptr<int>(),
            indices_b.data_ptr<int>(),
            N,
            C_a,
            C_b,
            K,
            split_k_factor);
#endif
  } else if (dt == torch::kFloat32) {
    // Convert A and B to half precision; accumulate into float32 C
    auto a_half = a.to(torch::kFloat16);
    auto b_half = b.to(torch::kFloat16);
    TORCH_CHECK(c.scalar_type() == torch::kFloat32, "For float32 case, c must be float32");
    status = ::warpconvnet::wmma_split_k_sm80::run_split_k_wmma_templated<half, float, int>(
        a_half.data_ptr(),
        b_half.data_ptr(),
        c.data_ptr(),
        indices_a.data_ptr<int>(),
        indices_b.data_ptr<int>(),
        N,
        C_a,
        C_b,
        K,
        split_k_factor);
  } else {
    TORCH_CHECK(false, "Unsupported data type for WMMA split-K implicit GEMM");
  }

  TORCH_CHECK(status == 0,
              "WMMA split-K implicit GEMM kernel failed with status: " + std::to_string(status));
  return status;
}

// Public API function instead of lambda
int wmma_implicit_gemm_sm80(torch::Tensor tensor_a,
                            torch::Tensor tensor_b,
                            torch::Tensor tensor_c,
                            torch::Tensor tensor_d,
                            torch::Tensor indices_a,
                            torch::Tensor indices_d,
                            float alpha,
                            float beta) {
  TORCH_CHECK(tensor_a.dim() == 2 && tensor_b.dim() == 2 && tensor_d.dim() == 2,
              "tensor_a, tensor_b, and tensor_d must be 2D");
  if (beta != 0.0f) {
    TORCH_CHECK(tensor_c.dim() == 2, "tensor_c must be 2D when beta != 0");
  }
  TORCH_CHECK(indices_a.dim() == 1 && indices_d.dim() == 1, "indices_a and indices_d must be 1D");
  TORCH_CHECK(tensor_a.is_cuda() && tensor_b.is_cuda() && tensor_d.is_cuda() &&
                  indices_a.is_cuda() && indices_d.is_cuda(),
              "A, B, D, indices_a, and indices_d must be CUDA tensors");
  if (beta != 0.0f) {
    TORCH_CHECK(tensor_c.is_cuda(), "C must be CUDA tensor when beta != 0");
  }
  tensor_a = tensor_a.contiguous();
  tensor_b = tensor_b.contiguous();
  if (beta != 0.0f) tensor_c = tensor_c.contiguous();
  tensor_d = tensor_d.contiguous();
  indices_a = indices_a.contiguous();
  indices_d = indices_d.contiguous();

  int64_t M = tensor_a.size(0);
  int64_t K = tensor_a.size(1);
  int64_t N = tensor_b.size(1);
  TORCH_CHECK(tensor_b.size(0) == K, "B.rows must equal A.cols");
  if (beta != 0.0f) {
    TORCH_CHECK(tensor_c.size(0) == M && tensor_c.size(1) == N, "C must be [M,N]");
  }
  int64_t Q = tensor_d.size(0);
  TORCH_CHECK(tensor_d.size(1) == N, "D must be [Q,N]");
  int64_t P = indices_a.size(0);
  TORCH_CHECK(indices_d.size(0) == P, "indices_a and indices_d must have same length");

  if (indices_a.scalar_type() != torch::kInt64) indices_a = indices_a.to(torch::kInt64);
  if (indices_d.scalar_type() != torch::kInt64) indices_d = indices_d.to(torch::kInt64);

  const void *Aptr = tensor_a.data_ptr();
  const void *Bptr = tensor_b.data_ptr();
  const void *Cptr = (beta == 0.0f) ? nullptr : tensor_c.data_ptr();
  void *Dptr = tensor_d.data_ptr();
  const long *a_inds_ptr = indices_a.data_ptr<long>();
  const long *d_inds_ptr = indices_d.data_ptr<long>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  auto dtA = tensor_a.scalar_type();
  auto dtB = tensor_b.scalar_type();
  auto dtD = tensor_d.scalar_type();
  TORCH_CHECK(dtA == dtB, "A and B dtypes must match");

  int status = 0;
  if (dtA == torch::kFloat16 && dtD == torch::kFloat32) {
    status = ::warpconvnet::wmma_implicit_gemm_sm80::run<half, float>(Aptr,
                                                                      Bptr,
                                                                      Cptr,
                                                                      Dptr,
                                                                      a_inds_ptr,
                                                                      d_inds_ptr,
                                                                      (int)M,
                                                                      (int)K,
                                                                      (int)N,
                                                                      (int)P,
                                                                      (int)Q,
                                                                      alpha,
                                                                      beta,
                                                                      stream);
  } else if (dtA == torch::kFloat16 && dtD == torch::kFloat16) {
    status = ::warpconvnet::wmma_implicit_gemm_sm80::run<half, half>(Aptr,
                                                                     Bptr,
                                                                     Cptr,
                                                                     Dptr,
                                                                     a_inds_ptr,
                                                                     d_inds_ptr,
                                                                     (int)M,
                                                                     (int)K,
                                                                     (int)N,
                                                                     (int)P,
                                                                     (int)Q,
                                                                     alpha,
                                                                     beta,
                                                                     stream);
#ifndef DISABLE_BFLOAT16
  } else if (dtA == torch::kBFloat16 && dtD == torch::kFloat32) {
    status = ::warpconvnet::wmma_implicit_gemm_sm80::run<nv_bfloat16, float>(Aptr,
                                                                             Bptr,
                                                                             Cptr,
                                                                             Dptr,
                                                                             a_inds_ptr,
                                                                             d_inds_ptr,
                                                                             (int)M,
                                                                             (int)K,
                                                                             (int)N,
                                                                             (int)P,
                                                                             (int)Q,
                                                                             alpha,
                                                                             beta,
                                                                             stream);
  } else if (dtA == torch::kBFloat16 && dtD == torch::kBFloat16) {
    status = ::warpconvnet::wmma_implicit_gemm_sm80::run<nv_bfloat16, nv_bfloat16>(Aptr,
                                                                                   Bptr,
                                                                                   Cptr,
                                                                                   Dptr,
                                                                                   a_inds_ptr,
                                                                                   d_inds_ptr,
                                                                                   (int)M,
                                                                                   (int)K,
                                                                                   (int)N,
                                                                                   (int)P,
                                                                                   (int)Q,
                                                                                   alpha,
                                                                                   beta,
                                                                                   stream);
#endif
  } else {
    TORCH_CHECK(false, "Unsupported dtype combination for WMMA SM80");
  }
  TORCH_CHECK(status == 0, "WMMA SM80 run failed");
  return status;
}

void register_gemm(py::module_ &m) {
  py::module_ gemm = m.def_submodule(
      "gemm", "CUTLASS GEMM with gather/scatter operations supporting multiple precisions");

  gemm.def("cutlass_gemm_AD_gather_scatter",
           &cutlass_gemm_AD_gather_scatter,
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_d"),
           py::arg("accumulator_type") = torch::kFloat32,
           py::arg("split_k_slices") = 1,
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 1.0f);

  gemm.def("cutlass_gemm_trAB_gather",
           &cutlass_gemm_trAB_gather,
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_b"),
           py::arg("accumulator_type") = torch::kFloat32,
           py::arg("split_k_slices") = 1,
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 1.0f);

  py::enum_<warpconvnet::gemm::GemmStatus>(gemm, "GemmStatus")
      .value("kSuccess", warpconvnet::gemm::GemmStatus::kSuccess)
      .value("kErrorProblemNotSupported", warpconvnet::gemm::GemmStatus::kErrorProblemNotSupported)
      .value("kErrorKernelInitialization",
             warpconvnet::gemm::GemmStatus::kErrorKernelInitialization)
      .value("kErrorKernelExecution", warpconvnet::gemm::GemmStatus::kErrorKernelExecution)
      .value("kErrorUnsupportedConfig", warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig)
      .value("kErrorInvalidParameters", warpconvnet::gemm::GemmStatus::kErrorInvalidParameters)
      .value("kErrorMixedInputUnsupported",
             warpconvnet::gemm::GemmStatus::kErrorMixedInputUnsupported)
      .export_values();

  gemm.def(
      "gemm_status_to_string",
      [](warpconvnet::gemm::GemmStatus status) {
        return std::string(warpconvnet::gemm::GemmStatusToString(status));
      },
      py::arg("status"));

  gemm.def("implicit_gemm",
           &implicit_gemm_cuda,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("in_map"),
           py::arg("out_map"),
           py::arg("kernel_type") = "basic",
           py::arg("block_size") = 16);

  gemm.def("split_k_implicit_gemm",
           &split_k_implicit_gemm_cuda,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("indices_a"),
           py::arg("indices_b"),
           py::arg("split_k_factor") = 4,
           py::arg("block_size") = 16);

  gemm.def("wmma_split_k_implicit_gemm_sm80",
           &wmma_split_k_implicit_gemm_sm80,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("indices_a"),
           py::arg("indices_b"),
           py::arg("split_k_factor") = 4);

  // Thin dispatcher that mirrors other CUTLASS-style helpers
  gemm.def("wmma_implicit_gemm_sm80",
           &wmma_implicit_gemm_sm80,
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_d"),
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 0.0f,
           "SM80 WMMA implicit GEMM with gather/scatter and CBLAS-style alpha/beta epilogue."
           " Skips loading C when beta == 0."
           "Default alpha = 1.0, beta = 0.0.");

  // Segmented WMMA SM80 implicit GEMM (per-segment B, per-row gates)
  gemm.def(
      "wmma_segmented_gemm_sm80",
      [](torch::Tensor tensor_a,
         torch::Tensor tensor_b_multi,
         torch::Tensor tensor_c,
         torch::Tensor tensor_d,
         torch::Tensor indices_a,
         torch::Tensor indices_d,
         torch::Tensor gates,
         torch::Tensor seg_offsets,
         float alpha,
         float beta) {
        TORCH_CHECK(tensor_a.dim() == 2, "tensor_a must be 2D [M,K]");
        TORCH_CHECK(tensor_b_multi.dim() == 3, "tensor_b_multi must be 3D [E,K,N]");
        if (beta != 0.0f) {
          TORCH_CHECK(tensor_c.dim() == 2, "tensor_c must be 2D when beta != 0");
        }
        TORCH_CHECK(tensor_d.dim() == 2, "tensor_d must be 2D [Q,N]");
        TORCH_CHECK(indices_a.dim() == 1 && indices_d.dim() == 1, "indices_a/d must be 1D");
        TORCH_CHECK(gates.dim() == 1, "gates must be 1D");
        TORCH_CHECK(seg_offsets.dim() == 1, "seg_offsets must be 1D [E+1]");

        TORCH_CHECK(tensor_a.is_cuda() && tensor_b_multi.is_cuda() && tensor_d.is_cuda() &&
                        indices_a.is_cuda() && indices_d.is_cuda() && gates.is_cuda() &&
                        seg_offsets.is_cuda(),
                    "All inputs must be CUDA tensors");
        if (beta != 0.0f) TORCH_CHECK(tensor_c.is_cuda(), "tensor_c must be CUDA when beta != 0");

        tensor_a = tensor_a.contiguous();
        tensor_b_multi = tensor_b_multi.contiguous();
        if (beta != 0.0f) tensor_c = tensor_c.contiguous();
        tensor_d = tensor_d.contiguous();
        indices_a = indices_a.contiguous();
        indices_d = indices_d.contiguous();
        gates = gates.contiguous();
        seg_offsets = seg_offsets.contiguous();

        int64_t M = tensor_a.size(0);
        int64_t K = tensor_a.size(1);
        int64_t E = tensor_b_multi.size(0);
        TORCH_CHECK(tensor_b_multi.size(1) == K, "B_multi[:,K,:] must match A.cols");
        int64_t N = tensor_b_multi.size(2);
        int64_t Q = tensor_d.size(0);
        TORCH_CHECK(tensor_d.size(1) == N, "tensor_d second dim must be N");
        int64_t P = indices_a.size(0);
        TORCH_CHECK(indices_d.size(0) == P, "indices_a and indices_d length mismatch");
        TORCH_CHECK(gates.size(0) == P, "gates length must equal P");
        TORCH_CHECK(seg_offsets.size(0) == E + 1, "seg_offsets must be length E+1");

        if (indices_a.scalar_type() != torch::kInt64) indices_a = indices_a.to(torch::kInt64);
        if (indices_d.scalar_type() != torch::kInt64) indices_d = indices_d.to(torch::kInt64);
        TORCH_CHECK(gates.scalar_type() == torch::kFloat32, "gates must be float32");
        if (seg_offsets.scalar_type() != torch::kInt32) seg_offsets = seg_offsets.to(torch::kInt32);

        auto dtA = tensor_a.scalar_type();
        auto dtB = tensor_b_multi.scalar_type();
        auto dtD = tensor_d.scalar_type();
        TORCH_CHECK(dtA == dtB, "A and B_multi dtypes must match");

        const void *Aptr = tensor_a.data_ptr();
        const void *Bptr = tensor_b_multi.data_ptr();
        const void *Cptr = (beta == 0.0f) ? nullptr : tensor_c.data_ptr();
        void *Dptr = tensor_d.data_ptr();
        const long *a_inds_ptr = indices_a.data_ptr<long>();
        const long *d_inds_ptr = indices_d.data_ptr<long>();
        const float *gates_ptr = gates.data_ptr<float>();
        const int *seg_offsets_ptr = seg_offsets.data_ptr<int>();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        int status = 0;
        if (dtA == torch::kFloat16 && dtD == torch::kFloat32) {
          status =
              ::warpconvnet::wmma_segmented_gemm_sm80::run_segmented<half, float>(Aptr,
                                                                                  Bptr,
                                                                                  Cptr,
                                                                                  Dptr,
                                                                                  a_inds_ptr,
                                                                                  d_inds_ptr,
                                                                                  gates_ptr,
                                                                                  seg_offsets_ptr,
                                                                                  (int)M,
                                                                                  (int)K,
                                                                                  (int)N,
                                                                                  (int)P,
                                                                                  (int)Q,
                                                                                  (int)E,
                                                                                  alpha,
                                                                                  beta,
                                                                                  stream);
        } else if (dtA == torch::kFloat16 && dtD == torch::kFloat16) {
          status =
              ::warpconvnet::wmma_segmented_gemm_sm80::run_segmented<half, half>(Aptr,
                                                                                 Bptr,
                                                                                 Cptr,
                                                                                 Dptr,
                                                                                 a_inds_ptr,
                                                                                 d_inds_ptr,
                                                                                 gates_ptr,
                                                                                 seg_offsets_ptr,
                                                                                 (int)M,
                                                                                 (int)K,
                                                                                 (int)N,
                                                                                 (int)P,
                                                                                 (int)Q,
                                                                                 (int)E,
                                                                                 alpha,
                                                                                 beta,
                                                                                 stream);
#ifndef DISABLE_BFLOAT16
        } else if (dtA == torch::kBFloat16 && dtD == torch::kFloat32) {
          status = ::warpconvnet::wmma_segmented_gemm_sm80::run_segmented<nv_bfloat16, float>(
              Aptr,
              Bptr,
              Cptr,
              Dptr,
              a_inds_ptr,
              d_inds_ptr,
              gates_ptr,
              seg_offsets_ptr,
              (int)M,
              (int)K,
              (int)N,
              (int)P,
              (int)Q,
              (int)E,
              alpha,
              beta,
              stream);
        } else if (dtA == torch::kBFloat16 && dtD == torch::kBFloat16) {
          status = ::warpconvnet::wmma_segmented_gemm_sm80::run_segmented<nv_bfloat16, nv_bfloat16>(
              Aptr,
              Bptr,
              Cptr,
              Dptr,
              a_inds_ptr,
              d_inds_ptr,
              gates_ptr,
              seg_offsets_ptr,
              (int)M,
              (int)K,
              (int)N,
              (int)P,
              (int)Q,
              (int)E,
              alpha,
              beta,
              stream);
#endif
        } else if (dtA == torch::kFloat32 && dtD == torch::kFloat32) {
          // Convert inputs to half; accumulate to float32
          auto a_half = tensor_a.to(torch::kFloat16);
          auto b_half = tensor_b_multi.to(torch::kFloat16);
          status =
              ::warpconvnet::wmma_segmented_gemm_sm80::run_segmented<half, float>(a_half.data_ptr(),
                                                                                  b_half.data_ptr(),
                                                                                  Cptr,
                                                                                  Dptr,
                                                                                  a_inds_ptr,
                                                                                  d_inds_ptr,
                                                                                  gates_ptr,
                                                                                  seg_offsets_ptr,
                                                                                  (int)M,
                                                                                  (int)K,
                                                                                  (int)N,
                                                                                  (int)P,
                                                                                  (int)Q,
                                                                                  (int)E,
                                                                                  alpha,
                                                                                  beta,
                                                                                  stream);
        } else {
          TORCH_CHECK(false, "Unsupported dtype combination for WMMA segmented GEMM");
        }
        TORCH_CHECK(status == 0, "WMMA segmented GEMM run failed");
        return status;
      },
      py::arg("tensor_a"),
      py::arg("tensor_b_multi"),
      py::arg("tensor_c"),
      py::arg("tensor_d"),
      py::arg("indices_a"),
      py::arg("indices_d"),
      py::arg("gates"),
      py::arg("seg_offsets"),
      py::arg("alpha") = 1.0f,
      py::arg("beta") = 0.0f,
      "Segmented implicit GEMM (SM80 WMMA): per-segment B tiles with per-row gates.");

  // Segmented tr(A)B gather-scatter for MoE dW (SM80 WMMA)
  gemm.def(
      "wmma_segmented_trab_gather_sm80",
      [](torch::Tensor tensor_x,     // [N_rows, C_in]
         torch::Tensor tensor_dy,    // [M_rows, C_out]
         torch::Tensor tensor_dw,    // [S, C_in, C_out] (usually S==E)
         torch::Tensor indices_a,    // [P] int64
         torch::Tensor indices_d,    // [P] int64
         torch::Tensor gates,        // [P] float32
         torch::Tensor seg_offsets,  // [S+1] int32
         float alpha,
         float beta) {
        TORCH_CHECK(tensor_x.dim() == 2, "tensor_x must be 2D [N_rows, C_in]");
        TORCH_CHECK(tensor_dy.dim() == 2, "tensor_dy must be 2D [M_rows, C_out]");
        TORCH_CHECK(tensor_dw.dim() == 3, "tensor_dw must be 3D [S, C_in, C_out]");
        TORCH_CHECK(indices_a.dim() == 1 && indices_d.dim() == 1, "indices must be 1D");
        TORCH_CHECK(gates.dim() == 1, "gates must be 1D");
        TORCH_CHECK(seg_offsets.dim() == 1, "seg_offsets must be 1D [S+1]");

        TORCH_CHECK(tensor_x.is_cuda() && tensor_dy.is_cuda() && tensor_dw.is_cuda() &&
                        indices_a.is_cuda() && indices_d.is_cuda() && gates.is_cuda() &&
                        seg_offsets.is_cuda(),
                    "All inputs must be CUDA tensors");

        // Make contiguous
        tensor_x = tensor_x.contiguous();
        tensor_dy = tensor_dy.contiguous();
        tensor_dw = tensor_dw.contiguous();
        indices_a = indices_a.contiguous();
        indices_d = indices_d.contiguous();
        gates = gates.contiguous();
        seg_offsets = seg_offsets.contiguous();

        const int64_t N_rows = tensor_x.size(0);
        const int64_t C_in = tensor_x.size(1);
        const int64_t M_rows = tensor_dy.size(0);
        const int64_t C_out = tensor_dy.size(1);
        TORCH_CHECK(tensor_dw.size(1) == C_in && tensor_dw.size(2) == C_out,
                    "tensor_dw must be [S, C_in, C_out]");
        const int64_t S = tensor_dw.size(0);
        const int64_t P = indices_a.size(0);
        TORCH_CHECK(indices_d.size(0) == P, "indices_a and indices_d must have same length");
        TORCH_CHECK(gates.size(0) == P, "gates length must equal P");
        TORCH_CHECK(seg_offsets.size(0) == S + 1,
                    "seg_offsets length must be S+1 matching tensor_dw.shape[0]");

        if (indices_a.scalar_type() != torch::kInt64) indices_a = indices_a.to(torch::kInt64);
        if (indices_d.scalar_type() != torch::kInt64) indices_d = indices_d.to(torch::kInt64);
        TORCH_CHECK(gates.scalar_type() == torch::kFloat32, "gates must be float32");
        if (seg_offsets.scalar_type() != torch::kInt32) seg_offsets = seg_offsets.to(torch::kInt32);

        auto dtX = tensor_x.scalar_type();
        auto dtDY = tensor_dy.scalar_type();
        auto dtDW = tensor_dw.scalar_type();
        TORCH_CHECK(dtX == dtDY, "X and DY dtypes must match");

        const void *Xptr = tensor_x.data_ptr();
        const void *DYptr = tensor_dy.data_ptr();
        void *DWptr = tensor_dw.data_ptr();
        const long *a_inds_ptr = indices_a.data_ptr<long>();
        const long *d_inds_ptr = indices_d.data_ptr<long>();
        const float *gates_ptr = gates.data_ptr<float>();
        const int *seg_offsets_ptr = seg_offsets.data_ptr<int>();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        int status = 0;
        if (dtX == torch::kFloat16 && dtDW == torch::kFloat32) {
          status = ::warpconvnet::wmma_segmented_trab_sm80::run_segmented_trab<half, float>(
              Xptr,
              DYptr,
              DWptr,
              a_inds_ptr,
              d_inds_ptr,
              gates_ptr,
              seg_offsets_ptr,
              (int)N_rows,
              (int)M_rows,
              (int)C_in,
              (int)C_out,
              (int)P,
              (int)S,
              alpha,
              beta,
              stream);
        } else if (dtX == torch::kFloat16 && dtDW == torch::kFloat16) {
          status = ::warpconvnet::wmma_segmented_trab_sm80::run_segmented_trab<half, half>(
              Xptr,
              DYptr,
              DWptr,
              a_inds_ptr,
              d_inds_ptr,
              gates_ptr,
              seg_offsets_ptr,
              (int)N_rows,
              (int)M_rows,
              (int)C_in,
              (int)C_out,
              (int)P,
              (int)S,
              alpha,
              beta,
              stream);
#ifndef DISABLE_BFLOAT16
        } else if (dtX == torch::kBFloat16 && dtDW == torch::kFloat32) {
          status = ::warpconvnet::wmma_segmented_trab_sm80::run_segmented_trab<nv_bfloat16, float>(
              Xptr,
              DYptr,
              DWptr,
              a_inds_ptr,
              d_inds_ptr,
              gates_ptr,
              seg_offsets_ptr,
              (int)N_rows,
              (int)M_rows,
              (int)C_in,
              (int)C_out,
              (int)P,
              (int)S,
              alpha,
              beta,
              stream);
        } else if (dtX == torch::kBFloat16 && dtDW == torch::kBFloat16) {
          status =
              ::warpconvnet::wmma_segmented_trab_sm80::run_segmented_trab<nv_bfloat16, nv_bfloat16>(
                  Xptr,
                  DYptr,
                  DWptr,
                  a_inds_ptr,
                  d_inds_ptr,
                  gates_ptr,
                  seg_offsets_ptr,
                  (int)N_rows,
                  (int)M_rows,
                  (int)C_in,
                  (int)C_out,
                  (int)P,
                  (int)S,
                  alpha,
                  beta,
                  stream);
#endif
        } else if (dtX == torch::kFloat32 && dtDW == torch::kFloat32) {
          // Convert X, DY to fp16 compute; accumulate into fp32 DW
          auto x_half = tensor_x.to(torch::kFloat16);
          auto dy_half = tensor_dy.to(torch::kFloat16);
          status = ::warpconvnet::wmma_segmented_trab_sm80::run_segmented_trab<half, float>(
              x_half.data_ptr(),
              dy_half.data_ptr(),
              DWptr,
              a_inds_ptr,
              d_inds_ptr,
              gates_ptr,
              seg_offsets_ptr,
              (int)N_rows,
              (int)M_rows,
              (int)C_in,
              (int)C_out,
              (int)P,
              (int)S,
              alpha,
              beta,
              stream);
        } else {
          TORCH_CHECK(false, "Unsupported dtype combination for WMMA segmented tr(A)B gather");
        }

        TORCH_CHECK(status == 0, "WMMA segmented tr(A)B kernel failed");
        return status;
      },
      py::arg("tensor_x"),
      py::arg("tensor_dy"),
      py::arg("tensor_dw"),
      py::arg("indices_a"),
      py::arg("indices_d"),
      py::arg("gates"),
      py::arg("seg_offsets"),
      py::arg("alpha") = 1.0f,
      py::arg("beta") = 0.0f,
      "Segmented tr(A)B gather-scatter for MoE dW using SM80 WMMA. Writes into [S,C_in,C_out].");
}

}  // namespace bindings
}  // namespace warpconvnet
