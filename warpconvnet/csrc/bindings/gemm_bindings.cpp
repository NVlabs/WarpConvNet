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
#include <cstddef>
#include <sstream>

#include "../include/gemm_error_codes.h"
#include "../include/gemm_mma_tiles.h"
#include "../include/grouped_gemm_params.h"  // GroupedGemmParams

namespace py = pybind11;

// Forward declarations for CUDA kernels and GEMM launchers in their namespaces
namespace warpconvnet {
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
template <typename ElementA, typename ElementB, typename ElementC, typename Itype>
int run_implicit_gemm_grouped_templated(const void *tensor_a,
                                        const void *tensor_b,
                                        void *tensor_c,
                                        const Itype *in_map,
                                        const Itype *out_map,
                                        const Itype *weight_idx,
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

namespace cute_gemm {
template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_ad_gather_scatter(const void *a,
                                    const void *b,
                                    const void *c,
                                    void *d,
                                    const int *idx_a,
                                    const int *idx_d,
                                    int M_A,
                                    int K,
                                    int N,
                                    int M_C,
                                    int idx_size,
                                    float alpha,
                                    float beta);

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_grouped_ad_gather_scatter(const void *a,
                                            void *d,
                                            const int *in_map,
                                            const int *out_map,
                                            const GroupedGemmParams &params,
                                            int total_m_tiles,
                                            int K,
                                            int N,
                                            float alpha);

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_trAB_gather(const void *a,
                              const void *b,
                              const void *c,
                              void *d,
                              const int *idx_a,
                              const int *idx_b,
                              int M_A,
                              int K,
                              int K_B,
                              int N,
                              int idx_size,
                              float alpha,
                              float beta);

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_ad_gather_scatter_staged(const void *a,
                                           const void *b,
                                           const void *c,
                                           void *d,
                                           const int *idx_a,
                                           const int *idx_d,
                                           int M_A,
                                           int K,
                                           int N,
                                           int M_C,
                                           int idx_size,
                                           float alpha,
                                           float beta,
                                           int num_stages,
                                           bool use_cp_async);

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_trAB_gather_staged(const void *a,
                                     const void *b,
                                     const void *c,
                                     void *d,
                                     const int *idx_a,
                                     const int *idx_b,
                                     int M_A,
                                     int K,
                                     int K_B,
                                     int N,
                                     int idx_size,
                                     float alpha,
                                     float beta,
                                     int num_stages,
                                     bool use_cp_async);

template <typename ElementInput, typename TileTag, typename ElementOutput>
int run_cute_gemm_grouped_ad_gather_scatter_staged(const void *a,
                                                   void *d,
                                                   const int *in_map,
                                                   const int *out_map,
                                                   const GroupedGemmParams &params,
                                                   int total_m_tiles,
                                                   int K,
                                                   int N,
                                                   float alpha,
                                                   int num_stages,
                                                   bool use_cp_async);
}  // namespace cute_gemm

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

// Grouped implicit GEMM: multiple weight matrices per launch via weight_idx
int implicit_gemm_grouped_cuda(torch::Tensor a,
                               torch::Tensor b,
                               torch::Tensor c,
                               torch::Tensor in_map,
                               torch::Tensor out_map,
                               torch::Tensor weight_idx,
                               const std::string &kernel_type,
                               int block_size) {
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(b.dim() == 3, "b must be 3D (num_weights x hB x wB)");
  TORCH_CHECK(in_map.dim() == 1 && out_map.dim() == 1 && weight_idx.dim() == 1,
              "in_map, out_map, and weight_idx must be 1D");
  TORCH_CHECK(in_map.scalar_type() == torch::kInt32 && out_map.scalar_type() == torch::kInt32 &&
                  weight_idx.scalar_type() == torch::kInt32,
              "in_map, out_map, and weight_idx must be int32");
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(),
              "a, b, and c must have the same type");

  int hA = a.size(0);
  int wA = a.size(1);
  int hB = b.size(1);
  int wB = b.size(2);
  TORCH_CHECK(wA == hB,
              "Matrix dimensions must be compatible. wA: " + std::to_string(wA) +
                  ", hB: " + std::to_string(hB));
  TORCH_CHECK(c.size(1) == wB, "c.size(1) must match wB");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda() && in_map.is_cuda() && out_map.is_cuda() &&
                  weight_idx.is_cuda(),
              "All tensors must be on GPU");

  a = a.contiguous();
  b = b.contiguous();
  c = c.contiguous();
  in_map = in_map.contiguous();
  out_map = out_map.contiguous();
  weight_idx = weight_idx.contiguous();

  int indices_size = in_map.size(0);
  TORCH_CHECK(indices_size == out_map.size(0) && indices_size == weight_idx.size(0),
              "in_map, out_map, and weight_idx must have the same length");

  int status = 0;
  if (a.scalar_type() == torch::kFloat32) {
    status =
        ::warpconvnet::implicit_gemm::run_implicit_gemm_grouped_templated<float, float, float, int>(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            in_map.data_ptr<int>(),
            out_map.data_ptr<int>(),
            weight_idx.data_ptr<int>(),
            wA,
            hA,
            wB,
            hB,
            indices_size,
            kernel_type,
            block_size);
  } else if (a.scalar_type() == torch::kFloat16) {
    status = ::warpconvnet::implicit_gemm::
        run_implicit_gemm_grouped_templated<__half, __half, __half, int>(a.data_ptr(),
                                                                         b.data_ptr(),
                                                                         c.data_ptr(),
                                                                         in_map.data_ptr<int>(),
                                                                         out_map.data_ptr<int>(),
                                                                         weight_idx.data_ptr<int>(),
                                                                         wA,
                                                                         hA,
                                                                         wB,
                                                                         hB,
                                                                         indices_size,
                                                                         kernel_type,
                                                                         block_size);
  } else if (a.scalar_type() == torch::kBFloat16) {
    status = ::warpconvnet::implicit_gemm::
        run_implicit_gemm_grouped_templated<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int>(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            in_map.data_ptr<int>(),
            out_map.data_ptr<int>(),
            weight_idx.data_ptr<int>(),
            wA,
            hA,
            wB,
            hB,
            indices_size,
            kernel_type,
            block_size);
  } else {
    TORCH_CHECK(false, "Unsupported data type for grouped implicit GEMM");
  }
  if (status != 0) {
    TORCH_CHECK(false,
                "Grouped implicit GEMM kernel failed with status: " + std::to_string(status));
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

// ------------------- CuTe 3.x dispatch helpers -------------------

// Dispatch helper macro — expands one case of the tile switch for AD gather-scatter.
#define CUTE_AD_GS_CASE(TILE_ENUM, TILE_TAG)                                                       \
  case warpconvnet::gemm::MMATile::TILE_ENUM:                                                      \
    return ::warpconvnet::cute_gemm::                                                              \
        run_cute_gemm_ad_gather_scatter<ElementInput, warpconvnet::gemm::TILE_TAG, ElementOutput>( \
            tensor_a.data_ptr(),                                                                   \
            tensor_b.data_ptr(),                                                                   \
            tensor_c.data_ptr(),                                                                   \
            tensor_d.data_ptr(),                                                                   \
            indices_a.data_ptr<int>(),                                                             \
            indices_d.data_ptr<int>(),                                                             \
            M_A,                                                                                   \
            K,                                                                                     \
            N,                                                                                     \
            M_C,                                                                                   \
            gather_ad_size,                                                                        \
            alpha,                                                                                 \
            beta);

template <torch::ScalarType SA, torch::ScalarType SC>
static int dispatch_cute_gemm_ad_gather_scatter(const torch::Tensor &tensor_a,
                                                const torch::Tensor &tensor_b,
                                                const torch::Tensor &tensor_c,
                                                torch::Tensor &tensor_d,
                                                const torch::Tensor &indices_a,
                                                const torch::Tensor &indices_d,
                                                int mma_tile,
                                                int M_A,
                                                int K,
                                                int N,
                                                int M_C,
                                                int gather_ad_size,
                                                float alpha,
                                                float beta) {
  using ElementInput = typename torch_to_cutlass<SA>::type;
  using ElementOutput = typename torch_to_cutlass<SC>::type;
  auto tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    CUTE_AD_GS_CASE(Tile128x128x32, Tile128x128x32)
    CUTE_AD_GS_CASE(Tile128x64x32, Tile128x64x32)
    CUTE_AD_GS_CASE(Tile64x128x32, Tile64x128x32)
    CUTE_AD_GS_CASE(Tile64x64x32, Tile64x64x32)
    CUTE_AD_GS_CASE(Tile64x64x64, Tile64x64x64)
    CUTE_AD_GS_CASE(Tile128x64x64, Tile128x64x64)
    CUTE_AD_GS_CASE(Tile64x128x64, Tile64x128x64)
    CUTE_AD_GS_CASE(Tile128x128x64, Tile128x128x64)
    CUTE_AD_GS_CASE(Tile256x64x32, Tile256x64x32)
    CUTE_AD_GS_CASE(Tile64x256x32, Tile64x256x32)
    default:
      TORCH_CHECK(false, "Unsupported mma_tile value");
  }
}
#undef CUTE_AD_GS_CASE

// ------------------- CuTe GEMM Python-facing functions -------------------

// Macro to dispatch AD gather-scatter on (input dtype, output dtype) pair.
// Supported output dtypes: float32, float16, bfloat16 (matching input type).
#define DISPATCH_CUTE_AD_GS(INPUT_SCALAR, OUTPUT_SCALAR)                                       \
  status =                                                                                     \
      dispatch_cute_gemm_ad_gather_scatter<INPUT_SCALAR, OUTPUT_SCALAR>(tensor_a,              \
                                                                        tensor_b,              \
                                                                        tensor_c,              \
                                                                        tensor_d,              \
                                                                        indices_a,             \
                                                                        indices_d,             \
                                                                        mma_tile,              \
                                                                        params.M_A,            \
                                                                        params.K,              \
                                                                        params.N,              \
                                                                        params.M_C,            \
                                                                        params.gather_ad_size, \
                                                                        alpha,                 \
                                                                        beta);

int cute_gemm_AD_gather_scatter(torch::Tensor tensor_a,
                                torch::Tensor tensor_b,
                                torch::Tensor tensor_c,
                                torch::Tensor tensor_d,
                                torch::Tensor indices_a,
                                torch::Tensor indices_d,
                                int mma_tile,
                                float alpha,
                                float beta) {
  const auto params =
      validate_ad_gather_scatter_args(tensor_a, tensor_b, tensor_c, tensor_d, indices_a, indices_d);
  TORCH_CHECK(tensor_c.scalar_type() == tensor_d.scalar_type(),
              "CuTe GEMM: C and D must have the same dtype");

  // CuTe kernel requires 128-bit aligned K and N (8 elements for fp16/bf16)
  int elem_size = (tensor_a.scalar_type() == torch::kFloat32) ? 2 : tensor_a.element_size();
  int vec_width = 16 / elem_size;  // 8 for fp16/bf16
  if (params.K % vec_width != 0 || params.N % vec_width != 0) {
    return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  // Downcast float32 inputs to float16 for compute
  if (scalar_a == torch::kFloat32) {
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    scalar_a = torch::kFloat16;
  }

  int status = 0;
  bool handled = false;

  // Dispatch on (input dtype, output dtype) — 6 combinations
  if (scalar_a == torch::kFloat16 && scalar_c == torch::kFloat32) {
    handled = true;
    DISPATCH_CUTE_AD_GS(torch::kFloat16, torch::kFloat32);
  } else if (scalar_a == torch::kFloat16 && scalar_c == torch::kFloat16) {
    handled = true;
    DISPATCH_CUTE_AD_GS(torch::kFloat16, torch::kFloat16);
  }
#ifndef DISABLE_BFLOAT16
  else if (scalar_a == torch::kBFloat16 && scalar_c == torch::kFloat32) {
    handled = true;
    DISPATCH_CUTE_AD_GS(torch::kBFloat16, torch::kFloat32);
  } else if (scalar_a == torch::kBFloat16 && scalar_c == torch::kBFloat16) {
    handled = true;
    DISPATCH_CUTE_AD_GS(torch::kBFloat16, torch::kBFloat16);
  }
#endif
  // float16 input, bfloat16 output (or vice versa) is not supported
  TORCH_CHECK(handled,
              "CuTe GEMM: unsupported input/output dtype combination (",
              tensor_a.scalar_type(),
              " input, ",
              tensor_c.scalar_type(),
              " output)");
  return status;
}
#undef DISPATCH_CUTE_AD_GS

// Dispatch helper macro — expands one case of the tile switch for TrAB gather.
#define CUTE_TRAB_CASE(TILE_ENUM, TILE_TAG)                                                  \
  case warpconvnet::gemm::MMATile::TILE_ENUM:                                                \
    return ::warpconvnet::cute_gemm::                                                        \
        run_cute_gemm_trAB_gather<ElementInput, warpconvnet::gemm::TILE_TAG, ElementOutput>( \
            tensor_a.data_ptr(),                                                             \
            tensor_b.data_ptr(),                                                             \
            tensor_c.data_ptr(),                                                             \
            tensor_d.data_ptr(),                                                             \
            indices_a.data_ptr<int>(),                                                       \
            indices_b.data_ptr<int>(),                                                       \
            M_A,                                                                             \
            K,                                                                               \
            K_B,                                                                             \
            N,                                                                               \
            gather_ab_size,                                                                  \
            alpha,                                                                           \
            beta);

template <torch::ScalarType SA, torch::ScalarType SC>
static int dispatch_cute_gemm_trAB_gather(const torch::Tensor &tensor_a,
                                          const torch::Tensor &tensor_b,
                                          const torch::Tensor &tensor_c,
                                          torch::Tensor &tensor_d,
                                          const torch::Tensor &indices_a,
                                          const torch::Tensor &indices_b,
                                          int mma_tile,
                                          int M_A,
                                          int K,
                                          int K_B,
                                          int N,
                                          int gather_ab_size,
                                          float alpha,
                                          float beta) {
  using ElementInput = typename torch_to_cutlass<SA>::type;
  using ElementOutput = typename torch_to_cutlass<SC>::type;
  auto tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    CUTE_TRAB_CASE(Tile128x128x32, Tile128x128x32)
    CUTE_TRAB_CASE(Tile128x64x32, Tile128x64x32)
    CUTE_TRAB_CASE(Tile64x128x32, Tile64x128x32)
    CUTE_TRAB_CASE(Tile64x64x32, Tile64x64x32)
    CUTE_TRAB_CASE(Tile64x64x64, Tile64x64x64)
    CUTE_TRAB_CASE(Tile128x64x64, Tile128x64x64)
    CUTE_TRAB_CASE(Tile64x128x64, Tile64x128x64)
    CUTE_TRAB_CASE(Tile128x128x64, Tile128x128x64)
    CUTE_TRAB_CASE(Tile256x64x32, Tile256x64x32)
    CUTE_TRAB_CASE(Tile64x256x32, Tile64x256x32)
    default:
      TORCH_CHECK(false, "Unsupported mma_tile value");
  }
}
#undef CUTE_TRAB_CASE

// Macro to dispatch TrAB gather on (input dtype, output dtype) pair.
#define DISPATCH_CUTE_TRAB(INPUT_SCALAR, OUTPUT_SCALAR)                                       \
  status = dispatch_cute_gemm_trAB_gather<INPUT_SCALAR, OUTPUT_SCALAR>(tensor_a,              \
                                                                       tensor_b,              \
                                                                       tensor_c,              \
                                                                       tensor_d,              \
                                                                       indices_a,             \
                                                                       indices_b,             \
                                                                       mma_tile,              \
                                                                       params.M_A,            \
                                                                       params.K,              \
                                                                       params.K_B,            \
                                                                       params.N,              \
                                                                       params.gather_ab_size, \
                                                                       alpha,                 \
                                                                       beta);

int cute_gemm_trAB_gather(torch::Tensor tensor_a,
                          torch::Tensor tensor_b,
                          torch::Tensor tensor_c,
                          torch::Tensor tensor_d,
                          torch::Tensor indices_a,
                          torch::Tensor indices_b,
                          int mma_tile,
                          float alpha,
                          float beta) {
  const auto params =
      validate_trAB_gather_args(tensor_a, tensor_b, tensor_c, tensor_d, indices_a, indices_b);
  TORCH_CHECK(tensor_c.scalar_type() == tensor_d.scalar_type(),
              "CuTe GEMM TrAB: C and D must have the same dtype");

  // CuTe TrAB kernel requires 128-bit aligned K and N (8 elements for fp16/bf16)
  int elem_size_trAB = (tensor_a.scalar_type() == torch::kFloat32) ? 2 : tensor_a.element_size();
  int vec_width_trAB = 16 / elem_size_trAB;
  if (params.K % vec_width_trAB != 0 || params.N % vec_width_trAB != 0) {
    return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  // Downcast float32 inputs to float16 for compute
  if (scalar_a == torch::kFloat32) {
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    scalar_a = torch::kFloat16;
  }

  int status = 0;
  bool handled = false;

  if (scalar_a == torch::kFloat16 && scalar_c == torch::kFloat32) {
    handled = true;
    DISPATCH_CUTE_TRAB(torch::kFloat16, torch::kFloat32);
  } else if (scalar_a == torch::kFloat16 && scalar_c == torch::kFloat16) {
    handled = true;
    DISPATCH_CUTE_TRAB(torch::kFloat16, torch::kFloat16);
  }
#ifndef DISABLE_BFLOAT16
  else if (scalar_a == torch::kBFloat16 && scalar_c == torch::kFloat32) {
    handled = true;
    DISPATCH_CUTE_TRAB(torch::kBFloat16, torch::kFloat32);
  } else if (scalar_a == torch::kBFloat16 && scalar_c == torch::kBFloat16) {
    handled = true;
    DISPATCH_CUTE_TRAB(torch::kBFloat16, torch::kBFloat16);
  }
#endif
  TORCH_CHECK(handled,
              "CuTe GEMM TrAB: unsupported input/output dtype combination (",
              tensor_a.scalar_type(),
              " input, ",
              tensor_c.scalar_type(),
              " output)");
  return status;
}
#undef DISPATCH_CUTE_TRAB

// ------------------- CuTe Staged GEMM dispatch (SM90+ autotunable) -------------------

// Only tiles 0-3 (tK=32) have staged instantiations.
static constexpr int kMaxStagedTile = 3;

#define CUTE_STAGED_AD_GS_CASE(TILE_ENUM, TILE_TAG)                          \
  case warpconvnet::gemm::MMATile::TILE_ENUM:                                \
    return ::warpconvnet::cute_gemm::run_cute_gemm_ad_gather_scatter_staged< \
        ElementInput,                                                        \
        warpconvnet::gemm::TILE_TAG,                                         \
        ElementOutput>(tensor_a.data_ptr(),                                  \
                       tensor_b.data_ptr(),                                  \
                       tensor_c.data_ptr(),                                  \
                       tensor_d.data_ptr(),                                  \
                       indices_a.data_ptr<int>(),                            \
                       indices_d.data_ptr<int>(),                            \
                       M_A,                                                  \
                       K,                                                    \
                       N,                                                    \
                       M_C,                                                  \
                       gather_ad_size,                                       \
                       alpha,                                                \
                       beta,                                                 \
                       num_stages,                                           \
                       use_cp_async);

template <torch::ScalarType SA, torch::ScalarType SC>
static int dispatch_cute_gemm_ad_gather_scatter_staged(const torch::Tensor &tensor_a,
                                                       const torch::Tensor &tensor_b,
                                                       const torch::Tensor &tensor_c,
                                                       torch::Tensor &tensor_d,
                                                       const torch::Tensor &indices_a,
                                                       const torch::Tensor &indices_d,
                                                       int mma_tile,
                                                       int M_A,
                                                       int K,
                                                       int N,
                                                       int M_C,
                                                       int gather_ad_size,
                                                       float alpha,
                                                       float beta,
                                                       int num_stages,
                                                       bool use_cp_async) {
  using ElementInput = typename torch_to_cutlass<SA>::type;
  using ElementOutput = typename torch_to_cutlass<SC>::type;
  auto tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    CUTE_STAGED_AD_GS_CASE(Tile128x128x32, Tile128x128x32)
    CUTE_STAGED_AD_GS_CASE(Tile128x64x32, Tile128x64x32)
    CUTE_STAGED_AD_GS_CASE(Tile64x128x32, Tile64x128x32)
    CUTE_STAGED_AD_GS_CASE(Tile64x64x32, Tile64x64x32)
    default:
      return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }
}
#undef CUTE_STAGED_AD_GS_CASE

#define DISPATCH_CUTE_STAGED_AD_GS(INPUT_SCALAR, OUTPUT_SCALAR)                      \
  status = dispatch_cute_gemm_ad_gather_scatter_staged<INPUT_SCALAR, OUTPUT_SCALAR>( \
      tensor_a,                                                                      \
      tensor_b,                                                                      \
      tensor_c,                                                                      \
      tensor_d,                                                                      \
      indices_a,                                                                     \
      indices_d,                                                                     \
      mma_tile,                                                                      \
      params.M_A,                                                                    \
      params.K,                                                                      \
      params.N,                                                                      \
      params.M_C,                                                                    \
      params.gather_ad_size,                                                         \
      alpha,                                                                         \
      beta,                                                                          \
      num_stages,                                                                    \
      use_cp_async);

int cute_gemm_AD_gather_scatter_staged(torch::Tensor tensor_a,
                                       torch::Tensor tensor_b,
                                       torch::Tensor tensor_c,
                                       torch::Tensor tensor_d,
                                       torch::Tensor indices_a,
                                       torch::Tensor indices_d,
                                       int mma_tile,
                                       float alpha,
                                       float beta,
                                       int num_stages,
                                       bool use_cp_async) {
  const auto params =
      validate_ad_gather_scatter_args(tensor_a, tensor_b, tensor_c, tensor_d, indices_a, indices_d);
  TORCH_CHECK(tensor_c.scalar_type() == tensor_d.scalar_type(),
              "CuTe GEMM staged: C and D must have the same dtype");
  TORCH_CHECK(mma_tile <= kMaxStagedTile, "Staged GEMM only supports mma_tile 0-3 (tK=32 tiles)");
  TORCH_CHECK(num_stages >= 2 && num_stages <= 4, "num_stages must be 2, 3, or 4");

  int elem_size = (tensor_a.scalar_type() == torch::kFloat32) ? 2 : tensor_a.element_size();
  int vec_width = 16 / elem_size;
  if (params.K % vec_width != 0 || params.N % vec_width != 0) {
    return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  if (scalar_a == torch::kFloat32) {
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    scalar_a = torch::kFloat16;
  }

  int status = 0;
  bool handled = false;

  if (scalar_a == torch::kFloat16 && scalar_c == torch::kFloat32) {
    handled = true;
    DISPATCH_CUTE_STAGED_AD_GS(torch::kFloat16, torch::kFloat32);
  } else if (scalar_a == torch::kFloat16 && scalar_c == torch::kFloat16) {
    handled = true;
    DISPATCH_CUTE_STAGED_AD_GS(torch::kFloat16, torch::kFloat16);
  }
#ifndef DISABLE_BFLOAT16
  else if (scalar_a == torch::kBFloat16 && scalar_c == torch::kFloat32) {
    handled = true;
    DISPATCH_CUTE_STAGED_AD_GS(torch::kBFloat16, torch::kFloat32);
  } else if (scalar_a == torch::kBFloat16 && scalar_c == torch::kBFloat16) {
    handled = true;
    DISPATCH_CUTE_STAGED_AD_GS(torch::kBFloat16, torch::kBFloat16);
  }
#endif
  TORCH_CHECK(handled, "CuTe GEMM staged: unsupported dtype combination");
  return status;
}
#undef DISPATCH_CUTE_STAGED_AD_GS

// --- Staged TrAB ---

#define CUTE_STAGED_TRAB_CASE(TILE_ENUM, TILE_TAG)                                                 \
  case warpconvnet::gemm::MMATile::TILE_ENUM:                                                      \
    return ::warpconvnet::cute_gemm::run_cute_gemm_trAB_gather_staged<ElementInput,                \
                                                                      warpconvnet::gemm::TILE_TAG, \
                                                                      ElementOutput>(              \
        tensor_a.data_ptr(),                                                                       \
        tensor_b.data_ptr(),                                                                       \
        tensor_c.data_ptr(),                                                                       \
        tensor_d.data_ptr(),                                                                       \
        indices_a.data_ptr<int>(),                                                                 \
        indices_b.data_ptr<int>(),                                                                 \
        M_A,                                                                                       \
        K,                                                                                         \
        K_B,                                                                                       \
        N,                                                                                         \
        gather_ab_size,                                                                            \
        alpha,                                                                                     \
        beta,                                                                                      \
        num_stages,                                                                                \
        use_cp_async);

template <torch::ScalarType SA, torch::ScalarType SC>
static int dispatch_cute_gemm_trAB_gather_staged(const torch::Tensor &tensor_a,
                                                 const torch::Tensor &tensor_b,
                                                 const torch::Tensor &tensor_c,
                                                 torch::Tensor &tensor_d,
                                                 const torch::Tensor &indices_a,
                                                 const torch::Tensor &indices_b,
                                                 int mma_tile,
                                                 int M_A,
                                                 int K,
                                                 int K_B,
                                                 int N,
                                                 int gather_ab_size,
                                                 float alpha,
                                                 float beta,
                                                 int num_stages,
                                                 bool use_cp_async) {
  using ElementInput = typename torch_to_cutlass<SA>::type;
  using ElementOutput = typename torch_to_cutlass<SC>::type;
  auto tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    CUTE_STAGED_TRAB_CASE(Tile128x128x32, Tile128x128x32)
    CUTE_STAGED_TRAB_CASE(Tile128x64x32, Tile128x64x32)
    CUTE_STAGED_TRAB_CASE(Tile64x128x32, Tile64x128x32)
    CUTE_STAGED_TRAB_CASE(Tile64x64x32, Tile64x64x32)
    default:
      return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }
}
#undef CUTE_STAGED_TRAB_CASE

#define DISPATCH_CUTE_STAGED_TRAB(INPUT_SCALAR, OUTPUT_SCALAR)                                  \
  status =                                                                                      \
      dispatch_cute_gemm_trAB_gather_staged<INPUT_SCALAR, OUTPUT_SCALAR>(tensor_a,              \
                                                                         tensor_b,              \
                                                                         tensor_c,              \
                                                                         tensor_d,              \
                                                                         indices_a,             \
                                                                         indices_b,             \
                                                                         mma_tile,              \
                                                                         params.M_A,            \
                                                                         params.K,              \
                                                                         params.K_B,            \
                                                                         params.N,              \
                                                                         params.gather_ab_size, \
                                                                         alpha,                 \
                                                                         beta,                  \
                                                                         num_stages,            \
                                                                         use_cp_async);

int cute_gemm_trAB_gather_staged(torch::Tensor tensor_a,
                                 torch::Tensor tensor_b,
                                 torch::Tensor tensor_c,
                                 torch::Tensor tensor_d,
                                 torch::Tensor indices_a,
                                 torch::Tensor indices_b,
                                 int mma_tile,
                                 float alpha,
                                 float beta,
                                 int num_stages,
                                 bool use_cp_async) {
  const auto params =
      validate_trAB_gather_args(tensor_a, tensor_b, tensor_c, tensor_d, indices_a, indices_b);
  TORCH_CHECK(tensor_c.scalar_type() == tensor_d.scalar_type(),
              "CuTe GEMM TrAB staged: C and D must have the same dtype");
  TORCH_CHECK(mma_tile <= kMaxStagedTile,
              "Staged TrAB GEMM only supports mma_tile 0-3 (tK=32 tiles)");
  TORCH_CHECK(num_stages >= 2 && num_stages <= 4, "num_stages must be 2, 3, or 4");

  int elem_size = (tensor_a.scalar_type() == torch::kFloat32) ? 2 : tensor_a.element_size();
  int vec_width = 16 / elem_size;
  if (params.K % vec_width != 0 || params.N % vec_width != 0) {
    return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_c = tensor_c.scalar_type();
  if (scalar_a == torch::kFloat32) {
    tensor_a = tensor_a.to(torch::kFloat16);
    tensor_b = tensor_b.to(torch::kFloat16);
    scalar_a = torch::kFloat16;
  }

  int status = 0;
  bool handled = false;

  if (scalar_a == torch::kFloat16 && scalar_c == torch::kFloat32) {
    handled = true;
    DISPATCH_CUTE_STAGED_TRAB(torch::kFloat16, torch::kFloat32);
  } else if (scalar_a == torch::kFloat16 && scalar_c == torch::kFloat16) {
    handled = true;
    DISPATCH_CUTE_STAGED_TRAB(torch::kFloat16, torch::kFloat16);
  }
#ifndef DISABLE_BFLOAT16
  else if (scalar_a == torch::kBFloat16 && scalar_c == torch::kFloat32) {
    handled = true;
    DISPATCH_CUTE_STAGED_TRAB(torch::kBFloat16, torch::kFloat32);
  } else if (scalar_a == torch::kBFloat16 && scalar_c == torch::kBFloat16) {
    handled = true;
    DISPATCH_CUTE_STAGED_TRAB(torch::kBFloat16, torch::kBFloat16);
  }
#endif
  TORCH_CHECK(handled, "CuTe GEMM TrAB staged: unsupported dtype combination");
  return status;
}
#undef DISPATCH_CUTE_STAGED_TRAB

// --- Staged Grouped AD ---

#define CUTE_STAGED_GROUPED_CASE(TILE_ENUM, TILE_TAG)                                \
  case warpconvnet::gemm::MMATile::TILE_ENUM:                                        \
    return ::warpconvnet::cute_gemm::run_cute_gemm_grouped_ad_gather_scatter_staged< \
        ElementInput,                                                                \
        warpconvnet::gemm::TILE_TAG,                                                 \
        ElementOutput>(ptr_A,                                                        \
                       ptr_D,                                                        \
                       in_map,                                                       \
                       out_map,                                                      \
                       params,                                                       \
                       total_m_tiles,                                                \
                       K,                                                            \
                       N,                                                            \
                       alpha,                                                        \
                       num_stages,                                                   \
                       use_cp_async);

template <torch::ScalarType SA, torch::ScalarType SC>
static int dispatch_cute_gemm_grouped_ad_gather_scatter_staged(
    const void *ptr_A,
    void *ptr_D,
    const int *in_map,
    const int *out_map,
    const warpconvnet::cute_gemm::GroupedGemmParams &params,
    int total_m_tiles,
    int K,
    int N,
    float alpha,
    int mma_tile,
    int num_stages,
    bool use_cp_async) {
  using ElementInput = typename torch_to_cutlass<SA>::type;
  using ElementOutput = typename torch_to_cutlass<SC>::type;
  auto tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    CUTE_STAGED_GROUPED_CASE(Tile128x128x32, Tile128x128x32)
    CUTE_STAGED_GROUPED_CASE(Tile128x64x32, Tile128x64x32)
    CUTE_STAGED_GROUPED_CASE(Tile64x128x32, Tile64x128x32)
    CUTE_STAGED_GROUPED_CASE(Tile64x64x32, Tile64x64x32)
    default:
      return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }
}
#undef CUTE_STAGED_GROUPED_CASE

int cute_gemm_grouped_AD_gather_scatter_staged(torch::Tensor tensor_a,
                                               torch::Tensor tensor_d,
                                               torch::Tensor in_map,
                                               torch::Tensor out_map,
                                               torch::Tensor weight_ptrs,
                                               torch::Tensor tile_offsets,
                                               torch::Tensor group_sizes,
                                               torch::Tensor map_offsets,
                                               int total_m_tiles,
                                               int mma_tile,
                                               float alpha,
                                               int num_stages,
                                               bool use_cp_async) {
  TORCH_CHECK(tensor_a.is_cuda() && tensor_d.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(in_map.is_cuda() && out_map.is_cuda(), "Maps must be on CUDA");
  TORCH_CHECK(weight_ptrs.is_cuda() && tile_offsets.is_cuda() && group_sizes.is_cuda() &&
                  map_offsets.is_cuda(),
              "Group params must be on CUDA");
  TORCH_CHECK(in_map.scalar_type() == torch::kInt32, "in_map must be int32");
  TORCH_CHECK(out_map.scalar_type() == torch::kInt32, "out_map must be int32");
  TORCH_CHECK(weight_ptrs.scalar_type() == torch::kInt64, "weight_ptrs must be int64");
  TORCH_CHECK(tile_offsets.scalar_type() == torch::kInt32, "tile_offsets must be int32");
  TORCH_CHECK(group_sizes.scalar_type() == torch::kInt32, "group_sizes must be int32");
  TORCH_CHECK(map_offsets.scalar_type() == torch::kInt32, "map_offsets must be int32");
  TORCH_CHECK(mma_tile <= kMaxStagedTile,
              "Staged grouped GEMM only supports mma_tile 0-3 (tK=32 tiles)");
  TORCH_CHECK(num_stages >= 2 && num_stages <= 4, "num_stages must be 2, 3, or 4");

  tensor_a = tensor_a.contiguous();
  tensor_d = tensor_d.contiguous();
  in_map = in_map.contiguous();
  out_map = out_map.contiguous();

  int K = tensor_a.size(1);
  int N = tensor_d.size(1);
  int num_groups = group_sizes.size(0);

  int elem_size = (tensor_a.scalar_type() == torch::kFloat32) ? 2 : tensor_a.element_size();
  int vec_width = 16 / elem_size;
  if (K % vec_width != 0 || N % vec_width != 0) {
    return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }

  warpconvnet::cute_gemm::GroupedGemmParams gparams;
  gparams.num_groups = num_groups;
  gparams.tile_offsets = tile_offsets.data_ptr<int>();
  gparams.group_sizes = group_sizes.data_ptr<int>();
  gparams.map_offsets = map_offsets.data_ptr<int>();
  gparams.ptr_B_array = reinterpret_cast<const void *const *>(weight_ptrs.data_ptr<int64_t>());

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_d = tensor_d.scalar_type();

  if (scalar_a == torch::kFloat32) {
    tensor_a = tensor_a.to(torch::kFloat16);
    scalar_a = torch::kFloat16;
  }

  int status = 0;
  bool handled = false;

  if (scalar_a == torch::kFloat16 && scalar_d == torch::kFloat32) {
    handled = true;
    status = dispatch_cute_gemm_grouped_ad_gather_scatter_staged<torch::kFloat16, torch::kFloat32>(
        tensor_a.data_ptr(),
        tensor_d.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        gparams,
        total_m_tiles,
        K,
        N,
        alpha,
        mma_tile,
        num_stages,
        use_cp_async);
  } else if (scalar_a == torch::kFloat16 && scalar_d == torch::kFloat16) {
    handled = true;
    status = dispatch_cute_gemm_grouped_ad_gather_scatter_staged<torch::kFloat16, torch::kFloat16>(
        tensor_a.data_ptr(),
        tensor_d.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        gparams,
        total_m_tiles,
        K,
        N,
        alpha,
        mma_tile,
        num_stages,
        use_cp_async);
  }
#ifndef DISABLE_BFLOAT16
  else if (scalar_a == torch::kBFloat16 && scalar_d == torch::kFloat32) {
    handled = true;
    status = dispatch_cute_gemm_grouped_ad_gather_scatter_staged<torch::kBFloat16, torch::kFloat32>(
        tensor_a.data_ptr(),
        tensor_d.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        gparams,
        total_m_tiles,
        K,
        N,
        alpha,
        mma_tile,
        num_stages,
        use_cp_async);
  } else if (scalar_a == torch::kBFloat16 && scalar_d == torch::kBFloat16) {
    handled = true;
    status =
        dispatch_cute_gemm_grouped_ad_gather_scatter_staged<torch::kBFloat16, torch::kBFloat16>(
            tensor_a.data_ptr(),
            tensor_d.data_ptr(),
            in_map.data_ptr<int>(),
            out_map.data_ptr<int>(),
            gparams,
            total_m_tiles,
            K,
            N,
            alpha,
            mma_tile,
            num_stages,
            use_cp_async);
  }
#endif

  TORCH_CHECK(handled, "CuTe grouped GEMM staged: unsupported dtype combination");
  return status;
}

// ------------------- CuTe Grouped GEMM dispatch -------------------

#define CUTE_GROUPED_CASE(TILE_ENUM, TILE_TAG)                                \
  case warpconvnet::gemm::MMATile::TILE_ENUM:                                 \
    return ::warpconvnet::cute_gemm::run_cute_gemm_grouped_ad_gather_scatter< \
        ElementInput,                                                         \
        warpconvnet::gemm::TILE_TAG,                                          \
        ElementOutput>(ptr_A, ptr_D, in_map, out_map, params, total_m_tiles, K, N, alpha);

template <torch::ScalarType SA, torch::ScalarType SC>
static int dispatch_cute_gemm_grouped_ad_gather_scatter(
    const void *ptr_A,
    void *ptr_D,
    const int *in_map,
    const int *out_map,
    const warpconvnet::cute_gemm::GroupedGemmParams &params,
    int total_m_tiles,
    int K,
    int N,
    float alpha,
    int mma_tile) {
  using ElementInput = typename torch_to_cutlass<SA>::type;
  using ElementOutput = typename torch_to_cutlass<SC>::type;
  auto tile = static_cast<warpconvnet::gemm::MMATile>(mma_tile);
  switch (tile) {
    CUTE_GROUPED_CASE(Tile128x128x32, Tile128x128x32)
    CUTE_GROUPED_CASE(Tile128x64x32, Tile128x64x32)
    CUTE_GROUPED_CASE(Tile64x128x32, Tile64x128x32)
    CUTE_GROUPED_CASE(Tile64x64x32, Tile64x64x32)
    CUTE_GROUPED_CASE(Tile64x64x64, Tile64x64x64)
    CUTE_GROUPED_CASE(Tile128x64x64, Tile128x64x64)
    CUTE_GROUPED_CASE(Tile64x128x64, Tile64x128x64)
    CUTE_GROUPED_CASE(Tile128x128x64, Tile128x128x64)
    CUTE_GROUPED_CASE(Tile256x64x32, Tile256x64x32)
    CUTE_GROUPED_CASE(Tile64x256x32, Tile64x256x32)
    default:
      TORCH_CHECK(false, "Unsupported mma_tile value for grouped GEMM");
  }
}
#undef CUTE_GROUPED_CASE

int cute_gemm_grouped_AD_gather_scatter(
    torch::Tensor tensor_a,      // [N_in, K] input features
    torch::Tensor tensor_d,      // [N_out, N] output (pre-zeroed)
    torch::Tensor in_map,        // [L] int32 concatenated gather indices
    torch::Tensor out_map,       // [L] int32 concatenated scatter indices
    torch::Tensor weight_ptrs,   // [num_groups] int64 device pointers to weight matrices
    torch::Tensor tile_offsets,  // [num_groups+1] int32 prefix sum of m_tiles
    torch::Tensor group_sizes,   // [num_groups] int32 M_g per group
    torch::Tensor map_offsets,   // [num_groups+1] int32 offsets into in_map/out_map
    int total_m_tiles,
    int mma_tile,
    float alpha) {
  TORCH_CHECK(tensor_a.is_cuda() && tensor_d.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(in_map.is_cuda() && out_map.is_cuda(), "Maps must be on CUDA");
  TORCH_CHECK(weight_ptrs.is_cuda() && tile_offsets.is_cuda() && group_sizes.is_cuda() &&
                  map_offsets.is_cuda(),
              "Group params must be on CUDA");
  TORCH_CHECK(in_map.scalar_type() == torch::kInt32, "in_map must be int32");
  TORCH_CHECK(out_map.scalar_type() == torch::kInt32, "out_map must be int32");
  TORCH_CHECK(weight_ptrs.scalar_type() == torch::kInt64, "weight_ptrs must be int64");
  TORCH_CHECK(tile_offsets.scalar_type() == torch::kInt32, "tile_offsets must be int32");
  TORCH_CHECK(group_sizes.scalar_type() == torch::kInt32, "group_sizes must be int32");
  TORCH_CHECK(map_offsets.scalar_type() == torch::kInt32, "map_offsets must be int32");

  tensor_a = tensor_a.contiguous();
  tensor_d = tensor_d.contiguous();
  in_map = in_map.contiguous();
  out_map = out_map.contiguous();

  int K = tensor_a.size(1);
  int N = tensor_d.size(1);
  int num_groups = group_sizes.size(0);

  // Alignment check
  int elem_size = (tensor_a.scalar_type() == torch::kFloat32) ? 2 : tensor_a.element_size();
  int vec_width = 16 / elem_size;
  if (K % vec_width != 0 || N % vec_width != 0) {
    return static_cast<int>(warpconvnet::gemm::GemmStatus::kErrorUnsupportedConfig);
  }

  // Build GroupedGemmParams
  warpconvnet::cute_gemm::GroupedGemmParams params;
  params.num_groups = num_groups;
  params.tile_offsets = tile_offsets.data_ptr<int>();
  params.group_sizes = group_sizes.data_ptr<int>();
  params.map_offsets = map_offsets.data_ptr<int>();
  params.ptr_B_array = reinterpret_cast<const void *const *>(weight_ptrs.data_ptr<int64_t>());

  auto scalar_a = tensor_a.scalar_type();
  auto scalar_d = tensor_d.scalar_type();

  // Downcast float32 inputs to float16 for compute
  if (scalar_a == torch::kFloat32) {
    tensor_a = tensor_a.to(torch::kFloat16);
    scalar_a = torch::kFloat16;
  }

  int status = 0;
  bool handled = false;

  if (scalar_a == torch::kFloat16 && scalar_d == torch::kFloat32) {
    handled = true;
    status = dispatch_cute_gemm_grouped_ad_gather_scatter<torch::kFloat16, torch::kFloat32>(
        tensor_a.data_ptr(),
        tensor_d.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        params,
        total_m_tiles,
        K,
        N,
        alpha,
        mma_tile);
  } else if (scalar_a == torch::kFloat16 && scalar_d == torch::kFloat16) {
    handled = true;
    status = dispatch_cute_gemm_grouped_ad_gather_scatter<torch::kFloat16, torch::kFloat16>(
        tensor_a.data_ptr(),
        tensor_d.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        params,
        total_m_tiles,
        K,
        N,
        alpha,
        mma_tile);
  }
#ifndef DISABLE_BFLOAT16
  else if (scalar_a == torch::kBFloat16 && scalar_d == torch::kFloat32) {
    handled = true;
    status = dispatch_cute_gemm_grouped_ad_gather_scatter<torch::kBFloat16, torch::kFloat32>(
        tensor_a.data_ptr(),
        tensor_d.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        params,
        total_m_tiles,
        K,
        N,
        alpha,
        mma_tile);
  } else if (scalar_a == torch::kBFloat16 && scalar_d == torch::kBFloat16) {
    handled = true;
    status = dispatch_cute_gemm_grouped_ad_gather_scatter<torch::kBFloat16, torch::kBFloat16>(
        tensor_a.data_ptr(),
        tensor_d.data_ptr(),
        in_map.data_ptr<int>(),
        out_map.data_ptr<int>(),
        params,
        total_m_tiles,
        K,
        N,
        alpha,
        mma_tile);
  }
#endif

  TORCH_CHECK(handled,
              "CuTe grouped GEMM: unsupported dtype combination (",
              tensor_a.scalar_type(),
              " input, ",
              tensor_d.scalar_type(),
              " output)");
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

  gemm.def("cute_gemm_AD_gather_scatter",
           &cute_gemm_AD_gather_scatter,
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_d"),
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 1.0f);

  gemm.def("cute_gemm_trAB_gather",
           &cute_gemm_trAB_gather,
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_b"),
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 1.0f);

  gemm.def("cute_gemm_grouped_AD_gather_scatter",
           &cute_gemm_grouped_AD_gather_scatter,
           py::arg("tensor_a"),
           py::arg("tensor_d"),
           py::arg("in_map"),
           py::arg("out_map"),
           py::arg("weight_ptrs"),
           py::arg("tile_offsets"),
           py::arg("group_sizes"),
           py::arg("map_offsets"),
           py::arg("total_m_tiles"),
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f);

  gemm.def("cute_gemm_AD_gather_scatter_staged",
           &cute_gemm_AD_gather_scatter_staged,
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_d"),
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 1.0f,
           py::arg("num_stages") = 2,
           py::arg("use_cp_async") = false);

  gemm.def("cute_gemm_trAB_gather_staged",
           &cute_gemm_trAB_gather_staged,
           py::arg("tensor_a"),
           py::arg("tensor_b"),
           py::arg("tensor_c"),
           py::arg("tensor_d"),
           py::arg("indices_a"),
           py::arg("indices_b"),
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("beta") = 1.0f,
           py::arg("num_stages") = 2,
           py::arg("use_cp_async") = false);

  gemm.def("cute_gemm_grouped_AD_gather_scatter_staged",
           &cute_gemm_grouped_AD_gather_scatter_staged,
           py::arg("tensor_a"),
           py::arg("tensor_d"),
           py::arg("in_map"),
           py::arg("out_map"),
           py::arg("weight_ptrs"),
           py::arg("tile_offsets"),
           py::arg("group_sizes"),
           py::arg("map_offsets"),
           py::arg("total_m_tiles"),
           py::arg("mma_tile") = 0,
           py::arg("alpha") = 1.0f,
           py::arg("num_stages") = 2,
           py::arg("use_cp_async") = false);

  gemm.def("implicit_gemm",
           &implicit_gemm_cuda,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("in_map"),
           py::arg("out_map"),
           py::arg("kernel_type") = "basic",
           py::arg("block_size") = 16);

  gemm.def("implicit_gemm_grouped",
           &implicit_gemm_grouped_cuda,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("in_map"),
           py::arg("out_map"),
           py::arg("weight_idx"),
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
}

}  // namespace bindings
}  // namespace warpconvnet
