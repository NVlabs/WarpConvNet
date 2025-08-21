// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cutlass/util/device_memory.h>

#include "include/gemm_error_codes.h"
#include "include/gemm_operations.h"

#ifdef DEBUG
#include <cxxabi.h>

#include <iomanip>
#include <iostream>
#include <typeinfo>

// Helper function to demangle type names
std::string demangle_type_name(const char *name) {
  int status = 0;
  char *demangled = abi::__cxa_demangle(name, 0, 0, &status);
  if (status == 0) {
    std::string result(demangled);
    free(demangled);
    return result;
  }
  return std::string(name);
}

// Helper function to print GemmShape dimensions
template <typename GemmShape>
std::string print_gemm_shape() {
  return std::to_string(GemmShape::kM) + "x" + std::to_string(GemmShape::kN) + "x" +
         std::to_string(GemmShape::kK);
}

// Debug function to print GEMM arguments and traits
template <typename Gemm, typename Traits, typename Config>
void debug_print_gemm_arguments(const typename Gemm::Arguments &arguments,
                                int problem_m,
                                int problem_n,
                                int problem_k,
                                int M_A,
                                int K,
                                int K_B,
                                int N,
                                int M_C,
                                int gather_a_size,
                                int scatter_d_size) {
  std::cout << "\n=== DEBUG: GEMM Arguments ===" << std::endl;
  std::cout << "Element types:" << std::endl;
  std::cout << "  ElementInputA: " << demangle_type_name(typeid(typename Gemm::ElementA).name())
            << std::endl;
  std::cout << "  ElementInputB: " << demangle_type_name(typeid(typename Gemm::ElementB).name())
            << std::endl;
  std::cout << "  ElementOutput: " << demangle_type_name(typeid(typename Gemm::ElementC).name())
            << std::endl;
  std::cout << "  ElementAccumulator: "
            << demangle_type_name(typeid(typename Gemm::ElementAccumulator).name()) << std::endl;

  std::cout << "\nTrait Shapes:" << std::endl;
  std::cout << "  ShapeMMAThreadBlock: " << print_gemm_shape<typename Traits::ShapeMMAThreadBlock>()
            << std::endl;
  std::cout << "  ShapeMMAWarp: " << print_gemm_shape<typename Traits::ShapeMMAWarp>() << std::endl;
  std::cout << "  ShapeMMAOp: " << print_gemm_shape<typename Traits::ShapeMMAOp>() << std::endl;

  std::cout << "\nTrait Properties:" << std::endl;
  std::cout << "  ConfigName: " << Traits::GetConfigName() << std::endl;
  std::cout << "  AlignmentA: " << int(Traits::AlignmentA) << std::endl;
  std::cout << "  AlignmentB: " << int(Traits::AlignmentB) << std::endl;
  std::cout << "  SupportsTensorOp: " << (Traits::SupportsTensorOp ? "true" : "false") << std::endl;
  std::cout << "  SupportsGatherA: " << (Traits::SupportsGatherA ? "true" : "false") << std::endl;
  std::cout << "  SupportsGatherB: " << (Traits::SupportsGatherB ? "true" : "false") << std::endl;
  std::cout << "  SupportsScatterD: " << (Traits::SupportsScatterD ? "true" : "false") << std::endl;
  std::cout << "  SupportsTransposeA: " << (Traits::SupportsTransposeA ? "true" : "false")
            << std::endl;
  std::cout << "  SupportsTransposeB: " << (Traits::SupportsTransposeB ? "true" : "false")
            << std::endl;
  std::cout << "  UseMixedInput: " << (Traits::UseMixedInput ? "true" : "false") << std::endl;
  std::cout << "  IsValidConfiguration: " << (Traits::IsValidConfiguration() ? "true" : "false")
            << std::endl;

  std::cout << "\nLayout types:" << std::endl;
  std::cout << "  LayoutInputA: "
            << demangle_type_name(typeid(typename Traits::LayoutInputA).name()) << std::endl;
  std::cout << "  LayoutInputB: "
            << demangle_type_name(typeid(typename Traits::LayoutInputB).name()) << std::endl;
  std::cout << "  LayoutOutput: "
            << demangle_type_name(typeid(typename Traits::LayoutOutput).name()) << std::endl;

  std::cout << "\nMMA Operation:" << std::endl;
  std::cout << "  MMAOp: " << demangle_type_name(typeid(typename Traits::MMAOp).name())
            << std::endl;
  std::cout << "  ArchitectureTag: "
            << demangle_type_name(typeid(typename Traits::ArchitectureTag).name()) << std::endl;

  std::cout << "\nConfiguration Analysis:" << std::endl;
  std::cout << "  Has operations: " << (Config::has_operations() ? "true" : "false") << std::endl;
  std::cout << "  Requires TensorOp: "
            << (Config::has_operations() && Traits::SupportsTensorOp ? "yes (supported)"
                : Config::has_operations()                           ? "yes (NOT supported)"
                                                                     : "no")
            << std::endl;
  std::cout << "  Config gather_a: " << (bool(Config::gather_a) ? "true" : "false") << std::endl;
  std::cout << "  Config gather_b: " << (bool(Config::gather_b) ? "true" : "false") << std::endl;
  std::cout << "  Config scatter_d: " << (bool(Config::scatter_d) ? "true" : "false") << std::endl;
  std::cout << "  Config transpose_a: " << (bool(Config::transpose_a) ? "true" : "false")
            << std::endl;
  std::cout << "  Config transpose_b: " << (bool(Config::transpose_b) ? "true" : "false")
            << std::endl;

  std::cout << "\nProblem size:" << std::endl;
  std::cout << "  problem_m: " << problem_m << ", problem_n: " << problem_n
            << ", problem_k: " << problem_k << std::endl;

  std::cout << "\nOriginal matrix dimensions:" << std::endl;
  std::cout << "  M_A (A rows): " << M_A << ", K (A cols): " << K << std::endl;
  std::cout << "  K_B (B rows): " << K_B << ", N (B cols): " << N << std::endl;
  std::cout << "  M_C (C rows): " << M_C << ", N (C cols): " << N << std::endl;

  std::cout << "\nGather/Scatter info:" << std::endl;
  std::cout << "  gather_a_size: " << gather_a_size << ", scatter_d_size: " << scatter_d_size
            << std::endl;

  std::cout << "\nMode and split:" << std::endl;
  std::cout << "  mode: " << static_cast<int>(arguments.mode) << std::endl;
  std::cout << "  split_k_slices: " << arguments.batch_count << std::endl;

  std::cout << "\nPointers (non-null check):" << std::endl;
  std::cout << "  ptr_A: " << (arguments.ptr_A ? "non-null" : "null") << std::endl;
  std::cout << "  ptr_B: " << (arguments.ptr_B ? "non-null" : "null") << std::endl;
  std::cout << "  ptr_C: " << (arguments.ptr_C ? "non-null" : "null") << std::endl;
  std::cout << "  ptr_D: " << (arguments.ptr_D ? "non-null" : "null") << std::endl;

  // Only print gather/scatter pointers if they're available for this configuration
  if constexpr (Config::gather_a || Config::gather_b || Config::scatter_d) {
    std::cout
        << "  NOTE: Gather/scatter pointer access skipped in debug to avoid compilation errors"
        << std::endl;
  }

  std::cout << "\nBatch strides:" << std::endl;
  std::cout << "  batch_stride_A: " << arguments.batch_stride_A << std::endl;
  std::cout << "  batch_stride_B: " << arguments.batch_stride_B << std::endl;
  std::cout << "  batch_stride_C: " << arguments.batch_stride_C << std::endl;
  std::cout << "  batch_stride_D: " << arguments.batch_stride_D << std::endl;

  std::cout << "\nLayout strides:" << std::endl;
  std::cout << "  lda: " << arguments.lda << std::endl;
  std::cout << "  ldb: " << arguments.ldb << std::endl;
  std::cout << "  ldc: " << arguments.ldc << std::endl;
  std::cout << "  ldd: " << arguments.ldd << std::endl;

  std::cout << "\nEpilogue:" << std::endl;
  std::cout << "  alpha: " << static_cast<float>(arguments.epilogue.alpha) << std::endl;
  std::cout << "  beta: " << static_cast<float>(arguments.epilogue.beta) << std::endl;

  std::cout << "================================================" << std::endl;
}
#endif

// Main templated function implementation - define in the warpconvnet::gemm namespace
namespace warpconvnet {
namespace gemm {

/*
 * @brief Run a GEMM operation with gather/scatter support.
 *
 * @param tensor_a: Pointer to the A matrix.
 * @param tensor_b: Pointer to the B matrix.
 * @param tensor_c: Pointer to the C matrix.
 * @param tensor_d: Pointer to the D matrix.
 *
 * @param indices_a: Indices for the A matrix.
 * @param indices_b: Indices for the B matrix.
 * @param indices_d: Indices for the D matrix.
 *
 * @param split_k_slices: Number of slices to split the K dimension into.
 * @param M_A: Original A matrix rows.
 * @param K: A matrix columns.
 * @param K_B: Original B matrix rows.
 * @param N: B matrix columns.
 * @param M_C: C matrix rows, equal to D matrix rows. (Regardless of whether C is transposed, M_C is
 * the number of rows of the original C matrix before transposition.)
 * @param gather_a_size: indices_a size, equal to indices_b when indices_b is not nullptr.
 *
 * @param alpha: Alpha value for the GEMM operation.
 * @param beta: Beta value for the GEMM operation.
 *
 * @return Status code indicating the success or failure of the operation.
 *
 * trAB gather:
 * D = \alpha * A[indices_a, :].T @ B[indices_b, :] + \beta * C
 *
 * AD gather scatter:
 * D[indices_d, :] = \alpha * A @ B[indices_b, :] + \beta * C[indices_d, :]
 *
 * Assume that the all inputs are row-major unless otherwise specified.
 * All transposition applied during GEMM to L1 load. So M_A is the number of rows of the original A
 * matrix before transposition. Same for M_C.
 *
 * A: M_A x K
 * B: K_B x N
 * C: M_C x N
 * D: M_C x N
 *
 *
 */
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename Config,
          typename TileTag,
          typename ArchTag>
int run_cutlass_gemm_with_operations_templated(
    const void *tensor_a,
    const void *tensor_b,
    const void *tensor_c,
    void *tensor_d,
    const int *indices_a,
    const int *indices_b,
    const int *indices_d,
    int split_k_slices,
    /* Row, col of A,B,C */ int M_A,  // Original A matrix rows
    int K,    // A matrix columns or B matrix rows when indices_b is not nullptr
    int K_B,  // Original B matrix rows when indices_b is not nullptr
    int N,    // B matrix columns or C matrix columns
    int M_C,  // C matrix rows when indices_d is not nullptr
    /* gather scatter size */ int gather_a_size,  // indices_a size, equal to indices_b when
                                                  // indices_b is not nullptr
    int scatter_d_size,                           // indices_d size
    float alpha,
    float beta) {
  using Traits = GemmOperationTraits<ElementInputA, ElementAccumulator, Config, TileTag, ArchTag>;
  using ElementComputeEpilogue = ElementAccumulator;

  // clang-format off
#ifdef DEBUG
  std::cout << "\n=== DEBUG: Entering run_cutlass_gemm_with_operations_templated ===" << std::endl;
  std::cout << "ElementInputA: " << demangle_type_name(typeid(ElementInputA).name()) << std::endl;
  std::cout << "ElementInputB: " << demangle_type_name(typeid(ElementInputB).name()) << std::endl;
  std::cout << "ElementOutput: " << demangle_type_name(typeid(ElementOutput).name()) << std::endl;
  std::cout << "ElementAccumulator: " << demangle_type_name(typeid(ElementAccumulator).name()) << std::endl;
  std::cout << "Config: " << demangle_type_name(typeid(Config).name()) << std::endl;
  std::cout << "TileTag: " << demangle_type_name(typeid(TileTag).name()) << std::endl;
  std::cout << "ArchTag: " << demangle_type_name(typeid(ArchTag).name()) << std::endl;
  std::cout << "=== DEBUG: Traits created ===" << std::endl;
  std::cout << "Traits::UseMixedInput: " << (Traits::UseMixedInput ? "true" : "false") << std::endl;
  std::cout << "Traits::SupportsTensorOp: " << (Traits::SupportsTensorOp ? "true" : "false") << std::endl;
  std::cout << "Traits::IsValidConfiguration(): " << (Traits::IsValidConfiguration() ? "true" : "false") << std::endl;

  const char* tile_name = "Unknown";
  if constexpr (std::is_same_v<TileTag, Tile128x128x32>) {
      tile_name = "Tile128x128x32";
  } else if constexpr (std::is_same_v<TileTag, Tile128x64x32>) {
      tile_name = "Tile128x64x32";
  } else if constexpr (std::is_same_v<TileTag, Tile64x64x32>) {
      tile_name = "Tile64x64x32";
  }

  std::cout << "\n=== DEBUG: GEMM Configuration Analysis ===" << std::endl;
  std::cout << "Tile: " << tile_name << std::endl;
  std::cout << "Arch: " << demangle_type_name(typeid(ArchTag).name()) << std::endl;

  std::cout << "\nTrait Shapes:" << std::endl;
  std::cout << "  ShapeMMAThreadBlock: " << print_gemm_shape<typename Traits::ShapeMMAThreadBlock>() << std::endl;
  std::cout << "  ShapeMMAWarp: " << print_gemm_shape<typename Traits::ShapeMMAWarp>() << std::endl;
  std::cout << "  ShapeMMAOp: " << print_gemm_shape<typename Traits::ShapeMMAOp>() << std::endl;

  std::cout << "\nTrait Properties:" << std::endl;
  std::cout << "  SupportsTensorOp: " << (Traits::SupportsTensorOp ? "true" : "false") << std::endl;
  std::cout << "  SupportsGatherA: " << (Traits::SupportsGatherA ? "true" : "false") << std::endl;
  std::cout << "  SupportsGatherB: " << (Traits::SupportsGatherB ? "true" : "false") << std::endl;
  std::cout << "  SupportsScatterD: " << (Traits::SupportsScatterD ? "true" : "false") << std::endl;
  std::cout << "  SupportsTransposeA: " << (Traits::SupportsTransposeA ? "true" : "false") << std::endl;
  std::cout << "  SupportsTransposeB: " << (Traits::SupportsTransposeB ? "true" : "false") << std::endl;
  std::cout << "  UseMixedInput: " << (Traits::UseMixedInput ? "true" : "false") << std::endl;
  std::cout << "  IsValidConfiguration: " << (Traits::IsValidConfiguration() ? "true" : "false") << std::endl;

  std::cout << "\nMMA Operation:" << std::endl;
  std::cout << "  MMAOp: " << demangle_type_name(typeid(typename Traits::MMAOp).name()) << std::endl;
  std::cout << "  ArchitectureTag: " << demangle_type_name(typeid(typename Traits::ArchitectureTag).name()) << std::endl;

  std::cout << "\nConfiguration Analysis:" << std::endl;
  std::cout << "  Has operations: " << (Config::has_operations() ? "true" : "false") << std::endl;
  std::cout << "  Requires TensorOp: " << (Config::has_operations() && Traits::SupportsTensorOp ? "yes (supported)" :
                                          Config::has_operations() ? "yes (NOT supported)" : "no") << std::endl;
  std::cout << "  Config gather_a: " << (bool(Config::gather_a) ? "true" : "false") << std::endl;
  std::cout << "  Config gather_b: " << (bool(Config::gather_b) ? "true" : "false") << std::endl;
  std::cout << "  Config scatter_d: " << (bool(Config::scatter_d) ? "true" : "false") << std::endl;
  std::cout << "  Config transpose_a: " << (bool(Config::transpose_a) ? "true" : "false") << std::endl;
  std::cout << "  Config transpose_b: " << (bool(Config::transpose_b) ? "true" : "false") << std::endl;
  std::cout << "================================================" << std::endl;
#endif
  // clang-format on

  if constexpr (Traits::UseMixedInput) {
    return static_cast<int>(GemmStatus::kErrorMixedInputUnsupported);
  }

  // Determine output vector length based on element type
  constexpr int OutputVectorLength = std::is_same_v<ElementOutput, cutlass::half_t>
                                         ? (128 / cutlass::sizeof_bits<ElementOutput>::value)
                                         : (128 / cutlass::sizeof_bits<ElementOutput>::value);

  // Define epilogue operation
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                                  OutputVectorLength,
                                                                  ElementAccumulator,
                                                                  ElementComputeEpilogue>;

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
                                                    typename Traits::LayoutInputA,
                                                    ElementInputB,
                                                    typename Traits::LayoutInputB,
                                                    ElementOutput,
                                                    typename Traits::LayoutOutput,
                                                    ElementAccumulator,
                                                    typename Traits::MMAOp,
                                                    ArchTag,
                                                    typename Traits::ShapeMMAThreadBlock,
                                                    typename Traits::ShapeMMAWarp,
                                                    typename Traits::ShapeMMAOp,
                                                    EpilogueOp,
                                                    SwizzleThreadBlock,
                                                    NumStages,
                                                    Traits::AlignmentA,
                                                    Traits::AlignmentB,
                                                    cutlass::arch::OpMultiplyAdd,
                                                    cutlass::ComplexTransform::kNone,
                                                    cutlass::ComplexTransform::kNone,
                                                    Traits::SupportsGatherA, /*GatherA*/
                                                    Traits::SupportsGatherB, /*GatherB*/
                                                    Traits::SupportsScatterD /*ScatterD*/
                                                    >;

  // Convert void pointers to appropriate types
  auto a_ptr = reinterpret_cast<const ElementInputA *>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementInputB *>(tensor_b);
  auto c_ptr = reinterpret_cast<const ElementOutput *>(tensor_c);
  auto d_ptr = reinterpret_cast<ElementOutput *>(tensor_d);

  if constexpr (Traits::IsValidConfiguration()) {
    // Native gather/scatter implementation
    // For transpose operation, the layout determines how the matrix is interpreted
    typename Traits::LayoutInputA layout_a(K);
    typename Traits::LayoutInputB layout_b(N);
    typename Traits::LayoutOutput layout_c(N);
    typename Traits::LayoutOutput layout_d(N);

    ElementComputeEpilogue alpha_cutlass = ElementComputeEpilogue(alpha);
    ElementComputeEpilogue beta_cutlass = ElementComputeEpilogue(beta);

    // Derive problem dimensions from original matrix dimensions
    int problem_m, problem_n, problem_k, N_B;

    // Currently only support trAB gather and AD gather scatter.
    if constexpr (Config::transpose_a && Config::gather_a && Config::gather_b) {
      // For A transpose: A[indices_a, :].T @ B[indices_b, :]
      // A[indices_a, :] is gather_a_size × K, transposed to K × gather_a_size
      // B[indices_b, :] is gather_a_size × N
      // Result: K × N
      assert(indices_a != nullptr);
      assert(indices_b != nullptr);
      assert(indices_d == nullptr);
      problem_m = K;  // rows in result (from transposed A)
      // TODO(cchoy): Should it be N instead of gather_a_size for problem_n
      problem_n = N;              // columns in result (from B)
      problem_k = gather_a_size;  // inner dimension
      N_B = gather_a_size;
    } else if constexpr (Config::gather_a && Config::scatter_d) {
      assert(gather_a_size == scatter_d_size);
      assert(indices_a != nullptr);
      assert(indices_b == nullptr);
      assert(indices_d != nullptr);
      // AD gather scatter: D[indices_D, :] = A[indices_A, :] @ B + C[indices_D, :]
      problem_m = gather_a_size;
      problem_n = N;
      problem_k = K;
      N_B = N;
    } else {
      // Standard GEMM without neither gather nor scatter
      assert(indices_a == nullptr);
      assert(indices_b == nullptr);
      assert(indices_d == nullptr);
      assert(K == K_B);
      assert(M_A == M_C);
      problem_m = M_A;
      problem_n = N;
      problem_k = K;
      N_B = N;
    }

    cutlass::gemm::GemmCoord problem_size(problem_m, problem_n, problem_k);

    // Calculate batch strides using original matrix dimensions.
    // Do not use the gather/scatter size.
    int64_t batch_stride_A = static_cast<int64_t>(M_A) * K * sizeof(ElementInputA);
    int64_t batch_stride_B = static_cast<int64_t>(K_B) * N * sizeof(ElementInputB);
    int64_t batch_stride_C = static_cast<int64_t>(M_C) * N * sizeof(ElementOutput);
    int64_t batch_stride_D = static_cast<int64_t>(M_C) * N * sizeof(ElementOutput);

    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                       problem_size,
                                       split_k_slices,
                                       {alpha_cutlass, beta_cutlass},
                                       a_ptr,
                                       b_ptr,
                                       c_ptr,
                                       d_ptr,
                                       /*batch strides*/ batch_stride_A,
                                       batch_stride_B,
                                       batch_stride_C,
                                       batch_stride_D,
                                       layout_a.stride(),
                                       layout_b.stride(),
                                       layout_c.stride(),
                                       layout_d.stride(),
                                       Config::gather_a ? indices_a : nullptr,
                                       Config::gather_b ? indices_b : nullptr,
                                       Config::scatter_d ? indices_d : nullptr};

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    Gemm gemm_op;

#ifdef DEBUG
    // Print detailed GEMM arguments for valid configurations only (config info already printed
    // above)
    debug_print_gemm_arguments<Gemm, Traits, Config>(arguments,
                                                     problem_m,
                                                     problem_n,
                                                     problem_k,
                                                     M_A,
                                                     K,
                                                     K_B,
                                                     N,
                                                     M_C,
                                                     gather_a_size,
                                                     scatter_d_size);
#endif

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(GemmStatus::kErrorProblemNotSupported);
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(GemmStatus::kErrorKernelInitialization);
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(GemmStatus::kErrorKernelExecution);
    }
  } else {
    // Configuration not supported
    return static_cast<int>(GemmStatus::kErrorUnsupportedConfig);
  }

  return static_cast<int>(GemmStatus::kSuccess);
}

}  // namespace gemm
}  // namespace warpconvnet

namespace warpconvnet {
namespace gemm {

// Helper function for AD gather scatter
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename TileTag,
          typename Arch = DefaultSmArch>
int run_cutlass_gemm_ad_gather_scatter(const void *tensor_a,
                                       const void *tensor_b,
                                       const void *tensor_c,
                                       void *tensor_d,
                                       const int *indices_a,
                                       const int *indices_d,
                                       int split_k_slices,
                                       int M_A,           // row of A
                                       int K,             // col of A
                                       int N,             // col of B
                                       int M_C,           // row of C
                                       int indices_size,  // indices_a and indices_d size
                                       float alpha,
                                       float beta) {
  // Forward to new templated implementation with ConfigAD (A gather + D scatter)
  return run_cutlass_gemm_with_operations_templated<ElementInputA,
                                                    ElementInputB,
                                                    ElementOutput,
                                                    ElementAccumulator,
                                                    ConfigAD,
                                                    TileTag,
                                                    Arch>(
      tensor_a,
      tensor_b,
      tensor_c,
      tensor_d,
      indices_a,
      nullptr,    // indices_b (no B gather in AD config)
      indices_d,  // indices_d for D scatter
      split_k_slices,
      M_A,           // M_A (original A matrix rows)
      K,             // K (A columns)
      K,             // K_B (B matrix rows)
      N,             // N (B columns)
      M_C,           // M_C (C matrix rows, different from M_A when indices_d is not nullptr)
      indices_size,  // indices_a size, equal to indices_b when indices_b is not nullptr
      indices_size,  // indices_d size
      alpha,
      beta);
}

// AB Gather with A Transpose
template <typename ElementInputA,
          typename ElementInputB,
          typename ElementOutput,
          typename ElementAccumulator,
          typename TileTag,
          typename Arch = DefaultSmArch>
int run_cutlass_gemm_trAB_gather(const void *tensor_a,
                                 const void *tensor_b,
                                 const void *tensor_c,
                                 void *tensor_d,
                                 const int *indices_a,
                                 const int *indices_b,
                                 int split_k_slices,
                                 int M_A,             // row of A (not trA)
                                 int K,               // col of A (not trA)
                                 int K_B,             // row of B (different from K since gathering)
                                 int N,               // col of B
                                 int gather_ab_size,  // indices_a and indices_b size
                                 float alpha,
                                 float beta) {
  // Forward to new templated implementation with ConfigTrAB (A gather + B gather + A transpose)
  return run_cutlass_gemm_with_operations_templated<ElementInputA,
                                                    ElementInputB,
                                                    ElementOutput,
                                                    ElementAccumulator,
                                                    ConfigTrAB,
                                                    TileTag,
                                                    Arch>(
      tensor_a,
      tensor_b,
      tensor_c,
      tensor_d,
      indices_a,
      indices_b,
      nullptr,  // indices_d (no D scatter in AB config)
      split_k_slices,
      M_A,  // M_A (original A matrix rows)
      K,    // K (A columns)
      K_B,  // M_B (original B matrix rows. Different from K when indices_b is not nullptr)
      N,    // N (B columns)
      K,    // M_C (C matrix rows. Since A is transposed, A columns are the same as C rows)
      gather_ab_size,
      0,  // scatter_d_size (no scatter)
      alpha,
      beta);
}

// Instantiate all configurations inside the warpconvnet::gemm namespace
// Instantiate all configurations for the default architecture (SM80 Tile128x128x32)
INSTANTIATE_AD_GS_FOR_ARCH(Tile128x128x32, DefaultSmArch)
INSTANTIATE_TRAB_FOR_ARCH(Tile128x128x32, DefaultSmArch)

// Instantiate all alternative tile configurations for SM80
INSTANTIATE_AD_GS_FOR_ARCH(Tile128x64x32, DefaultSmArch)
INSTANTIATE_TRAB_FOR_ARCH(Tile128x64x32, DefaultSmArch)

INSTANTIATE_AD_GS_FOR_ARCH(Tile64x128x32, DefaultSmArch)
INSTANTIATE_TRAB_FOR_ARCH(Tile64x128x32, DefaultSmArch)

INSTANTIATE_AD_GS_FOR_ARCH(Tile64x64x32, DefaultSmArch)
INSTANTIATE_TRAB_FOR_ARCH(Tile64x64x32, DefaultSmArch)

}  // namespace gemm
}  // namespace warpconvnet
