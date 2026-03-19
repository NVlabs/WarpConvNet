// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if defined(WARPCONVNET_SM80_ENABLED)

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory_sm80.h>  // cp.async helpers
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/permute.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////
// 1.  Custom iterator: load float → convert to half → store to fragment
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Vector‑width (must be 2 or 4 to enable f16x2 conversions)
static int const kPack = 2;

using FragmentFP16x2 = cutlass::Array<cutlass::half_t, kPack>;
using FragmentFP32x2 = cutlass::Array<float, kPack>;

template <typename BaseIterator>
struct F32ToF16TileIterator : public BaseIterator {
  using Base = BaseIterator;
  using Fragment = FragmentFP16x2;
  using FragmentF32 = FragmentFP32x2;

  using Base::Base;  // inherit ctors

  CUTLASS_DEVICE void load_with_pointer_offset(Fragment &frag, int32_t offset) const {
    FragmentF32 tmp;
    Base::load_with_pointer_offset(tmp, offset);
    cutlass::NumericArrayConverter<cutlass::half_t, float, kPack> cvt;
    frag = cvt(tmp);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// 2.  Kernel config for SM80
///////////////////////////////////////////////////////////////////////////////////////////////////

using ArchTag = cutlass::arch::Sm80;
// using OpClass = cutlass::arch::OpClassTensorOp;
using OpClass = cutlass::arch::OpClassTensorOp;

// GMEM element types (float32)
using ElementA_gmem = float;
using ElementB_gmem = float;
// Internal dtype
using ElementInternal = cutlass::half_t;
// Accumulator / output
using ElementAccumulator = float;
using ElementOutput = float;

// Layouts
using LayoutA_gmem = cutlass::layout::RowMajor;
using LayoutB_gmem = cutlass::layout::RowMajor;
using LayoutC_gmem = cutlass::layout::RowMajor;
using LayoutD_gmem = cutlass::layout::RowMajor;

// Tile & pipeline shapes (keep modest for 128‑thread TB)
using TileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<32>>;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
static constexpr int Stages = 3;

///////////////////////////////////////////////////////////////////////////////////////////////////
// 3.  DefaultMmaCore (hard-coded sizes) and 2.x GEMM kernel
///////////////////////////////////////////////////////////////////////////////////////////////////
// Base CUTLASS core defining thread-block shapes, thread maps, etc.
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using BaseMmaCore = cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                               WarpShape,
                                                               InstructionShape,
                                                               ElementInternal,
                                                               LayoutA_gmem,
                                                               ElementInternal,
                                                               LayoutB_gmem,
                                                               ElementAccumulator,
                                                               LayoutC_gmem,
                                                               OpClass,
                                                               Stages,
                                                               cutlass::arch::OpMultiplyAdd>;

// ---------------------------------------------------------------------------------------------
// Custom MmaCore that hijacks the global-memory iterators so they read FP32 and convert to FP16
// on the fly via the F32ToF16TileIterator wrapper defined above.
// ---------------------------------------------------------------------------------------------

struct MmaCore : public BaseMmaCore {
  // Thread maps coming from the base definition
  using ThreadMapA = typename BaseMmaCore::IteratorThreadMapA;
  using ThreadMapB = typename BaseMmaCore::IteratorThreadMapB;

  // --- A operand iterator: loads 2 fp32 values and converts them to fp16 ---
  using IteratorA_Base = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<BaseMmaCore::Shape::kM, BaseMmaCore::Shape::kK>,
      ElementA_gmem,    // what resides in global memory (fp32)
      LayoutA_gmem,     // layout
      1,                // AdvanceRank — contiguous dimension for row-major A
      ThreadMapA,       // per-thread map
      kPack,            // vector width (see kPack above)
      /*Gather*/ true,  // we are gathering rows of A
      cutlass::layout::NoPermute>;

  using IteratorA = F32ToF16TileIterator<IteratorA_Base>;

  // --- B operand iterator: loads 2 fp32 values and converts them to fp16 ---
  using IteratorB_Base = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<BaseMmaCore::Shape::kK, BaseMmaCore::Shape::kN>,
      ElementB_gmem,     // fp32 in global memory
      LayoutB_gmem,      // layout
      0,                 // AdvanceRank — strided dimension for row-major B
      ThreadMapB,        // per-thread map
      kPack,             // vector width
      /*Gather*/ false,  // no gather on B
      cutlass::layout::NoPermute>;

  using IteratorB = F32ToF16TileIterator<IteratorB_Base>;
};

// Epilogue (simple alpha*acc + beta*C)
using OutputOp = cutlass::epilogue::thread::LinearCombination<ElementOutput,  // D element type
                                                              8,              // elements per thread
                                                              ElementAccumulator,  // Accumulator
                                                              ElementAccumulator>;

// 2.x GEMM kernel with gather A / scatter D flags
using GemmKernel = cutlass::gemm::kernel::DefaultGemmUniversal<
    ElementInternal,
    LayoutA_gmem,
    cutlass::ComplexTransform::kNone,
    1,
    ElementInternal,
    LayoutB_gmem,
    cutlass::ComplexTransform::kNone,
    1,
    ElementOutput,
    LayoutC_gmem,
    ElementAccumulator,
    OpClass,
    ArchTag,
    cutlass::gemm::GemmShape<128, 128, 32>,  // thread-block
    cutlass::gemm::GemmShape<64, 64, 32>,    // warp
    cutlass::gemm::GemmShape<16, 8, 16>,     // instruction
    OutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    true,  /* GatherA */
    false, /* GatherB  */
    true /* ScatterD */>::GemmKernel;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// 5.  Thin launcher identical to Ampere version
///////////////////////////////////////////////////////////////////////////////////////////////////
int run_f32_to_f16_gemm_gather_scatter_sm80(const float *dA,
                                            const float *dB,
                                            const float *dC,
                                            float *dD,
                                            const int *gatherA_indices,
                                            const int *scatterD_indices,
                                            int split_k_slices,
                                            int M,
                                            int N,
                                            int K,
                                            int gather_rows,
                                            int scatter_rows,
                                            float alpha = 1.f,
                                            float beta = 0.f,
                                            cudaStream_t stream = 0) {
  cutlass::gemm::GemmCoord problem_size(gather_rows, N, K);

  int64_t lda = K;
  int64_t ldb = N;
  int64_t ldc = N;
  int64_t ldd = N;

  using StrideA = LayoutA_gmem::Stride;
  using StrideB = LayoutB_gmem::Stride;
  using StrideC = LayoutC_gmem::Stride;

  OutputOp::Params epilogue_params(alpha, beta);

  // assert gatherA_indices and scatterD_indices are not null
  assert(gatherA_indices != nullptr);
  assert(scatterD_indices != nullptr);

  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
                                problem_size, /*batch_count*/
                                split_k_slices,
                                epilogue_params,
                                dA,
                                dB,
                                dC,
                                dD,
                                /*batch strides*/ int64_t(0),
                                int64_t(0),
                                int64_t(0),
                                int64_t(0),
                                StrideA(lda),
                                StrideB(ldb),
                                StrideC(ldc),
                                StrideC(ldd),
                                gatherA_indices,
                                nullptr,
                                scatterD_indices);

  Gemm op;
  size_t workspace_bytes = op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);

  if (op.can_implement(args) != cutlass::Status::kSuccess) return -1;
  if (op.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) return -2;
  if (op(stream) != cutlass::Status::kSuccess) return -3;
  return 0;
}

// -------------------------------------------------------------------------------------------------
//  Patch: provide 2-byte cp.async fallbacks (CUTLASS only supports 4/8/16 by
//  default) --------------
// -------------------------------------------------------------------------------------------------
// Some iterator paths in older SM80 kernels vectorise to 2-byte accesses
// (single half_t).  The stock cp_async helper hard-errors via static_assert for
// that case.  We override it with a trivial C++ fallback (direct LD/ST) so
// compilation succeeds.  Efficiency loss is negligible for the experimental
// path.

namespace cutlass::arch {

// Only compile this patch for SM80+ and when CUTLASS hasn't already provided a
// specialisation.
#if CUDA_CP_ASYNC_ACTIVATED

// Specialisation: cp_async<2, CacheOperation::Always>
template <>
struct cp_async<2, CacheOperation::Always> {
  CUTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    using AccessType = uint16_t;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    }
  }
};

// Specialisation: cp_async_zfill<2, CacheOperation::Always>
template <>
struct cp_async_zfill<2, CacheOperation::Always> {
  CUTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    using AccessType = uint16_t;
    *static_cast<AccessType *>(smem_ptr) =
        pred_guard ? *static_cast<AccessType const *>(global_ptr) : AccessType{0};
  }
};

#endif  // CUDA_CP_ASYNC_ACTIVATED

}  // namespace cutlass::arch

// -------------------------------------------------------------------------------------------------
// Minimal test driver (dummy)
// ---------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
#ifdef BUILD_STANDALONE
int main() {
  // Compilation sanity-check – we don't run the kernel here.
  printf("fp32_to_fp16_gemm_gather_scatter_sm80 compiled successfully.\n");
  return 0;
}
#endif

#else  // !WARPCONVNET_SM80_ENABLED

// Empty translation unit when SM80 CUTLASS is not available.
void cutlass_gemm_sm80_fp32_stub() {}

#endif  // WARPCONVNET_SM80_ENABLED
