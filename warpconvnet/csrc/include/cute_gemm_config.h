// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CuTe tile configurations for GEMM with gather/scatter.
// Each specialization provides TiledMMA, smem layout/copy atoms
// for a specific (ElementType, TileTag) combination.
//
// All configurations use 2×2 warp layout (128 threads) following
// CUTLASS's DefaultGemmConfigurationToCutlass3Types for SM80 FP16/BF16.
// Larger tiles (128×64, 64×128, 128×128) use more MMA iterations
// per thread rather than more warps.
//
// The gmem → smem copy is done manually by the kernel (element-wise
// with gather indices), so no GmemTiledCopy types are defined here.
// The SmemLayoutAtom and SmemCopyAtom are used for smem → register
// copies via LDSM instructions.

#pragma once

#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"

#include "gemm_mma_tiles.h"

namespace warpconvnet {
namespace cute_gemm {

using namespace cute;

// Primary template — must be specialized per (Element, Tile) pair
template <class ElementInput, class TileTag>
struct CuteTileConfig;

// SmemLayoutAtom for operand A: K-contiguous (row-major) with Swizzle<2,3,3>.
// 8 rows × 32 columns, K is the contiguous (stride-1) dimension.
// From CUTLASS DefaultGemm_TensorOpSm80_OperandA<half_t, RowMajor>.
using SmemLayoutAtomA_FP16 = decltype(composition(
    Swizzle<2, 3, 3>{},
    Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

// SmemLayoutAtom for operand B: N-contiguous (column-major) with Swizzle<3,3,3>.
// 64 rows × 8 columns, N is the contiguous (stride-1) dimension.
// From CUTLASS DefaultGemm_TensorOpSm80_OperandB<half_t, RowMajor> which maps to
// OperandA<half_t, ColumnMajor>. Used with SM75_U16x8_LDSM_T (transposing LDSM)
// that transposes N-contiguous smem data to K-contiguous registers for the MMA.
using SmemLayoutAtomB_FP16 = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_64, _8>, Stride<_1, _64>>{}));

// ============================================================================
// half_t specializations — all use 2×2 warps (128 threads)
// ============================================================================

// --- Tile64x64x32 × half_t ---
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile64x64x32> {
  using ElementInput = cutlass::half_t;
  using TileShape = Shape<_64, _64, _32>;

  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                             Layout<Shape<_2, _2, _1>>,
                             Tile<_32, _32, _16>>;

  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

// --- Tile128x64x32 × half_t ---
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile128x64x32> {
  using ElementInput = cutlass::half_t;
  using TileShape = Shape<_128, _64, _32>;

  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                             Layout<Shape<_2, _2, _1>>,
                             Tile<_32, _32, _16>>;

  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

// --- Tile64x128x32 × half_t ---
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile64x128x32> {
  using ElementInput = cutlass::half_t;
  using TileShape = Shape<_64, _128, _32>;

  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                             Layout<Shape<_2, _2, _1>>,
                             Tile<_32, _32, _16>>;

  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

// --- Tile128x128x32 × half_t ---
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile128x128x32> {
  using ElementInput = cutlass::half_t;
  using TileShape = Shape<_128, _128, _32>;

  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                             Layout<Shape<_2, _2, _1>>,
                             Tile<_32, _32, _16>>;

  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

// ============================================================================
// bfloat16_t specializations — same 2×2 warp pattern
// ============================================================================

template <>
struct CuteTileConfig<cutlass::bfloat16_t, gemm::Tile64x64x32> {
  using ElementInput = cutlass::bfloat16_t;
  using TileShape = Shape<_64, _64, _32>;
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,
                             Layout<Shape<_2, _2, _1>>,
                             Tile<_32, _32, _16>>;
  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

template <>
struct CuteTileConfig<cutlass::bfloat16_t, gemm::Tile128x64x32> {
  using ElementInput = cutlass::bfloat16_t;
  using TileShape = Shape<_128, _64, _32>;
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,
                             Layout<Shape<_2, _2, _1>>,
                             Tile<_32, _32, _16>>;
  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

template <>
struct CuteTileConfig<cutlass::bfloat16_t, gemm::Tile64x128x32> {
  using ElementInput = cutlass::bfloat16_t;
  using TileShape = Shape<_64, _128, _32>;
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,
                             Layout<Shape<_2, _2, _1>>,
                             Tile<_32, _32, _16>>;
  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

template <>
struct CuteTileConfig<cutlass::bfloat16_t, gemm::Tile128x128x32> {
  using ElementInput = cutlass::bfloat16_t;
  using TileShape = Shape<_128, _128, _32>;
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,
                             Layout<Shape<_2, _2, _1>>,
                             Tile<_32, _32, _16>>;
  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

}  // namespace cute_gemm
}  // namespace warpconvnet
