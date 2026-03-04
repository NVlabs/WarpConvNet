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

// SmemLayoutAtom for FP16/BF16 with K=32 (from CUTLASS DefaultGemm_TensorOpSm80_OperandA):
// 8 rows × 32 columns with K-contiguous storage and bank-conflict-free swizzle.
// Swizzle<2,3,3> = 4-way swizzle on bits [3:5] → eliminates bank conflicts for 16-bit elements.
using SmemLayoutAtomFP16 = decltype(composition(
    Swizzle<2, 3, 3>{},
    Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

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

  using SmemLayoutAtomA = SmemLayoutAtomFP16;
  using SmemLayoutAtomB = SmemLayoutAtomA;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = SmemCopyAtomA;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  static constexpr int NumStages = 1;
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

  using SmemLayoutAtomA = SmemLayoutAtomFP16;
  using SmemLayoutAtomB = SmemLayoutAtomA;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = SmemCopyAtomA;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  static constexpr int NumStages = 1;
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

  using SmemLayoutAtomA = SmemLayoutAtomFP16;
  using SmemLayoutAtomB = SmemLayoutAtomA;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = SmemCopyAtomA;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  static constexpr int NumStages = 1;
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

  using SmemLayoutAtomA = SmemLayoutAtomFP16;
  using SmemLayoutAtomB = SmemLayoutAtomA;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = SmemCopyAtomA;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  static constexpr int NumStages = 1;
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
  using SmemLayoutAtomA = SmemLayoutAtomFP16;
  using SmemLayoutAtomB = SmemLayoutAtomA;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = SmemCopyAtomA;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 1;
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
  using SmemLayoutAtomA = SmemLayoutAtomFP16;
  using SmemLayoutAtomB = SmemLayoutAtomA;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = SmemCopyAtomA;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 1;
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
  using SmemLayoutAtomA = SmemLayoutAtomFP16;
  using SmemLayoutAtomB = SmemLayoutAtomA;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = SmemCopyAtomA;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 1;
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
  using SmemLayoutAtomA = SmemLayoutAtomFP16;
  using SmemLayoutAtomB = SmemLayoutAtomA;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = SmemCopyAtomA;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 1;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
};

}  // namespace cute_gemm
}  // namespace warpconvnet
