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
using SmemLayoutAtomA_FP16 =
    decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

// SmemLayoutAtom for operand B: N-contiguous (column-major) with Swizzle<3,3,3>.
// 64 rows × 8 columns, N is the contiguous (stride-1) dimension.
// From CUTLASS DefaultGemm_TensorOpSm80_OperandB<half_t, RowMajor> which maps to
// OperandA<half_t, ColumnMajor>. Used with SM75_U16x8_LDSM_T (transposing LDSM)
// that transposes N-contiguous smem data to K-contiguous registers for the MMA.
using SmemLayoutAtomB_FP16 =
    decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));

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
  static constexpr bool UseCpAsyncGatherA = false;
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
  static constexpr bool UseCpAsyncGatherA = false;
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
  static constexpr bool UseCpAsyncGatherA = false;
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
  static constexpr bool UseCpAsyncGatherA = false;
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
  static constexpr bool UseCpAsyncGatherA = false;
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
  static constexpr bool UseCpAsyncGatherA = false;
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
  static constexpr bool UseCpAsyncGatherA = false;
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
  static constexpr bool UseCpAsyncGatherA = false;
};

// ============================================================================
// Additional tile specializations — generated via macro since only TileShape
// differs (all SM80 configs share the same MMA atom, smem layout, copy atoms).
// ============================================================================

#define DEFINE_CUTE_TILE_CONFIG_FP16(TileTag, M_DIM, N_DIM, K_DIM)    \
  template <>                                                         \
  struct CuteTileConfig<cutlass::half_t, gemm::TileTag> {             \
    using ElementInput = cutlass::half_t;                             \
    using TileShape = Shape<Int<M_DIM>, Int<N_DIM>, Int<K_DIM>>;      \
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>, \
                              Layout<Shape<_2, _2, _1>>,              \
                              Tile<_32, _32, _16>>;                   \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP16;                     \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP16;                     \
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>; \
    using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>; \
    using GmemTiledCopyA = void;                                      \
    using GmemTiledCopyB = void;                                      \
    static constexpr int NumStages = 2;                               \
    static constexpr int AlignmentA = 4;                              \
    static constexpr int AlignmentB = 4;                              \
    static constexpr bool UseCpAsyncGatherA = false;                  \
  };

#define DEFINE_CUTE_TILE_CONFIG_BF16(TileTag, M_DIM, N_DIM, K_DIM)      \
  template <>                                                           \
  struct CuteTileConfig<cutlass::bfloat16_t, gemm::TileTag> {           \
    using ElementInput = cutlass::bfloat16_t;                           \
    using TileShape = Shape<Int<M_DIM>, Int<N_DIM>, Int<K_DIM>>;        \
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>, \
                              Layout<Shape<_2, _2, _1>>,                \
                              Tile<_32, _32, _16>>;                     \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP16;                       \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP16;                       \
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;   \
    using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>;   \
    using GmemTiledCopyA = void;                                        \
    using GmemTiledCopyB = void;                                        \
    static constexpr int NumStages = 2;                                 \
    static constexpr int AlignmentA = 4;                                \
    static constexpr int AlignmentB = 4;                                \
    static constexpr bool UseCpAsyncGatherA = false;                    \
  };

// tK=64 tiles
DEFINE_CUTE_TILE_CONFIG_FP16(Tile64x64x64, 64, 64, 64)
DEFINE_CUTE_TILE_CONFIG_FP16(Tile128x64x64, 128, 64, 64)
DEFINE_CUTE_TILE_CONFIG_FP16(Tile64x128x64, 64, 128, 64)
DEFINE_CUTE_TILE_CONFIG_FP16(Tile128x128x64, 128, 128, 64)

DEFINE_CUTE_TILE_CONFIG_BF16(Tile64x64x64, 64, 64, 64)
DEFINE_CUTE_TILE_CONFIG_BF16(Tile128x64x64, 128, 64, 64)
DEFINE_CUTE_TILE_CONFIG_BF16(Tile64x128x64, 64, 128, 64)
DEFINE_CUTE_TILE_CONFIG_BF16(Tile128x128x64, 128, 128, 64)

// Asymmetric M/N tiles (tK=32)
DEFINE_CUTE_TILE_CONFIG_FP16(Tile256x64x32, 256, 64, 32)
DEFINE_CUTE_TILE_CONFIG_FP16(Tile64x256x32, 64, 256, 32)

DEFINE_CUTE_TILE_CONFIG_BF16(Tile256x64x32, 256, 64, 32)
DEFINE_CUTE_TILE_CONFIG_BF16(Tile64x256x32, 64, 256, 32)

#undef DEFINE_CUTE_TILE_CONFIG_FP16
#undef DEFINE_CUTE_TILE_CONFIG_BF16

/// Override wrapper: inherits all types from BaseConfig, overrides NumStages and UseCpAsyncGatherA.
template <class BaseConfig, int NumStages_, bool UseCpAsyncGatherA_>
struct CuteTileConfigOverride {
  using ElementInput = typename BaseConfig::ElementInput;
  using TileShape = typename BaseConfig::TileShape;
  using TiledMma = typename BaseConfig::TiledMma;
  using SmemLayoutAtomA = typename BaseConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename BaseConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename BaseConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename BaseConfig::SmemCopyAtomB;
  using GmemTiledCopyA = typename BaseConfig::GmemTiledCopyA;
  using GmemTiledCopyB = typename BaseConfig::GmemTiledCopyB;
  static constexpr int AlignmentA = BaseConfig::AlignmentA;
  static constexpr int AlignmentB = BaseConfig::AlignmentB;
  static constexpr int NumStages = NumStages_;
  static constexpr bool UseCpAsyncGatherA = UseCpAsyncGatherA_;
};

}  // namespace cute_gemm
}  // namespace warpconvnet
