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

// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
// clang-format on
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "gemm_mma_tiles.h"

namespace warpconvnet {
namespace cute_gemm {

// Primary template — must be specialized per (Element, Tile) pair
template <class ElementInput, class TileTag>
struct CuteTileConfig;

// SmemLayoutAtom for operand A: K-contiguous (row-major) with Swizzle<2,3,3>.
// 8 rows × 32 columns, K is the contiguous (stride-1) dimension.
// From CUTLASS DefaultGemm_TensorOpSm80_OperandA<half_t, RowMajor>.
using SmemLayoutAtomA_FP16 = decltype(cute::composition(
    cute::Swizzle<2, 3, 3>{},
    cute::Layout<cute::Shape<cute::_8, cute::_32>, cute::Stride<cute::_32, cute::_1>>{}));

// SmemLayoutAtom for operand B: N-contiguous (column-major) with Swizzle<3,3,3>.
// 64 rows × 8 columns, N is the contiguous (stride-1) dimension.
// From CUTLASS DefaultGemm_TensorOpSm80_OperandB<half_t, RowMajor> which maps to
// OperandA<half_t, ColumnMajor>. Used with SM75_U16x8_LDSM_T (transposing LDSM)
// that transposes N-contiguous smem data to K-contiguous registers for the MMA.
using SmemLayoutAtomB_FP16 = decltype(cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_64, cute::_8>, cute::Stride<cute::_1, cute::_64>>{}));

// SmemLayoutAtom for FP32/TF32 operand A: K-contiguous (row-major).
// Shape matches MMA fragment: 16 M rows × 8 K columns of float.
// Plain layout (no swizzle) — accepts bank conflicts for correctness-first.
using SmemLayoutAtomA_FP32 =
    cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_8, cute::_1>>;

// SmemLayoutAtom for FP32/TF32 operand B: N-contiguous (column-major).
// 8 N rows × 8 K columns, N stride-1.
using SmemLayoutAtomB_FP32 =
    cute::Layout<cute::Shape<cute::_8, cute::_8>, cute::Stride<cute::_1, cute::_8>>;

// ============================================================================
// Macro for FP16 tile configs — reduces boilerplate with cute:: qualification
// ============================================================================

#define DEFINE_CUTE_TILE_CONFIG_FP16(TileTag, M_DIM, N_DIM, K_DIM)                           \
  template <>                                                                                \
  struct CuteTileConfig<cutlass::half_t, gemm::TileTag> {                                    \
    using ElementInput = cutlass::half_t;                                                    \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;     \
    using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,      \
                                    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>, \
                                    cute::Tile<cute::_32, cute::_32, cute::_16>>;            \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP16;                                            \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP16;                                            \
    using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;            \
    using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, ElementInput>;            \
    using GmemTiledCopyA = void;                                                             \
    using GmemTiledCopyB = void;                                                             \
    static constexpr int NumStages = 2;                                                      \
    static constexpr int AlignmentA = 4;                                                     \
    static constexpr int AlignmentB = 4;                                                     \
    static constexpr bool UseCpAsyncGatherA = false;                                         \
  };

#define DEFINE_CUTE_TILE_CONFIG_BF16(TileTag, M_DIM, N_DIM, K_DIM)                           \
  template <>                                                                                \
  struct CuteTileConfig<cutlass::bfloat16_t, gemm::TileTag> {                                \
    using ElementInput = cutlass::bfloat16_t;                                                \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;     \
    using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>,    \
                                    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>, \
                                    cute::Tile<cute::_32, cute::_32, cute::_16>>;            \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP16;                                            \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP16;                                            \
    using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;            \
    using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, ElementInput>;            \
    using GmemTiledCopyA = void;                                                             \
    using GmemTiledCopyB = void;                                                             \
    static constexpr int NumStages = 2;                                                      \
    static constexpr int AlignmentA = 4;                                                     \
    static constexpr int AlignmentB = 4;                                                     \
    static constexpr bool UseCpAsyncGatherA = false;                                         \
  };

// FP16 accumulator + k=8 MMA atom (mma.sync.m16n8k8)
// 4 k-blocks per tile (vs 2 with k=16): more interleaving points for MMA/load overlap

#define DEFINE_CUTE_TILE_CONFIG_FP16_K8(TileTag, M_DIM, N_DIM, K_DIM)                        \
  template <>                                                                                \
  struct CuteTileConfig<cutlass::half_t, gemm::TileTag##_F16K8> {                            \
    using ElementInput = cutlass::half_t;                                                    \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;     \
    using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x8_F16F16F16F16_TN>,       \
                                    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>, \
                                    cute::Tile<cute::_32, cute::_32, cute::_8>>;             \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP16;                                            \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP16;                                            \
    /* k=8 needs half-width smem copy atoms to match mma_k */                                \
    using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, ElementInput>;            \
    using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x4_LDSM_T, ElementInput>;            \
    using GmemTiledCopyA = void;                                                             \
    using GmemTiledCopyB = void;                                                             \
    static constexpr int NumStages = 2;                                                      \
    static constexpr int AlignmentA = 4;                                                     \
    static constexpr int AlignmentB = 4;                                                     \
    static constexpr bool UseCpAsyncGatherA = false;                                         \
  };

// FP16 accumulator variant — 2x tensor core throughput vs F32 accum on Ada SM89
// Used for forward and dgrad where fp16 precision suffices.
// Tag: CuteTileConfig<cutlass::half_t, gemm::TileTag_F16Accum>
// We use a separate tag type to distinguish from the F32 accum variant.

#define DEFINE_CUTE_TILE_CONFIG_FP16_ACCUM(TileTag, M_DIM, N_DIM, K_DIM)                     \
  template <>                                                                                \
  struct CuteTileConfig<cutlass::half_t, gemm::TileTag##_F16Accum> {                         \
    using ElementInput = cutlass::half_t;                                                    \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;     \
    using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x16_F16F16F16F16_TN>,      \
                                    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>, \
                                    cute::Tile<cute::_32, cute::_32, cute::_16>>;            \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP16;                                            \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP16;                                            \
    using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;            \
    using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, ElementInput>;            \
    using GmemTiledCopyA = void;                                                             \
    using GmemTiledCopyB = void;                                                             \
    static constexpr int NumStages = 2;                                                      \
    static constexpr int AlignmentA = 4;                                                     \
    static constexpr int AlignmentB = 4;                                                     \
    static constexpr bool UseCpAsyncGatherA = false;                                         \
  };

// FP32 input (TF32 tensor core) — for full-precision training.
// SM80_16x8x8: K=8 per MMA, float inputs truncated to TF32 by hardware.
// kVec = 16/4 = 4 floats per 128-bit vector.
// Uses UniversalCopy for smem→reg (no LDSM for 32-bit element transpose).
#define DEFINE_CUTE_TILE_CONFIG_FP32(TileTag, M_DIM, N_DIM, K_DIM)                           \
  template <>                                                                                \
  struct CuteTileConfig<float, gemm::TileTag> {                                              \
    using ElementInput = float;                                                              \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;     \
    using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,     \
                                    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>, \
                                    cute::Tile<cute::_32, cute::_32, cute::_8>>;             \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP32;                                            \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP32;                                            \
    using SmemCopyAtomA = cute::Copy_Atom<cute::UniversalCopy<ElementInput>, ElementInput>;  \
    using SmemCopyAtomB = cute::Copy_Atom<cute::UniversalCopy<ElementInput>, ElementInput>;  \
    using GmemTiledCopyA = void;                                                             \
    using GmemTiledCopyB = void;                                                             \
    static constexpr int NumStages = 2;                                                      \
    static constexpr int AlignmentA = 4;                                                     \
    static constexpr int AlignmentB = 4;                                                     \
    static constexpr bool UseCpAsyncGatherA = false;                                         \
  };

// F32 accumulator + k=8 MMA atom — for wgrad where f32 precision is needed
// but 4 interleaving points help with load/MMA overlap.
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile64x64x32_F32K8> {
  using ElementInput = cutlass::half_t;
  using TileShape = cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<32>>;
  using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x8_F32F16F16F32_TN>,
                                  cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
                                  cute::Tile<cute::_32, cute::_32, cute::_8>>;
  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16;
  // k=8 needs half-width smem copy atoms to match mma_k
  using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, ElementInput>;
  using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x4_LDSM_T, ElementInput>;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
  static constexpr bool UseCpAsyncGatherA = false;
};

// k=8 MMA with FULL-WIDTH LDSM — for dgrad where both A and B use K-contiguous
// smem layout (SmemLayoutAtomA). The standard SmemLayoutAtomA + SM75_U32x4_LDSM_N
// pair is compatible with the swizzle pattern. CuTe's make_tiled_copy_A adapts
// the copy layout to the k=8 MMA fragment shape automatically.
//
// Why not use half-width LDSM (SM75_U32x2)?
// Half-width ldmatrix.x2 reads 8 bytes from swizzle-designed-for-16-byte addresses,
// producing misaligned smem reads → NaN. Full-width ldmatrix.x4 reads 16 bytes
// which matches the Swizzle<2,3,3> atom perfectly.

// Dgrad k=8 config: Uses k=8 MMA atom with half-width LDSM but a K-contiguous
// smem layout adapted for x2 access. The swizzle is the same as SmemLayoutAtomA_FP16
// but the copy atom matches the k=8 fragment size.
//
// Key: The MMA uses k=8 (SM80_16x8x8), so SmemCopyAtom must load k=8 worth of data.
// SM75_U32x2_LDSM_N loads 2 registers = 4 fp16 elements = half a k=8 fragment.
// Two iterations of the copy atom fill one k=8 MMA fragment.

#define DEFINE_CUTE_TILE_CONFIG_DGRAD_K8(TileTag, M_DIM, N_DIM, K_DIM)                       \
  template <>                                                                                \
  struct CuteTileConfig<cutlass::half_t, gemm::TileTag##_DgradK8> {                          \
    using ElementInput = cutlass::half_t;                                                    \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;     \
    using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x8_F32F16F16F32_TN>,       \
                                    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>, \
                                    cute::Tile<cute::_32, cute::_32, cute::_8>>;             \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP16;                                            \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP16;                                            \
    /* k=8 half-width atoms — match MMA fragment size */                                   \
    using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, ElementInput>;            \
    using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x4_LDSM_T, ElementInput>;            \
    using GmemTiledCopyA = void;                                                             \
    using GmemTiledCopyB = void;                                                             \
    static constexpr int NumStages = 2;                                                      \
    static constexpr int AlignmentA = 4;                                                     \
    static constexpr int AlignmentB = 4;                                                     \
    static constexpr bool UseCpAsyncGatherA = false;                                         \
  };

DEFINE_CUTE_TILE_CONFIG_DGRAD_K8(Tile64x64x32, 64, 64, 32)
DEFINE_CUTE_TILE_CONFIG_DGRAD_K8(Tile64x128x32, 64, 128, 32)
#undef DEFINE_CUTE_TILE_CONFIG_DGRAD_K8

// ============================================================================
// All tile specializations
// ============================================================================

// tK=32 tiles
DEFINE_CUTE_TILE_CONFIG_FP16(Tile64x64x32, 64, 64, 32)
DEFINE_CUTE_TILE_CONFIG_FP16(Tile128x64x32, 128, 64, 32)
DEFINE_CUTE_TILE_CONFIG_FP16(Tile64x128x32, 64, 128, 32)
DEFINE_CUTE_TILE_CONFIG_FP16(Tile128x128x32, 128, 128, 32)

DEFINE_CUTE_TILE_CONFIG_BF16(Tile64x64x32, 64, 64, 32)
DEFINE_CUTE_TILE_CONFIG_BF16(Tile128x64x32, 128, 64, 32)
DEFINE_CUTE_TILE_CONFIG_BF16(Tile64x128x32, 64, 128, 32)
DEFINE_CUTE_TILE_CONFIG_BF16(Tile128x128x32, 128, 128, 32)

// SmemLayoutAtomB for N=32 tiles (smaller atom: 32×8 instead of 64×8)
using SmemLayoutAtomB_FP16_N32 = decltype(cute::composition(
    cute::Swizzle<2, 3, 3>{},
    cute::Layout<cute::Shape<cute::_32, cute::_8>, cute::Stride<cute::_1, cute::_32>>{}));

// 32x32x32 tile with 2-warp (64 thread) layout for small-C configs
// Lower register pressure → higher occupancy → better for C=32
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile32x32x32> {
  using ElementInput = cutlass::half_t;
  using TileShape = cute::Shape<cute::Int<32>, cute::Int<32>, cute::Int<32>>;
  using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
                                  cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>,
                                  cute::Tile<cute::_32, cute::_32, cute::_16>>;
  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16_N32;
  using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, ElementInput>;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
  static constexpr bool UseCpAsyncGatherA = false;
};

// 32x32x32 with fp16 accumulator (2x tensor throughput)
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile32x32x32_F16Accum> {
  using ElementInput = cutlass::half_t;
  using TileShape = cute::Shape<cute::Int<32>, cute::Int<32>, cute::Int<32>>;
  using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x16_F16F16F16F16_TN>,
                                  cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>,
                                  cute::Tile<cute::_32, cute::_32, cute::_16>>;
  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16_N32;
  using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, ElementInput>;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
  static constexpr bool UseCpAsyncGatherA = false;
};

// 32×16×32 tile: tN=16 → 2× blocks for C=32 (better wave scheduling)
// SmemLayoutAtomB needs N=16 compatible atom
using SmemLayoutAtomB_FP16_N16 = decltype(cute::composition(
    cute::Swizzle<2, 3, 3>{},
    cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_1, cute::_16>>{}));

template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile32x16x32_F16Accum> {
  using ElementInput = cutlass::half_t;
  using TileShape = cute::Shape<cute::Int<32>, cute::Int<16>, cute::Int<32>>;
  using TiledMma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x16_F16F16F16F16_TN>,
                                  cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>,
                                  cute::Tile<cute::_32, cute::_16, cute::_16>>;
  using SmemLayoutAtomA = SmemLayoutAtomA_FP16;
  using SmemLayoutAtomB = SmemLayoutAtomB_FP16_N16;
  using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;
  using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, ElementInput>;
  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;
  static constexpr int NumStages = 2;
  static constexpr int AlignmentA = 4;
  static constexpr int AlignmentB = 4;
  static constexpr bool UseCpAsyncGatherA = false;
};

// FP16 accum + k=8 MMA atom tiles (m16n8k8)
DEFINE_CUTE_TILE_CONFIG_FP16_K8(Tile64x64x32, 64, 64, 32)
DEFINE_CUTE_TILE_CONFIG_FP16_K8(Tile64x128x32, 64, 128, 32)

// FP16 accumulator tiles (2x tensor throughput, for fwd/dgrad)
DEFINE_CUTE_TILE_CONFIG_FP16_ACCUM(Tile64x64x32, 64, 64, 32)
DEFINE_CUTE_TILE_CONFIG_FP16_ACCUM(Tile128x64x32, 128, 64, 32)
DEFINE_CUTE_TILE_CONFIG_FP16_ACCUM(Tile64x128x32, 64, 128, 32)

// FP32 input (TF32 tensor cores) — for full-precision training
DEFINE_CUTE_TILE_CONFIG_FP32(Tile64x64x32, 64, 64, 32)
DEFINE_CUTE_TILE_CONFIG_FP32(Tile128x64x32, 128, 64, 32)
DEFINE_CUTE_TILE_CONFIG_FP32(Tile64x128x32, 64, 128, 32)

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
