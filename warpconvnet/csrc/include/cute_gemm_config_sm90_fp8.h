// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SM90 FP8 (E4M3 / E5M2) tile configurations for CuTe GEMM with gather/scatter.
//
// FP8 WGMMA atoms on Hopper have shape 64xNx32 (K=32 per atom, same as FP16).
// However, because FP8 elements are 8-bit (half the size of FP16), we can
// double the K dimension of the CTA tile while consuming the same shared
// memory bandwidth, yielding tiles with tK=64 or tK=128.
//
// All configurations accumulate in F32 — FP8 WGMMA always uses F32 accumulators.
//
// Shared-memory layout atoms use CUTLASS's GMMA Layout_K_SW128_Atom which
// provides the correct 128-byte swizzled layout for WGMMA descriptor-based
// access.  The smem copy atom is SM75_U32x4_LDSM_N (non-transposing LDSM)
// because both A and B are K-major when used with GMMA descriptors.

#pragma once

#if defined(WARPCONVNET_SM90_ENABLED)

#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"  // MUST come first for CUDA 12.9 compat
#include "cute_gemm_config.h"
#include "cutlass/float8.h"
#include "cutlass/numeric_types.h"
#include "gemm_mma_tiles.h"

namespace warpconvnet {
namespace cute_gemm {

// ============================================================================
// Shared-memory layout atoms for FP8 operands (8-bit elements)
// ============================================================================

// K-major (row-major) smem layout with 128-byte swizzle for 8-bit types.
// Layout_K_SW128_Atom is defined in cute/atom/mma_traits_sm90_gmma.hpp and
// provides the canonical swizzled layout accepted by GMMA descriptors.
using SmemLayoutAtomA_FP8 = cute::GMMA::Layout_K_SW128_Atom<cutlass::float_e4m3_t>;
using SmemLayoutAtomB_FP8 = cute::GMMA::Layout_K_SW128_Atom<cutlass::float_e4m3_t>;

// E5M2 uses the same element size (8-bit), so the layout atoms are identical.
// We define aliases for clarity when used with E5M2 configs.
using SmemLayoutAtomA_FP8_E5M2 = cute::GMMA::Layout_K_SW128_Atom<cutlass::float_e5m2_t>;
using SmemLayoutAtomB_FP8_E5M2 = cute::GMMA::Layout_K_SW128_Atom<cutlass::float_e5m2_t>;

// ============================================================================
// Macro for FP8 E4M3 tile configs (SM90 WGMMA, F32 accumulator)
// ============================================================================
//
// The WGMMA atom is SM90_64x{N_DIM}x32_F32E4M3E4M3_SS_TN.
// For RS (register-smem) variants, A is in registers and B in smem; for SS
// both operands are in smem.  We use SS_TN here because the gather kernel
// loads A through smem anyway.

#define DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3(TileTag, M_DIM, N_DIM, K_DIM, MMA_N)              \
  template <>                                                                                   \
  struct CuteTileConfig<cutlass::float_e4m3_t, gemm::TileTag> {                                 \
    using ElementInput = cutlass::float_e4m3_t;                                                 \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;        \
    using TiledMma =                                                                            \
        cute::TiledMMA<cute::MMA_Atom<cute::SM90_64x##MMA_N##x32_F32E4M3E4M3_SS_TN<>>,          \
                       cute::Layout<cute::Shape<cute::Int<(M_DIM) / 64>, cute::_1, cute::_1>>>; \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP8;                                                \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP8;                                                \
    using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;               \
    using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;               \
    using GmemTiledCopyA = void;                                                                \
    using GmemTiledCopyB = void;                                                                \
    static constexpr int NumStages = 2;                                                         \
    static constexpr int AlignmentA = 16; /* 16 bytes = 16 FP8 elements */                      \
    static constexpr int AlignmentB = 16;                                                       \
    static constexpr bool UseCpAsyncGatherA = false;                                            \
  };

// ============================================================================
// Macro for FP8 E5M2 tile configs (SM90 WGMMA, F32 accumulator)
// ============================================================================

#define DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2(TileTag, M_DIM, N_DIM, K_DIM, MMA_N)              \
  template <>                                                                                   \
  struct CuteTileConfig<cutlass::float_e5m2_t, gemm::TileTag> {                                 \
    using ElementInput = cutlass::float_e5m2_t;                                                 \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;        \
    using TiledMma =                                                                            \
        cute::TiledMMA<cute::MMA_Atom<cute::SM90_64x##MMA_N##x32_F32E5M2E5M2_SS_TN<>>,          \
                       cute::Layout<cute::Shape<cute::Int<(M_DIM) / 64>, cute::_1, cute::_1>>>; \
    using SmemLayoutAtomA = SmemLayoutAtomA_FP8_E5M2;                                           \
    using SmemLayoutAtomB = SmemLayoutAtomB_FP8_E5M2;                                           \
    using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;               \
    using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementInput>;               \
    using GmemTiledCopyA = void;                                                                \
    using GmemTiledCopyB = void;                                                                \
    static constexpr int NumStages = 2;                                                         \
    static constexpr int AlignmentA = 16;                                                       \
    static constexpr int AlignmentB = 16;                                                       \
    static constexpr bool UseCpAsyncGatherA = false;                                            \
  };

// ============================================================================
// FP8 E4M3 tile specializations
// ============================================================================

// tK=64 tiles (2 WGMMA K=32 atoms along K)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3(SM90_FP8_Tile64x64x64, 64, 64, 64, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3(SM90_FP8_Tile64x128x64, 64, 128, 64, 128)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3(SM90_FP8_Tile128x64x64, 128, 64, 64, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3(SM90_FP8_Tile128x128x64, 128, 128, 64, 128)

// tK=128 tiles (4 WGMMA K=32 atoms along K — high throughput for large channels)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3(SM90_FP8_Tile64x128x128, 64, 128, 128, 128)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3(SM90_FP8_Tile128x128x128, 128, 128, 128, 128)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3(SM90_FP8_Tile64x256x128, 64, 256, 128, 256)

// ============================================================================
// FP8 E5M2 tile specializations
// ============================================================================

// tK=64 tiles
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2(SM90_FP8_Tile64x64x64, 64, 64, 64, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2(SM90_FP8_Tile64x128x64, 64, 128, 64, 128)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2(SM90_FP8_Tile128x64x64, 128, 64, 64, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2(SM90_FP8_Tile128x128x64, 128, 128, 64, 128)

// tK=128 tiles
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2(SM90_FP8_Tile64x128x128, 64, 128, 128, 128)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2(SM90_FP8_Tile128x128x128, 128, 128, 128, 128)
DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2(SM90_FP8_Tile64x256x128, 64, 256, 128, 256)

#undef DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E4M3
#undef DEFINE_CUTE_TILE_CONFIG_SM90_FP8_E5M2

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
