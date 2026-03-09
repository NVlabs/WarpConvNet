// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CuTe tile configurations for SM90 (Hopper) WGMMA-based GEMM with gather/scatter.
// Each specialization provides TiledMMA and smem layout atoms for a specific
// (ElementType, TileTag) combination using SM90 GMMA instructions.
//
// Key differences from SM80 configs:
// - MMA atoms use WGMMA (warp-group MMA) instructions that operate on
//   128 threads (1 warp group = 4 warps).
// - Both operands are read directly from shared memory (SS variant),
//   so no SmemCopyAtom is needed (set to void).
// - Shared memory layouts use GMMA-compatible 128-byte swizzle patterns
//   (Layout_K_SW128_Atom for K-major A, Layout_MN_SW128_Atom for N-major B).
// - Larger default tiles and more pipeline stages (4) due to SM90's
//   228KB shared memory capacity.
//
// The gmem → smem copy is done manually by the kernel (element-wise
// with gather indices), so no GmemTiledCopy types are defined here.

#pragma once

#if defined(WARPCONVNET_SM90_ENABLED)

#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"     // MUST come first for CUDA 12.9 compat
#include "cute_gemm_config.h"  // CuteTileConfig primary template
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "gemm_mma_tiles.h"  // SM90 tile tag structs

namespace warpconvnet {
namespace cute_gemm {

// ============================================================================
// SM90 GMMA smem layout atoms
// ============================================================================

// SmemLayoutAtom for operand A: K-major (row-major) with 128-byte swizzle.
// GMMA reads A directly from shared memory via descriptors.
// From CUTLASS GMMA::Layout_K_SW128_Atom<half_t>.
using SmemLayoutAtomA_SM90_FP16 = cute::GMMA::Layout_K_SW128_Atom<cutlass::half_t>;
using SmemLayoutAtomA_SM90_BF16 = cute::GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>;

// SmemLayoutAtom for operand B: MN-major (N-contiguous) with 128-byte swizzle.
// GMMA reads B directly from shared memory via descriptors.
// From CUTLASS GMMA::Layout_MN_SW128_Atom<half_t>.
using SmemLayoutAtomB_SM90_FP16 = cute::GMMA::Layout_MN_SW128_Atom<cutlass::half_t>;
using SmemLayoutAtomB_SM90_BF16 = cute::GMMA::Layout_MN_SW128_Atom<cutlass::bfloat16_t>;

// ============================================================================
// Macro for SM90 FP16 tile configs (F32 accumulator, SS variant)
// ============================================================================

#define DEFINE_CUTE_TILE_CONFIG_SM90_FP16(TileTag, M_DIM, N_DIM, K_DIM)                            \
  template <>                                                                                      \
  struct CuteTileConfig<cutlass::half_t, gemm::TileTag> {                                          \
    using ElementInput = cutlass::half_t;                                                          \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;           \
    using TiledMma = cute::TiledMMA<                                                               \
        cute::MMA_Atom<                                                                            \
            cute::SM90_64x##N_DIM##x16_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::MN>>, \
        cute::Layout<cute::Shape<cute::Int<(M_DIM) / 64>, cute::_1, cute::_1>>>;                   \
    using SmemLayoutAtomA = SmemLayoutAtomA_SM90_FP16;                                             \
    using SmemLayoutAtomB = SmemLayoutAtomB_SM90_FP16;                                             \
    using SmemCopyAtomA = void;                                                                    \
    using SmemCopyAtomB = void;                                                                    \
    using GmemTiledCopyA = void;                                                                   \
    using GmemTiledCopyB = void;                                                                   \
    static constexpr int NumStages = 4;                                                            \
    static constexpr int AlignmentA = 8;                                                           \
    static constexpr int AlignmentB = 8;                                                           \
    static constexpr bool UseCpAsyncGatherA = true;                                                \
    static constexpr bool UseTmaLoadB = true;                                                      \
  };

#define DEFINE_CUTE_TILE_CONFIG_SM90_BF16(TileTag, M_DIM, N_DIM, K_DIM)                   \
  template <>                                                                             \
  struct CuteTileConfig<cutlass::bfloat16_t, gemm::TileTag> {                             \
    using ElementInput = cutlass::bfloat16_t;                                             \
    using TileShape = cute::Shape<cute::Int<M_DIM>, cute::Int<N_DIM>, cute::Int<K_DIM>>;  \
    using TiledMma = cute::TiledMMA<                                                      \
        cute::MMA_Atom<cute::SM90_64x##N_DIM##x16_F32BF16BF16_SS<cute::GMMA::Major::K,    \
                                                                 cute::GMMA::Major::MN>>, \
        cute::Layout<cute::Shape<cute::Int<(M_DIM) / 64>, cute::_1, cute::_1>>>;          \
    using SmemLayoutAtomA = SmemLayoutAtomA_SM90_BF16;                                    \
    using SmemLayoutAtomB = SmemLayoutAtomB_SM90_BF16;                                    \
    using SmemCopyAtomA = void;                                                           \
    using SmemCopyAtomB = void;                                                           \
    using GmemTiledCopyA = void;                                                          \
    using GmemTiledCopyB = void;                                                          \
    static constexpr int NumStages = 4;                                                   \
    static constexpr int AlignmentA = 8;                                                  \
    static constexpr int AlignmentB = 8;                                                  \
    static constexpr bool UseCpAsyncGatherA = true;                                       \
    static constexpr bool UseTmaLoadB = true;                                              \
  };

// ============================================================================
// All SM90 tile specializations
// ============================================================================

// SM90_Tile64x64x64: small channels (C_out=32 or 64)
DEFINE_CUTE_TILE_CONFIG_SM90_FP16(SM90_Tile64x64x64, 64, 64, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_BF16(SM90_Tile64x64x64, 64, 64, 64)

// SM90_Tile64x128x64: small problems
DEFINE_CUTE_TILE_CONFIG_SM90_FP16(SM90_Tile64x128x64, 64, 128, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_BF16(SM90_Tile64x128x64, 64, 128, 64)

// SM90_Tile128x128x64: primary workhorse
DEFINE_CUTE_TILE_CONFIG_SM90_FP16(SM90_Tile128x128x64, 128, 128, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_BF16(SM90_Tile128x128x64, 128, 128, 64)

// SM90_Tile128x256x64: large N
DEFINE_CUTE_TILE_CONFIG_SM90_FP16(SM90_Tile128x256x64, 128, 256, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_BF16(SM90_Tile128x256x64, 128, 256, 64)

// SM90_Tile256x128x64: large M, for grouped GEMM
DEFINE_CUTE_TILE_CONFIG_SM90_FP16(SM90_Tile256x128x64, 256, 128, 64)
DEFINE_CUTE_TILE_CONFIG_SM90_BF16(SM90_Tile256x128x64, 256, 128, 64)

#undef DEFINE_CUTE_TILE_CONFIG_SM90_FP16
#undef DEFINE_CUTE_TILE_CONFIG_SM90_BF16

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
