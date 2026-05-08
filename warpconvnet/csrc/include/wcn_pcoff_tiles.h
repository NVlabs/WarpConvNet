// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// wcn-only Pcoff (E1 offset-precompute) tile tags + CuteTileConfig
// specializations. Pcoff_* variants are warpconvnet-side experimental
// kernels (warpgemm tile_ids 54-59, 63) that compose the E1 axis
// (offset-index precompute) on top of canonical fwd configs.
//
// The base tile_tags (Tile64x64x32_F16Accum, Tile64x64x32_F16K8, etc.)
// live in csrc/include/{gemm_mma_tiles.h, cute_gemm_config.h} (committed
// canonical snapshot, optionally regenerated from warpgemm via
// WARPGEMM_REGEN=1). Pcoff_* tags declared here are wcn-only and inherit
// their CuteTileConfig from a matching canonical base.

#pragma once

// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
// clang-format on
#include "cute_gemm_config.h"  // canonical (snapshot in csrc/include/)
#include "gemm_mma_tiles.h"    // canonical (snapshot in csrc/include/)

namespace warpconvnet {
namespace gemm {

// --- Pcoff (E1 offset-index precompute) variants — wcn-only.
// Each mirrors warpgemm fwd tile IDs 54-59, 63. Config inherits from a
// matching canonical base tile.
struct Tile64x64x32_Pcoff {};      // tile 54: F16Accum base
struct Tile64x64x32_Pcoff_K8 {};   // tile 55: F16K8 base
struct Tile64x128x32_Pcoff_K8 {};  // tile 56: F16K8 base (A100 flagship)
struct Tile64x128x32_Pcoff {};     // tile 57: F16Accum base
struct Tile64x64x32_Pcoff_3s {};   // tile 58: F32 accum, 3-stage override
struct Tile64x64x32_Pcoff_WS {};   // tile 59: F32 accum, warp-spec
struct Tile64x128x32_Pcoff_WS {};  // tile 63: F32 accum, warp-spec 64x128
struct Tile64x128x32_Pcoff_3s {};  // dgrad tile 69: F32 accum, 3-stage 64x128

}  // namespace gemm

namespace cute_gemm {

// -----------------------------------------------------------------------------
// Pcoff tile Config specializations — wcn-only. Each Pcoff TileTag inherits
// from the matching canonical base config. Kernel class carries the _pcoff
// suffix (offset-precompute variant).
// -----------------------------------------------------------------------------

// tile 54: flat_pcoff + F16Accum base
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile64x64x32_Pcoff>
    : CuteTileConfig<cutlass::half_t, gemm::Tile64x64x32_F16Accum> {};

// tile 55: flat_pcoff + F16K8 base
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile64x64x32_Pcoff_K8>
    : CuteTileConfig<cutlass::half_t, gemm::Tile64x64x32_F16K8> {};

// tile 56: flat_pcoff + F16K8 base (64x128, A100 flagship)
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile64x128x32_Pcoff_K8>
    : CuteTileConfig<cutlass::half_t, gemm::Tile64x128x32_F16K8> {};

// tile 57: flat_pcoff + F16Accum base (64x128)
template <>
struct CuteTileConfig<cutlass::half_t, gemm::Tile64x128x32_Pcoff>
    : CuteTileConfig<cutlass::half_t, gemm::Tile64x128x32_F16Accum> {};

// tile 58: 3s_pcoff — F32 accum base with NumStages=3 override
template <typename ElemIn>
struct CuteTileConfig<ElemIn, gemm::Tile64x64x32_Pcoff_3s>
    : CuteTileConfigOverride<CuteTileConfig<ElemIn, gemm::Tile64x64x32>, 3, false> {};

// tile 59: 2s_warp_spec_pcoff — F32 accum base
template <typename ElemIn>
struct CuteTileConfig<ElemIn, gemm::Tile64x64x32_Pcoff_WS>
    : CuteTileConfig<ElemIn, gemm::Tile64x64x32> {};

// tile 63: 2s_warp_spec_pcoff — F32 accum base (64x128)
template <typename ElemIn>
struct CuteTileConfig<ElemIn, gemm::Tile64x128x32_Pcoff_WS>
    : CuteTileConfig<ElemIn, gemm::Tile64x128x32> {};

// dgrad tile 69: 3s_pcoff — F32 accum base (64x128) with NumStages=3 override
template <typename ElemIn>
struct CuteTileConfig<ElemIn, gemm::Tile64x128x32_Pcoff_3s>
    : CuteTileConfigOverride<CuteTileConfig<ElemIn, gemm::Tile64x128x32>, 3, false> {};

}  // namespace cute_gemm
}  // namespace warpconvnet
