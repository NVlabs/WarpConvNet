// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace warpconvnet {
namespace gemm {

// -----------------------------------------------------------------------------
// Tile selector enum classes — bond #22 Option 1 SoT.
//
// AUTO-GENERATED from warpgemm registries. Run `python -m warpgemm.codegen`
// or call `write_mask_to(out_dir)` to regenerate. Do NOT edit by hand.
//
// Member names derived from kernel_struct (strip MaskGemm_<op>_ prefix,
// prepend `_`, append tile_tag flavor for F16Accum/F16K8/DgradK8/8warp,
// append _MW2/_MW4 for high-mask-words tiles, append _wt for dgrad-via-fwd
// aliases). Numeric values are the canonical warpgemm tile_ids.
//
// Replaces the legacy hand-maintained ProdFwdTile / ProdDgradTile /
// ProdWgradTile enums in warpconvnet/csrc/include/gemm_mma_tiles.h
// (dropped during bond #22 PR #2).
// -----------------------------------------------------------------------------

enum class FwdTile : int {
  // Forward GEMM tile selectors. Member name derived from kernel_struct.
  _64x64x32_2s = 0,
  _64x128x32_2s = 1,
  _128x64x32_2s = 2,
  _64x128x32_3s = 3,
  _64x128x32_4s = 4,
  _64x64x32_2s_pipelined = 8,
  _64x128x32_2s_pipelined = 9,
  _64x64x32_3s = 10,
  _128x64x32_2s_pipelined = 11,
  _64x128x32_2s_fused = 13,
  _64x64x32_1s_flat_F16Accum = 17,
  _64x64x32_2s_fused_F16Accum = 18,
  _64x128x32_2s_fused_F16Accum = 19,
  _64x64x32_1s_flat_F16K8 = 22,
  _64x128x32_2s_fused_F16K8 = 24,
  _32x32x32_1s_flat_F16Accum = 28,
  _32x32x32_1s_flat_direpi = 32,
  _32x32x32_1s_flat_direpi_F16Accum = 33,
  _64x64x32_1s_flat_direpi = 34,
  _64x64x32_1s_flat_direpi_F16Accum = 35,
  _64x64x32_1s_flat_sab_se = 40,
  _64x64x32_1s_flat_sa = 41,
  _64x64x32_1s_flat_sb_se = 42,
  _64x64x32_2s_warp_spec = 48,
  _64x128x32_2s_warp_spec = 49,
  _64x128x32_1s_flat_sb_se = 50,
  _128x64x32_1s_flat_sb_se = 51,
  _64x64x32_1s_flat_direpi_sb = 52,
  _64x128x32_1s_flat_direpi_sb = 53,
  _64x64x32_1s_flat_pcoff_F16Accum = 54,
  _64x64x32_1s_flat_pcoff_F16K8 = 55,
  _64x128x32_1s_flat_pcoff_F16K8 = 56,
  _64x128x32_1s_flat_pcoff_F16Accum = 57,
  _64x64x32_3s_pcoff = 58,
  _64x64x32_2s_warp_spec_pcoff = 59,
  _64x128x32_2s_warp_spec_pcoff = 63,
};

enum class DgradTile : int {
  // Dgrad tile selectors. Members marked '_wt' are dgrad-via-fwd aliases (bond #22 T4).
  _64x64x32_2s = 0,
  _64x128x32_2s = 1,
  _128x64x32_3s = 2,
  _64x128x32_3s = 3,
  _64x128x32_4s = 4,
  _64x64x32_2s_fused = 5,
  _64x128x32_2s_fused = 6,
  _64x64x32_1s_flat = 7,
  _64x128x32_1s_flat = 8,
  _128x64x32_1s_flat = 9,
  _128x64x32_2s = 10,
  _128x64x32_2s_fused = 11,
  _32x32x32_1s_flat = 12,
  _64x64x32_1s_flat_DgradK8 = 13,
  _64x64x32_2s_DgradK8 = 14,
  _64x128x32_1s_flat_DgradK8 = 15,
  _64x64x32_1s_flat_F16Accum = 22,
  _64x128x32_1s_flat_F16Accum = 23,
  _64x128x32_2s_F16Accum = 24,
  _64x128x32_1s_flat_direpi = 25,
  _64x128x32_2s_direpi = 26,
  _64x128x32_1s_flat_direpi_F16Accum = 27,
  _64x64x32_2s_pipelined = 30,
  _64x128x32_2s_pipelined = 31,
  _128x64x32_2s_pipelined = 32,
  _64x64x32_1s_flat_sab_se = 40,
  _64x64x32_1s_flat_sa = 41,
  _64x64x32_1s_flat_sb_se = 42,
  _64x128x32_1s_flat_sb_se = 50,
  _128x64x32_1s_flat_sb_se = 51,
  _64x64x32_1s_flat_direpi_sb = 52,
  _64x128x32_1s_flat_direpi_sb = 53,
  _64x64x32_1s_flat_sa_wt = 900,
  _64x128x32_3s_wt = 901,
  _128x64x32_2s_wt = 902,
  _32x32x32_1s_flat_F16Accum_wt = 903,
  _64x128x32_2s_fused_F16Accum_wt = 904,
  _64x64x32_1s_flat_pcoff_F16Accum_wt = 905,
  _64x64x32_1s_flat_pcoff_F16K8_wt = 906,
  _64x128x32_1s_flat_pcoff_F16K8_wt = 907,
  _64x128x32_1s_flat_pcoff_F16Accum_wt = 908,
  _64x64x32_3s_pcoff_wt = 909,
  _64x64x32_2s_warp_spec_pcoff_wt = 910,
  _64x128x32_2s_warp_spec_pcoff_wt = 911,
};

enum class WgradTile : int {
  // Wgrad tile selectors. Members ending '_compact_segment_*' consume valid_signal arrays.
  _64x64x32_2s_f32 = 0,
  _64x64x32_2s_f32_workspace = 1,
  _64x64x32_3s_f32_workspace = 2,
  _64x128x32_2s_f32_workspace = 3,
  _64x64x32_2s_f32_atomic = 4,
  _64x64x32_2s_f32_strided_atomic = 5,
  _64x64x32_2s_f32_semaphore = 6,
  _64x128x32_2s_f32_atomic = 7,
  _64x64x32_2s_f32_sab = 8,
  _64x64x32_3s_f32_atomic = 9,
  _64x64x32_2s_compact_segment_f32_atomic = 100,
  _64x64x32_3s_compact_segment_f32_atomic = 101,
  _64x128x32_2s_compact_segment_f32_atomic = 102,
  _64x128x32_3s_compact_segment_f32_atomic = 103,
  _128x64x32_2s_compact_segment_f32_atomic = 104,
  _128x64x32_3s_compact_segment_f32_atomic = 105,
};

}  // namespace gemm
}  // namespace warpconvnet
