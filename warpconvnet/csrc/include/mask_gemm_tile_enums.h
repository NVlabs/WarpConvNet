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
  _64x64x32_1s_flat = 5,   // experimental
  _64x128x32_1s_flat = 6,  // experimental
  _128x64x32_1s_flat = 7,  // experimental
  _64x64x32_2s_pipelined = 8,
  _64x128x32_2s_pipelined = 9,
  _64x64x32_3s = 10,
  _128x64x32_2s_pipelined = 11,
  _64x64x32_2s_fused = 12,  // experimental
  _64x128x32_2s_fused = 13,
  _128x64x32_2s_fused = 14,  // experimental
  _64x64x32_2s_raked = 15,   // experimental
  _64x128x32_2s_raked = 16,  // experimental
  _64x64x32_1s_flat_F16Accum = 17,
  _64x64x32_2s_fused_F16Accum = 18,
  _64x128x32_2s_fused_F16Accum = 19,
  _64x64x32_1s_ldg = 20,           // experimental
  _64x64x32_1s_ldg_F16Accum = 21,  // experimental
  _64x64x32_1s_flat_F16K8 = 22,
  _64x64x32_2s_fused_F16K8 = 23,  // experimental
  _64x128x32_2s_fused_F16K8 = 24,
  _64x64x32_2s_direct_F16Accum = 25,   // experimental
  _64x128x32_2s_direct_F16Accum = 26,  // experimental
  _32x32x32_1s_flat = 27,              // experimental
  _32x32x32_1s_flat_F16Accum = 28,
  _32x32x32_1s_flat_direpi = 32,
  _32x32x32_1s_flat_direpi_F16Accum = 33,
  _64x64x32_1s_flat_direpi = 34,
  _64x64x32_1s_flat_direpi_F16Accum = 35,
  _64x64x32_2s_bkb = 36,         // experimental
  _64x128x32_2s_bkb = 37,        // experimental
  _64x64x32_2s_fused_bkb = 38,   // experimental
  _64x128x32_2s_fused_bkb = 39,  // experimental
  _64x64x32_1s_flat_sab_se = 40,
  _64x64x32_1s_flat_sa = 41,
  _64x64x32_1s_flat_sb_se = 42,
  _64x64x32_1s_flat_ptpf_F16Accum = 43,  // null-result
  _64x64x32_1s_flat_rawmma = 44,         // experimental
  _64x128x32_1s_flat_rawmma = 45,        // experimental
  _64x64x32_1s_flat_mb2 = 46,            // experimental
  _64x64x32_2s_pipelined_mb2 = 47,       // experimental
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
  _64x64x32_1s_flat_MW2 = 60,         // experimental
  _64x64x32_1s_flat_sab_se_MW2 = 61,  // experimental
  _64x64x32_1s_flat_MW4 = 62,         // experimental
  _64x128x32_2s_warp_spec_pcoff = 63,
  _128x128x32_2s_fused_8warp = 70,                   // experimental
  _128x128x32_2s_fused_8warp_F16Accum = 71,          // experimental
  _128x128x32_2s_fused_persist_8warp = 72,           // experimental
  _128x128x32_2s_fused_persist_8warp_F16Accum = 73,  // experimental
  _64x128x32_2s_fused_persist_F16Accum = 88,         // experimental
  _64x128x32_2s_fused_mb2_F16Accum = 89,             // null-result
  _64x128x32_2s_fused_mb3_F16Accum = 90,             // null-result
  _64x64x32_2s_pipelined_splitk_workspace = 200,     // null-result
  _64x64x32_3s_pipelined_splitk_workspace = 201,     // null-result
  _64x128x32_2s_pipelined_splitk_workspace = 202,    // null-result
  _64x128x32_3s_pipelined_splitk_workspace = 203,    // null-result
  _128x64x32_2s_pipelined_splitk_workspace = 204,    // null-result
  _64x128x32_2s_fused_splitk_workspace = 205,        // null-result
  _64x64x32_2s_pipelined_strided = 300,              // experimental
  _64x64x32_3s_pipelined_strided = 301,              // experimental
  _64x128x32_2s_pipelined_strided = 302,             // experimental
  _64x128x32_3s_pipelined_strided = 303,             // experimental
  _128x64x32_2s_pipelined_strided = 304,             // experimental
  _64x64x32_2s_fused_strided = 305,                  // experimental
  _64x128x32_2s_fused_strided = 306,                 // experimental
  _128x64x32_2s_fused_strided = 307,                 // experimental
  _64x64x32_1s_flat_sa_wt = 400,
  _64x128x32_3s_wt = 401,
  _128x64x32_2s_wt = 402,
  _32x32x32_1s_flat_wt_F16Accum = 403,
  _64x128x32_2s_fused_wt_F16Accum = 404,
  _64x64x32_1s_flat_pcoff_wt_F16Accum = 405,
  _64x64x32_1s_flat_pcoff_wt_F16K8 = 406,
  _64x128x32_1s_flat_pcoff_wt_F16K8 = 407,
  _64x128x32_1s_flat_pcoff_wt_F16Accum = 408,
  _64x64x32_3s_pcoff_wt = 409,
  _64x64x32_2s_warp_spec_pcoff_wt = 410,
  _64x128x32_2s_warp_spec_pcoff_wt = 411,
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
  _64x64x32_2s_bkb = 16,         // experimental
  _64x128x32_2s_bkb = 17,        // experimental
  _64x128x32_3s_bkb = 18,        // experimental
  _64x128x32_4s_bkb = 19,        // experimental
  _64x64x32_2s_fused_bkb = 20,   // experimental
  _64x128x32_2s_fused_bkb = 21,  // experimental
  _64x64x32_1s_flat_F16Accum = 22,
  _64x128x32_1s_flat_F16Accum = 23,
  _64x128x32_2s_F16Accum = 24,
  _64x128x32_1s_flat_direpi = 25,
  _64x128x32_2s_direpi = 26,
  _64x128x32_1s_flat_direpi_F16Accum = 27,
  _64x128x32_1s_flat_direpi_so2_F16Accum = 28,  // experimental
  _64x128x32_1s_flat_direpi_so4_F16Accum = 29,  // experimental
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
  _64x64x32_1s_flat_MW2 = 60,              // experimental
  _64x64x32_1s_flat_sab_se_MW2 = 61,       // experimental
  _64x64x32_1s_flat_MW4 = 62,              // experimental
  _64x64x32_1s_flat_pcoff_F16Accum = 64,   // experimental
  _64x64x32_1s_flat_pcoff_F16K8 = 65,      // experimental
  _64x128x32_1s_flat_pcoff_F16K8 = 66,     // experimental
  _64x128x32_1s_flat_pcoff_F16Accum = 67,  // experimental
  _64x64x32_3s_pcoff = 68,                 // experimental
  _64x128x32_3s_pcoff = 69,                // experimental
  _64x64x32_1s_flat_sa_wt = 900,
  _64x128x32_3s_wt = 901,
  _128x64x32_2s_wt = 902,
  _32x32x32_1s_flat_wt_F16Accum = 903,
  _64x128x32_2s_fused_wt_F16Accum = 904,
  _64x64x32_1s_flat_pcoff_wt_F16Accum = 905,
  _64x64x32_1s_flat_pcoff_wt_F16K8 = 906,
  _64x128x32_1s_flat_pcoff_wt_F16K8 = 907,
  _64x128x32_1s_flat_pcoff_wt_F16Accum = 908,
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
  _64x128x32_2s_f32_atomic_mb2 = 91,  // experimental
  _64x128x32_2s_f32_atomic_mb3 = 92,  // experimental
  _64x64x32_2s_compact_segment_f32_atomic = 100,
  _64x64x32_3s_compact_segment_f32_atomic = 101,
  _64x128x32_2s_compact_segment_f32_atomic = 102,
  _64x128x32_3s_compact_segment_f32_atomic = 103,
  _128x64x32_2s_compact_segment_f32_atomic = 104,
  _128x64x32_3s_compact_segment_f32_atomic = 105,
  _64x64x32_2s_compact_segment_f32_workspace = 106,   // experimental
  _64x64x32_3s_compact_segment_f32_workspace = 107,   // experimental
  _64x128x32_2s_compact_segment_f32_workspace = 108,  // experimental
  _64x128x32_3s_compact_segment_f32_workspace = 109,  // experimental
  _128x64x32_2s_compact_segment_f32_workspace = 110,  // experimental
  _128x64x32_3s_compact_segment_f32_workspace = 111,  // experimental
};

}  // namespace gemm
}  // namespace warpconvnet
