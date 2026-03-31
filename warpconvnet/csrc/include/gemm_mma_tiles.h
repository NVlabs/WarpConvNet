// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace warpconvnet {
namespace gemm {

// -----------------------------------------------------------------------------
//  Tile tag types — used as template parameters to select CTA tile shapes.
//  The naming convention is Tile{M}x{N}x{K}.
// -----------------------------------------------------------------------------

// --- tK=32 tiles (original) ---
struct Tile128x128x32 {};
struct Tile128x64x32 {};
struct Tile64x128x32 {};
struct Tile64x64x32 {};

// --- tK=64 tiles (fewer mainloop iterations for large channels) ---
struct Tile64x64x64 {};
struct Tile128x64x64 {};
struct Tile64x128x64 {};
struct Tile128x128x64 {};

// --- Asymmetric M/N tiles (tK=32) ---
struct Tile256x64x32 {};
struct Tile64x256x32 {};

// --- FP16 accumulator variants (2x tensor throughput vs F32 accum) ---
struct Tile64x64x32_F16Accum {};
struct Tile128x64x32_F16Accum {};
struct Tile64x128x32_F16Accum {};

// --- FP16 accum + k=8 MMA atom (m16n8k8) ---
struct Tile64x64x32_F16K8 {};
struct Tile64x128x32_F16K8 {};

// --- F32 accum + k=8 MMA atom (for wgrad numerics) ---
struct Tile64x64x32_F32K8 {};

// --- k=8 MMA with full-width LDSM (for dgrad K-contiguous B) ---
struct Tile64x64x32_DgradK8 {};
struct Tile64x128x32_DgradK8 {};

// --- Small tiles for 2-warp configs (lower register pressure) ---
struct Tile32x32x32 {};
struct Tile32x32x32_F16Accum {};
struct Tile32x16x32_F16Accum {};

// --- SM90 (Hopper) WGMMA tiles ---
#if defined(WARPCONVNET_SM90_ENABLED)
struct SM90_Tile64x64x64 {};
struct SM90_Tile64x128x64 {};
struct SM90_Tile128x128x64 {};
struct SM90_Tile128x256x64 {};
struct SM90_Tile256x128x64 {};

// --- SM90 FP8 tile tags (WGMMA with K=32 atom, larger CTA-K) ---
// tK=64 tiles (2x WGMMA K=32 atoms)
struct SM90_FP8_Tile64x64x64 {};
struct SM90_FP8_Tile64x128x64 {};
struct SM90_FP8_Tile128x64x64 {};
struct SM90_FP8_Tile128x128x64 {};
// tK=128 tiles (4x WGMMA K=32 atoms)
struct SM90_FP8_Tile64x128x128 {};
struct SM90_FP8_Tile128x128x128 {};
struct SM90_FP8_Tile64x256x128 {};
#endif  // WARPCONVNET_SM90_ENABLED

// Runtime tile selector (integer maps to tile tag via switch)
enum class MMATile : int {
  Tile128x128x32 = 0,
  Tile128x64x32 = 1,
  Tile64x128x32 = 2,
  Tile64x64x32 = 3,
  Tile64x64x64 = 4,
  Tile128x64x64 = 5,
  Tile64x128x64 = 6,
  Tile128x128x64 = 7,
  Tile256x64x32 = 8,
  Tile64x256x32 = 9,
  // Production mask kernels (warp shuffle + precomp rows + double-buffered MMA)
  // Forward
  Prod_Fwd_32x32x32_F16Acc = 40,   // C<=48, fp16 only
  Prod_Fwd_64x64x32 = 41,          // C=64 or C<=48 bf16
  Prod_Fwd_64x128x32_F16Acc = 42,  // C>=128, C_in>=C_out, fp16
  Prod_Fwd_64x128x32_3s = 43,      // C>=128, C_in>=C_out, bf16
  Prod_Fwd_128x64x32 = 44,         // C>=128, C_in<C_out
  // Dgrad
  Prod_Dgrad_32x32x32 = 50,          // C<=48
  Prod_Dgrad_64x64x32 = 51,          // C=64
  Prod_Dgrad_64x128x32 = 52,         // C>=128
  Prod_Dgrad_64x64x32_F16Acc = 53,   // C=64, fp16
  Prod_Dgrad_64x128x32_F16Acc = 54,  // C>=128, fp16
  // Wgrad
  Prod_Wgrad_64x64x32_f32 = 60,  // All configs
  // Scalar variants for unaligned C (no padding needed)
  Prod_Scalar_SAB_SE = 70,     // Both C_in, C_out unaligned (fwd/dgrad)
  Prod_Scalar_SA = 71,         // C_in unaligned, C_out aligned (fwd/dgrad)
  Prod_Scalar_SB_SE = 72,      // C_in aligned, C_out unaligned (fwd/dgrad)
  Prod_Wgrad_Scalar_SAB = 73,  // Wgrad with scalar A+B loads (any C)
#if defined(WARPCONVNET_SM90_ENABLED)
  // SM90 (Hopper) WGMMA tiles
  SM90_Tile64x128x64 = 100,
  SM90_Tile128x128x64 = 101,
  SM90_Tile128x256x64 = 102,
  SM90_Tile256x128x64 = 103,
  SM90_Tile64x64x64 = 104,

  // SM90 FP8 tiles (starting at 200)
  SM90_FP8_Tile64x64x64 = 200,
  SM90_FP8_Tile64x128x64 = 201,
  SM90_FP8_Tile128x64x64 = 202,
  SM90_FP8_Tile128x128x64 = 203,
  SM90_FP8_Tile64x128x128 = 204,
  SM90_FP8_Tile128x128x128 = 205,
  SM90_FP8_Tile64x256x128 = 206,
#endif  // WARPCONVNET_SM90_ENABLED
};

}  // namespace gemm
}  // namespace warpconvnet
