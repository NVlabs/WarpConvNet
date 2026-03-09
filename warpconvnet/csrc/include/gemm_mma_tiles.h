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
