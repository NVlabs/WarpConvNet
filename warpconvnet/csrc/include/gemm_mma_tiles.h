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

// Runtime tile selector (integer maps to tile tag via switch)
enum class MMATile : int {
  Tile128x128x32 = 0,
  Tile128x64x32  = 1,
  Tile64x128x32  = 2,
  Tile64x64x32   = 3,
  Tile64x64x64   = 4,
  Tile128x64x64  = 5,
  Tile64x128x64  = 6,
  Tile128x128x64 = 7,
  Tile256x64x32  = 8,
  Tile64x256x32  = 9,
};

}  // namespace gemm
}  // namespace warpconvnet
