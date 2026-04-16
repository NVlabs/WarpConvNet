// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// nvtx_range.cuh - RAII NVTX range scope helper.
//
// Usage:
//   void launch_foo(...) {
//     CUHASH_NVTX_SCOPE("launch_foo");
//     ...
//   }
//
// When building without -DCUHASH_USE_NVTX the macro expands to a no-op, so
// there is zero runtime cost. Opt in by passing `-DCUHASH_USE_NVTX` through
// CMake / setup.py `extra_compile_args`, and link against `nvToolsExt` if
// your CUDA version requires it (CUDA 12+ uses header-only nvtx3).
#pragma once

#ifdef CUHASH_USE_NVTX
#include <nvtx3/nvToolsExt.h>
namespace cuhash {
struct NvtxScope {
  NvtxScope(const char *name) { nvtxRangePushA(name); }
  ~NvtxScope() { nvtxRangePop(); }
};
}  // namespace cuhash
#define CUHASH_NVTX_SCOPE(name) ::cuhash::NvtxScope _cuhash_nvtx_scope_(name)
#else
#define CUHASH_NVTX_SCOPE(name) ((void)0)
#endif
