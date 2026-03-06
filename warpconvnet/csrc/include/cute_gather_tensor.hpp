// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Thin wrapper that includes the CUTLASS example gather_tensor.hpp and
// re-exports its types into the warpconvnet::cute_gemm namespace.

#pragma once

// Include the original CUTLASS gather tensor utilities — this brings in
// IndexedGather, CustomStride, make_gather_tensor, NoGather, and the
// ComposedLayout upcast specializations into the correct namespaces.
#include "gather_tensor.hpp"

namespace warpconvnet {
namespace cute_gemm {

// Re-export from the `example` namespace used by CUTLASS's gather_tensor.hpp
using example::NoGather;
using example::IndexedGather;
using example::CustomStride;
using example::StridedGather;
using example::make_custom_stride_layout;
using example::make_gather_tensor;

}  // namespace cute_gemm
}  // namespace warpconvnet
