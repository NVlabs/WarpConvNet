// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cutlass/arch/arch.h>
#include <cutlass/numeric_types.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "bindings/register.h"
#include "include/gemm_error_codes.h"

namespace py = pybind11;

// Build-time commit hash injected via -DWARPCONVNET_BUILD_COMMIT="<sha>".
#ifndef WARPCONVNET_BUILD_COMMIT
#define WARPCONVNET_BUILD_COMMIT "unknown"
#endif

PYBIND11_MODULE(_C, m) {
  m.doc() = "CUDA kernels exposed through PyBind11";
  m.attr("__build_commit__") = py::str(WARPCONVNET_BUILD_COMMIT);
  warpconvnet::bindings::register_gemm(m);
  warpconvnet::bindings::register_fma(m);
  warpconvnet::bindings::register_utils(m);
  warpconvnet::bindings::register_sampling(m);
  warpconvnet::bindings::register_coords(m);
  warpconvnet::bindings::register_cuhash(m);
  warpconvnet::bindings::register_mask_gemm(m);
}
