// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace warpconvnet {
namespace bindings {

// Each registration function should add a submodule to the given parent module
// and register all of its bindings.
void register_gemm(pybind11::module_ &m);
void register_fma(pybind11::module_ &m);
void register_utils(pybind11::module_ &m);

}  // namespace bindings
}  // namespace warpconvnet
