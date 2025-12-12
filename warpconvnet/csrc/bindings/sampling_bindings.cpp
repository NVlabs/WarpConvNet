// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace warpconvnet {

void farthest_point_sampling_cuda(
    at::Tensor points, at::Tensor offsets, at::Tensor temp, at::Tensor idxs, int K);

namespace bindings {

void register_sampling(pybind11::module_ &m) {
  pybind11::module_ sampling = m.def_submodule("sampling", "Sampling operations");

  sampling.def("farthest_point_sampling",
               &farthest_point_sampling_cuda,
               "Farthest Point Sampling (CUDA)",
               pybind11::arg("points"),
               pybind11::arg("offsets"),
               pybind11::arg("temp"),
               pybind11::arg("idxs"),
               pybind11::arg("K"));
}

}  // namespace bindings
}  // namespace warpconvnet
