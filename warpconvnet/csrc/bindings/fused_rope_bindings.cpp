// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace warpconvnet {
namespace fused_rope {

void run_fused_rope_qkv(const at::Tensor& qkv,
                        const at::Tensor& coords,
                        const at::Tensor& theta,
                        at::Tensor& out,
                        int num_heads,
                        int rope_dim,
                        int conjugate);

}  // namespace fused_rope

namespace bindings {

static void fused_rope_qkv_cuda(at::Tensor qkv,
                                at::Tensor coords,
                                at::Tensor theta,
                                at::Tensor out,
                                int64_t num_heads,
                                int64_t rope_dim,
                                int64_t conjugate) {
  qkv = qkv.contiguous();
  coords = coords.contiguous();
  theta = theta.contiguous();
  out = out.contiguous();
  warpconvnet::fused_rope::run_fused_rope_qkv(qkv,
                                              coords,
                                              theta,
                                              out,
                                              static_cast<int>(num_heads),
                                              static_cast<int>(rope_dim),
                                              static_cast<int>(conjugate));
}

void register_fused_rope(py::module_& m) {
  py::module_ fr = m.def_submodule("fused_rope", "Fused RoPE + QKV reshape kernels");
  fr.def("qkv",
         &fused_rope_qkv_cuda,
         py::arg("qkv"),
         py::arg("coords"),
         py::arg("theta"),
         py::arg("out"),
         py::arg("num_heads"),
         py::arg("rope_dim"),
         py::arg("conjugate") = 0);
}

}  // namespace bindings
}  // namespace warpconvnet
