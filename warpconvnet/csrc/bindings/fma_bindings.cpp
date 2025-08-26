// Copyright 2025 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <sstream>

namespace py = pybind11;

// Kernel entry points implemented in .cu files (correct namespaces)
namespace warpconvnet {
namespace implicit_fma {
template <typename ElementA, typename ElementB, typename ElementC>
int run_implicit_fma_templated(const void *tensor_a,
                               const void *tensor_b,
                               void *tensor_c,
                               const int *in_indices,
                               const int *out_indices,
                               int num_ops,
                               int C,
                               int N_A,
                               int N_C,
                               const std::string &kernel_type);
}  // namespace implicit_fma
namespace implicit_reduction {
template <typename ElementA, typename ElementB, typename ElementOutput>
int run_implicit_reduction_templated(const void *tensor_a,
                                     const int *a_indices,
                                     const void *tensor_b,
                                     const int *b_indices,
                                     void *result,
                                     int M,
                                     int C,
                                     int N_A,
                                     int N_B,
                                     const std::string &kernel_type);
}  // namespace implicit_reduction
}  // namespace warpconvnet

namespace warpconvnet {
namespace bindings {

// Implementations moved from warpconvnet_pybind.cpp
static void implicit_fma_cuda(torch::Tensor a,
                              torch::Tensor b,
                              torch::Tensor c,
                              torch::Tensor in_indices,
                              torch::Tensor out_indices,
                              const std::string &kernel_type) {
  // Validate dimensions and types
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(b.dim() == 1, "b must be 1D");
  TORCH_CHECK(c.dim() == 2, "c must be 2D");
  TORCH_CHECK(in_indices.dim() == 1, "in_indices must be 1D");
  TORCH_CHECK(out_indices.dim() == 1, "out_indices must be 1D");
  TORCH_CHECK(in_indices.scalar_type() == torch::kInt32, "in_indices must be int32");
  TORCH_CHECK(out_indices.scalar_type() == torch::kInt32, "out_indices must be int32");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same type");
  TORCH_CHECK(a.scalar_type() == c.scalar_type(), "a and c must have the same type");

  // Validate dimensions
  int N_A = a.size(0);
  int C_dim = a.size(1);
  int N_C = c.size(0);
  int num_ops = in_indices.size(0);
  TORCH_CHECK(b.size(0) == C_dim, "b.size(0) must be match a.size(1)");
  TORCH_CHECK(c.size(1) == C_dim, "c.size(1) must be match a.size(1)");
  TORCH_CHECK(out_indices.size(0) == num_ops,
              "in_indices and out_indices must have the same number of rows");
  TORCH_CHECK(
      a.is_cuda() && b.is_cuda() && c.is_cuda() && in_indices.is_cuda() && out_indices.is_cuda(),
      "a, b, c, in_indices, and out_indices must be on GPU");

  // Contiguous tensors
  a = a.contiguous();
  b = b.contiguous();
  c = c.contiguous();
  in_indices = in_indices.contiguous();
  out_indices = out_indices.contiguous();

  // Dispatch based on tensor types
  int status = 0;
  if (a.scalar_type() == torch::kFloat32) {
    status = warpconvnet::implicit_fma::run_implicit_fma_templated<float, float, float>(
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        in_indices.data_ptr<int>(),
        out_indices.data_ptr<int>(),
        num_ops,
        C_dim,
        N_A,
        N_C,
        kernel_type);
  } else if (a.scalar_type() == torch::kFloat16) {
    status = warpconvnet::implicit_fma::
        run_implicit_fma_templated<cutlass::half_t, cutlass::half_t, cutlass::half_t>(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            in_indices.data_ptr<int>(),
            out_indices.data_ptr<int>(),
            num_ops,
            C_dim,
            N_A,
            N_C,
            kernel_type);
  } else if (a.scalar_type() == torch::kBFloat16) {
    status = warpconvnet::implicit_fma::
        run_implicit_fma_templated<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t>(
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            in_indices.data_ptr<int>(),
            out_indices.data_ptr<int>(),
            num_ops,
            C_dim,
            N_A,
            N_C,
            kernel_type);
  } else if (a.scalar_type() == torch::kFloat64) {
    status = warpconvnet::implicit_fma::run_implicit_fma_templated<double, double, double>(
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        in_indices.data_ptr<int>(),
        out_indices.data_ptr<int>(),
        num_ops,
        C_dim,
        N_A,
        N_C,
        kernel_type);
  } else {
    TORCH_CHECK(false, "Unsupported data type for implicit FMA");
  }
  if (status != 0) {
    TORCH_CHECK(false, "Implicit FMA kernel failed with status: " + std::to_string(status));
  }
}

static void implicit_reduction_cuda(torch::Tensor a,
                                    torch::Tensor a_indices,
                                    const py::object &b,
                                    const py::object &b_indices,
                                    torch::Tensor result,
                                    const std::string &kernel_type) {
  // Validate dimensions and types
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(a_indices.dim() == 1, "a_indices must be 1D");
  TORCH_CHECK(result.dim() == 1, "result must be 1D");
  TORCH_CHECK(a_indices.scalar_type() == torch::kInt32, "a_indices must be int32");

  // Validate dimensions
  int N_A = a.size(0);
  int C_dim = a.size(1);
  int M = a_indices.size(0);
  TORCH_CHECK(result.size(0) == C_dim, "result.size(0) must be match a.size(1)");

  // Handle optional b and b_indices
  int N_B = 0;
  bool has_b = !b.is_none();

  torch::Tensor b_tensor;
  torch::Tensor b_indices_tensor;

  if (has_b) {
    b_tensor = b.cast<torch::Tensor>();
    b_indices_tensor = b_indices.cast<torch::Tensor>();
    TORCH_CHECK(b_tensor.dim() == 2, "b must be 2D");
    TORCH_CHECK(b_indices_tensor.dim() == 1, "b_indices must be 1D");
    TORCH_CHECK(b_indices_tensor.scalar_type() == torch::kInt32, "b_indices must be int32");
    TORCH_CHECK(b_indices_tensor.size(0) == M, "b_indices.size(0) must be match a_indices.size(0)");
    TORCH_CHECK(b_tensor.size(1) == C_dim, "b.size(1) must be match a.size(1)");
    TORCH_CHECK(a.scalar_type() == b_tensor.scalar_type(), "a and b must have the same type");
    N_B = b_tensor.size(0);
  }
  TORCH_CHECK(a.scalar_type() == result.scalar_type(), "a and result must have the same type");
  TORCH_CHECK(a.is_cuda() && a_indices.is_cuda() && result.is_cuda(),
              "a, a_indices, and result must be on GPU");

  // Contiguous tensors
  a = a.contiguous();
  a_indices = a_indices.contiguous();
  result = result.contiguous();

  if (has_b) {
    TORCH_CHECK(b_tensor.is_cuda() && b_indices_tensor.is_cuda(), "b and b_indices must be on GPU");
    b_tensor = b_tensor.contiguous();
    b_indices_tensor = b_indices_tensor.contiguous();
  }

  // Dispatch based on tensor types
  int status = 0;
  if (a.scalar_type() == torch::kFloat32) {
    status = warpconvnet::implicit_reduction::run_implicit_reduction_templated<float, float, float>(
        a.data_ptr(),
        a_indices.data_ptr<int>(),
        has_b ? b_tensor.data_ptr() : nullptr,
        has_b ? b_indices_tensor.data_ptr<int>() : nullptr,
        result.data_ptr(),
        M,
        C_dim,
        N_A,
        N_B,
        kernel_type);
  } else if (a.scalar_type() == torch::kFloat16) {
    status = warpconvnet::implicit_reduction::
        run_implicit_reduction_templated<cutlass::half_t, cutlass::half_t, cutlass::half_t>(
            a.data_ptr(),
            a_indices.data_ptr<int>(),
            has_b ? b_tensor.data_ptr() : nullptr,
            has_b ? b_indices_tensor.data_ptr<int>() : nullptr,
            result.data_ptr(),
            M,
            C_dim,
            N_A,
            N_B,
            kernel_type);
  } else if (a.scalar_type() == torch::kBFloat16) {
    status = warpconvnet::implicit_reduction::run_implicit_reduction_templated<cutlass::bfloat16_t,
                                                                               cutlass::bfloat16_t,
                                                                               cutlass::bfloat16_t>(
        a.data_ptr(),
        a_indices.data_ptr<int>(),
        has_b ? b_tensor.data_ptr() : nullptr,
        has_b ? b_indices_tensor.data_ptr<int>() : nullptr,
        result.data_ptr(),
        M,
        C_dim,
        N_A,
        N_B,
        kernel_type);
  } else if (a.scalar_type() == torch::kFloat64) {
    status =
        warpconvnet::implicit_reduction::run_implicit_reduction_templated<double, double, double>(
            a.data_ptr(),
            a_indices.data_ptr<int>(),
            has_b ? b_tensor.data_ptr() : nullptr,
            has_b ? b_indices_tensor.data_ptr<int>() : nullptr,
            result.data_ptr(),
            M,
            C_dim,
            N_A,
            N_B,
            kernel_type);
  } else {
    TORCH_CHECK(false, "Unsupported data type for implicit reduction");
  }
  if (status != 0) {
    TORCH_CHECK(false, "Implicit reduction kernel failed with status: " + std::to_string(status));
  }
}

void register_fma(py::module_ &m) {
  py::module_ fma =
      m.def_submodule("fma", "Fused Multiply-Add operations and related functionality");

  fma.def("implicit_fma",
          &implicit_fma_cuda,
          py::arg("a"),
          py::arg("b"),
          py::arg("c"),
          py::arg("in_indices"),
          py::arg("out_indices"),
          py::arg("kernel_type") = "basic");

  fma.def("implicit_reduction",
          &implicit_reduction_cuda,
          py::arg("a"),
          py::arg("a_indices"),
          py::arg("b") = py::none(),
          py::arg("b_indices") = py::none(),
          py::arg("result"),
          py::arg("kernel_type") = "basic");
}

}  // namespace bindings
}  // namespace warpconvnet
