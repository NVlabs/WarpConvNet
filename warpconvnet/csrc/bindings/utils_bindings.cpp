// Copyright 2025 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0

#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// Implementations of utility kernels (from .cu)
py::object cub_segmented_sort(const torch::Tensor &keys,
                              const torch::Tensor &segment_offsets,
                              const py::object &values,
                              bool descending,
                              bool return_indices);

torch::Tensor points_to_closest_voxel_mapping(torch::Tensor points,
                                              torch::Tensor offsets,
                                              torch::Tensor grid_shape,
                                              torch::Tensor bounds_min,
                                              torch::Tensor bounds_max);

namespace warpconvnet {
namespace segmented_arithmetic {
template <typename ElementB, typename ElementC, typename ElementD>
int run_segmented_arithmetic_templated(const void *tensor_b,
                                       const void *tensor_c,
                                       void *tensor_d,
                                       const int *offsets,
                                       int N,
                                       int C,
                                       int K,
                                       const std::string &operation,
                                       const std::string &kernel_type);
}  // namespace segmented_arithmetic
}  // namespace warpconvnet

static void segmented_arithmetic_cuda(torch::Tensor tensor_b,
                                      torch::Tensor tensor_c,
                                      torch::Tensor tensor_d,
                                      torch::Tensor offsets,
                                      const std::string &operation,
                                      const std::string &kernel_type) {
  // Validate input tensors
  TORCH_CHECK(tensor_b.dim() == 2, "tensor_b must be 2D");
  TORCH_CHECK(tensor_c.dim() == 2, "tensor_c must be 2D");
  TORCH_CHECK(tensor_d.dim() == 2, "tensor_d must be 2D");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32 || offsets.scalar_type() == torch::kInt64,
              "offsets must be int32 or int64");
  TORCH_CHECK(tensor_b.scalar_type() == tensor_c.scalar_type(),
              "tensor_b and tensor_c must have the same type");
  TORCH_CHECK(tensor_b.scalar_type() == tensor_d.scalar_type(),
              "tensor_b and tensor_d must have the same type");

  // Validate dimensions
  int N = tensor_b.size(0);
  int C = tensor_b.size(1);
  int K = tensor_c.size(0);
  TORCH_CHECK(tensor_c.size(1) == C, "tensor_c.size(1) must be match C");
  TORCH_CHECK(tensor_d.size(0) == N && tensor_d.size(1) == C,
              "tensor_d.size(0) and tensor_d.size(1) must be match N and C");
  TORCH_CHECK(offsets.size(0) == K + 1, "offsets.size(0) must be match K + 1");
  TORCH_CHECK(tensor_b.is_cuda() && tensor_c.is_cuda() && tensor_d.is_cuda() && offsets.is_cuda(),
              "tensor_b, tensor_c, tensor_d, and offsets must be on GPU");

  // Contiguous tensors
  tensor_b = tensor_b.contiguous();
  tensor_c = tensor_c.contiguous();
  tensor_d = tensor_d.contiguous();
  offsets = offsets.contiguous().to(torch::kInt32);

  // Dispatch based on tensor types
  int status = 0;
  if (tensor_b.scalar_type() == torch::kFloat32) {
    status =
        warpconvnet::segmented_arithmetic::run_segmented_arithmetic_templated<float, float, float>(
            tensor_b.data_ptr(),
            tensor_c.data_ptr(),
            tensor_d.data_ptr(),
            offsets.data_ptr<int>(),
            N,
            C,
            K,
            operation,
            kernel_type);
  } else if (tensor_b.scalar_type() == torch::kFloat16) {
    status = warpconvnet::segmented_arithmetic::
        run_segmented_arithmetic_templated<cutlass::half_t, cutlass::half_t, cutlass::half_t>(
            tensor_b.data_ptr(),
            tensor_c.data_ptr(),
            tensor_d.data_ptr(),
            offsets.data_ptr<int>(),
            N,
            C,
            K,
            operation,
            kernel_type);
  } else if (tensor_b.scalar_type() == torch::kBFloat16) {
    status =
        warpconvnet::segmented_arithmetic::run_segmented_arithmetic_templated<cutlass::bfloat16_t,
                                                                              cutlass::bfloat16_t,
                                                                              cutlass::bfloat16_t>(
            tensor_b.data_ptr(),
            tensor_c.data_ptr(),
            tensor_d.data_ptr(),
            offsets.data_ptr<int>(),
            N,
            C,
            K,
            operation,
            kernel_type);
  } else if (tensor_b.scalar_type() == torch::kFloat64) {
    status = warpconvnet::segmented_arithmetic::
        run_segmented_arithmetic_templated<double, double, double>(tensor_b.data_ptr(),
                                                                   tensor_c.data_ptr(),
                                                                   tensor_d.data_ptr(),
                                                                   offsets.data_ptr<int>(),
                                                                   N,
                                                                   C,
                                                                   K,
                                                                   operation,
                                                                   kernel_type);
  } else {
    TORCH_CHECK(false, "Unsupported data type for segmented arithmetic");
  }
  if (status != 0) {
    TORCH_CHECK(false, "Segmented arithmetic kernel failed with status: " + std::to_string(status));
  }
}

namespace warpconvnet {
namespace bindings {

void register_utils(py::module_ &m) {
  py::module_ utils = m.def_submodule("utils", "Utility functions including sorting operations");

  utils.def("segmented_sort",
            &cub_segmented_sort,
            py::arg("keys"),
            py::arg("segment_offsets"),
            py::arg("values") = py::none(),
            py::arg("descending") = false,
            py::arg("return_indices") = false);

  utils.def("points_to_closest_voxel_mapping",
            &points_to_closest_voxel_mapping,
            py::arg("points"),
            py::arg("offsets"),
            py::arg("grid_shape"),
            py::arg("bounds_min"),
            py::arg("bounds_max"));

  utils.def("segmented_arithmetic",
            &segmented_arithmetic_cuda,
            py::arg("tensor_b"),
            py::arg("tensor_c"),
            py::arg("tensor_d"),
            py::arg("offsets"),
            py::arg("operation"),
            py::arg("kernel_type") = "basic");
}

}  // namespace bindings
}  // namespace warpconvnet
