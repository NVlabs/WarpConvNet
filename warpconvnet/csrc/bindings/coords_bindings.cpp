// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pybind11 bindings for coordinate search and utility kernels.
// Exposes _C.coords submodule.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace py = pybind11;

// Forward declarations of host wrapper functions from coords_launch.cu
void coords_morton_code_16bit(torch::Tensor bcoords, int num_points, torch::Tensor result);
void coords_morton_code_20bit(torch::Tensor coords, int num_points, torch::Tensor result);
void coords_find_first_gt_bsearch(
    torch::Tensor offsets_tensor, int M, torch::Tensor indices, int N, torch::Tensor output);
void coords_coord_to_code(torch::Tensor grid_coord,
                          torch::Tensor coord_offset,
                          torch::Tensor min_coord,
                          torch::Tensor window_size,
                          int N,
                          torch::Tensor codes);
void coords_radius_search_count(torch::Tensor points,
                                torch::Tensor queries,
                                torch::Tensor sorted_indices,
                                torch::Tensor cell_starts,
                                torch::Tensor cell_counts,
                                torch::Tensor keys,
                                torch::Tensor values,
                                torch::Tensor result_count,
                                int N,
                                int M,
                                int num_cells,
                                float radius,
                                float cell_size,
                                int capacity);
void coords_radius_search_write(torch::Tensor points,
                                torch::Tensor queries,
                                torch::Tensor sorted_indices,
                                torch::Tensor cell_starts,
                                torch::Tensor cell_counts,
                                torch::Tensor keys,
                                torch::Tensor values,
                                torch::Tensor result_offsets,
                                torch::Tensor result_indices,
                                torch::Tensor result_distances,
                                int N,
                                int M,
                                int num_cells,
                                float radius,
                                float cell_size,
                                int capacity);

// Forward declarations: window grouping (counting sort)
void coords_window_group_histogram(torch::Tensor grid_coord,
                                   torch::Tensor batch_offsets,
                                   torch::Tensor coord_offset,
                                   torch::Tensor min_coord,
                                   torch::Tensor window_size,
                                   torch::Tensor grid_shape,
                                   torch::Tensor codes,
                                   torch::Tensor histogram,
                                   int N,
                                   int B,
                                   int W);
void coords_window_group_scatter(torch::Tensor codes,
                                 torch::Tensor window_offsets_dense,
                                 torch::Tensor scatter_counters,
                                 torch::Tensor perm,
                                 torch::Tensor inverse_perm,
                                 int N);

namespace warpconvnet {
namespace bindings {

void register_coords(py::module_ &m) {
  py::module_ coords = m.def_submodule("coords", "Coordinate search and utility operations");

  // --- Morton code operations ---
  coords.def("morton_code_16bit",
             &coords_morton_code_16bit,
             py::arg("bcoords"),
             py::arg("num_points"),
             py::arg("result"));
  coords.def("morton_code_20bit",
             &coords_morton_code_20bit,
             py::arg("coords"),
             py::arg("num_points"),
             py::arg("result"));

  // --- Binary search ---
  coords.def("find_first_gt_bsearch",
             &coords_find_first_gt_bsearch,
             py::arg("offsets"),
             py::arg("M"),
             py::arg("indices"),
             py::arg("N"),
             py::arg("output"));

  // --- Coord to code ---
  coords.def("coord_to_code",
             &coords_coord_to_code,
             py::arg("grid_coord"),
             py::arg("coord_offset"),
             py::arg("min_coord"),
             py::arg("window_size"),
             py::arg("N"),
             py::arg("codes"));

  // --- Radius search (PackedHashTable / cuhash) ---
  coords.def("radius_search_count",
             &coords_radius_search_count,
             py::arg("points"),
             py::arg("queries"),
             py::arg("sorted_indices"),
             py::arg("cell_starts"),
             py::arg("cell_counts"),
             py::arg("keys"),
             py::arg("values"),
             py::arg("result_count"),
             py::arg("N"),
             py::arg("M"),
             py::arg("num_cells"),
             py::arg("radius"),
             py::arg("cell_size"),
             py::arg("capacity"));
  coords.def("radius_search_write",
             &coords_radius_search_write,
             py::arg("points"),
             py::arg("queries"),
             py::arg("sorted_indices"),
             py::arg("cell_starts"),
             py::arg("cell_counts"),
             py::arg("keys"),
             py::arg("values"),
             py::arg("result_offsets"),
             py::arg("result_indices"),
             py::arg("result_distances"),
             py::arg("N"),
             py::arg("M"),
             py::arg("num_cells"),
             py::arg("radius"),
             py::arg("cell_size"),
             py::arg("capacity"));

  // --- Window grouping (counting sort) ---
  coords.def("window_group_histogram",
             &coords_window_group_histogram,
             py::arg("grid_coord"),
             py::arg("batch_offsets"),
             py::arg("coord_offset"),
             py::arg("min_coord"),
             py::arg("window_size"),
             py::arg("grid_shape"),
             py::arg("codes"),
             py::arg("histogram"),
             py::arg("N"),
             py::arg("B"),
             py::arg("W"));
  coords.def("window_group_scatter",
             &coords_window_group_scatter,
             py::arg("codes"),
             py::arg("window_offsets_dense"),
             py::arg("scatter_counters"),
             py::arg("perm"),
             py::arg("inverse_perm"),
             py::arg("N"));
}

}  // namespace bindings
}  // namespace warpconvnet
