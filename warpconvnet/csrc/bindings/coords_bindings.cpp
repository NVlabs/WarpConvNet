// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pybind11 bindings for coordinate hash table and search kernels.
// Exposes _C.coords submodule.

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// Forward declarations of host wrapper functions from coords_launch.cu
void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity);
void coords_hashmap_insert(torch::Tensor table_kvs, torch::Tensor vector_keys,
                            int num_keys, int key_dim, int capacity, int hash_method);
void coords_hashmap_search(torch::Tensor table_kvs, torch::Tensor vector_keys,
                            torch::Tensor search_keys, torch::Tensor results,
                            int num_search, int key_dim, int capacity, int hash_method);
void coords_hashmap_expand(torch::Tensor table_kvs, torch::Tensor vector_keys,
                            torch::Tensor base_coords, torch::Tensor offsets,
                            int num_base, int num_offsets, int key_dim,
                            int capacity, int vector_capacity,
                            torch::Tensor num_entries_tensor, torch::Tensor status_tensor,
                            int hash_method);
void coords_kernel_map_offset(torch::Tensor table_kvs, torch::Tensor vector_keys,
                               torch::Tensor query_coords, torch::Tensor kernel_offsets,
                               torch::Tensor output,
                               int num_query, int key_dim, int num_offsets, int capacity,
                               int hash_method, int threads_x, int threads_y);
void coords_map_found_indices_to_maps(torch::Tensor found, torch::Tensor mapped,
                                       torch::Tensor offsets, torch::Tensor in_maps,
                                       torch::Tensor out_maps, int K, int M);
void coords_kernel_map_size_4d(torch::Tensor table_kvs, torch::Tensor vector_keys,
                                torch::Tensor query_coords, torch::Tensor kernel_sizes,
                                torch::Tensor output,
                                int num_query, int capacity, int num_kernels,
                                int hash_method, int threads_x, int threads_y);
void coords_morton_code_16bit(torch::Tensor bcoords, int num_points, torch::Tensor result);
void coords_morton_code_20bit(torch::Tensor coords, int num_points, torch::Tensor result);
void coords_find_first_gt_bsearch(torch::Tensor offsets_tensor, int M,
                                    torch::Tensor indices, int N,
                                    torch::Tensor output);
void coords_coord_to_code(torch::Tensor grid_coord, torch::Tensor coord_offset,
                           torch::Tensor min_coord, torch::Tensor window_size,
                           int N, torch::Tensor codes);

namespace warpconvnet {
namespace bindings {

void register_coords(py::module_ &m) {
  py::module_ coords = m.def_submodule("coords", "Coordinate hash table and search operations");

  // --- Hash table operations ---
  coords.def("hashmap_prepare", &coords_hashmap_prepare,
             py::arg("table_kvs"), py::arg("capacity"));
  coords.def("hashmap_insert", &coords_hashmap_insert,
             py::arg("table_kvs"), py::arg("vector_keys"),
             py::arg("num_keys"), py::arg("key_dim"),
             py::arg("capacity"), py::arg("hash_method"));
  coords.def("hashmap_search", &coords_hashmap_search,
             py::arg("table_kvs"), py::arg("vector_keys"),
             py::arg("search_keys"), py::arg("results"),
             py::arg("num_search"), py::arg("key_dim"),
             py::arg("capacity"), py::arg("hash_method"));
  coords.def("hashmap_expand", &coords_hashmap_expand,
             py::arg("table_kvs"), py::arg("vector_keys"),
             py::arg("base_coords"), py::arg("offsets"),
             py::arg("num_base"), py::arg("num_offsets"),
             py::arg("key_dim"), py::arg("capacity"),
             py::arg("vector_capacity"),
             py::arg("num_entries_tensor"), py::arg("status_tensor"),
             py::arg("hash_method"));

  // --- Discrete search operations ---
  coords.def("kernel_map_offset", &coords_kernel_map_offset,
             py::arg("table_kvs"), py::arg("vector_keys"),
             py::arg("query_coords"), py::arg("kernel_offsets"),
             py::arg("output"),
             py::arg("num_query"), py::arg("key_dim"),
             py::arg("num_offsets"), py::arg("capacity"),
             py::arg("hash_method"),
             py::arg("threads_x") = 128, py::arg("threads_y") = 8);
  coords.def("map_found_indices_to_maps", &coords_map_found_indices_to_maps,
             py::arg("found"), py::arg("mapped"),
             py::arg("offsets"), py::arg("in_maps"),
             py::arg("out_maps"), py::arg("K"), py::arg("M"));
  coords.def("kernel_map_size_4d", &coords_kernel_map_size_4d,
             py::arg("table_kvs"), py::arg("vector_keys"),
             py::arg("query_coords"), py::arg("kernel_sizes"),
             py::arg("output"),
             py::arg("num_query"), py::arg("capacity"),
             py::arg("num_kernels"), py::arg("hash_method"),
             py::arg("threads_x") = 128, py::arg("threads_y") = 8);

  // --- Morton code operations ---
  coords.def("morton_code_16bit", &coords_morton_code_16bit,
             py::arg("bcoords"), py::arg("num_points"), py::arg("result"));
  coords.def("morton_code_20bit", &coords_morton_code_20bit,
             py::arg("coords"), py::arg("num_points"), py::arg("result"));

  // --- Binary search ---
  coords.def("find_first_gt_bsearch", &coords_find_first_gt_bsearch,
             py::arg("offsets"), py::arg("M"),
             py::arg("indices"), py::arg("N"),
             py::arg("output"));

  // --- Coord to code ---
  coords.def("coord_to_code", &coords_coord_to_code,
             py::arg("grid_coord"), py::arg("coord_offset"),
             py::arg("min_coord"), py::arg("window_size"),
             py::arg("N"), py::arg("codes"));
}

}  // namespace bindings
}  // namespace warpconvnet
