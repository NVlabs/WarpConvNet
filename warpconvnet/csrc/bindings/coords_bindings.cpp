// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pybind11 bindings for coordinate hash table and search kernels.
// Exposes _C.coords submodule.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <tuple>
#include <vector>

namespace py = pybind11;

// Forward declarations of host wrapper functions from coords_launch.cu
void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity);
void coords_hashmap_insert(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           int num_keys,
                           int key_dim,
                           int capacity,
                           int hash_method);
void coords_hashmap_search(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           torch::Tensor search_keys,
                           torch::Tensor results,
                           int num_search,
                           int key_dim,
                           int capacity,
                           int hash_method);
void coords_hashmap_warp_search(torch::Tensor table_kvs,
                                torch::Tensor vector_keys,
                                torch::Tensor search_keys,
                                torch::Tensor results,
                                int num_search,
                                int key_dim,
                                int capacity,
                                int hash_method);
void coords_hashmap_expand(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           torch::Tensor base_coords,
                           torch::Tensor offsets,
                           int num_base,
                           int num_offsets,
                           int key_dim,
                           int capacity,
                           int vector_capacity,
                           torch::Tensor num_entries_tensor,
                           torch::Tensor status_tensor,
                           int hash_method);
void coords_kernel_map_offset(torch::Tensor table_kvs,
                              torch::Tensor vector_keys,
                              torch::Tensor query_coords,
                              torch::Tensor kernel_offsets,
                              torch::Tensor output,
                              int num_query,
                              int key_dim,
                              int num_offsets,
                              int capacity,
                              int hash_method,
                              int threads_x,
                              int threads_y);
void coords_map_found_indices_to_maps(torch::Tensor found,
                                      torch::Tensor mapped,
                                      torch::Tensor offsets,
                                      torch::Tensor in_maps,
                                      torch::Tensor out_maps,
                                      int K,
                                      int M);
void coords_kernel_map_size_4d(torch::Tensor table_kvs,
                               torch::Tensor vector_keys,
                               torch::Tensor query_coords,
                               torch::Tensor kernel_sizes,
                               torch::Tensor output,
                               int num_query,
                               int capacity,
                               int num_kernels,
                               int hash_method,
                               int threads_x,
                               int threads_y);
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
void coords_kernel_map_size_4d_count(torch::Tensor table_kvs,
                                     torch::Tensor vector_keys,
                                     torch::Tensor query_coords,
                                     torch::Tensor kernel_sizes,
                                     torch::Tensor counts,
                                     int num_query,
                                     int capacity,
                                     int num_kernels,
                                     int hash_method,
                                     int threads_x,
                                     int threads_y);
void coords_kernel_map_size_4d_scatter(torch::Tensor table_kvs,
                                       torch::Tensor vector_keys,
                                       torch::Tensor query_coords,
                                       torch::Tensor kernel_sizes,
                                       torch::Tensor offsets,
                                       torch::Tensor scatter_counters,
                                       torch::Tensor in_maps,
                                       torch::Tensor out_maps,
                                       int num_query,
                                       int capacity,
                                       int num_kernels,
                                       int hash_method,
                                       int threads_x,
                                       int threads_y);
void coords_postprocess_count(torch::Tensor found, torch::Tensor counts, int K, int M);
void coords_postprocess_scatter(torch::Tensor found,
                                torch::Tensor offsets,
                                torch::Tensor scatter_counters,
                                torch::Tensor in_maps,
                                torch::Tensor out_maps,
                                int K,
                                int M);
void coords_radius_search_count(torch::Tensor points,
                                torch::Tensor queries,
                                torch::Tensor sorted_indices,
                                torch::Tensor cell_starts,
                                torch::Tensor cell_counts,
                                torch::Tensor table_kvs,
                                torch::Tensor vector_keys,
                                torch::Tensor result_count,
                                int N,
                                int M,
                                int num_cells,
                                float radius,
                                float cell_size,
                                int table_capacity,
                                int hash_method);
void coords_radius_search_write(torch::Tensor points,
                                torch::Tensor queries,
                                torch::Tensor sorted_indices,
                                torch::Tensor cell_starts,
                                torch::Tensor cell_counts,
                                torch::Tensor table_kvs,
                                torch::Tensor vector_keys,
                                torch::Tensor result_offsets,
                                torch::Tensor result_indices,
                                torch::Tensor result_distances,
                                int N,
                                int M,
                                int num_cells,
                                float radius,
                                float cell_size,
                                int table_capacity,
                                int hash_method);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> launch_fused_kernel_map(
    torch::Tensor output_coords,
    torch::Tensor table_kvs,
    torch::Tensor vector_keys,
    int table_capacity,
    std::vector<int> kernel_size,
    int hash_method);

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
  py::module_ coords = m.def_submodule("coords", "Coordinate hash table and search operations");

  // --- Hash table operations ---
  coords.def("hashmap_prepare", &coords_hashmap_prepare, py::arg("table_kvs"), py::arg("capacity"));
  coords.def("hashmap_insert",
             &coords_hashmap_insert,
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("num_keys"),
             py::arg("key_dim"),
             py::arg("capacity"),
             py::arg("hash_method"));
  coords.def("hashmap_search",
             &coords_hashmap_search,
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("search_keys"),
             py::arg("results"),
             py::arg("num_search"),
             py::arg("key_dim"),
             py::arg("capacity"),
             py::arg("hash_method"));
  coords.def("hashmap_warp_search",
             &coords_hashmap_warp_search,
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("search_keys"),
             py::arg("results"),
             py::arg("num_search"),
             py::arg("key_dim"),
             py::arg("capacity"),
             py::arg("hash_method"));
  coords.def("hashmap_expand",
             &coords_hashmap_expand,
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("base_coords"),
             py::arg("offsets"),
             py::arg("num_base"),
             py::arg("num_offsets"),
             py::arg("key_dim"),
             py::arg("capacity"),
             py::arg("vector_capacity"),
             py::arg("num_entries_tensor"),
             py::arg("status_tensor"),
             py::arg("hash_method"));

  // --- Discrete search operations ---
  coords.def("kernel_map_offset",
             &coords_kernel_map_offset,
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("query_coords"),
             py::arg("kernel_offsets"),
             py::arg("output"),
             py::arg("num_query"),
             py::arg("key_dim"),
             py::arg("num_offsets"),
             py::arg("capacity"),
             py::arg("hash_method"),
             py::arg("threads_x") = 64,
             py::arg("threads_y") = 8);
  coords.def("map_found_indices_to_maps",
             &coords_map_found_indices_to_maps,
             py::arg("found"),
             py::arg("mapped"),
             py::arg("offsets"),
             py::arg("in_maps"),
             py::arg("out_maps"),
             py::arg("K"),
             py::arg("M"));
  coords.def("kernel_map_size_4d",
             &coords_kernel_map_size_4d,
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("query_coords"),
             py::arg("kernel_sizes"),
             py::arg("output"),
             py::arg("num_query"),
             py::arg("capacity"),
             py::arg("num_kernels"),
             py::arg("hash_method"),
             py::arg("threads_x") = 64,
             py::arg("threads_y") = 8);

  // --- Fused count/scatter kernel map operations ---
  coords.def("kernel_map_size_4d_count",
             &coords_kernel_map_size_4d_count,
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("query_coords"),
             py::arg("kernel_sizes"),
             py::arg("counts"),
             py::arg("num_query"),
             py::arg("capacity"),
             py::arg("num_kernels"),
             py::arg("hash_method"),
             py::arg("threads_x") = 64,
             py::arg("threads_y") = 8);
  coords.def("kernel_map_size_4d_scatter",
             &coords_kernel_map_size_4d_scatter,
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("query_coords"),
             py::arg("kernel_sizes"),
             py::arg("offsets"),
             py::arg("scatter_counters"),
             py::arg("in_maps"),
             py::arg("out_maps"),
             py::arg("num_query"),
             py::arg("capacity"),
             py::arg("num_kernels"),
             py::arg("hash_method"),
             py::arg("threads_x") = 64,
             py::arg("threads_y") = 8);

  // --- Postprocess operations (search-once pipeline) ---
  coords.def("postprocess_count",
             &coords_postprocess_count,
             py::arg("found"),
             py::arg("counts"),
             py::arg("K"),
             py::arg("M"));
  coords.def("postprocess_scatter",
             &coords_postprocess_scatter,
             py::arg("found"),
             py::arg("offsets"),
             py::arg("scatter_counters"),
             py::arg("in_maps"),
             py::arg("out_maps"),
             py::arg("K"),
             py::arg("M"));

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

  // --- Radius search ---
  coords.def("radius_search_count",
             &coords_radius_search_count,
             py::arg("points"),
             py::arg("queries"),
             py::arg("sorted_indices"),
             py::arg("cell_starts"),
             py::arg("cell_counts"),
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("result_count"),
             py::arg("N"),
             py::arg("M"),
             py::arg("num_cells"),
             py::arg("radius"),
             py::arg("cell_size"),
             py::arg("table_capacity"),
             py::arg("hash_method"));
  coords.def("radius_search_write",
             &coords_radius_search_write,
             py::arg("points"),
             py::arg("queries"),
             py::arg("sorted_indices"),
             py::arg("cell_starts"),
             py::arg("cell_counts"),
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("result_offsets"),
             py::arg("result_indices"),
             py::arg("result_distances"),
             py::arg("N"),
             py::arg("M"),
             py::arg("num_cells"),
             py::arg("radius"),
             py::arg("cell_size"),
             py::arg("table_capacity"),
             py::arg("hash_method"));

  // --- Fused kernel map (2 CUDA launches instead of 4) ---
  coords.def("fused_kernel_map",
             &launch_fused_kernel_map,
             "Fused kernel map generation (search + count + scatter in 2 passes)",
             py::arg("output_coords"),
             py::arg("table_kvs"),
             py::arg("vector_keys"),
             py::arg("table_capacity"),
             py::arg("kernel_size"),
             py::arg("hash_method") = 1);
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
