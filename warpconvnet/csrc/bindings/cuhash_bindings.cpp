// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// pybind11 bindings for cuhash packed hash table and kernel map kernels.
// Registers everything under the _C.cuhash submodule.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "register.h"

namespace py = pybind11;

// Forward declarations from cuhash_hash_table.cu
namespace cuhash {
void launch_packed_prepare(torch::Tensor keys, torch::Tensor values, int capacity);
void launch_packed_insert(torch::Tensor keys,
                          torch::Tensor values,
                          torch::Tensor coords,
                          int num_keys,
                          int capacity,
                          bool use_double_hash,
                          torch::Tensor status_tensor);
void launch_packed_search(torch::Tensor keys,
                          torch::Tensor values,
                          torch::Tensor search_coords,
                          torch::Tensor results,
                          int num_search,
                          int capacity,
                          int search_mode);
void launch_generic_prepare(torch::Tensor table_kvs, int capacity);
void launch_generic_insert(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           int num_keys,
                           int key_dim,
                           int capacity,
                           int hash_method);
void launch_generic_search(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           torch::Tensor search_keys,
                           torch::Tensor results,
                           int num_search,
                           int key_dim,
                           int capacity,
                           int hash_method);

// Forward declarations from cuhash_kernel_map.cu
void launch_packed_kernel_map_offset(torch::Tensor keys,
                                     torch::Tensor values,
                                     torch::Tensor query_coords,
                                     torch::Tensor kernel_offsets,
                                     torch::Tensor output,
                                     int num_query,
                                     int num_offsets,
                                     int capacity,
                                     int threads_x,
                                     int threads_y);
void launch_packed_kernel_map_size(torch::Tensor keys,
                                   torch::Tensor values,
                                   torch::Tensor query_coords,
                                   torch::Tensor kernel_sizes,
                                   torch::Tensor output,
                                   int num_query,
                                   int num_kernels,
                                   int capacity,
                                   int threads_x,
                                   int threads_y);
void launch_packed_kernel_map_count(torch::Tensor keys,
                                    torch::Tensor values,
                                    torch::Tensor query_coords,
                                    torch::Tensor kernel_sizes,
                                    torch::Tensor counts,
                                    int num_query,
                                    int num_kernels,
                                    int capacity,
                                    int threads_x,
                                    int threads_y);
void launch_packed_kernel_map_scatter(torch::Tensor keys,
                                      torch::Tensor values,
                                      torch::Tensor query_coords,
                                      torch::Tensor kernel_sizes,
                                      torch::Tensor offsets,
                                      torch::Tensor scatter_counters,
                                      torch::Tensor in_maps,
                                      torch::Tensor out_maps,
                                      int num_query,
                                      int num_kernels,
                                      int capacity,
                                      int threads_x,
                                      int threads_y);
void launch_postprocess_count(torch::Tensor found, torch::Tensor counts, int K, int M);
void launch_postprocess_scatter(torch::Tensor found,
                                torch::Tensor offsets,
                                torch::Tensor scatter_counters,
                                torch::Tensor in_maps,
                                torch::Tensor out_maps,
                                int K,
                                int M);

// Per-query loop kernels
void launch_packed_kernel_map_loop(torch::Tensor keys,
                                   torch::Tensor values,
                                   torch::Tensor query_coords,
                                   torch::Tensor spatial_offsets,
                                   torch::Tensor output,
                                   int num_query,
                                   int num_kernels,
                                   int capacity);
void launch_packed_kernel_map_count_loop(torch::Tensor keys,
                                         torch::Tensor values,
                                         torch::Tensor query_coords,
                                         torch::Tensor spatial_offsets,
                                         torch::Tensor counts,
                                         int num_query,
                                         int num_kernels,
                                         int capacity);
void launch_packed_kernel_map_scatter_loop(torch::Tensor keys,
                                           torch::Tensor values,
                                           torch::Tensor query_coords,
                                           torch::Tensor spatial_offsets,
                                           torch::Tensor offsets,
                                           torch::Tensor scatter_counters,
                                           torch::Tensor in_maps,
                                           torch::Tensor out_maps,
                                           int num_query,
                                           int num_kernels,
                                           int capacity);
void launch_packed_kernel_map_onepass(torch::Tensor keys,
                                      torch::Tensor values,
                                      torch::Tensor query_coords,
                                      torch::Tensor spatial_offsets,
                                      torch::Tensor offsets,
                                      torch::Tensor scatter_counters,
                                      torch::Tensor in_maps,
                                      torch::Tensor out_maps,
                                      int num_query,
                                      int num_kernels,
                                      int capacity);

// Mask data
void launch_csr_to_pair_table(torch::Tensor in_maps,
                              torch::Tensor out_maps,
                              torch::Tensor offsets,
                              torch::Tensor pair_table,
                              int N_out,
                              int K,
                              int L);
void launch_build_pair_mask(torch::Tensor pair_table, torch::Tensor pair_mask, int N, int K);

// Expand with offsets
void launch_packed_expand_insert(torch::Tensor keys,
                                 torch::Tensor values,
                                 torch::Tensor coords_store,
                                 torch::Tensor base_coords,
                                 torch::Tensor offsets,
                                 int num_base,
                                 int num_offsets,
                                 int capacity,
                                 int vector_capacity,
                                 torch::Tensor num_entries_tensor,
                                 torch::Tensor status_tensor);

// Build coarse table directly from fine coords
void launch_packed_build_coarse(torch::Tensor keys,
                                torch::Tensor values,
                                torch::Tensor fine_coords,
                                int num_fine,
                                int stride_shift,
                                int capacity,
                                torch::Tensor num_entries_tensor);

// Fused hierarchical kernel map (single C++ call)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_hierarchical_kernel_map(torch::Tensor fine_keys,
                               torch::Tensor fine_values,
                               torch::Tensor fine_coords,
                               torch::Tensor query_coords,
                               std::vector<int> kernel_size,
                               int stride,
                               int fine_capacity);

// Hierarchical search (individual kernels)
void launch_coarse_probe(torch::Tensor keys_c,
                         torch::Tensor values_c,
                         torch::Tensor query_coords,
                         torch::Tensor coarse_spatial_offsets,
                         torch::Tensor coarse_masks,
                         int num_query,
                         int num_coarse_offsets,
                         int stride_shift,
                         int coarse_capacity);
void launch_fine_search_pruned(torch::Tensor keys_f,
                               torch::Tensor values_f,
                               torch::Tensor query_coords,
                               torch::Tensor fine_spatial_offsets,
                               torch::Tensor coarse_masks,
                               torch::Tensor found,
                               int num_query,
                               int num_fine_offsets,
                               int stride_shift,
                               int coarse_min,
                               int coarse_dy,
                               int coarse_dz,
                               int fine_capacity,
                               int threads_x,
                               int threads_y);
}  // namespace cuhash

namespace warpconvnet {
namespace bindings {

void register_cuhash(pybind11::module_ &m) {
  auto mod = m.def_submodule("cuhash", "Packed hash table and kernel map");

  // --- Packed Key Hash Table ---
  mod.def("packed_prepare",
          &cuhash::launch_packed_prepare,
          py::arg("keys"),
          py::arg("values"),
          py::arg("capacity"));
  mod.def("packed_insert",
          &cuhash::launch_packed_insert,
          py::arg("keys"),
          py::arg("values"),
          py::arg("coords"),
          py::arg("num_keys"),
          py::arg("capacity"),
          py::arg("use_double_hash") = false,
          py::arg("status_tensor"));
  mod.def("packed_search",
          &cuhash::launch_packed_search,
          py::arg("keys"),
          py::arg("values"),
          py::arg("search_coords"),
          py::arg("results"),
          py::arg("num_search"),
          py::arg("capacity"),
          py::arg("search_mode") = 0);

  // --- Generic Key Hash Table ---
  mod.def("generic_prepare",
          &cuhash::launch_generic_prepare,
          py::arg("table_kvs"),
          py::arg("capacity"));
  mod.def("generic_insert",
          &cuhash::launch_generic_insert,
          py::arg("table_kvs"),
          py::arg("vector_keys"),
          py::arg("num_keys"),
          py::arg("key_dim"),
          py::arg("capacity"),
          py::arg("hash_method") = 1);
  mod.def("generic_search",
          &cuhash::launch_generic_search,
          py::arg("table_kvs"),
          py::arg("vector_keys"),
          py::arg("search_keys"),
          py::arg("results"),
          py::arg("num_search"),
          py::arg("key_dim"),
          py::arg("capacity"),
          py::arg("hash_method") = 1);

  // --- Packed Kernel Map ---
  mod.def("packed_kernel_map_offset",
          &cuhash::launch_packed_kernel_map_offset,
          py::arg("keys"),
          py::arg("values"),
          py::arg("query_coords"),
          py::arg("kernel_offsets"),
          py::arg("output"),
          py::arg("num_query"),
          py::arg("num_offsets"),
          py::arg("capacity"),
          py::arg("threads_x") = 64,
          py::arg("threads_y") = 8);
  mod.def("packed_kernel_map_size",
          &cuhash::launch_packed_kernel_map_size,
          py::arg("keys"),
          py::arg("values"),
          py::arg("query_coords"),
          py::arg("kernel_sizes"),
          py::arg("output"),
          py::arg("num_query"),
          py::arg("num_kernels"),
          py::arg("capacity"),
          py::arg("threads_x") = 64,
          py::arg("threads_y") = 8);
  mod.def("packed_kernel_map_count",
          &cuhash::launch_packed_kernel_map_count,
          py::arg("keys"),
          py::arg("values"),
          py::arg("query_coords"),
          py::arg("kernel_sizes"),
          py::arg("counts"),
          py::arg("num_query"),
          py::arg("num_kernels"),
          py::arg("capacity"),
          py::arg("threads_x") = 64,
          py::arg("threads_y") = 8);
  mod.def("packed_kernel_map_scatter",
          &cuhash::launch_packed_kernel_map_scatter,
          py::arg("keys"),
          py::arg("values"),
          py::arg("query_coords"),
          py::arg("kernel_sizes"),
          py::arg("offsets"),
          py::arg("scatter_counters"),
          py::arg("in_maps"),
          py::arg("out_maps"),
          py::arg("num_query"),
          py::arg("num_kernels"),
          py::arg("capacity"),
          py::arg("threads_x") = 64,
          py::arg("threads_y") = 8);

  // --- Postprocess ---
  mod.def("postprocess_count",
          &cuhash::launch_postprocess_count,
          py::arg("found"),
          py::arg("counts"),
          py::arg("K"),
          py::arg("M"));
  mod.def("postprocess_scatter",
          &cuhash::launch_postprocess_scatter,
          py::arg("found"),
          py::arg("offsets"),
          py::arg("scatter_counters"),
          py::arg("in_maps"),
          py::arg("out_maps"),
          py::arg("K"),
          py::arg("M"));

  // --- Per-query loop kernels ---
  mod.def("packed_kernel_map_loop",
          &cuhash::launch_packed_kernel_map_loop,
          py::arg("keys"),
          py::arg("values"),
          py::arg("query_coords"),
          py::arg("spatial_offsets"),
          py::arg("output"),
          py::arg("num_query"),
          py::arg("num_kernels"),
          py::arg("capacity"));
  mod.def("packed_kernel_map_count_loop",
          &cuhash::launch_packed_kernel_map_count_loop,
          py::arg("keys"),
          py::arg("values"),
          py::arg("query_coords"),
          py::arg("spatial_offsets"),
          py::arg("counts"),
          py::arg("num_query"),
          py::arg("num_kernels"),
          py::arg("capacity"));
  mod.def("packed_kernel_map_scatter_loop",
          &cuhash::launch_packed_kernel_map_scatter_loop,
          py::arg("keys"),
          py::arg("values"),
          py::arg("query_coords"),
          py::arg("spatial_offsets"),
          py::arg("offsets"),
          py::arg("scatter_counters"),
          py::arg("in_maps"),
          py::arg("out_maps"),
          py::arg("num_query"),
          py::arg("num_kernels"),
          py::arg("capacity"));
  mod.def("packed_kernel_map_onepass",
          &cuhash::launch_packed_kernel_map_onepass,
          py::arg("keys"),
          py::arg("values"),
          py::arg("query_coords"),
          py::arg("spatial_offsets"),
          py::arg("offsets"),
          py::arg("scatter_counters"),
          py::arg("in_maps"),
          py::arg("out_maps"),
          py::arg("num_query"),
          py::arg("num_kernels"),
          py::arg("capacity"));

  // --- Build coarse table ---
  mod.def("packed_build_coarse",
          &cuhash::launch_packed_build_coarse,
          py::arg("keys"),
          py::arg("values"),
          py::arg("fine_coords"),
          py::arg("num_fine"),
          py::arg("stride_shift"),
          py::arg("capacity"),
          py::arg("num_entries_tensor"));

  // --- Expand with offsets ---
  mod.def("packed_expand_insert",
          &cuhash::launch_packed_expand_insert,
          py::arg("keys"),
          py::arg("values"),
          py::arg("coords_store"),
          py::arg("base_coords"),
          py::arg("offsets"),
          py::arg("num_base"),
          py::arg("num_offsets"),
          py::arg("capacity"),
          py::arg("vector_capacity"),
          py::arg("num_entries_tensor"),
          py::arg("status_tensor"));

  // --- Mask data ---
  mod.def("csr_to_pair_table",
          &cuhash::launch_csr_to_pair_table,
          py::arg("in_maps"),
          py::arg("out_maps"),
          py::arg("offsets"),
          py::arg("pair_table"),
          py::arg("N_out"),
          py::arg("K"),
          py::arg("L"));
  mod.def("build_pair_mask",
          &cuhash::launch_build_pair_mask,
          py::arg("pair_table"),
          py::arg("pair_mask"),
          py::arg("N"),
          py::arg("K"));

  // --- Fused hierarchical kernel map ---
  mod.def("hierarchical_kernel_map",
          &cuhash::launch_hierarchical_kernel_map,
          "Fused hierarchical: coarse build + probe + fine search + postprocess",
          py::arg("fine_keys"),
          py::arg("fine_values"),
          py::arg("fine_coords"),
          py::arg("query_coords"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("fine_capacity"));

  // --- Hierarchical search (individual kernels) ---
  mod.def("coarse_probe",
          &cuhash::launch_coarse_probe,
          py::arg("keys_c"),
          py::arg("values_c"),
          py::arg("query_coords"),
          py::arg("coarse_spatial_offsets"),
          py::arg("coarse_masks"),
          py::arg("num_query"),
          py::arg("num_coarse_offsets"),
          py::arg("stride_shift"),
          py::arg("coarse_capacity"));
  mod.def("fine_search_pruned",
          &cuhash::launch_fine_search_pruned,
          py::arg("keys_f"),
          py::arg("values_f"),
          py::arg("query_coords"),
          py::arg("fine_spatial_offsets"),
          py::arg("coarse_masks"),
          py::arg("found"),
          py::arg("num_query"),
          py::arg("num_fine_offsets"),
          py::arg("stride_shift"),
          py::arg("coarse_min"),
          py::arg("coarse_dy"),
          py::arg("coarse_dz"),
          py::arg("fine_capacity"),
          py::arg("threads_x") = 64,
          py::arg("threads_y") = 8);
}

}  // namespace bindings
}  // namespace warpconvnet
