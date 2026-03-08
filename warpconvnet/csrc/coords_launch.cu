// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host wrapper functions for coordinate hash table and search kernels.
// These wrap the extern "C" __global__ kernels from hashmap_kernels.cu,
// discrete_kernels.cu, morton_code.cu, and find_first_gt_bsearch.cu.

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

// ============================================================================
// Forward declarations of extern "C" __global__ kernels from other .cu files
// ============================================================================

// --- hashmap_kernels.cu ---
extern "C" __global__ void prepare_key_value_pairs_kernel(int* table_kvs, int capacity);

extern "C" __global__ void insert_kernel_fnv1a(
    int* table_kvs, const int* vector_keys, int num_keys, int key_dim, int table_capacity);
extern "C" __global__ void insert_kernel_city(
    int* table_kvs, const int* vector_keys, int num_keys, int key_dim, int table_capacity);
extern "C" __global__ void insert_kernel_murmur(
    int* table_kvs, const int* vector_keys, int num_keys, int key_dim, int table_capacity);

extern "C" __global__ void search_kernel_fnv1a(const int* table_kvs,
                                               const int* vector_keys,
                                               const int* search_keys,
                                               int* results,
                                               int num_search_keys,
                                               int key_dim,
                                               int table_capacity);
extern "C" __global__ void search_kernel_city(const int* table_kvs,
                                              const int* vector_keys,
                                              const int* search_keys,
                                              int* results,
                                              int num_search_keys,
                                              int key_dim,
                                              int table_capacity);
extern "C" __global__ void search_kernel_murmur(const int* table_kvs,
                                                const int* vector_keys,
                                                const int* search_keys,
                                                int* results,
                                                int num_search_keys,
                                                int key_dim,
                                                int table_capacity);

extern "C" __global__ void expand_insert_kernel_fnv1a(int* table_kvs,
                                                      int* vector_keys,
                                                      const int* base_coords,
                                                      const int* offsets,
                                                      int num_base_coords,
                                                      int num_offsets,
                                                      int key_dim,
                                                      int table_capacity,
                                                      int vector_capacity,
                                                      int* num_entries_ptr,
                                                      int* status_ptr);
extern "C" __global__ void expand_insert_kernel_city(int* table_kvs,
                                                     int* vector_keys,
                                                     const int* base_coords,
                                                     const int* offsets,
                                                     int num_base_coords,
                                                     int num_offsets,
                                                     int key_dim,
                                                     int table_capacity,
                                                     int vector_capacity,
                                                     int* num_entries_ptr,
                                                     int* status_ptr);
extern "C" __global__ void expand_insert_kernel_murmur(int* table_kvs,
                                                       int* vector_keys,
                                                       const int* base_coords,
                                                       const int* offsets,
                                                       int num_base_coords,
                                                       int num_offsets,
                                                       int key_dim,
                                                       int table_capacity,
                                                       int vector_capacity,
                                                       int* num_entries_ptr,
                                                       int* status_ptr);

// --- discrete_kernels.cu ---
extern "C" __global__ void kernel_map_offset_fnv1a(const int* table_kvs,
                                                   const int* vector_keys,
                                                   const int* query_coords,
                                                   const int* kernel_offsets,
                                                   int* found_in_coord_index,
                                                   int num_query_coords,
                                                   int key_dim,
                                                   int num_kernel_offsets,
                                                   int table_capacity);
extern "C" __global__ void kernel_map_offset_city(const int* table_kvs,
                                                  const int* vector_keys,
                                                  const int* query_coords,
                                                  const int* kernel_offsets,
                                                  int* found_in_coord_index,
                                                  int num_query_coords,
                                                  int key_dim,
                                                  int num_kernel_offsets,
                                                  int table_capacity);
extern "C" __global__ void kernel_map_offset_murmur(const int* table_kvs,
                                                    const int* vector_keys,
                                                    const int* query_coords,
                                                    const int* kernel_offsets,
                                                    int* found_in_coord_index,
                                                    int num_query_coords,
                                                    int key_dim,
                                                    int num_kernel_offsets,
                                                    int table_capacity);

extern "C" __global__ void map_found_indices_to_maps_cuda(const int* found_in_coord_index,
                                                          const int* mapped_indices,
                                                          const int* offsets,
                                                          int* out_in_maps,
                                                          int* out_out_maps,
                                                          int num_kernel_offsets,
                                                          int num_query_coords);

extern "C" __global__ void kernel_map_size_4d_fnv1a(const int* table_kvs,
                                                    const int* vector_keys,
                                                    const int* query_coords,
                                                    const int* kernel_sizes,
                                                    int* found_in_coord_index,
                                                    int num_query_coords,
                                                    int table_capacity,
                                                    int num_kernels);
extern "C" __global__ void kernel_map_size_4d_city(const int* table_kvs,
                                                   const int* vector_keys,
                                                   const int* query_coords,
                                                   const int* kernel_sizes,
                                                   int* found_in_coord_index,
                                                   int num_query_coords,
                                                   int table_capacity,
                                                   int num_kernels);
extern "C" __global__ void kernel_map_size_4d_murmur(const int* table_kvs,
                                                     const int* vector_keys,
                                                     const int* query_coords,
                                                     const int* kernel_sizes,
                                                     int* found_in_coord_index,
                                                     int num_query_coords,
                                                     int table_capacity,
                                                     int num_kernels);

// --- Fused count/scatter kernels (discrete_kernels.cu) ---
extern "C" __global__ void kernel_map_size_4d_count_fnv1a(const int* table_kvs,
                                                          const int* vector_keys,
                                                          const int* query_coords,
                                                          const int* kernel_sizes,
                                                          int* counts,
                                                          int num_query_coords,
                                                          int table_capacity,
                                                          int num_kernels);
extern "C" __global__ void kernel_map_size_4d_count_city(const int* table_kvs,
                                                         const int* vector_keys,
                                                         const int* query_coords,
                                                         const int* kernel_sizes,
                                                         int* counts,
                                                         int num_query_coords,
                                                         int table_capacity,
                                                         int num_kernels);
extern "C" __global__ void kernel_map_size_4d_count_murmur(const int* table_kvs,
                                                           const int* vector_keys,
                                                           const int* query_coords,
                                                           const int* kernel_sizes,
                                                           int* counts,
                                                           int num_query_coords,
                                                           int table_capacity,
                                                           int num_kernels);

extern "C" __global__ void kernel_map_size_4d_scatter_fnv1a(const int* table_kvs,
                                                            const int* vector_keys,
                                                            const int* query_coords,
                                                            const int* kernel_sizes,
                                                            const int* offsets,
                                                            int* scatter_counters,
                                                            int* in_maps,
                                                            int* out_maps,
                                                            int num_query_coords,
                                                            int table_capacity,
                                                            int num_kernels);
extern "C" __global__ void kernel_map_size_4d_scatter_city(const int* table_kvs,
                                                           const int* vector_keys,
                                                           const int* query_coords,
                                                           const int* kernel_sizes,
                                                           const int* offsets,
                                                           int* scatter_counters,
                                                           int* in_maps,
                                                           int* out_maps,
                                                           int num_query_coords,
                                                           int table_capacity,
                                                           int num_kernels);
extern "C" __global__ void kernel_map_size_4d_scatter_murmur(const int* table_kvs,
                                                             const int* vector_keys,
                                                             const int* query_coords,
                                                             const int* kernel_sizes,
                                                             const int* offsets,
                                                             int* scatter_counters,
                                                             int* in_maps,
                                                             int* out_maps,
                                                             int num_query_coords,
                                                             int table_capacity,
                                                             int num_kernels);

// --- Postprocess kernels (discrete_kernels.cu) ---
extern "C" __global__ void postprocess_count_kernel(const int* found_in_coord_index,
                                                    int* counts,
                                                    int K,
                                                    int M);
extern "C" __global__ void postprocess_scatter_kernel(const int* found_in_coord_index,
                                                      const int* offsets,
                                                      int* scatter_counters,
                                                      int* in_maps,
                                                      int* out_maps,
                                                      int K,
                                                      int M);

// --- morton_code.cu ---
extern "C" __global__ void assign_order_discrete_16bit_kernel(const int* bcoords_data,
                                                              int num_points,
                                                              int64_t* result_order);
extern "C" __global__ void assign_order_discrete_20bit_kernel_4points(const int* coords_data,
                                                                      int num_points,
                                                                      int64_t* result_order);

// --- find_first_gt_bsearch.cu ---
extern "C" __global__ void find_first_gt_bsearch(
    const int* srcM, int M, const int* srcN, int N, int* out);

// --- radius_search_kernels.cu ---
extern "C" __global__ void radius_search_count_kernel_fnv1a(const float* points,
                                                            const float* queries,
                                                            const int* sorted_indices,
                                                            const int* cell_starts,
                                                            const int* cell_counts,
                                                            const int* table_kvs,
                                                            const int* vector_keys,
                                                            int* result_count,
                                                            int N,
                                                            int M,
                                                            int num_cells,
                                                            float radius,
                                                            float cell_size,
                                                            int table_capacity);
extern "C" __global__ void radius_search_count_kernel_city(const float* points,
                                                           const float* queries,
                                                           const int* sorted_indices,
                                                           const int* cell_starts,
                                                           const int* cell_counts,
                                                           const int* table_kvs,
                                                           const int* vector_keys,
                                                           int* result_count,
                                                           int N,
                                                           int M,
                                                           int num_cells,
                                                           float radius,
                                                           float cell_size,
                                                           int table_capacity);
extern "C" __global__ void radius_search_count_kernel_murmur(const float* points,
                                                             const float* queries,
                                                             const int* sorted_indices,
                                                             const int* cell_starts,
                                                             const int* cell_counts,
                                                             const int* table_kvs,
                                                             const int* vector_keys,
                                                             int* result_count,
                                                             int N,
                                                             int M,
                                                             int num_cells,
                                                             float radius,
                                                             float cell_size,
                                                             int table_capacity);

extern "C" __global__ void radius_search_write_kernel_fnv1a(const float* points,
                                                            const float* queries,
                                                            const int* sorted_indices,
                                                            const int* cell_starts,
                                                            const int* cell_counts,
                                                            const int* table_kvs,
                                                            const int* vector_keys,
                                                            const int* result_offsets,
                                                            int* result_indices,
                                                            float* result_distances,
                                                            int N,
                                                            int M,
                                                            int num_cells,
                                                            float radius,
                                                            float cell_size,
                                                            int table_capacity);
extern "C" __global__ void radius_search_write_kernel_city(const float* points,
                                                           const float* queries,
                                                           const int* sorted_indices,
                                                           const int* cell_starts,
                                                           const int* cell_counts,
                                                           const int* table_kvs,
                                                           const int* vector_keys,
                                                           const int* result_offsets,
                                                           int* result_indices,
                                                           float* result_distances,
                                                           int N,
                                                           int M,
                                                           int num_cells,
                                                           float radius,
                                                           float cell_size,
                                                           int table_capacity);
extern "C" __global__ void radius_search_write_kernel_murmur(const float* points,
                                                             const float* queries,
                                                             const int* sorted_indices,
                                                             const int* cell_starts,
                                                             const int* cell_counts,
                                                             const int* table_kvs,
                                                             const int* vector_keys,
                                                             const int* result_offsets,
                                                             int* result_indices,
                                                             float* result_distances,
                                                             int N,
                                                             int M,
                                                             int num_cells,
                                                             float radius,
                                                             float cell_size,
                                                             int table_capacity);

// ============================================================================
// coord_to_code kernel (moved from inline Python string in voxel_encode.py)
// ============================================================================

__device__ static int64_t part1by2_long_ctc(int64_t n) {
  n &= 0x1fffffLL;  // mask to 21 bits
  n = (n | (n << 32)) & 0x1f00000000ffffLL;
  n = (n | (n << 16)) & 0x1f0000ff0000ffLL;
  n = (n | (n << 8)) & 0x100f00f00f00f00fLL;
  n = (n | (n << 4)) & 0x10c30c30c30c30c3LL;
  n = (n | (n << 2)) & 0x1249249249249249LL;
  return n;
}

__device__ static int64_t morton_code_20bit_device_ctc(int64_t coord_x,
                                                       int64_t coord_y,
                                                       int64_t coord_z) {
  return (part1by2_long_ctc(coord_z) << 2) | (part1by2_long_ctc(coord_y) << 1) |
         part1by2_long_ctc(coord_x);
}

__global__ void coord_to_code_kernel_impl(const int* grid_coord,
                                          const int* coord_offset,
                                          const int* min_coord,
                                          const int* window_size,
                                          const int N,
                                          int64_t* codes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  __shared__ int s_coord_offset[3];
  __shared__ int s_min_coord[3];
  __shared__ int s_window_size[3];
  if (threadIdx.x < 3) {
    s_coord_offset[threadIdx.x] = coord_offset[threadIdx.x];
    s_min_coord[threadIdx.x] = min_coord[threadIdx.x];
    s_window_size[threadIdx.x] = window_size[threadIdx.x];
  }
  __syncthreads();

  int grid_coord_x = grid_coord[idx * 3 + 0];
  int grid_coord_y = grid_coord[idx * 3 + 1];
  int grid_coord_z = grid_coord[idx * 3 + 2];

  int64_t voxel_x = (grid_coord_x + s_coord_offset[0] - s_min_coord[0]) / s_window_size[0];
  int64_t voxel_y = (grid_coord_y + s_coord_offset[1] - s_min_coord[1]) / s_window_size[1];
  int64_t voxel_z = (grid_coord_z + s_coord_offset[2] - s_min_coord[2]) / s_window_size[2];

  codes[idx] = morton_code_20bit_device_ctc(voxel_x, voxel_y, voxel_z);
}

// ============================================================================
// Host wrapper functions (callable from coords_bindings.cpp)
// ============================================================================

void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (capacity + threads - 1) / threads;
  prepare_key_value_pairs_kernel<<<blocks, threads, 0, stream>>>(table_kvs.data_ptr<int>(),
                                                                 capacity);
}

void coords_hashmap_insert(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           int num_keys,
                           int key_dim,
                           int capacity,
                           int hash_method) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (num_keys + threads - 1) / threads;
  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  switch (hash_method) {
    case 0:
      insert_kernel_fnv1a<<<blocks, threads, 0, stream>>>(tbl, vk, num_keys, key_dim, capacity);
      break;
    case 1:
      insert_kernel_city<<<blocks, threads, 0, stream>>>(tbl, vk, num_keys, key_dim, capacity);
      break;
    case 2:
      insert_kernel_murmur<<<blocks, threads, 0, stream>>>(tbl, vk, num_keys, key_dim, capacity);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

void coords_hashmap_search(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           torch::Tensor search_keys,
                           torch::Tensor results,
                           int num_search,
                           int key_dim,
                           int capacity,
                           int hash_method) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (num_search + threads - 1) / threads;
  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* sk = search_keys.data_ptr<int>();
  auto* res = results.data_ptr<int>();
  switch (hash_method) {
    case 0:
      search_kernel_fnv1a<<<blocks, threads, 0, stream>>>(
          tbl, vk, sk, res, num_search, key_dim, capacity);
      break;
    case 1:
      search_kernel_city<<<blocks, threads, 0, stream>>>(
          tbl, vk, sk, res, num_search, key_dim, capacity);
      break;
    case 2:
      search_kernel_murmur<<<blocks, threads, 0, stream>>>(
          tbl, vk, sk, res, num_search, key_dim, capacity);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

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
                           int hash_method) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  long long total = (long long)num_base * (long long)num_offsets;
  int blocks = (int)((total + threads - 1) / threads);
  int shared_mem = threads * key_dim * 4;  // sizeof(int) * key_dim * threads

  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* bc = base_coords.data_ptr<int>();
  auto* off = offsets.data_ptr<int>();
  auto* ne = num_entries_tensor.data_ptr<int>();
  auto* st = status_tensor.data_ptr<int>();

  switch (hash_method) {
    case 0:
      expand_insert_kernel_fnv1a<<<blocks, threads, shared_mem, stream>>>(
          tbl, vk, bc, off, num_base, num_offsets, key_dim, capacity, vector_capacity, ne, st);
      break;
    case 1:
      expand_insert_kernel_city<<<blocks, threads, shared_mem, stream>>>(
          tbl, vk, bc, off, num_base, num_offsets, key_dim, capacity, vector_capacity, ne, st);
      break;
    case 2:
      expand_insert_kernel_murmur<<<blocks, threads, shared_mem, stream>>>(
          tbl, vk, bc, off, num_base, num_offsets, key_dim, capacity, vector_capacity, ne, st);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

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
                              int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_offsets + threads_y - 1) / threads_y);

  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* qc = query_coords.data_ptr<int>();
  auto* ko = kernel_offsets.data_ptr<int>();
  auto* out = output.data_ptr<int>();

  switch (hash_method) {
    case 0:
      kernel_map_offset_fnv1a<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ko, out, num_query, key_dim, num_offsets, capacity);
      break;
    case 1:
      kernel_map_offset_city<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ko, out, num_query, key_dim, num_offsets, capacity);
      break;
    case 2:
      kernel_map_offset_murmur<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ko, out, num_query, key_dim, num_offsets, capacity);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

void coords_map_found_indices_to_maps(torch::Tensor found,
                                      torch::Tensor mapped,
                                      torch::Tensor offsets,
                                      torch::Tensor in_maps,
                                      torch::Tensor out_maps,
                                      int K,
                                      int M) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int total = K * M;
  int blocks = (total + threads - 1) / threads;
  map_found_indices_to_maps_cuda<<<blocks, threads, 0, stream>>>(found.data_ptr<int>(),
                                                                 mapped.data_ptr<int>(),
                                                                 offsets.data_ptr<int>(),
                                                                 in_maps.data_ptr<int>(),
                                                                 out_maps.data_ptr<int>(),
                                                                 K,
                                                                 M);
}

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
                               int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_kernels + threads_y - 1) / threads_y);

  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* qc = query_coords.data_ptr<int>();
  auto* ks = kernel_sizes.data_ptr<int>();
  auto* out = output.data_ptr<int>();

  switch (hash_method) {
    case 0:
      kernel_map_size_4d_fnv1a<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, out, num_query, capacity, num_kernels);
      break;
    case 1:
      kernel_map_size_4d_city<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, out, num_query, capacity, num_kernels);
      break;
    case 2:
      kernel_map_size_4d_murmur<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, out, num_query, capacity, num_kernels);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

void coords_morton_code_16bit(torch::Tensor bcoords, int num_points, torch::Tensor result) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (num_points + threads - 1) / threads;
  assign_order_discrete_16bit_kernel<<<blocks, threads, 0, stream>>>(
      bcoords.data_ptr<int>(), num_points, result.data_ptr<int64_t>());
}

void coords_morton_code_20bit(torch::Tensor coords, int num_points, torch::Tensor result) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  // The kernel processes 4 points per thread
  int blocks = (num_points + threads * 4 - 1) / (threads * 4);
  assign_order_discrete_20bit_kernel_4points<<<blocks, threads, 0, stream>>>(
      coords.data_ptr<int>(), num_points, result.data_ptr<int64_t>());
}

void coords_find_first_gt_bsearch(
    torch::Tensor offsets_tensor, int M, torch::Tensor indices, int N, torch::Tensor output) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  int shared_mem = M * sizeof(int);
  find_first_gt_bsearch<<<blocks, threads, shared_mem, stream>>>(
      offsets_tensor.data_ptr<int>(), M, indices.data_ptr<int>(), N, output.data_ptr<int>());
}

void coords_coord_to_code(torch::Tensor grid_coord,
                          torch::Tensor coord_offset,
                          torch::Tensor min_coord,
                          torch::Tensor window_size,
                          int N,
                          torch::Tensor codes) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  coord_to_code_kernel_impl<<<blocks, threads, 0, stream>>>(grid_coord.data_ptr<int>(),
                                                            coord_offset.data_ptr<int>(),
                                                            min_coord.data_ptr<int>(),
                                                            window_size.data_ptr<int>(),
                                                            N,
                                                            codes.data_ptr<int64_t>());
}

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
                                     int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_kernels + threads_y - 1) / threads_y);

  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* qc = query_coords.data_ptr<int>();
  auto* ks = kernel_sizes.data_ptr<int>();
  auto* cnt = counts.data_ptr<int>();

  switch (hash_method) {
    case 0:
      kernel_map_size_4d_count_fnv1a<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, cnt, num_query, capacity, num_kernels);
      break;
    case 1:
      kernel_map_size_4d_count_city<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, cnt, num_query, capacity, num_kernels);
      break;
    case 2:
      kernel_map_size_4d_count_murmur<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, cnt, num_query, capacity, num_kernels);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

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
                                       int threads_y) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_kernels + threads_y - 1) / threads_y);

  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* qc = query_coords.data_ptr<int>();
  auto* ks = kernel_sizes.data_ptr<int>();
  auto* off = offsets.data_ptr<int>();
  auto* sc = scatter_counters.data_ptr<int>();
  auto* im = in_maps.data_ptr<int>();
  auto* om = out_maps.data_ptr<int>();

  switch (hash_method) {
    case 0:
      kernel_map_size_4d_scatter_fnv1a<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, off, sc, im, om, num_query, capacity, num_kernels);
      break;
    case 1:
      kernel_map_size_4d_scatter_city<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, off, sc, im, om, num_query, capacity, num_kernels);
      break;
    case 2:
      kernel_map_size_4d_scatter_murmur<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, off, sc, im, om, num_query, capacity, num_kernels);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

void coords_postprocess_count(torch::Tensor found, torch::Tensor counts, int K, int M) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  long long total = (long long)K * (long long)M;
  int blocks = (int)((total + threads - 1) / threads);
  postprocess_count_kernel<<<blocks, threads, 0, stream>>>(
      found.data_ptr<int>(), counts.data_ptr<int>(), K, M);
}

void coords_postprocess_scatter(torch::Tensor found,
                                torch::Tensor offsets,
                                torch::Tensor scatter_counters,
                                torch::Tensor in_maps,
                                torch::Tensor out_maps,
                                int K,
                                int M) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  long long total = (long long)K * (long long)M;
  int blocks = (int)((total + threads - 1) / threads);
  postprocess_scatter_kernel<<<blocks, threads, 0, stream>>>(found.data_ptr<int>(),
                                                             offsets.data_ptr<int>(),
                                                             scatter_counters.data_ptr<int>(),
                                                             in_maps.data_ptr<int>(),
                                                             out_maps.data_ptr<int>(),
                                                             K,
                                                             M);
}

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
                                int hash_method) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (M + threads - 1) / threads;

  auto* pts = points.data_ptr<float>();
  auto* qrs = queries.data_ptr<float>();
  auto* si = sorted_indices.data_ptr<int>();
  auto* cs = cell_starts.data_ptr<int>();
  auto* cc = cell_counts.data_ptr<int>();
  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* rc = result_count.data_ptr<int>();

  switch (hash_method) {
    case 0:
      radius_search_count_kernel_fnv1a<<<blocks, threads, 0, stream>>>(
          pts, qrs, si, cs, cc, tbl, vk, rc, N, M, num_cells, radius, cell_size, table_capacity);
      break;
    case 1:
      radius_search_count_kernel_city<<<blocks, threads, 0, stream>>>(
          pts, qrs, si, cs, cc, tbl, vk, rc, N, M, num_cells, radius, cell_size, table_capacity);
      break;
    case 2:
      radius_search_count_kernel_murmur<<<blocks, threads, 0, stream>>>(
          pts, qrs, si, cs, cc, tbl, vk, rc, N, M, num_cells, radius, cell_size, table_capacity);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

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
                                int hash_method) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (M + threads - 1) / threads;

  auto* pts = points.data_ptr<float>();
  auto* qrs = queries.data_ptr<float>();
  auto* si = sorted_indices.data_ptr<int>();
  auto* cs = cell_starts.data_ptr<int>();
  auto* cc = cell_counts.data_ptr<int>();
  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* ro = result_offsets.data_ptr<int>();
  auto* ri = result_indices.data_ptr<int>();
  auto* rd = result_distances.data_ptr<float>();

  switch (hash_method) {
    case 0:
      radius_search_write_kernel_fnv1a<<<blocks, threads, 0, stream>>>(pts,
                                                                       qrs,
                                                                       si,
                                                                       cs,
                                                                       cc,
                                                                       tbl,
                                                                       vk,
                                                                       ro,
                                                                       ri,
                                                                       rd,
                                                                       N,
                                                                       M,
                                                                       num_cells,
                                                                       radius,
                                                                       cell_size,
                                                                       table_capacity);
      break;
    case 1:
      radius_search_write_kernel_city<<<blocks, threads, 0, stream>>>(pts,
                                                                      qrs,
                                                                      si,
                                                                      cs,
                                                                      cc,
                                                                      tbl,
                                                                      vk,
                                                                      ro,
                                                                      ri,
                                                                      rd,
                                                                      N,
                                                                      M,
                                                                      num_cells,
                                                                      radius,
                                                                      cell_size,
                                                                      table_capacity);
      break;
    case 2:
      radius_search_write_kernel_murmur<<<blocks, threads, 0, stream>>>(pts,
                                                                        qrs,
                                                                        si,
                                                                        cs,
                                                                        cc,
                                                                        tbl,
                                                                        vk,
                                                                        ro,
                                                                        ri,
                                                                        rd,
                                                                        N,
                                                                        M,
                                                                        num_cells,
                                                                        radius,
                                                                        cell_size,
                                                                        table_capacity);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }
}

// ============================================================================
// Window grouping kernels (counting sort)
// ============================================================================

extern "C" __global__ void window_group_histogram_kernel(
    const int* __restrict__ grid_coord,
    const int* __restrict__ batch_offsets,
    const int* __restrict__ coord_offset,
    const int* __restrict__ min_coord,
    const int* __restrict__ window_size,
    const int* __restrict__ grid_shape,
    int64_t* __restrict__ codes,
    int* __restrict__ histogram,
    int N,
    int B,
    int W);

extern "C" __global__ void window_group_scatter_kernel(
    const int64_t* __restrict__ codes,
    const int* __restrict__ window_offsets_dense,
    int* __restrict__ scatter_counters,
    int64_t* __restrict__ perm,
    int64_t* __restrict__ inverse_perm,
    int N);

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
                                   int W) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  window_group_histogram_kernel<<<blocks, threads, 0, stream>>>(
      grid_coord.data_ptr<int>(),
      batch_offsets.data_ptr<int>(),
      coord_offset.data_ptr<int>(),
      min_coord.data_ptr<int>(),
      window_size.data_ptr<int>(),
      grid_shape.data_ptr<int>(),
      codes.data_ptr<int64_t>(),
      histogram.data_ptr<int>(),
      N,
      B,
      W);
}

void coords_window_group_scatter(torch::Tensor codes,
                                 torch::Tensor window_offsets_dense,
                                 torch::Tensor scatter_counters,
                                 torch::Tensor perm,
                                 torch::Tensor inverse_perm,
                                 int N) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  window_group_scatter_kernel<<<blocks, threads, 0, stream>>>(
      codes.data_ptr<int64_t>(),
      window_offsets_dense.data_ptr<int>(),
      scatter_counters.data_ptr<int>(),
      perm.data_ptr<int64_t>(),
      inverse_perm.data_ptr<int64_t>(),
      N);
}
