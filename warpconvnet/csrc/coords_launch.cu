// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host wrapper functions for coordinate hash table and search kernels.
// These wrap the extern "C" __global__ kernels from hashmap_kernels.cu,
// discrete_kernels.cu, morton_code.cu, and find_first_gt_bsearch.cu.

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cub/cub.cuh>
#include <tuple>
#include <vector>

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

extern "C" __global__ void warp_search_kernel_fnv1a(const int* table_kvs,
                                                    const int* vector_keys,
                                                    const int* search_keys,
                                                    int* results,
                                                    int num_search_keys,
                                                    int key_dim,
                                                    int table_capacity);
extern "C" __global__ void warp_search_kernel_city(const int* table_kvs,
                                                   const int* vector_keys,
                                                   const int* search_keys,
                                                   int* results,
                                                   int num_search_keys,
                                                   int key_dim,
                                                   int table_capacity);
extern "C" __global__ void warp_search_kernel_murmur(const int* table_kvs,
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

void coords_hashmap_warp_search(torch::Tensor table_kvs,
                                torch::Tensor vector_keys,
                                torch::Tensor search_keys,
                                torch::Tensor results,
                                int num_search,
                                int key_dim,
                                int capacity,
                                int hash_method) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int threads = 256;
  // Each warp (32 threads) handles one query, so we need num_search warps total.
  // Total threads = num_search * 32
  int total_threads = num_search * 32;
  int blocks = (total_threads + threads - 1) / threads;
  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* sk = search_keys.data_ptr<int>();
  auto* res = results.data_ptr<int>();
  switch (hash_method) {
    case 0:
      warp_search_kernel_fnv1a<<<blocks, threads, 0, stream>>>(
          tbl, vk, sk, res, num_search, key_dim, capacity);
      break;
    case 1:
      warp_search_kernel_city<<<blocks, threads, 0, stream>>>(
          tbl, vk, sk, res, num_search, key_dim, capacity);
      break;
    case 2:
      warp_search_kernel_murmur<<<blocks, threads, 0, stream>>>(
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
  // shared_candidates: threads * key_dim * sizeof(int)
  // smem_cache: 256 * sizeof(long long) = 2048 bytes
  // smem_cache_tags: 256 * sizeof(int) = 1024 bytes
  int shared_mem = threads * key_dim * 4 + 256 * 8 + 256 * 4;

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

// ============================================================================
// Fused kernel map: count + cumsum + scatter in a single host function call.
// Eliminates the K*M intermediate and reduces to 2 CUDA kernel launches
// plus a torch::cumsum on a small (K,) tensor.
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> launch_fused_kernel_map(
    torch::Tensor output_coords,
    torch::Tensor table_kvs,
    torch::Tensor vector_keys,
    int table_capacity,
    std::vector<int> kernel_size,
    int hash_method) {
  TORCH_CHECK(kernel_size.size() == 3, "kernel_size must have 3 elements");
  TORCH_CHECK(output_coords.dim() == 2 && output_coords.size(1) == 4,
              "output_coords must be [M, 4]");
  TORCH_CHECK(output_coords.dtype() == torch::kInt32, "output_coords must be int32");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto device = output_coords.device();
  int num_query = output_coords.size(0);
  int num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2];

  // Prepare kernel size tensor on device
  auto kernel_size_tensor =
      torch::tensor({kernel_size[0], kernel_size[1], kernel_size[2]},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));

  // Thread block configuration matching existing kernels
  int threads_x = 64;
  int threads_y = 8;
  dim3 block(threads_x, threads_y);
  dim3 grid((num_query + threads_x - 1) / threads_x, (num_kernels + threads_y - 1) / threads_y);

  auto* tbl = table_kvs.data_ptr<int>();
  auto* vk = vector_keys.data_ptr<int>();
  auto* qc = output_coords.data_ptr<int>();
  auto* ks = kernel_size_tensor.data_ptr<int>();

  // --- Pass 1: Count valid pairs per kernel offset ---
  auto counts =
      torch::zeros({num_kernels}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto* cnt = counts.data_ptr<int>();

  switch (hash_method) {
    case 0:
      kernel_map_size_4d_count_fnv1a<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, cnt, num_query, table_capacity, num_kernels);
      break;
    case 1:
      kernel_map_size_4d_count_city<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, cnt, num_query, table_capacity, num_kernels);
      break;
    case 2:
      kernel_map_size_4d_count_murmur<<<grid, block, 0, stream>>>(
          tbl, vk, qc, ks, cnt, num_query, table_capacity, num_kernels);
      break;
    default:
      TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
  }

  // --- Prefix sum to compute offsets ---
  auto offsets =
      torch::zeros({num_kernels + 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto cumsum_result = torch::cumsum(counts, 0, torch::kInt32);
  offsets.slice(0, 1, num_kernels + 1).copy_(cumsum_result);

  // Get total number of pairs (last element of offsets)
  int num_total_maps = offsets[num_kernels].item<int>();

  // --- Allocate output arrays ---
  auto in_maps =
      torch::empty({num_total_maps}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto out_maps =
      torch::empty({num_total_maps}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  if (num_total_maps > 0) {
    // --- Pass 2: Scatter valid pairs ---
    auto scatter_counters =
        torch::zeros({num_kernels}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto* off = offsets.data_ptr<int>();
    auto* sc = scatter_counters.data_ptr<int>();
    auto* im = in_maps.data_ptr<int>();
    auto* om = out_maps.data_ptr<int>();

    switch (hash_method) {
      case 0:
        kernel_map_size_4d_scatter_fnv1a<<<grid, block, 0, stream>>>(
            tbl, vk, qc, ks, off, sc, im, om, num_query, table_capacity, num_kernels);
        break;
      case 1:
        kernel_map_size_4d_scatter_city<<<grid, block, 0, stream>>>(
            tbl, vk, qc, ks, off, sc, im, om, num_query, table_capacity, num_kernels);
        break;
      case 2:
        kernel_map_size_4d_scatter_murmur<<<grid, block, 0, stream>>>(
            tbl, vk, qc, ks, off, sc, im, om, num_query, table_capacity, num_kernels);
        break;
      default:
        TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
    }
  }

  return std::make_tuple(in_maps, out_maps, offsets);
}

// ============================================================================
// Fused kernel map + mask data: kernel_map → pair_table → mask → CUB sort
// All in one C++ call, no Python round-trips between steps.
// ============================================================================

// Forward declarations from mask_data_kernels.cu
namespace warpconvnet {
namespace mask_data {
void csr_to_pair_table(const int* in_maps,
                       const int* out_maps,
                       const int* offsets,
                       int* pair_table,
                       int N_out,
                       int K,
                       int L);

void build_pair_mask(const int* pair_table, uint32_t* pair_mask, int N_out, int K);
}  // namespace mask_data
}  // namespace warpconvnet

// Helper: fill array with 0, 1, 2, ..., N-1
static __global__ void iota_kernel(int* vals, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) vals[idx] = idx;
}

// CUB radix sort wrapper for mask_argsort (replaces torch.argsort, no CPU sync)
void cub_argsort_uint32(const uint32_t* d_keys_in, int* d_values_out, int N, cudaStream_t stream) {
  auto* alloc = c10::cuda::CUDACachingAllocator::get();

  auto keys_buf = alloc->allocate(N * sizeof(uint32_t));
  auto vals_buf = alloc->allocate(N * sizeof(int));
  auto* d_keys_out = static_cast<uint32_t*>(keys_buf.get());
  auto* d_values_in = static_cast<int*>(vals_buf.get());

  {
    int threads = 256, blocks = (N + threads - 1) / threads;
    iota_kernel<<<blocks, threads, 0, stream>>>(d_values_in, N);
  }

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, N, 0, 32, stream);

  auto temp_buf = alloc->allocate(temp_bytes);
  cub::DeviceRadixSort::SortPairs(temp_buf.get(),
                                  temp_bytes,
                                  d_keys_in,
                                  d_keys_out,
                                  d_values_in,
                                  d_values_out,
                                  N,
                                  0,
                                  32,
                                  stream);
}

using MaskResult = std::tuple<torch::Tensor,
                              torch::Tensor,
                              torch::Tensor,  // in_maps, out_maps, offsets
                              torch::Tensor,
                              torch::Tensor,
                              torch::Tensor,  // pair_table, pair_mask, mask_argsort
                              torch::Tensor,
                              torch::Tensor,
                              torch::Tensor>;  // rev_pair_table, rev_mask, rev_argsort

MaskResult launch_fused_kernel_map_with_mask(torch::Tensor output_coords,
                                             torch::Tensor table_kvs,
                                             torch::Tensor vector_keys,
                                             int table_capacity,
                                             std::vector<int> kernel_size,
                                             int hash_method,
                                             bool build_reverse) {
  // Step 1: Generate kernel map (reuses existing two-pass implementation)
  auto [in_maps, out_maps, offsets] = launch_fused_kernel_map(
      output_coords, table_kvs, vector_keys, table_capacity, kernel_size, hash_method);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto device = output_coords.device();
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
  int N = output_coords.size(0);
  int K = kernel_size[0] * kernel_size[1] * kernel_size[2];

  // Step 2: Build pair_table from CSR kernel map
  auto pair_table = torch::full({K * N}, -1, opts);
  int num_pairs = offsets[K].item<int>();
  if (num_pairs > 0 && K <= 32) {
    auto offsets_gpu = offsets.to(device).to(torch::kInt32).contiguous();
    warpconvnet::mask_data::csr_to_pair_table(in_maps.data_ptr<int>(),
                                              out_maps.data_ptr<int>(),
                                              offsets_gpu.data_ptr<int>(),
                                              pair_table.data_ptr<int>(),
                                              N,
                                              K,
                                              num_pairs);
  }

  // Step 3: Build pair_mask from pair_table
  auto pair_mask = torch::zeros({N}, opts);
  if (num_pairs > 0 && K <= 32) {
    warpconvnet::mask_data::build_pair_mask(
        pair_table.data_ptr<int>(), reinterpret_cast<uint32_t*>(pair_mask.data_ptr<int>()), N, K);
  }

  // Step 4: CUB sort for mask_argsort
  auto mask_argsort = torch::empty({N}, opts);
  cub_argsort_uint32(reinterpret_cast<const uint32_t*>(pair_mask.data_ptr<int>()),
                     mask_argsort.data_ptr<int>(),
                     N,
                     stream);

  // Step 5: Build reverse mask data (for dgrad)
  auto rev_pair_table = torch::full({K * N}, -1, opts);
  auto rev_mask = torch::zeros({N}, opts);
  auto rev_argsort = torch::empty({N}, opts);

  if (build_reverse && num_pairs > 0 && K <= 32) {
    // Build reverse pair_table: for each valid entry in pair_table,
    // swap in_row and out_row. Vectorized via torch ops (no Python loop).
    auto pt_2d = pair_table.view({K, N});
    auto valid = pt_2d.ge(0);              // [K, N] bool
    auto indices = torch::nonzero(valid);  // [num_valid, 2] — (k, out_row)
    if (indices.numel() > 0) {
      auto k_idx = indices.select(1, 0);            // offset indices
      auto out_idx = indices.select(1, 1);          // out_row indices
      auto in_idx = pt_2d.index({k_idx, out_idx});  // in_row values
      // Scatter: rev_pair_table[k, in_row] = out_row
      auto rev_2d = rev_pair_table.view({K, N});
      rev_2d.index_put_({k_idx, in_idx.to(torch::kInt64)}, out_idx.to(torch::kInt32));
    }
    // Reverse mask + sort
    warpconvnet::mask_data::build_pair_mask(rev_pair_table.data_ptr<int>(),
                                            reinterpret_cast<uint32_t*>(rev_mask.data_ptr<int>()),
                                            N,
                                            K);
    cub_argsort_uint32(reinterpret_cast<const uint32_t*>(rev_mask.data_ptr<int>()),
                       rev_argsort.data_ptr<int>(),
                       N,
                       stream);
  }

  return {in_maps,
          out_maps,
          offsets,
          pair_table,
          pair_mask,
          mask_argsort,
          rev_pair_table,
          rev_mask,
          rev_argsort};
}

// =============================================================================
// Direct single-pass fused kernel map with mask data
// Replaces the multi-kernel pipeline (count + scatter + csr_to_pair + mask)
// with a single kernel that uses atomics for direct-write output.
// =============================================================================

// Forward declarations for the direct-write kernels (defined in fused_kernel_map.cu)
extern "C" __global__ void fused_kernel_map_direct_fnv1a(const int*,
                                                         const int*,
                                                         const int*,
                                                         int,
                                                         const int*,
                                                         int,
                                                         int,
                                                         int*,
                                                         int*,
                                                         int*,
                                                         int,
                                                         int*,
                                                         uint32_t*,
                                                         int*,
                                                         uint32_t*);
extern "C" __global__ void fused_kernel_map_direct_city(const int*,
                                                        const int*,
                                                        const int*,
                                                        int,
                                                        const int*,
                                                        int,
                                                        int,
                                                        int*,
                                                        int*,
                                                        int*,
                                                        int,
                                                        int*,
                                                        uint32_t*,
                                                        int*,
                                                        uint32_t*);
extern "C" __global__ void fused_kernel_map_direct_murmur(const int*,
                                                          const int*,
                                                          const int*,
                                                          int,
                                                          const int*,
                                                          int,
                                                          int,
                                                          int*,
                                                          int*,
                                                          int*,
                                                          int,
                                                          int*,
                                                          uint32_t*,
                                                          int*,
                                                          uint32_t*);

MaskResult launch_fused_kernel_map_direct(torch::Tensor output_coords,
                                          torch::Tensor table_kvs,
                                          torch::Tensor vector_keys,
                                          int table_capacity,
                                          std::vector<int> kernel_size,
                                          int hash_method,
                                          bool build_reverse) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto device = output_coords.device();
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(device);

  int M = output_coords.size(0);
  int K = kernel_size[0] * kernel_size[1] * kernel_size[2];

  // Allocate all outputs
  auto pair_counts = torch::zeros({K}, opts);
  auto pair_table = torch::full({K * M}, -1, opts);
  auto pair_mask = torch::zeros({M}, opts);

  // CSR output: use M as max_pairs_per_offset (conservative upper bound)
  auto in_maps_padded = torch::empty({K * M}, opts);
  auto out_maps_padded = torch::empty({K * M}, opts);

  // Reverse outputs
  int N_in = vector_keys.size(0);
  auto rev_pair_table = torch::full({K * N_in}, -1, opts);
  auto rev_mask = torch::zeros({N_in}, opts);

  // Kernel size on GPU
  auto kernel_size_tensor = torch::tensor({kernel_size[0], kernel_size[1], kernel_size[2]}, opts);

  // Launch 2D grid: (output voxels, kernel offsets)
  dim3 block(256, 1);
  dim3 grid((M + block.x - 1) / block.x, K);

  auto launch_kernel = [&](auto kernel_fn) {
    kernel_fn<<<grid, block, 0, stream>>>(
        output_coords.data_ptr<int>(),
        table_kvs.data_ptr<int>(),
        vector_keys.data_ptr<int>(),
        table_capacity,
        kernel_size_tensor.data_ptr<int>(),
        M,
        K,
        in_maps_padded.data_ptr<int>(),
        out_maps_padded.data_ptr<int>(),
        pair_counts.data_ptr<int>(),
        M,  // max_pairs_per_offset
        pair_table.data_ptr<int>(),
        reinterpret_cast<uint32_t*>(pair_mask.data_ptr<int>()),
        build_reverse ? rev_pair_table.data_ptr<int>() : nullptr,
        build_reverse ? reinterpret_cast<uint32_t*>(rev_mask.data_ptr<int>()) : nullptr);
  };

  switch (hash_method) {
    case 0:
      launch_kernel(fused_kernel_map_direct_fnv1a);
      break;
    case 1:
      launch_kernel(fused_kernel_map_direct_city);
      break;
    case 2:
      launch_kernel(fused_kernel_map_direct_murmur);
      break;
    default:
      TORCH_CHECK(false, "Unknown hash method: ", hash_method);
  }

  // Compact CSR from padded layout using pair_counts
  auto counts_cpu = pair_counts.cpu();
  auto counts_ptr = counts_cpu.data_ptr<int>();
  int64_t total_pairs = 0;
  std::vector<int64_t> offsets_vec(K + 1);
  offsets_vec[0] = 0;
  for (int i = 0; i < K; ++i) {
    total_pairs += counts_ptr[i];
    offsets_vec[i + 1] = total_pairs;
  }

  auto offsets = torch::tensor(offsets_vec, torch::TensorOptions().dtype(torch::kInt64));
  auto in_maps = torch::empty({total_pairs}, opts);
  auto out_maps = torch::empty({total_pairs}, opts);

  // Copy compacted CSR data
  for (int i = 0; i < K; ++i) {
    int count = counts_ptr[i];
    if (count > 0) {
      in_maps.slice(0, offsets_vec[i], offsets_vec[i + 1])
          .copy_(in_maps_padded.slice(0, (int64_t)i * M, (int64_t)i * M + count));
      out_maps.slice(0, offsets_vec[i], offsets_vec[i + 1])
          .copy_(out_maps_padded.slice(0, (int64_t)i * M, (int64_t)i * M + count));
    }
  }

  // Argsort for pair_mask
  auto mask_argsort = torch::empty({M}, opts);
  cub_argsort_uint32(reinterpret_cast<const uint32_t*>(pair_mask.data_ptr<int>()),
                     mask_argsort.data_ptr<int>(),
                     M,
                     stream);

  // Argsort for reverse mask
  auto rev_argsort = torch::empty({N_in}, opts);
  if (build_reverse) {
    warpconvnet::mask_data::build_pair_mask(rev_pair_table.data_ptr<int>(),
                                            reinterpret_cast<uint32_t*>(rev_mask.data_ptr<int>()),
                                            N_in,
                                            K);
    cub_argsort_uint32(reinterpret_cast<const uint32_t*>(rev_mask.data_ptr<int>()),
                       rev_argsort.data_ptr<int>(),
                       N_in,
                       stream);
  }

  return {in_maps,
          out_maps,
          offsets,
          pair_table,
          pair_mask,
          mask_argsort,
          rev_pair_table,
          rev_mask,
          rev_argsort};
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

extern "C" __global__ void window_group_histogram_kernel(const int* __restrict__ grid_coord,
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

extern "C" __global__ void window_group_scatter_kernel(const int64_t* __restrict__ codes,
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
  window_group_histogram_kernel<<<blocks, threads, 0, stream>>>(grid_coord.data_ptr<int>(),
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
  window_group_scatter_kernel<<<blocks, threads, 0, stream>>>(codes.data_ptr<int64_t>(),
                                                              window_offsets_dense.data_ptr<int>(),
                                                              scatter_counters.data_ptr<int>(),
                                                              perm.data_ptr<int64_t>(),
                                                              inverse_perm.data_ptr<int64_t>(),
                                                              N);
}
