// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// cuhash - Optimized CUDA Hash Table Library
// hash_table.cu - Hash table CUDA kernels and host launcher functions
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cuhash/hash_table.cuh"

namespace cuhash {

// ============================================================================
// Packed Key Hash Table Kernels
// ============================================================================

__global__ void packed_prepare_kernel(uint64_t *keys, int *values, int capacity) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < capacity) {
    keys[tid] = kEmpty;
    values[tid] = -1;
  }
}

__global__ void packed_insert_kernel(uint64_t *__restrict__ keys,
                                     int *__restrict__ values,
                                     const int *__restrict__ coords,
                                     int num_keys,
                                     uint32_t capacity_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_keys) return;

  const int *key = &coords[idx * 4];
  uint64_t packed = pack_key_4d(key[0], key[1], key[2], key[3]);
  packed_insert(keys, values, packed, idx, capacity_mask);
}

__global__ void packed_insert_double_kernel(uint64_t *__restrict__ keys,
                                            int *__restrict__ values,
                                            const int *__restrict__ coords,
                                            int num_keys,
                                            uint32_t capacity_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_keys) return;

  const int *key = &coords[idx * 4];
  uint64_t packed = pack_key_4d(key[0], key[1], key[2], key[3]);
  packed_insert_double(keys, values, packed, idx, capacity_mask);
}

__global__ void packed_search_kernel(const uint64_t *__restrict__ keys,
                                     const int *__restrict__ values,
                                     const int *__restrict__ search_coords,
                                     int *__restrict__ results,
                                     int num_search,
                                     uint32_t capacity_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_search) return;

  const int *key = &search_coords[idx * 4];
  uint64_t packed = pack_key_4d(key[0], key[1], key[2], key[3]);
  results[idx] = packed_search(keys, values, packed, capacity_mask);
}

__global__ void packed_search_double_kernel(const uint64_t *__restrict__ keys,
                                            const int *__restrict__ values,
                                            const int *__restrict__ search_coords,
                                            int *__restrict__ results,
                                            int num_search,
                                            uint32_t capacity_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_search) return;

  const int *key = &search_coords[idx * 4];
  uint64_t packed = pack_key_4d(key[0], key[1], key[2], key[3]);
  results[idx] = packed_search_double(keys, values, packed, capacity_mask);
}

// Warp-cooperative search: each WARP handles one query
__global__ void packed_warp_search_kernel(const uint64_t *__restrict__ keys,
                                          const int *__restrict__ values,
                                          const int *__restrict__ search_coords,
                                          int *__restrict__ results,
                                          int num_search,
                                          uint32_t capacity_mask) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x & 31;
  if (warp_id >= num_search) return;

  const int *key = &search_coords[warp_id * 4];
  uint64_t packed = pack_key_4d(key[0], key[1], key[2], key[3]);
  int result = packed_warp_search(keys, values, packed, capacity_mask);
  if (lane == 0) {
    results[warp_id] = result;
  }
}

// ============================================================================
// Generic Key Hash Table Kernels (for arbitrary key_dim, backward compat)
// ============================================================================

__global__ void generic_prepare_kernel(int *table_kvs, int capacity) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < capacity) {
    table_kvs[2 * tid + 0] = -1;
    table_kvs[2 * tid + 1] = -1;
  }
}

template <typename HashFuncT>
__global__ void generic_insert_kernel(
    int *table_kvs, const int *vector_keys, int num_keys, int key_dim, uint32_t capacity_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_keys) return;

  const int *key = &vector_keys[idx * key_dim];
  uint32_t slot = HashFuncT::hash(key, key_dim, capacity_mask);

  while (true) {
    // 64-bit CAS to claim slot and set value atomically
    unsigned long long *slot_addr = reinterpret_cast<unsigned long long *>(&table_kvs[slot * 2]);
    unsigned long long old_val = *slot_addr;
    int slot_marker = static_cast<int>(old_val & 0xFFFFFFFF);

    if (slot_marker == -1) {
      unsigned long long new_val =
          (static_cast<unsigned long long>(idx) << 32) | static_cast<unsigned int>(slot);
      unsigned long long prev = atomicCAS(slot_addr, old_val, new_val);
      if (prev == old_val) return;  // Success
      old_val = prev;
      slot_marker = static_cast<int>(old_val & 0xFFFFFFFF);
    }

    if (slot_marker != -1) {
      int vector_index = static_cast<int>(old_val >> 32);
      if (vector_index >= 0) {
        const int *existing = &vector_keys[vector_index * key_dim];
        if (key_dim == 4 ? vec_equal_4d(existing, key) : vec_equal(existing, key, key_dim)) {
          return;  // Dedup
        }
      }
      slot = (slot + 1) & capacity_mask;
      continue;
    }
  }
}

template <typename HashFuncT>
__global__ void generic_search_kernel(const int *__restrict__ table_kvs,
                                      const int *__restrict__ vector_keys,
                                      const int *__restrict__ search_keys,
                                      int *__restrict__ results,
                                      int num_search,
                                      int key_dim,
                                      uint32_t capacity_mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_search) return;

  const int *query = &search_keys[idx * key_dim];
  results[idx] = generic_search<HashFuncT>(table_kvs, vector_keys, query, key_dim, capacity_mask);
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

static constexpr int kBlockSize = 256;

// --- Packed key table ---

void launch_packed_prepare(torch::Tensor keys, torch::Tensor values, int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (capacity + kBlockSize - 1) / kBlockSize;
  packed_prepare_kernel<<<blocks, kBlockSize, 0, stream>>>(
      reinterpret_cast<uint64_t *>(keys.data_ptr<int64_t>()), values.data_ptr<int>(), capacity);
}

void launch_packed_insert(torch::Tensor keys,
                          torch::Tensor values,
                          torch::Tensor coords,
                          int num_keys,
                          int capacity,
                          bool use_double_hash) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (num_keys + kBlockSize - 1) / kBlockSize;
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  if (use_double_hash) {
    packed_insert_double_kernel<<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<uint64_t *>(keys.data_ptr<int64_t>()),
        values.data_ptr<int>(),
        coords.data_ptr<int>(),
        num_keys,
        mask);
  } else {
    packed_insert_kernel<<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<uint64_t *>(keys.data_ptr<int64_t>()),
        values.data_ptr<int>(),
        coords.data_ptr<int>(),
        num_keys,
        mask);
  }
}

void launch_packed_search(torch::Tensor keys,
                          torch::Tensor values,
                          torch::Tensor search_coords,
                          torch::Tensor results,
                          int num_search,
                          int capacity,
                          int search_mode) {
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t mask = static_cast<uint32_t>(capacity - 1);

  if (search_mode == 2) {
    // Warp-cooperative: need 32 threads per query
    int total_threads = num_search * 32;
    int blocks = (total_threads + kBlockSize - 1) / kBlockSize;
    packed_warp_search_kernel<<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
        values.data_ptr<int>(),
        search_coords.data_ptr<int>(),
        results.data_ptr<int>(),
        num_search,
        mask);
  } else if (search_mode == 1) {
    // Double hash search
    int blocks = (num_search + kBlockSize - 1) / kBlockSize;
    packed_search_double_kernel<<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
        values.data_ptr<int>(),
        search_coords.data_ptr<int>(),
        results.data_ptr<int>(),
        num_search,
        mask);
  } else {
    // Linear probe search
    int blocks = (num_search + kBlockSize - 1) / kBlockSize;
    packed_search_kernel<<<blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const uint64_t *>(keys.data_ptr<int64_t>()),
        values.data_ptr<int>(),
        search_coords.data_ptr<int>(),
        results.data_ptr<int>(),
        num_search,
        mask);
  }
}

// --- Generic key table ---

void launch_generic_prepare(torch::Tensor table_kvs, int capacity) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (capacity + kBlockSize - 1) / kBlockSize;
  generic_prepare_kernel<<<blocks, kBlockSize, 0, stream>>>(table_kvs.data_ptr<int>(), capacity);
}

void launch_generic_insert(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           int num_keys,
                           int key_dim,
                           int capacity,
                           int hash_method) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (num_keys + kBlockSize - 1) / kBlockSize;
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  if (hash_method == 0) {
    generic_insert_kernel<FNV1AHash><<<blocks, kBlockSize, 0, stream>>>(
        table_kvs.data_ptr<int>(), vector_keys.data_ptr<int>(), num_keys, key_dim, mask);
  } else {
    generic_insert_kernel<MurmurHash><<<blocks, kBlockSize, 0, stream>>>(
        table_kvs.data_ptr<int>(), vector_keys.data_ptr<int>(), num_keys, key_dim, mask);
  }
}

void launch_generic_search(torch::Tensor table_kvs,
                           torch::Tensor vector_keys,
                           torch::Tensor search_keys,
                           torch::Tensor results,
                           int num_search,
                           int key_dim,
                           int capacity,
                           int hash_method) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (num_search + kBlockSize - 1) / kBlockSize;
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  if (hash_method == 0) {
    generic_search_kernel<FNV1AHash><<<blocks, kBlockSize, 0, stream>>>(table_kvs.data_ptr<int>(),
                                                                        vector_keys.data_ptr<int>(),
                                                                        search_keys.data_ptr<int>(),
                                                                        results.data_ptr<int>(),
                                                                        num_search,
                                                                        key_dim,
                                                                        mask);
  } else {
    generic_search_kernel<MurmurHash>
        <<<blocks, kBlockSize, 0, stream>>>(table_kvs.data_ptr<int>(),
                                            vector_keys.data_ptr<int>(),
                                            search_keys.data_ptr<int>(),
                                            results.data_ptr<int>(),
                                            num_search,
                                            key_dim,
                                            mask);
  }
}

// ============================================================================
// Packed expand_with_offsets: for each (base, offset) pair, insert
// base+offset into the packed hash table if not already present.
// New entries get a fresh index via atomicAdd on num_entries_ptr.
// The unpacked coordinates are stored in coords_store for backward compat.
// ============================================================================

__global__ void packed_expand_insert_kernel(uint64_t *__restrict__ keys,
                                            int *__restrict__ values,
                                            int *__restrict__ coords_store,
                                            const int *__restrict__ base_coords,
                                            const int *__restrict__ offsets,
                                            int num_base,
                                            int num_offsets,
                                            uint32_t capacity_mask,
                                            int vector_capacity,
                                            int *num_entries_ptr,
                                            int *status_ptr) {
  long long gidx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  long long total = static_cast<long long>(num_base) * static_cast<long long>(num_offsets);
  if (gidx >= total) return;

  int offset_idx = static_cast<int>(gidx / num_base);
  int base_idx = static_cast<int>(gidx % num_base);

  // Compute candidate coordinate
  int b = base_coords[base_idx * 4 + 0] + offsets[offset_idx * 4 + 0];
  int x = base_coords[base_idx * 4 + 1] + offsets[offset_idx * 4 + 1];
  int y = base_coords[base_idx * 4 + 2] + offsets[offset_idx * 4 + 2];
  int z = base_coords[base_idx * 4 + 3] + offsets[offset_idx * 4 + 3];
  uint64_t packed = pack_key_4d(b, x, y, z);

  // Probe and insert
  uint32_t slot = Splitmix64Hash::hash(packed, capacity_mask);
  uint32_t attempts = 0;
  while (attempts <= capacity_mask) {
    uint64_t prev = atomicCAS_u64(&keys[slot], kEmpty, packed);
    if (prev == kEmpty) {
      // Claimed empty slot — allocate a new index
      int idx = atomicAdd(num_entries_ptr, 1);
      if (idx >= vector_capacity) {
        atomicMax(status_ptr, 1);  // VECTOR_OVERFLOW
        return;
      }
      values[slot] = idx;
      coords_store[idx * 4 + 0] = b;
      coords_store[idx * 4 + 1] = x;
      coords_store[idx * 4 + 2] = y;
      coords_store[idx * 4 + 3] = z;
      return;
    }
    if (prev == packed) {
      return;  // Already exists
    }
    slot = (slot + 1) & capacity_mask;
    ++attempts;
  }
  // Table full
  atomicMax(status_ptr, 2);  // TABLE_FULL
}

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
                                 torch::Tensor status_tensor) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long long total = static_cast<long long>(num_base) * static_cast<long long>(num_offsets);
  int blocks = (total + kBlockSize - 1) / kBlockSize;
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  packed_expand_insert_kernel<<<blocks, kBlockSize, 0, stream>>>(
      reinterpret_cast<uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      coords_store.data_ptr<int>(),
      base_coords.data_ptr<int>(),
      offsets.data_ptr<int>(),
      num_base,
      num_offsets,
      mask,
      vector_capacity,
      num_entries_tensor.data_ptr<int>(),
      status_tensor.data_ptr<int>());
}

// ============================================================================
// Build coarse hash table directly from fine coordinates via floor-div.
// Each thread takes one fine coord, computes coarse = floor(coord / stride),
// packs and inserts with dedup. Replaces torch.unique + from_coords.
// ============================================================================

__global__ void packed_build_coarse_kernel(uint64_t *__restrict__ keys,
                                           int *__restrict__ values,
                                           const int *__restrict__ fine_coords,
                                           int num_fine,
                                           int stride_shift,
                                           uint32_t capacity_mask,
                                           int *num_entries_ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_fine) return;

  int b = fine_coords[idx * 4 + 0];
  int x = fine_coords[idx * 4 + 1] >> stride_shift;
  int y = fine_coords[idx * 4 + 2] >> stride_shift;
  int z = fine_coords[idx * 4 + 3] >> stride_shift;
  uint64_t packed = pack_key_4d(b, x, y, z);

  uint32_t slot = Splitmix64Hash::hash(packed, capacity_mask);
  uint32_t attempts = 0;
  while (attempts <= capacity_mask) {
    uint64_t prev = atomicCAS_u64(&keys[slot], kEmpty, packed);
    if (prev == kEmpty) {
      int entry_idx = atomicAdd(num_entries_ptr, 1);
      values[slot] = entry_idx;
      return;
    }
    if (prev == packed) {
      return;
    }
    slot = (slot + 1) & capacity_mask;
    ++attempts;
  }
}

void launch_packed_build_coarse(torch::Tensor keys,
                                torch::Tensor values,
                                torch::Tensor fine_coords,
                                int num_fine,
                                int stride_shift,
                                int capacity,
                                torch::Tensor num_entries_tensor) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int blocks = (num_fine + kBlockSize - 1) / kBlockSize;
  uint32_t mask = static_cast<uint32_t>(capacity - 1);
  packed_build_coarse_kernel<<<blocks, kBlockSize, 0, stream>>>(
      reinterpret_cast<uint64_t *>(keys.data_ptr<int64_t>()),
      values.data_ptr<int>(),
      fine_coords.data_ptr<int>(),
      num_fine,
      stride_shift,
      mask,
      num_entries_tensor.data_ptr<int>());
}

}  // namespace cuhash
