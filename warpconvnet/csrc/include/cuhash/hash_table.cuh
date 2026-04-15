// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// cuhash - Optimized CUDA Hash Table Library
// hash_table.cuh - Hash table device functions for packed and generic keys
#pragma once

#include "cuhash/hash_functions.cuh"

namespace cuhash {

// ============================================================================
// Packed Key Hash Table (4D coordinates packed into uint64)
//
// Layout:
//   keys[capacity]   : uint64_t packed keys (kEmpty = sentinel)
//   values[capacity] : int32_t  original indices
//
// Key advantages over separate table_kvs + vector_keys:
//   - No vector_keys indirection: key is inline, single load per probe
//   - Single uint64 comparison instead of 4-element vec_equal
//   - Hash computed from single uint64 instead of 4-element loop
//   - ~50% less memory for keys (8 bytes vs 16 bytes per key)
// ============================================================================

// Helper: atomicCAS on uint64_t via unsigned long long cast
__device__ __forceinline__ uint64_t atomicCAS_u64(uint64_t *addr, uint64_t compare, uint64_t val) {
  return static_cast<uint64_t>(atomicCAS(reinterpret_cast<unsigned long long *>(addr),
                                         static_cast<unsigned long long>(compare),
                                         static_cast<unsigned long long>(val)));
}

// --- Insert (per-thread, linear probing) ---
__device__ __forceinline__ void packed_insert(uint64_t *__restrict__ keys,
                                              int *__restrict__ values,
                                              uint64_t packed_key,
                                              int value,
                                              uint32_t capacity_mask) {
  uint32_t slot = Splitmix64Hash::hash(packed_key, capacity_mask);
  uint32_t attempts = 0;
  while (attempts <= capacity_mask) {
    uint64_t prev = atomicCAS_u64(&keys[slot], kEmpty, packed_key);
    if (prev == kEmpty) {
      // Claimed empty slot
      values[slot] = value;
      return;
    }
    if (prev == packed_key) {
      // Key already exists (dedup)
      return;
    }
    // Collision: linear probe
    slot = (slot + 1) & capacity_mask;
    ++attempts;
  }
  // Table full - should not happen with proper sizing
}

// --- Insert (per-thread, double hashing probe) ---
__device__ __forceinline__ void packed_insert_double(uint64_t *__restrict__ keys,
                                                     int *__restrict__ values,
                                                     uint64_t packed_key,
                                                     int value,
                                                     uint32_t capacity_mask) {
  uint32_t slot = Splitmix64Hash::hash(packed_key, capacity_mask);
  uint32_t stride = double_hash_stride(packed_key, capacity_mask);
  uint32_t attempts = 0;
  while (attempts <= capacity_mask) {
    uint64_t prev = atomicCAS_u64(&keys[slot], kEmpty, packed_key);
    if (prev == kEmpty) {
      values[slot] = value;
      return;
    }
    if (prev == packed_key) {
      return;  // Dedup
    }
    slot = (slot + stride) & capacity_mask;
    ++attempts;
  }
}

// --- Search (per-thread, linear probing) ---
__device__ __forceinline__ int packed_search(const uint64_t *__restrict__ keys,
                                             const int *__restrict__ values,
                                             uint64_t query_key,
                                             uint32_t capacity_mask) {
  uint32_t slot = Splitmix64Hash::hash(query_key, capacity_mask);
  uint32_t attempts = 0;
  while (attempts <= capacity_mask) {
    uint64_t k = keys[slot];
    if (k == kEmpty) return -1;
    if (k == query_key) return values[slot];
    slot = (slot + 1) & capacity_mask;
    ++attempts;
  }
  return -1;
}

// --- Search (per-thread, double hashing) ---
__device__ __forceinline__ int packed_search_double(const uint64_t *__restrict__ keys,
                                                    const int *__restrict__ values,
                                                    uint64_t query_key,
                                                    uint32_t capacity_mask) {
  uint32_t slot = Splitmix64Hash::hash(query_key, capacity_mask);
  uint32_t stride = double_hash_stride(query_key, capacity_mask);
  uint32_t attempts = 0;
  while (attempts <= capacity_mask) {
    uint64_t k = keys[slot];
    if (k == kEmpty) return -1;
    if (k == query_key) return values[slot];
    slot = (slot + stride) & capacity_mask;
    ++attempts;
  }
  return -1;
}

// --- Warp-Cooperative Search (32 threads probe 32 slots in parallel) ---
__device__ __forceinline__ int packed_warp_search(const uint64_t *__restrict__ keys,
                                                  const int *__restrict__ values,
                                                  uint64_t query_key,
                                                  uint32_t capacity_mask) {
  int lane = threadIdx.x & 31;
  uint32_t slot = Splitmix64Hash::hash(query_key, capacity_mask);

  for (uint32_t attempt = 0; attempt <= capacity_mask; attempt += 32) {
    uint32_t my_slot = (slot + lane + attempt) & capacity_mask;
    uint64_t k = keys[my_slot];

    // Check for empty slot
    unsigned empty_mask = __ballot_sync(0xFFFFFFFF, k == kEmpty);
    // Check for match
    bool match = (k == query_key);
    unsigned match_mask = __ballot_sync(0xFFFFFFFF, match);

    if (match_mask != 0) {
      int winner = __ffs(match_mask) - 1;
      int val = values[my_slot];
      return __shfl_sync(0xFFFFFFFF, val, winner);
    }
    if (empty_mask != 0) {
      return -1;  // Empty before match means key absent
    }
  }
  return -1;
}

// ============================================================================
// Generic Key Hash Table (arbitrary key_dim, same layout as warpconvnet)
//
// Layout:
//   table_kvs[capacity * 2]    : interleaved (slot_marker, vector_index)
//   vector_keys[N * key_dim]   : stored keys
// ============================================================================

template <typename HashFuncT>
__device__ __forceinline__ int generic_search(const int *__restrict__ table_kvs,
                                              const int *__restrict__ vector_keys,
                                              const int *__restrict__ query_key,
                                              int key_dim,
                                              uint32_t capacity_mask) {
  uint32_t slot = HashFuncT::hash(query_key, key_dim, capacity_mask);
  uint32_t initial_slot = slot;
  uint32_t attempts = 0;

  while (attempts <= capacity_mask) {
    // 64-bit load for (slot_marker, vector_index)
    long long pair = *reinterpret_cast<const long long *>(&table_kvs[slot * 2]);
    int slot_marker = static_cast<int>(pair & 0xFFFFFFFF);
    int vector_index = static_cast<int>(pair >> 32);

    if (slot_marker == -1) return -1;
    if (vector_index >= 0) {
      bool match = (key_dim == 4)
                       ? vec_equal_4d(&vector_keys[vector_index * 4], query_key)
                       : vec_equal(&vector_keys[vector_index * key_dim], query_key, key_dim);
      if (match) return vector_index;
    }
    slot = (slot + 1) & capacity_mask;
    if (slot == initial_slot) return -1;
    ++attempts;
  }
  return -1;
}

}  // namespace cuhash
