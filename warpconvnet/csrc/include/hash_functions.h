// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Shared hash function definitions used by hashmap_kernels.cu and radius_search_kernels.cu.
// Also included by cuhash_hash_table.cu for packed hash table operations.

#pragma once

#include <cstdint>

// --- Hash Helper Functions ---

__device__ inline uint32_t murmur_32_scramble(uint32_t k) {
  k *= 0xCC9E2D51;
  k = (k << 15) | (k >> 17);
  k *= 0x1B873593;
  return k;
}

__device__ inline uint32_t _hash_murmur_impl(uint32_t h, uint32_t k) {
  h ^= murmur_32_scramble(k);
  h = (h << 13) | (h >> 19);
  h = h * 5 + 0xE6546B64;
  return h;
}

__device__ inline uint32_t _hash_murmur_finalize(uint32_t h, int length_bytes) {
  h ^= length_bytes;
  h ^= h >> 16;
  h *= 0x85EBCA6B;
  h ^= h >> 13;
  h *= 0xC2B2AE35;
  h ^= h >> 16;
  return h;
}

__device__ inline uint32_t _hash_fnv1a_impl(uint32_t hash_val, uint32_t key) {
  hash_val ^= key;
  hash_val *= 16777619;  // FNV prime
  return hash_val;
}

__device__ inline uint32_t _hash_city_impl(uint32_t hash_val, uint32_t key) {
  hash_val += key * 0x9E3779B9;
  hash_val ^= hash_val >> 16;
  hash_val *= 0x85EBCA6B;
  hash_val ^= hash_val >> 13;
  hash_val *= 0xC2B2AE35;
  hash_val ^= hash_val >> 16;
  return hash_val;
}

// --- Hash Function Structs/Functors ---

struct FNV1AHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t hash_val = 2166136261u;  // FNV offset basis
    for (int i = 0; i < key_dim; ++i) {
      hash_val = _hash_fnv1a_impl(hash_val, (uint32_t)key[i]);
    }
    int signed_hash = (int)hash_val;
    return ((signed_hash % capacity) + capacity) % capacity;
  }
};

struct CityHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t hash_val = 0;
    for (int i = 0; i < key_dim; ++i) {
      hash_val = _hash_city_impl(hash_val, (uint32_t)key[i]);
    }
    int signed_hash = (int)hash_val;
    return ((signed_hash % capacity) + capacity) % capacity;
  }
};

struct MurmurHash {
  __device__ inline static int hash(const int* key, int key_dim, int capacity) {
    uint32_t h = 0x9747B28Cu;  // Seed
    for (int i = 0; i < key_dim; ++i) {
      h = _hash_murmur_impl(h, (uint32_t)key[i]);
    }
    h = _hash_murmur_finalize(h, key_dim * 4);
    int signed_hash = (int)h;
    return ((signed_hash % capacity) + capacity) % capacity;
  }
};

// --- Vector Comparison ---
template <typename T>
__device__ inline bool vec_equal(const T* a, const int* b, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// Non-templated overload for int-int comparison
__device__ inline bool vec_equal(const int* a, const int* b, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// --- Device Function for Hash Table Search ---
template <typename HashFuncT>
__device__ inline int search_hash_table(const int* __restrict__ table_kvs,
                                        const int* __restrict__ vector_keys,
                                        const int* __restrict__ query_key,
                                        int key_dim,
                                        int table_capacity) {
  int slot = HashFuncT::hash(query_key, key_dim, table_capacity);
  int initial_slot = slot;
  int attempts = 0;

  while (attempts < table_capacity) {
    int slot_marker = table_kvs[slot * 2 + 0];
    if (slot_marker == -1) {
      return -1;
    }

    int vector_index = table_kvs[slot * 2 + 1];
    if (vector_index < 0) {
      continue;
    }

    const int* candidate_key = &vector_keys[vector_index * key_dim];
    if (vec_equal(candidate_key, query_key, key_dim)) {
      return vector_index;
    }
    slot = (slot + 1) % table_capacity;

    if (slot == initial_slot) {
      return -1;
    }
    attempts++;
  }
  return -1;
}
