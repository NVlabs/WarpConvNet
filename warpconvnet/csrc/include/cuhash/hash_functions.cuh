// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// cuhash - Optimized CUDA Hash Table Library
// hash_functions.cuh - Hash functions and key packing utilities
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cuhash {

// ============================================================================
// 4D Key Packing: (batch, x, y, z) -> uint64_t
//
// Bit layout (MSB to LSB):
//   batch:  10 bits unsigned [0, 1023]         — bits 63..54
//   x:      18 bits signed   [-131072, 131071] — bits 53..36
//   y:      18 bits signed   [-131072, 131071] — bits 35..18
//   z:      18 bits signed   [-131072, 131071] — bits 17..0
//
// Rationale: batch is typically tiny (< 32), spatial coords need wider range.
// 18-bit signed covers ±131071, which at 0.01m voxel size spans ±1.3 km.
// ============================================================================

static constexpr int kBatchBits = 10;
static constexpr int kCoordBits = 18;
static constexpr uint32_t kBatchMask = (1u << kBatchBits) - 1;  // 0x3FF
static constexpr uint32_t kCoordMask = (1u << kCoordBits) - 1;  // 0x3FFFF
static constexpr int kCoordMax = (1 << (kCoordBits - 1)) - 1;   // 131071
static constexpr int kCoordMin = -(1 << (kCoordBits - 1));      // -131072

static constexpr uint64_t kEmpty = 0xFFFFFFFFFFFFFFFFull;

__host__ __device__ __forceinline__ uint64_t pack_key_4d(int b, int x, int y, int z) {
  return (static_cast<uint64_t>(b & kBatchMask) << 54) |
         (static_cast<uint64_t>(x & kCoordMask) << 36) |
         (static_cast<uint64_t>(y & kCoordMask) << 18) | static_cast<uint64_t>(z & kCoordMask);
}

__host__ __device__ __forceinline__ uint64_t pack_key_4d_from_ptr(const int *key) {
  return pack_key_4d(key[0], key[1], key[2], key[3]);
}

// Sign-extend an N-bit value stored in the low N bits of v
__device__ __forceinline__ int sign_extend(uint32_t v, int bits) {
  uint32_t sign_bit = 1u << (bits - 1);
  return static_cast<int>((v ^ sign_bit) - sign_bit);
}

__device__ __forceinline__ void unpack_key_4d(uint64_t packed, int &b, int &x, int &y, int &z) {
  b = static_cast<int>((packed >> 54) & kBatchMask);
  x = sign_extend(static_cast<uint32_t>((packed >> 36) & kCoordMask), kCoordBits);
  y = sign_extend(static_cast<uint32_t>((packed >> 18) & kCoordMask), kCoordBits);
  z = sign_extend(static_cast<uint32_t>(packed & kCoordMask), kCoordBits);
}

// Compute packed key for (base + offset) without full unpack/repack
__device__ __forceinline__ uint64_t offset_packed_key_4d(uint64_t base, int ox, int oy, int oz) {
  int b, x, y, z;
  unpack_key_4d(base, b, x, y, z);
  return pack_key_4d(b, x + ox, y + oy, z + oz);
}

// ============================================================================
// Hash Functions for Packed uint64 Keys
// ============================================================================

// Splitmix64 - excellent avalanche properties for 64-bit keys
struct Splitmix64Hash {
  __device__ __forceinline__ static uint32_t hash(uint64_t key, uint32_t capacity_mask) {
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9ull;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBull;
    key ^= key >> 31;
    return static_cast<uint32_t>(key) & capacity_mask;
  }
};

// Fibonacci hashing - fast single multiply
struct FibonacciHash {
  __device__ __forceinline__ static uint32_t hash(uint64_t key, uint32_t capacity_mask) {
    // Golden ratio constant for 64-bit
    return static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> 32) & capacity_mask;
  }
};

// Double hash: secondary hash for double-hashing probe strategy
// Returns an odd stride to ensure full table coverage with power-of-2 capacity
__device__ __forceinline__ uint32_t double_hash_stride(uint64_t key, uint32_t capacity_mask) {
  key ^= key >> 33;
  key *= 0xFF51AFD7ED558CCDull;
  key ^= key >> 33;
  // Ensure odd stride (|1) so it's coprime with power-of-2 capacity
  return (static_cast<uint32_t>(key) & capacity_mask) | 1u;
}

// ============================================================================
// Hash Functions for Generic int32 Array Keys (backward compat)
// ============================================================================

__device__ __forceinline__ uint32_t murmur_scramble(uint32_t k) {
  k *= 0xCC9E2D51;
  k = (k << 15) | (k >> 17);
  k *= 0x1B873593;
  return k;
}

struct MurmurHash {
  __device__ __forceinline__ static uint32_t hash(const int *key,
                                                  int key_dim,
                                                  uint32_t capacity_mask) {
    uint32_t h = 0x9747B28Cu;
    for (int i = 0; i < key_dim; ++i) {
      h ^= murmur_scramble(static_cast<uint32_t>(key[i]));
      h = (h << 13) | (h >> 19);
      h = h * 5 + 0xE6546B64;
    }
    // Finalize
    h ^= static_cast<uint32_t>(key_dim * 4);
    h ^= h >> 16;
    h *= 0x85EBCA6B;
    h ^= h >> 13;
    h *= 0xC2B2AE35;
    h ^= h >> 16;
    return h & capacity_mask;
  }
};

struct FNV1AHash {
  __device__ __forceinline__ static uint32_t hash(const int *key,
                                                  int key_dim,
                                                  uint32_t capacity_mask) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < key_dim; ++i) {
      h ^= static_cast<uint32_t>(key[i]);
      h *= 16777619u;
    }
    return h & capacity_mask;
  }
};

// ============================================================================
// Key Comparison Helpers
// ============================================================================

__device__ __forceinline__ bool vec_equal(const int *a, const int *b, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

__device__ __forceinline__ bool vec_equal_4d(const int *a, const int *b) {
  int4 va = *reinterpret_cast<const int4 *>(a);
  int4 vb = *reinterpret_cast<const int4 *>(b);
  return (va.x == vb.x) && (va.y == vb.y) && (va.z == vb.z) && (va.w == vb.w);
}

// ============================================================================
// Floor-division helpers (arithmetic right shift for power-of-2 stride)
// CUDA guarantees arithmetic right shift for signed integers.
// ============================================================================

__device__ __forceinline__ int floor_div_pow2(int x, int shift) { return x >> shift; }

}  // namespace cuhash
