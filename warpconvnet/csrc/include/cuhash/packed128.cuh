// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// cuhash - 128-bit packed key hash table for high-dimensional integer coords.
//
// Layout: a key is D signed integers, each occupying CoordBits, packed
// little-endian into a 128-bit (lo, hi) pair. Bit 127 of the packed value
// is reserved as a validity flag; an unoccupied slot has hi == 0.
//
// Contract: caller MUST guarantee distinct keys. Insert path does NOT
// dedup — duplicate inserts will allocate separate slots and search will
// nondeterministically return one of them. Run torch.unique upstream
// (or equivalent) before insert.
//
// Concurrency: insert and search must not overlap on the same stream.
// The pattern is insert kernel -> stream sync -> search kernel. The
// kernel-boundary fence makes lo/val writes from insert visible to search
// without an explicit __threadfence. Concurrent insert+search produces
// undefined behavior (search may observe a partially-written slot).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cuhash {
namespace packed128 {

// Compile-time configuration: D axes of CoordBits each, plus 1 validity bit.
// D * CoordBits must be <= 127 to leave room for the validity flag.
template <int D, int CoordBits>
struct PackedKeyConfig {
  static_assert(D >= 1 && D <= 8, "D must be in [1, 8]");
  static_assert(CoordBits >= 1 && CoordBits <= 32, "CoordBits in [1, 32]");
  static_assert(D * CoordBits <= 127, "Need bit 127 for validity flag");
  static constexpr int kDim = D;
  static constexpr int kCoordBits = CoordBits;
  static constexpr uint32_t kCoordMaskU =
      (CoordBits == 32) ? 0xFFFFFFFFu : ((1u << CoordBits) - 1u);
  static constexpr int kCoordMax = (1 << (CoordBits - 1)) - 1;
  static constexpr int kCoordMin = -(1 << (CoordBits - 1));
};

// 16-byte key. lo holds bits 0..63 of the packed coord; hi holds bits 64..126
// of the packed coord plus the validity flag at bit 63 (i.e. global bit 127).
struct __align__(16) PackedKey128 {
  uint64_t lo;
  uint64_t hi;
};

// Validity bit lives in the top bit of hi (= bit 127 of the 128-bit value).
static constexpr uint64_t kValidBit128 = 1ull << 63;
static constexpr uint64_t kEmptyHi = 0ull;

// Pack D signed coords stored as int32[D] into a 128-bit key. Each coord
// is masked to CoordBits and placed at offset i * CoordBits. The validity
// bit is OR'd into the top of hi so that hi == 0 unambiguously means
// "empty slot". Coords outside [kCoordMin, kCoordMax] silently truncate;
// callers must validate range before calling (Python wrapper does so).
template <int D, int CoordBits>
__host__ __device__ __forceinline__ PackedKey128 pack_keyN(const int32_t *key) {
  using Cfg = PackedKeyConfig<D, CoordBits>;
  uint64_t lo = 0, hi = 0;
#pragma unroll
  for (int i = 0; i < D; ++i) {
    uint64_t bits = static_cast<uint64_t>(static_cast<uint32_t>(key[i]) & Cfg::kCoordMaskU);
    int shift = i * CoordBits;
    if (shift < 64) {
      lo |= bits << shift;
      // Spill into hi if this axis straddles the lo/hi boundary.
      // Guarded by shift > (64 - CoordBits), i.e. shift + CoordBits > 64.
      if (shift + CoordBits > 64) {
        hi |= bits >> (64 - shift);
      }
    } else {
      hi |= bits << (shift - 64);
    }
  }
  PackedKey128 k;
  k.lo = lo;
  k.hi = hi | kValidBit128;
  return k;
}

__device__ __forceinline__ bool key_equal(PackedKey128 a, PackedKey128 b) {
  return a.lo == b.lo && a.hi == b.hi;
}

// Splitmix64 finalizer applied to (lo + Phi*hi). Phi is the golden-ratio
// 64-bit constant; this combines lo and hi with an irrational multiplier
// before running splitmix64's three xor-mul-shift rounds.
__device__ __forceinline__ uint32_t hash128(PackedKey128 k, uint32_t capacity_mask) {
  constexpr uint64_t kPhi = 0x9E3779B97F4A7C15ull;
  uint64_t x = k.lo + kPhi * k.hi;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return static_cast<uint32_t>(x) & capacity_mask;
}

__device__ __forceinline__ uint64_t atomicCAS_u64_p128(uint64_t *addr, uint64_t cmp, uint64_t val) {
  return static_cast<uint64_t>(atomicCAS(reinterpret_cast<unsigned long long *>(addr),
                                         static_cast<unsigned long long>(cmp),
                                         static_cast<unsigned long long>(val)));
}

// Insert. Probe sequence: hash128 -> linear. Claim a slot via atomicCAS on
// hi (cmp = 0, val = key.hi which has the validity bit set). On success,
// write lo and val. No dedup — the contract is that the caller has already
// deduplicated upstream.
__device__ __forceinline__ void packed128_insert(PackedKey128 *__restrict__ keys,
                                                 int32_t *__restrict__ vals,
                                                 PackedKey128 key,
                                                 int32_t value,
                                                 uint32_t capacity_mask,
                                                 int *status_ptr = nullptr) {
  uint32_t slot = hash128(key, capacity_mask);
  uint32_t attempts = 0;
  while (attempts <= capacity_mask) {
    uint64_t prev = atomicCAS_u64_p128(&keys[slot].hi, kEmptyHi, key.hi);
    if (prev == kEmptyHi) {
      keys[slot].lo = key.lo;
      vals[slot] = value;
      return;
    }
    // No dedup: another (different or same) key occupies this slot, probe on.
    slot = (slot + 1) & capacity_mask;
    ++attempts;
  }
  if (status_ptr != nullptr) {
    atomicMax(status_ptr, 1);
  }
}

// Search. Probe sequence matches insert. Returns vals[slot] on hit, -1 on
// miss (either unoccupied slot reached, or the full table scanned without
// match — the latter only at load factor close to 1.0).
__device__ __forceinline__ int32_t packed128_search(const PackedKey128 *__restrict__ keys,
                                                    const int32_t *__restrict__ vals,
                                                    PackedKey128 key,
                                                    uint32_t capacity_mask) {
  uint32_t slot = hash128(key, capacity_mask);
  uint32_t attempts = 0;
  while (attempts <= capacity_mask) {
    uint64_t hi = keys[slot].hi;
    if (hi == kEmptyHi) return -1;
    if (hi == key.hi && keys[slot].lo == key.lo) return vals[slot];
    slot = (slot + 1) & capacity_mask;
    ++attempts;
  }
  return -1;
}

}  // namespace packed128
}  // namespace cuhash
