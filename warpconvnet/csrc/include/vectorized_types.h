// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <string>
#include <type_traits>

/**
 * Custom vectorized data types for efficient CUDA memory operations
 * These types enable vectorized loads/stores for different precisions
 * Following a principled template-based design pattern
 */

// ============================================================================
// Template-based vector types
// ============================================================================

template <typename T, int N>
struct __align__(N * sizeof(T)) vector_type {
  T data[N];

  // Constructors
  __device__ __host__ vector_type() {}

  __device__ __host__ vector_type(T val) {
#pragma unroll
    for (int i = 0; i < N; i++) {
      data[i] = val;
    }
  }

  __device__ __host__ vector_type(const T* ptr) {
#pragma unroll
    for (int i = 0; i < N; i++) {
      data[i] = ptr[i];
    }
  }

  // Variadic constructor
  template <typename... Args>
  __device__ __host__ vector_type(Args... args) : data{static_cast<T>(args)...} {
    static_assert(sizeof...(args) == N, "Wrong number of arguments");
  }

  // Array access operators
  __device__ __host__ T& operator[](int i) { return data[i]; }
  __device__ __host__ const T& operator[](int i) const { return data[i]; }

  // Arithmetic operators
  __device__ vector_type operator+(const vector_type& other) const {
    vector_type result;
#pragma unroll
    for (int i = 0; i < N; i++) {
      result.data[i] = data[i] + other.data[i];
    }
    return result;
  }

  __device__ vector_type operator-(const vector_type& other) const {
    vector_type result;
#pragma unroll
    for (int i = 0; i < N; i++) {
      result.data[i] = data[i] - other.data[i];
    }
    return result;
  }

  __device__ vector_type operator*(const vector_type& other) const {
    vector_type result;
#pragma unroll
    for (int i = 0; i < N; i++) {
      result.data[i] = data[i] * other.data[i];
    }
    return result;
  }

  __device__ vector_type operator/(const vector_type& other) const {
    vector_type result;
#pragma unroll
    for (int i = 0; i < N; i++) {
      result.data[i] = data[i] / other.data[i];
    }
    return result;
  }

  __device__ vector_type& operator+=(const vector_type& other) {
#pragma unroll
    for (int i = 0; i < N; i++) {
      data[i] += other.data[i];
    }
    return *this;
  }
};

// ============================================================================
// Specialized vector types with proper alignment and precision-specific operations
// ============================================================================

// Legacy compatibility types (4-element vectors)
struct __align__(8) half4 {
  __half x, y, z, w;

  __device__ __host__ half4() {}
  __device__ __host__ half4(__half x_, __half y_, __half z_, __half w_)
      : x(x_), y(y_), z(z_), w(w_) {}
};

struct __align__(8) bfloat4 {
  __nv_bfloat16 x, y, z, w;

  __device__ __host__ bfloat4() {}
  __device__ __host__ bfloat4(
      __nv_bfloat16 x_, __nv_bfloat16 y_, __nv_bfloat16 z_, __nv_bfloat16 w_)
      : x(x_), y(y_), z(z_), w(w_) {}
};

// ============================================================================
// Half-precision arithmetic helpers (supports both intrinsics and operators)
// ============================================================================

// Half8 - 8 half-precision floats (16 bytes, 128-bit aligned)
struct __align__(16) half8 : public vector_type<__half, 8> {
  using base = vector_type<__half, 8>;
  using base::base;

  // Explicit constructor
  __device__ __host__ half8(
      __half a, __half b, __half c, __half d, __half e, __half f, __half g, __half h) {
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
    data[4] = e;
    data[5] = f;
    data[6] = g;
    data[7] = h;
  }

  // Half-precision specific arithmetic operators using optimized operations
  __device__ half8 operator+(const half8& other) const {
    half8 result;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      result.data[i] = __hadd(data[i], other.data[i]);
    }
    return result;
  }

  __device__ half8 operator-(const half8& other) const {
    half8 result;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      result.data[i] = __hsub(data[i], other.data[i]);
    }
    return result;
  }

  __device__ half8 operator*(const half8& other) const {
    half8 result;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      result.data[i] = __hmul(data[i], other.data[i]);
    }
    return result;
  }

  __device__ half8 operator/(const half8& other) const {
    half8 result;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      result.data[i] = __hdiv(data[i], other.data[i]);
    }
    return result;
  }

  __device__ half8& operator+=(const half8& other) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      data[i] = __hadd(data[i], other.data[i]);
    }
    return *this;
  }
};

// Bfloat16_8 - 8 bfloat16 floats (16 bytes, 128-bit aligned)
struct __align__(16) bfloat16_8 : public vector_type<__nv_bfloat16, 8> {
  using base = vector_type<__nv_bfloat16, 8>;
  using base::base;

  // Explicit constructor
  __device__ __host__ bfloat16_8(__nv_bfloat16 a,
                                 __nv_bfloat16 b,
                                 __nv_bfloat16 c,
                                 __nv_bfloat16 d,
                                 __nv_bfloat16 e,
                                 __nv_bfloat16 f,
                                 __nv_bfloat16 g,
                                 __nv_bfloat16 h) {
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
    data[4] = e;
    data[5] = f;
    data[6] = g;
    data[7] = h;
  }

  // Bfloat16-precision specific arithmetic operators (use operators since no specific intrinsics)
  __device__ bfloat16_8 operator+(const bfloat16_8& other) const {
    bfloat16_8 result;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      result.data[i] = data[i] + other.data[i];
    }
    return result;
  }

  __device__ bfloat16_8 operator-(const bfloat16_8& other) const {
    bfloat16_8 result;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      result.data[i] = data[i] - other.data[i];
    }
    return result;
  }

  __device__ bfloat16_8 operator*(const bfloat16_8& other) const {
    bfloat16_8 result;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      result.data[i] = data[i] * other.data[i];
    }
    return result;
  }

  __device__ bfloat16_8 operator/(const bfloat16_8& other) const {
    bfloat16_8 result;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      result.data[i] = data[i] / other.data[i];
    }
    return result;
  }

  __device__ bfloat16_8& operator+=(const bfloat16_8& other) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      data[i] = data[i] + other.data[i];
    }
    return *this;
  }
};

// ============================================================================
// Utility Functions
// ============================================================================

// Helper functions to create vectorized types
__device__ __forceinline__ half8
make_half8(__half x, __half y, __half z, __half w, __half x2, __half y2, __half z2, __half w2) {
  return half8(x, y, z, w, x2, y2, z2, w2);
}

__device__ __forceinline__ bfloat16_8 make_bfloat16_8(__nv_bfloat16 x,
                                                      __nv_bfloat16 y,
                                                      __nv_bfloat16 z,
                                                      __nv_bfloat16 w,
                                                      __nv_bfloat16 x2,
                                                      __nv_bfloat16 y2,
                                                      __nv_bfloat16 z2,
                                                      __nv_bfloat16 w2) {
  return bfloat16_8(x, y, z, w, x2, y2, z2, w2);
}

// 128-bit vectorized load for half8
__device__ inline half8 load_half8(const __half* ptr) {
  half8 result;
  // Use 128-bit vectorized load (4x uint32_t = 16 bytes = 8 halves)
  const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
  uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
  *result_uint4 = *ptr_uint4;
  return result;
}

// 128-bit vectorized store for half8
__device__ inline void store_half8(__half* ptr, const half8& val) {
  const uint4* val_uint4 = reinterpret_cast<const uint4*>(&val);
  uint4* ptr_uint4 = reinterpret_cast<uint4*>(ptr);
  *ptr_uint4 = *val_uint4;
}

// Read-only cache load for half8 (using regular load)
__device__ inline half8 ldg_half8(const __half* ptr) {
  half8 result;
  const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
  uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
  *result_uint4 = *ptr_uint4;
  return result;
}

// 128-bit vectorized load for bfloat16_8
__device__ inline bfloat16_8 load_bfloat16_8(const __nv_bfloat16* ptr) {
  bfloat16_8 result;
  // Use 128-bit vectorized load (4x uint32_t = 16 bytes = 8 bfloat16)
  const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
  uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
  *result_uint4 = *ptr_uint4;
  return result;
}

// 128-bit vectorized store for bfloat16_8
__device__ inline void store_bfloat16_8(__nv_bfloat16* ptr, const bfloat16_8& val) {
  const uint4* val_uint4 = reinterpret_cast<const uint4*>(&val);
  uint4* ptr_uint4 = reinterpret_cast<uint4*>(ptr);
  *ptr_uint4 = *val_uint4;
}

// Read-only cache load for bfloat16_8 (using regular load)
__device__ inline bfloat16_8 ldg_bfloat16_8(const __nv_bfloat16* ptr) {
  bfloat16_8 result;
  const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
  uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
  *result_uint4 = *ptr_uint4;
  return result;
}

// Standalone arithmetic functions for convenience (for backward compatibility)
__device__ __forceinline__ half8 add_half8(const half8& a, const half8& b) { return a + b; }

__device__ __forceinline__ half8 sub_half8(const half8& a, const half8& b) { return a - b; }

__device__ __forceinline__ half8 mul_half8(const half8& a, const half8& b) { return a * b; }

__device__ __forceinline__ half8 div_half8(const half8& a, const half8& b) { return a / b; }

__device__ __forceinline__ bfloat16_8 add_bfloat16_8(const bfloat16_8& a, const bfloat16_8& b) {
  return a + b;
}

__device__ __forceinline__ bfloat16_8 sub_bfloat16_8(const bfloat16_8& a, const bfloat16_8& b) {
  return a - b;
}

__device__ __forceinline__ bfloat16_8 mul_bfloat16_8(const bfloat16_8& a, const bfloat16_8& b) {
  return a * b;
}

__device__ __forceinline__ bfloat16_8 div_bfloat16_8(const bfloat16_8& a, const bfloat16_8& b) {
  return a / b;
}

// ============================================================================
// Type traits and size information
// ============================================================================

template <typename T>
struct vector_info;

template <>
struct vector_info<half8> {
  static constexpr int elements = 8;
  static constexpr int size_bytes = 16;
  using element_type = __half;
};

template <>
struct vector_info<bfloat16_8> {
  static constexpr int elements = 8;
  static constexpr int size_bytes = 16;
  using element_type = __nv_bfloat16;
};

template <>
struct vector_info<half4> {
  static constexpr int elements = 4;
  static constexpr int size_bytes = 8;
  using element_type = __half;
};

template <>
struct vector_info<bfloat4> {
  static constexpr int elements = 4;
  static constexpr int size_bytes = 8;
  using element_type = __nv_bfloat16;
};

template <>
struct vector_info<float4> {
  static constexpr int elements = 4;
  static constexpr int size_bytes = 16;
  using element_type = float;
};

template <>
struct vector_info<float2> {
  static constexpr int elements = 2;
  static constexpr int size_bytes = 8;
  using element_type = float;
};
