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

#include <string>
#include <type_traits>

struct __align__(8) half4 { __half x, y, z, w; };

struct __align__(8) bfloat4 { __nv_bfloat16 x, y, z, w; };

// Vectorized type definitions for 128-bit operations
struct __align__(16) half8 {
  __half x, y, z, w;      // First 4 halves
  __half x2, y2, z2, w2;  // Second 4 halves

  __device__ __forceinline__ half8() {}

  __device__ __forceinline__ half8(
      __half x_, __half y_, __half z_, __half w_, __half x2_, __half y2_, __half z2_, __half w2_)
      : x(x_), y(y_), z(z_), w(w_), x2(x2_), y2(y2_), z2(z2_), w2(w2_) {}

  // Arithmetic operations
  __device__ __forceinline__ half8 operator+(const half8& other) const {
    return half8(__hadd(x, other.x),
                 __hadd(y, other.y),
                 __hadd(z, other.z),
                 __hadd(w, other.w),
                 __hadd(x2, other.x2),
                 __hadd(y2, other.y2),
                 __hadd(z2, other.z2),
                 __hadd(w2, other.w2));
  }

  __device__ __forceinline__ half8 operator-(const half8& other) const {
    return half8(__hsub(x, other.x),
                 __hsub(y, other.y),
                 __hsub(z, other.z),
                 __hsub(w, other.w),
                 __hsub(x2, other.x2),
                 __hsub(y2, other.y2),
                 __hsub(z2, other.z2),
                 __hsub(w2, other.w2));
  }

  __device__ __forceinline__ half8 operator*(const half8& other) const {
    return half8(__hmul(x, other.x),
                 __hmul(y, other.y),
                 __hmul(z, other.z),
                 __hmul(w, other.w),
                 __hmul(x2, other.x2),
                 __hmul(y2, other.y2),
                 __hmul(z2, other.z2),
                 __hmul(w2, other.w2));
  }

  __device__ __forceinline__ half8 operator/(const half8& other) const {
    return half8(__hdiv(x, other.x),
                 __hdiv(y, other.y),
                 __hdiv(z, other.z),
                 __hdiv(w, other.w),
                 __hdiv(x2, other.x2),
                 __hdiv(y2, other.y2),
                 __hdiv(z2, other.z2),
                 __hdiv(w2, other.w2));
  }
};

struct __align__(16) bfloat16_8 {
  __nv_bfloat16 x, y, z, w;      // First 4 bfloat16s
  __nv_bfloat16 x2, y2, z2, w2;  // Second 4 bfloat16s

  __device__ __forceinline__ bfloat16_8() {}

  __device__ __forceinline__ bfloat16_8(__nv_bfloat16 x_,
                                        __nv_bfloat16 y_,
                                        __nv_bfloat16 z_,
                                        __nv_bfloat16 w_,
                                        __nv_bfloat16 x2_,
                                        __nv_bfloat16 y2_,
                                        __nv_bfloat16 z2_,
                                        __nv_bfloat16 w2_)
      : x(x_), y(y_), z(z_), w(w_), x2(x2_), y2(y2_), z2(z2_), w2(w2_) {}

  // Arithmetic operations
  __device__ __forceinline__ bfloat16_8 operator+(const bfloat16_8& other) const {
    return bfloat16_8(x + other.x,
                      y + other.y,
                      z + other.z,
                      w + other.w,
                      x2 + other.x2,
                      y2 + other.y2,
                      z2 + other.z2,
                      w2 + other.w2);
  }

  __device__ __forceinline__ bfloat16_8 operator-(const bfloat16_8& other) const {
    return bfloat16_8(x - other.x,
                      y - other.y,
                      z - other.z,
                      w - other.w,
                      x2 - other.x2,
                      y2 - other.y2,
                      z2 - other.z2,
                      w2 - other.w2);
  }

  __device__ __forceinline__ bfloat16_8 operator*(const bfloat16_8& other) const {
    return bfloat16_8(x * other.x,
                      y * other.y,
                      z * other.z,
                      w * other.w,
                      x2 * other.x2,
                      y2 * other.y2,
                      z2 * other.z2,
                      w2 * other.w2);
  }

  __device__ __forceinline__ bfloat16_8 operator/(const bfloat16_8& other) const {
    return bfloat16_8(x / other.x,
                      y / other.y,
                      z / other.z,
                      w / other.w,
                      x2 / other.x2,
                      y2 / other.y2,
                      z2 / other.z2,
                      w2 / other.w2);
  }
};

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

// Standalone arithmetic functions for convenience
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
