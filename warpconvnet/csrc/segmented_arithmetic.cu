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

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <string>
#include <type_traits>

#include "vectorized_types.h"

/**
 * Templated Segmented Arithmetic CUDA Kernels with Shared Memory Optimization
 *
 * Clean template-based design following CUTLASS patterns:
 * - Single templated kernel implementation for all data types (float, half, bfloat16, double)
 * - Templated arithmetic operations (Add, Subtract, Multiply, Divide)
 * - No explicit type names in function names
 * - Organized in warpconvnet::segmented_arithmetic namespace
 * - Status-based error handling with clean propagation to Python
 * - Vectorized specializations for improved memory bandwidth
 * - Reduced code duplication and improved maintainability
 *
 * Operation: D = B OP C (segmented arithmetic)
 * Where:
 * - B: Input matrix B (N x C)
 * - C: Segment vectors matrix C (K x C) - cached in shared memory
 * - D: Output matrix D (N x C)
 * - OP: Arithmetic operation (Add, Subtract, Multiply, Divide)
 * - offsets: Segment boundaries (K+1,) where offsets[i] to offsets[i+1] defines segment i
 * - K: Number of segments
 *
 * Each segment of B gets the corresponding row from C applied:
 * D[offsets[i]:offsets[i+1], :] = B[offsets[i]:offsets[i+1], :] OP C[i, :]
 *
 * Matrix C is loaded into shared memory to reduce global memory accesses:
 * - Shared memory requirement: K * C * sizeof(dtype) bytes per thread block
 * - C is loaded cooperatively by all threads in a thread block
 * - This reduces global memory accesses from O(N) to O(1) per element of C
 */

// Define arithmetic operation types
struct Add {};
struct Subtract {};
struct Multiply {};
struct Divide {};

// Templated arithmetic operations
template <typename T, typename Op>
struct Arithmetic;

// Addition specialization
template <typename T>
struct Arithmetic<T, Add> {
  __device__ __forceinline__ static T apply(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, __half>) {
      return __hadd(a, b);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return a + b;
    } else {
      return a + b;
    }
  }
};

// Subtraction specialization
template <typename T>
struct Arithmetic<T, Subtract> {
  __device__ __forceinline__ static T apply(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, __half>) {
      return __hsub(a, b);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return a - b;
    } else {
      return a - b;
    }
  }
};

// Multiplication specialization
template <typename T>
struct Arithmetic<T, Multiply> {
  __device__ __forceinline__ static T apply(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, __half>) {
      return __hmul(a, b);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return a * b;
    } else {
      return a * b;
    }
  }
};

// Division specialization
template <typename T>
struct Arithmetic<T, Divide> {
  __device__ __forceinline__ static T apply(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, __half>) {
      return __hdiv(a, b);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return a / b;
    } else {
      return a / b;
    }
  }
};

// Float4 specializations
template <>
struct Arithmetic<float4, Add> {
  __device__ __forceinline__ static float4 apply(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
};

template <>
struct Arithmetic<float4, Subtract> {
  __device__ __forceinline__ static float4 apply(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
  }
};

template <>
struct Arithmetic<float4, Multiply> {
  __device__ __forceinline__ static float4 apply(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
  }
};

template <>
struct Arithmetic<float4, Divide> {
  __device__ __forceinline__ static float4 apply(const float4& a, const float4& b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
  }
};

// Half8 specializations - using standalone functions from types.h
template <>
struct Arithmetic<half8, Add> {
  __device__ __forceinline__ static half8 apply(const half8& a, const half8& b) {
    return add_half8(a, b);
  }
};

template <>
struct Arithmetic<half8, Subtract> {
  __device__ __forceinline__ static half8 apply(const half8& a, const half8& b) {
    return sub_half8(a, b);
  }
};

template <>
struct Arithmetic<half8, Multiply> {
  __device__ __forceinline__ static half8 apply(const half8& a, const half8& b) {
    return mul_half8(a, b);
  }
};

template <>
struct Arithmetic<half8, Divide> {
  __device__ __forceinline__ static half8 apply(const half8& a, const half8& b) {
    return div_half8(a, b);
  }
};

// Bfloat16_8 specializations - using standalone functions from types.h
template <>
struct Arithmetic<bfloat16_8, Add> {
  __device__ __forceinline__ static bfloat16_8 apply(const bfloat16_8& a, const bfloat16_8& b) {
    return add_bfloat16_8(a, b);
  }
};

template <>
struct Arithmetic<bfloat16_8, Subtract> {
  __device__ __forceinline__ static bfloat16_8 apply(const bfloat16_8& a, const bfloat16_8& b) {
    return sub_bfloat16_8(a, b);
  }
};

template <>
struct Arithmetic<bfloat16_8, Multiply> {
  __device__ __forceinline__ static bfloat16_8 apply(const bfloat16_8& a, const bfloat16_8& b) {
    return mul_bfloat16_8(a, b);
  }
};

template <>
struct Arithmetic<bfloat16_8, Divide> {
  __device__ __forceinline__ static bfloat16_8 apply(const bfloat16_8& a, const bfloat16_8& b) {
    return div_bfloat16_8(a, b);
  }
};

// Define error codes for segmented arithmetic operations
enum class SegmentedArithmeticStatus {
  kSuccess = 0,
  kErrorInvalidKernelType = 1,
  kErrorUnsupportedDataType = 2,
  kErrorKernelExecution = 3,
  kErrorInvalidOperation = 4
};

/**
 * Binary search function for finding segment index in shared memory offsets
 *
 * @param shared_offsets Offsets array in shared memory
 * @param row_idx        Row index to search for
 * @param K              Number of segments
 * @return               Segment index that contains row_idx
 */
__device__ __forceinline__ int binary_search_segment(const int* shared_offsets,
                                                     int row_idx,
                                                     int K) {
  int left = 0;
  int right = K - 1;

  while (left <= right) {
    int mid = (left + right) / 2;
    if (row_idx >= shared_offsets[mid] && row_idx < shared_offsets[mid + 1]) {
      return mid;
    } else if (row_idx < shared_offsets[mid]) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }

  // Fallback - should not reach here if offsets are valid
  return 0;
}

/**
 * Segmented arithmetic kernel with shared memory for matrix C and offsets
 *
 * @param b           Input matrix B (N x C)
 * @param c           Segment vectors matrix C (K x C)
 * @param d           Output matrix D (N x C)
 * @param offsets     Segment boundaries (K+1,)
 * @param N           Number of rows in B and D
 * @param C           Number of channels/columns
 * @param K           Number of segments
 */
template <typename T, typename Op>
__global__ void segmented_arithmetic_kernel(const T* __restrict__ b,
                                            const T* __restrict__ c,
                                            T* __restrict__ d,
                                            const int* __restrict__ offsets,
                                            int N,
                                            int C,
                                            int K) {
  // Chunking parameters for shared memory optimization
  constexpr int CHUNK_SIZE = 32;

  // Get current chunk from blockIdx.y
  int chunk_idx = blockIdx.y;
  int chunk_start = chunk_idx * CHUNK_SIZE;
  int chunk_end = min(chunk_start + CHUNK_SIZE, C);
  int chunk_actual_size = chunk_end - chunk_start;

  // Shared memory layout: [matrix C chunk data][offsets data]
  extern __shared__ char shared_mem_raw[];
  T* shared_c = reinterpret_cast<T*>(shared_mem_raw);

  // Calculate offset for shared offsets (aligned to int boundary)
  size_t c_chunk_size = K * CHUNK_SIZE * sizeof(T);
  size_t aligned_c_size = (c_chunk_size + sizeof(int) - 1) & ~(sizeof(int) - 1);
  int* shared_offsets = reinterpret_cast<int*>(shared_mem_raw + aligned_c_size);

  // Load current chunk of matrix C into shared memory cooperatively
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Load segment-by-segment to maintain proper memory layout
  for (int seg_idx = 0; seg_idx < K; ++seg_idx) {
    for (int ch_offset = tid; ch_offset < chunk_actual_size; ch_offset += threads_per_block) {
      int global_channel = chunk_start + ch_offset;
      int shared_idx = seg_idx * chunk_actual_size + ch_offset;
      shared_c[shared_idx] = c[seg_idx * C + global_channel];
    }
  }

  // Load offsets into shared memory cooperatively (K+1 elements)
  for (int i = tid; i <= K; i += threads_per_block) {
    if (i <= K) {
      shared_offsets[i] = offsets[i];
    }
  }
  __syncthreads();

  // Process current chunk
  int tid_in_block = threadIdx.x;
  int threads_per_block_local = blockDim.x;

  // Process elements in current chunk
  int total_chunk_elements = N * chunk_actual_size;
  for (int i = tid_in_block; i < total_chunk_elements; i += threads_per_block_local) {
    int row_idx = i / chunk_actual_size;
    int ch_offset = i % chunk_actual_size;

    // Bounds checking
    if (row_idx >= N) continue;

    // Find which segment this row belongs to using binary search on shared memory
    int segment_idx = binary_search_segment(shared_offsets, row_idx, K);

    // Calculate global channel index
    int global_ch_idx = chunk_start + ch_offset;

    // Get the values
    T b_val = b[row_idx * C + global_ch_idx];
    T c_val = shared_c[segment_idx * chunk_actual_size + ch_offset];

    // Perform the arithmetic operation using templated arithmetic
    T result = Arithmetic<T, Op>::apply(b_val, c_val);

    // Store result
    d[row_idx * C + global_ch_idx] = result;
  }
}

/**
 * Float4 specialization of segmented arithmetic kernel for improved memory bandwidth
 * Uses vectorized loads/stores to process 4 floats at once
 */
template <typename Op>
__global__ void segmented_arithmetic_kernel_float4(const float* __restrict__ b,
                                                   const float* __restrict__ c,
                                                   float* __restrict__ d,
                                                   const int* __restrict__ offsets,
                                                   int N,
                                                   int C,
                                                   int K) {
  // Chunking parameters for shared memory optimization
  constexpr int CHUNK_SIZE = 32;

  // Get current chunk from blockIdx.y
  int chunk_idx = blockIdx.y;

  // Shared memory layout: [matrix C chunk data][offsets data]
  extern __shared__ char shared_mem_raw[];
  float* shared_c = reinterpret_cast<float*>(shared_mem_raw);

  // Calculate offset for shared offsets (aligned to int boundary)
  size_t c_chunk_size = K * CHUNK_SIZE * sizeof(float);
  size_t aligned_c_size = (c_chunk_size + sizeof(int) - 1) & ~(sizeof(int) - 1);
  int* shared_offsets = reinterpret_cast<int*>(shared_mem_raw + aligned_c_size);

  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Load offsets into shared memory once (K+1 elements)
  for (int i = tid; i <= K; i += threads_per_block) {
    if (i <= K) {
      shared_offsets[i] = offsets[i];
    }
  }
  __syncthreads();

  // Process current chunk of the C dimension
  int chunk_start = chunk_idx * CHUNK_SIZE;
  int chunk_end = min(chunk_start + CHUNK_SIZE, C);
  int chunk_actual_size = chunk_end - chunk_start;

  // Load current chunk of matrix C into shared memory cooperatively
  // Load segment-by-segment to maintain proper memory layout
  for (int seg_idx = 0; seg_idx < K; ++seg_idx) {
    for (int ch_offset = tid; ch_offset < chunk_actual_size; ch_offset += threads_per_block) {
      int global_channel = chunk_start + ch_offset;
      int shared_idx = seg_idx * chunk_actual_size + ch_offset;
      shared_c[shared_idx] = c[seg_idx * C + global_channel];
    }
  }
  __syncthreads();

  // Process current chunk - calculate chunk processing parameters
  int chunk_vec4 = chunk_actual_size / 4;
  int chunk_remainder = chunk_actual_size % 4;

  int tid_in_block = threadIdx.x;
  int threads_per_block_local = blockDim.x;

  // Process vectorized channels (groups of 4)
  if (chunk_vec4 > 0) {
    int total_vec4_elements = N * chunk_vec4;
    for (int i = tid_in_block; i < total_vec4_elements; i += threads_per_block_local) {
      int row_idx = i / chunk_vec4;
      int vec_idx = i % chunk_vec4;

      // Bounds checking
      if (row_idx >= N) continue;

      // Find which segment this row belongs to using binary search on shared memory
      int segment_idx = binary_search_segment(shared_offsets, row_idx, K);

      // Check alignment for float4 operations within chunk
      int global_channel_start = chunk_start + vec_idx * 4;
      if (global_channel_start + 4 <= chunk_end) {
        const float* b_ptr = &b[row_idx * C + global_channel_start];
        float* d_ptr = &d[row_idx * C + global_channel_start];
        const float* c_ptr = &shared_c[segment_idx * chunk_actual_size + vec_idx * 4];

        if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(d_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0) {
          // Aligned - use vectorized operations
          float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
          float4 c_vec = *reinterpret_cast<const float4*>(c_ptr);
          float4 d_vec;

          // Perform vectorized arithmetic
          d_vec = Arithmetic<float4, Op>::apply(b_vec, c_vec);

          // Store result back
          *reinterpret_cast<float4*>(d_ptr) = d_vec;
        } else {
          // Not aligned - fall back to scalar operations
          for (int j = 0; j < 4; ++j) {
            int global_ch_idx = global_channel_start + j;
            float b_val = b[row_idx * C + global_ch_idx];
            float c_val = shared_c[segment_idx * chunk_actual_size + vec_idx * 4 + j];
            d[row_idx * C + global_ch_idx] = Arithmetic<float, Op>::apply(b_val, c_val);
          }
        }
      }
    }
  }

  // Process remaining channels (non-multiple of 4)
  if (chunk_remainder > 0) {
    int start_ch_offset = chunk_vec4 * 4;
    for (int row_idx = 0; row_idx < N; ++row_idx) {
      // Find which segment this row belongs to using binary search on shared memory
      int segment_idx = binary_search_segment(shared_offsets, row_idx, K);

      // Process remaining channels cooperatively
      for (int ch_offset = tid_in_block; ch_offset < chunk_remainder;
           ch_offset += threads_per_block_local) {
        int global_ch_idx = chunk_start + start_ch_offset + ch_offset;
        float b_val = b[row_idx * C + global_ch_idx];
        float c_val = shared_c[segment_idx * chunk_actual_size + start_ch_offset + ch_offset];
        d[row_idx * C + global_ch_idx] = Arithmetic<float, Op>::apply(b_val, c_val);
      }
    }
  }
}

/**
 * Half8 specialization of segmented arithmetic kernel for improved memory bandwidth
 * Uses vectorized loads/stores to process 8 halves at once (128 bits = 16 bytes)
 */
template <typename Op>
__global__ void segmented_arithmetic_kernel_half8(const __half* __restrict__ b,
                                                  const __half* __restrict__ c,
                                                  __half* __restrict__ d,
                                                  const int* __restrict__ offsets,
                                                  int N,
                                                  int C,
                                                  int K) {
  // Chunking parameters for shared memory optimization
  constexpr int CHUNK_SIZE = 32;

  // Get current chunk from blockIdx.y
  int chunk_idx = blockIdx.y;
  int chunk_start = chunk_idx * CHUNK_SIZE;
  int chunk_end = min(chunk_start + CHUNK_SIZE, C);
  int chunk_actual_size = chunk_end - chunk_start;

  // Shared memory layout: [matrix C chunk data][aligned padding][offsets data]
  extern __shared__ char shared_mem_raw[];
  __half* shared_c = reinterpret_cast<__half*>(shared_mem_raw);

  // Calculate offset for shared_offsets with proper alignment
  size_t c_chunk_size = K * CHUNK_SIZE * sizeof(__half);
  size_t aligned_c_size = ((c_chunk_size + 15) / 16) * 16;  // 16-byte alignment
  int* shared_offsets = reinterpret_cast<int*>(shared_mem_raw + aligned_c_size);

  // Load current chunk of matrix C into shared memory cooperatively
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Load segment-by-segment to maintain proper memory layout
  for (int seg_idx = 0; seg_idx < K; ++seg_idx) {
    for (int ch_offset = tid; ch_offset < chunk_actual_size; ch_offset += threads_per_block) {
      int global_channel = chunk_start + ch_offset;
      int shared_idx = seg_idx * chunk_actual_size + ch_offset;
      shared_c[shared_idx] = c[seg_idx * C + global_channel];
    }
  }

  // Load offsets into shared memory cooperatively (K+1 elements)
  for (int i = tid; i <= K; i += threads_per_block) {
    if (i <= K) {
      shared_offsets[i] = offsets[i];
    }
  }
  __syncthreads();

  // Process current chunk - calculate chunk processing parameters
  int chunk_vec8 = chunk_actual_size / 8;
  int chunk_remainder = chunk_actual_size % 8;

  int tid_in_block = threadIdx.x;
  int threads_per_block_local = blockDim.x;

  // Process vectorized channels (groups of 8)
  if (chunk_vec8 > 0) {
    int total_vec8_elements = N * chunk_vec8;
    for (int i = tid_in_block; i < total_vec8_elements; i += threads_per_block_local) {
      int row_idx = i / chunk_vec8;
      int vec_idx = i % chunk_vec8;

      // Bounds checking
      if (row_idx >= N) continue;

      // Find which segment this row belongs to using binary search on shared memory
      int segment_idx = binary_search_segment(shared_offsets, row_idx, K);

      // Check alignment for half8 operations within chunk
      int global_channel_start = chunk_start + vec_idx * 8;
      if (global_channel_start + 8 <= chunk_end) {
        const __half* b_ptr = &b[row_idx * C + global_channel_start];
        __half* d_ptr = &d[row_idx * C + global_channel_start];
        const __half* c_ptr = &shared_c[segment_idx * chunk_actual_size + vec_idx * 8];

        if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(d_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0) {
          // Aligned - use uint4 for 8 halves (16 bytes)
          uint4 b_vec = *reinterpret_cast<const uint4*>(b_ptr);
          uint4 c_vec = *reinterpret_cast<const uint4*>(c_ptr);

          // Convert to half8 for arithmetic, then back to uint4 for storage
          half8 b_half8 = *reinterpret_cast<const half8*>(&b_vec);
          half8 c_half8 = *reinterpret_cast<const half8*>(&c_vec);
          half8 d_half8 = Arithmetic<half8, Op>::apply(b_half8, c_half8);

          // Store result back as uint4
          *reinterpret_cast<uint4*>(d_ptr) = *reinterpret_cast<const uint4*>(&d_half8);
        } else {
// Not aligned - fall back to scalar operations
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            int global_ch_idx = global_channel_start + j;
            __half b_val = b[row_idx * C + global_ch_idx];
            __half c_val = shared_c[segment_idx * chunk_actual_size + vec_idx * 8 + j];
            d[row_idx * C + global_ch_idx] = Arithmetic<__half, Op>::apply(b_val, c_val);
          }
        }
      }
    }
  }

  // Process remaining channels (non-multiple of 8)
  if (chunk_remainder > 0) {
    int start_ch_offset = chunk_vec8 * 8;
    for (int row_idx = 0; row_idx < N; ++row_idx) {
      // Find which segment this row belongs to using binary search on shared memory
      int segment_idx = binary_search_segment(shared_offsets, row_idx, K);

      // Process remaining channels cooperatively
      for (int ch_offset = tid_in_block; ch_offset < chunk_remainder;
           ch_offset += threads_per_block_local) {
        int global_ch_idx = chunk_start + start_ch_offset + ch_offset;
        __half b_val = b[row_idx * C + global_ch_idx];
        __half c_val = shared_c[segment_idx * chunk_actual_size + start_ch_offset + ch_offset];
        d[row_idx * C + global_ch_idx] = Arithmetic<__half, Op>::apply(b_val, c_val);
      }
    }
  }
}

/**
 * Bfloat16_8 specialization of segmented arithmetic kernel for improved memory bandwidth
 * Uses vectorized loads/stores to process 8 bfloat16s at once (128 bits = 16 bytes)
 */
template <typename Op>
__global__ void segmented_arithmetic_kernel_bfloat16(const __nv_bfloat16* __restrict__ b,
                                                     const __nv_bfloat16* __restrict__ c,
                                                     __nv_bfloat16* __restrict__ d,
                                                     const int* __restrict__ offsets,
                                                     int N,
                                                     int C,
                                                     int K) {
  // Chunking parameters for shared memory optimization
  constexpr int CHUNK_SIZE = 32;

  // Get current chunk from blockIdx.y
  int chunk_idx = blockIdx.y;
  int chunk_start = chunk_idx * CHUNK_SIZE;
  int chunk_end = min(chunk_start + CHUNK_SIZE, C);
  int chunk_actual_size = chunk_end - chunk_start;

  // Shared memory layout: [matrix C chunk data][aligned padding][offsets data]
  extern __shared__ char shared_mem_raw[];
  __nv_bfloat16* shared_c = reinterpret_cast<__nv_bfloat16*>(shared_mem_raw);

  // Calculate offset for shared_offsets with proper alignment
  size_t c_chunk_size = K * CHUNK_SIZE * sizeof(__nv_bfloat16);
  size_t aligned_c_size = ((c_chunk_size + 15) / 16) * 16;  // 16-byte alignment
  int* shared_offsets = reinterpret_cast<int*>(shared_mem_raw + aligned_c_size);

  // Load current chunk of matrix C into shared memory cooperatively
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Load segment-by-segment to maintain proper memory layout
  for (int seg_idx = 0; seg_idx < K; ++seg_idx) {
    for (int ch_offset = tid; ch_offset < chunk_actual_size; ch_offset += threads_per_block) {
      int global_channel = chunk_start + ch_offset;
      int shared_idx = seg_idx * chunk_actual_size + ch_offset;
      shared_c[shared_idx] = c[seg_idx * C + global_channel];
    }
  }

  // Load offsets into shared memory cooperatively (K+1 elements)
  for (int i = tid; i <= K; i += threads_per_block) {
    if (i <= K) {
      shared_offsets[i] = offsets[i];
    }
  }
  __syncthreads();

  // Process current chunk - calculate chunk processing parameters
  int chunk_vec8 = chunk_actual_size / 8;
  int chunk_remainder = chunk_actual_size % 8;

  int tid_in_block = threadIdx.x;
  int threads_per_block_local = blockDim.x;

  // Process vectorized channels (groups of 8)
  if (chunk_vec8 > 0) {
    int total_vec8_elements = N * chunk_vec8;
    for (int i = tid_in_block; i < total_vec8_elements; i += threads_per_block_local) {
      int row_idx = i / chunk_vec8;
      int vec_idx = i % chunk_vec8;

      // Bounds checking
      if (row_idx >= N) continue;

      // Find which segment this row belongs to using binary search on shared memory
      int segment_idx = binary_search_segment(shared_offsets, row_idx, K);

      // Check alignment for bfloat16_8 operations within chunk
      int global_channel_start = chunk_start + vec_idx * 8;
      if (global_channel_start + 8 <= chunk_end) {
        const __nv_bfloat16* b_ptr = &b[row_idx * C + global_channel_start];
        __nv_bfloat16* d_ptr = &d[row_idx * C + global_channel_start];
        const __nv_bfloat16* c_ptr = &shared_c[segment_idx * chunk_actual_size + vec_idx * 8];

        if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(d_ptr) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0) {
          // Aligned - use uint4 for 8 bfloat16s (16 bytes)
          uint4 b_vec = *reinterpret_cast<const uint4*>(b_ptr);
          uint4 c_vec = *reinterpret_cast<const uint4*>(c_ptr);

          // Convert to bfloat16_8 for arithmetic, then back to uint4 for storage
          bfloat16_8 b_bfloat16_8 = *reinterpret_cast<const bfloat16_8*>(&b_vec);
          bfloat16_8 c_bfloat16_8 = *reinterpret_cast<const bfloat16_8*>(&c_vec);
          bfloat16_8 d_bfloat16_8 = Arithmetic<bfloat16_8, Op>::apply(b_bfloat16_8, c_bfloat16_8);

          // Store result back as uint4
          *reinterpret_cast<uint4*>(d_ptr) = *reinterpret_cast<const uint4*>(&d_bfloat16_8);
        } else {
// Not aligned - fall back to scalar operations
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            int global_ch_idx = global_channel_start + j;
            __nv_bfloat16 b_val = b[row_idx * C + global_ch_idx];
            __nv_bfloat16 c_val = shared_c[segment_idx * chunk_actual_size + vec_idx * 8 + j];
            d[row_idx * C + global_ch_idx] = Arithmetic<__nv_bfloat16, Op>::apply(b_val, c_val);
          }
        }
      }
    }
  }

  // Process remaining channels (non-multiple of 8)
  if (chunk_remainder > 0) {
    int start_ch_offset = chunk_vec8 * 8;
    for (int row_idx = 0; row_idx < N; ++row_idx) {
      // Find which segment this row belongs to using binary search on shared memory
      int segment_idx = binary_search_segment(shared_offsets, row_idx, K);

      // Process remaining channels cooperatively
      for (int ch_offset = tid_in_block; ch_offset < chunk_remainder;
           ch_offset += threads_per_block_local) {
        int global_ch_idx = chunk_start + start_ch_offset + ch_offset;
        __nv_bfloat16 b_val = b[row_idx * C + global_ch_idx];
        __nv_bfloat16 c_val =
            shared_c[segment_idx * chunk_actual_size + start_ch_offset + ch_offset];
        d[row_idx * C + global_ch_idx] = Arithmetic<__nv_bfloat16, Op>::apply(b_val, c_val);
      }
    }
  }
}

// Main templated function implementation
namespace warpconvnet {
namespace segmented_arithmetic {

/**
 * @brief Run a segmented arithmetic operation with templated operation type.
 *
 * @param tensor_b: Pointer to the B matrix (N x C).
 * @param tensor_c: Pointer to the C matrix (K x C).
 * @param tensor_d: Pointer to the D matrix (N x C).
 * @param offsets: Segment boundaries (K+1,).
 * @param N: Number of rows in B and D.
 * @param C: Number of channels/columns.
 * @param K: Number of segments.
 * @param kernel_type: Type of kernel to use ("basic").
 *
 * @return Status code indicating the success or failure of the operation.
 *
 * Operation: D[offsets[i]:offsets[i+1], :] = B[offsets[i]:offsets[i+1], :] OP C[i, :]
 */
template <typename ElementB, typename ElementC, typename ElementD, typename Op>
int run_segmented_arithmetic_templated_impl(const void* tensor_b,
                                            const void* tensor_c,
                                            void* tensor_d,
                                            const int* offsets,
                                            int N,
                                            int C,
                                            int K,
                                            const std::string& kernel_type) {
  // Convert void pointers to appropriate types
  auto b_ptr = reinterpret_cast<const ElementB*>(tensor_b);
  auto c_ptr = reinterpret_cast<const ElementC*>(tensor_c);
  auto d_ptr = reinterpret_cast<ElementD*>(tensor_d);

  // Operation type is now determined at compile time via template parameter Op

  // Get CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Launch kernel configuration
  const int threads_per_block = 256;

  // Calculate shared memory size for matrix C chunk and offsets (using CHUNK_SIZE=32)
  constexpr int CHUNK_SIZE = 32;
  size_t c_chunk_mem_size = K * CHUNK_SIZE * sizeof(ElementC);
  size_t aligned_c_size =
      ((c_chunk_mem_size + 15) / 16) * 16;  // 16-byte alignment for vectorized access
  size_t offsets_mem_size = (K + 1) * sizeof(int);
  int shared_mem_size = static_cast<int>(aligned_c_size + offsets_mem_size);

  // Calculate number of chunks and 2D grid dimensions
  int num_chunks = (C + CHUNK_SIZE - 1) / CHUNK_SIZE;
  dim3 blocks, grid_dim;

  // Dispatch based on kernel type
  if (kernel_type == "basic") {
    if constexpr (std::is_same_v<ElementB, float>) {
      // Calculate grid dimensions for chunked approach
      int total_threads = N;
      int blocks_x = (total_threads + threads_per_block - 1) / threads_per_block;
      grid_dim = dim3(blocks_x, num_chunks, 1);

      segmented_arithmetic_kernel_float4<Op>
          <<<grid_dim, threads_per_block, shared_mem_size, stream>>>(
              b_ptr, c_ptr, d_ptr, offsets, N, C, K);
    } else if constexpr (std::is_same_v<ElementB, cutlass::half_t>) {
      // Calculate grid dimensions for chunked approach
      int total_threads = N;
      int blocks_x = (total_threads + threads_per_block - 1) / threads_per_block;
      grid_dim = dim3(blocks_x, num_chunks, 1);

      segmented_arithmetic_kernel_half8<Op>
          <<<grid_dim, threads_per_block, shared_mem_size, stream>>>(
              reinterpret_cast<const __half*>(b_ptr),
              reinterpret_cast<const __half*>(c_ptr),
              reinterpret_cast<__half*>(d_ptr),
              offsets,
              N,
              C,
              K);
    } else if constexpr (std::is_same_v<ElementB, cutlass::bfloat16_t>) {
      // Calculate grid dimensions for chunked approach
      int total_threads = N;
      int blocks_x = (total_threads + threads_per_block - 1) / threads_per_block;
      grid_dim = dim3(blocks_x, num_chunks, 1);

      segmented_arithmetic_kernel_bfloat16<Op>
          <<<grid_dim, threads_per_block, shared_mem_size, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(b_ptr),
              reinterpret_cast<const __nv_bfloat16*>(c_ptr),
              reinterpret_cast<__nv_bfloat16*>(d_ptr),
              offsets,
              N,
              C,
              K);
    } else if constexpr (std::is_same_v<ElementB, double>) {
      // Calculate grid dimensions for chunked approach
      int total_threads = N;
      int blocks_x = (total_threads + threads_per_block - 1) / threads_per_block;
      grid_dim = dim3(blocks_x, num_chunks, 1);

      segmented_arithmetic_kernel<double, Op>
          <<<grid_dim, threads_per_block, shared_mem_size, stream>>>(
              b_ptr, c_ptr, d_ptr, offsets, N, C, K);
    } else {
      return static_cast<int>(SegmentedArithmeticStatus::kErrorUnsupportedDataType);
    }
  } else {
    return static_cast<int>(SegmentedArithmeticStatus::kErrorInvalidKernelType);
  }

  // Check for CUDA errors
  cudaStreamSynchronize(stream);
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(SegmentedArithmeticStatus::kErrorKernelExecution);
  }

  return static_cast<int>(SegmentedArithmeticStatus::kSuccess);
}

/**
 * @brief Public wrapper for segmented arithmetic operations with runtime operation dispatch.
 *
 * @param tensor_b: Pointer to the B matrix (N x C).
 * @param tensor_c: Pointer to the C matrix (K x C).
 * @param tensor_d: Pointer to the D matrix (N x C).
 * @param offsets: Segment boundaries (K+1,).
 * @param N: Number of rows in B and D.
 * @param C: Number of channels/columns.
 * @param K: Number of segments.
 * @param operation: Operation type ("add", "subtract", "multiply", "divide").
 * @param kernel_type: Type of kernel to use ("basic").
 *
 * @return Status code indicating the success or failure of the operation.
 */
template <typename ElementB, typename ElementC, typename ElementD>
int run_segmented_arithmetic_templated(const void* tensor_b,
                                       const void* tensor_c,
                                       void* tensor_d,
                                       const int* offsets,
                                       int N,
                                       int C,
                                       int K,
                                       const std::string& operation,
                                       const std::string& kernel_type) {
  // Dispatch to the appropriate templated implementation based on operation
  if (operation == "add") {
    return run_segmented_arithmetic_templated_impl<ElementB, ElementC, ElementD, Add>(
        tensor_b, tensor_c, tensor_d, offsets, N, C, K, kernel_type);
  } else if (operation == "subtract" || operation == "sub") {
    return run_segmented_arithmetic_templated_impl<ElementB, ElementC, ElementD, Subtract>(
        tensor_b, tensor_c, tensor_d, offsets, N, C, K, kernel_type);
  } else if (operation == "multiply" || operation == "mul") {
    return run_segmented_arithmetic_templated_impl<ElementB, ElementC, ElementD, Multiply>(
        tensor_b, tensor_c, tensor_d, offsets, N, C, K, kernel_type);
  } else if (operation == "divide" || operation == "div") {
    return run_segmented_arithmetic_templated_impl<ElementB, ElementC, ElementD, Divide>(
        tensor_b, tensor_c, tensor_d, offsets, N, C, K, kernel_type);
  } else {
    return static_cast<int>(SegmentedArithmeticStatus::kErrorInvalidOperation);
  }
}

}  // namespace segmented_arithmetic
}  // namespace warpconvnet

// Use the namespace for convenience in the rest of the file
using namespace warpconvnet::segmented_arithmetic;

// Expose the template instantiations for use in pybind
template int
warpconvnet::segmented_arithmetic::run_segmented_arithmetic_templated<float, float, float>(
    const void*,
    const void*,
    void*,
    const int*,
    int,
    int,
    int,
    const std::string&,
    const std::string&);

template int warpconvnet::segmented_arithmetic::
    run_segmented_arithmetic_templated<cutlass::half_t, cutlass::half_t, cutlass::half_t>(
        const void*,
        const void*,
        void*,
        const int*,
        int,
        int,
        int,
        const std::string&,
        const std::string&);

template int warpconvnet::segmented_arithmetic::run_segmented_arithmetic_templated<
    cutlass::bfloat16_t,
    cutlass::bfloat16_t,
    cutlass::bfloat16_t>(const void*,
                         const void*,
                         void*,
                         const int*,
                         int,
                         int,
                         int,
                         const std::string&,
                         const std::string&);

template int
warpconvnet::segmented_arithmetic::run_segmented_arithmetic_templated<double, double, double>(
    const void*,
    const void*,
    void*,
    const int*,
    int,
    int,
    int,
    const std::string&,
    const std::string&);
