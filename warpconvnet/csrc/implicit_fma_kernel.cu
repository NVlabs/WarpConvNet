// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <type_traits>

/**
 * Templated Implicit FMA CUDA Kernels with Shared Memory Optimization
 *
 * Clean template-based design following CUTLASS GEMM patterns:
 * - Single templated kernel implementation for all data types (float, half, double)
 * - No explicit type names in function names (no _float, _half, _double suffixes)
 * - Organized in warpconvnet::implicit_fma namespace
 * - Status-based error handling with clean propagation to Python
 * - No direct CUDA imports in Python bindings
 * - Compile-time optimization with automatic type deduction
 * - Reduced code duplication and improved maintainability
 *
 * All kernels use shared memory to cache vector B for improved performance:
 * - Vector B is loaded cooperatively by all threads in a thread block
 * - This reduces global memory accesses from O(num_ops) to O(1) per element of B
 * - Shared memory requirement: C * sizeof(dtype) bytes per thread block
 *
 * Operation: c[out_index] += a[in_index] * b
 * Where:
 * - a: Input matrix A (N_A x C)
 * - b: Input vector B (C,) - cached in shared memory
 * - c: Output matrix C (N_C x C) - modified in-place
 * - in_index/out_index: Gather-scatter indices
 */

// Define error codes for implicit FMA operations
enum class ImplicitFmaStatus {
  kSuccess = 0,
  kErrorInvalidKernelType = 1,
  kErrorUnsupportedDataType = 2,
  kErrorKernelExecution = 3
};

/**
 * Implicit FMA kernel with shared memory for vector B: c[out_index] += a[in_index] * b
 *
 * @param a           Input matrix A (N_A x C)
 * @param b           Input vector B (C,)
 * @param c           Output matrix C (N_C x C), modified in-place
 * @param in_indices  Input indices for gathering from A (num_ops,)
 * @param out_indices Output indices for scattering to C (num_ops,)
 * @param num_ops     Number of operations (length of indices)
 * @param C           Number of channels/columns
 * @param N_A         Number of rows in A
 * @param N_C         Number of rows in C
 */
template <typename T>
__global__ void implicit_fma_kernel(const T* __restrict__ a,
                                    const T* __restrict__ b,
                                    T* __restrict__ c,
                                    const int* __restrict__ in_indices,
                                    const int* __restrict__ out_indices,
                                    int num_ops,
                                    int C,
                                    int N_A,
                                    int N_C) {
  // Shared memory for vector B
  extern __shared__ char shared_mem_raw[];
  T* shared_b = reinterpret_cast<T*>(shared_mem_raw);

  // Load vector B into shared memory cooperatively
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Each thread loads multiple elements of B if necessary
  for (int i = tid; i < C; i += threads_per_block) {
    shared_b[i] = b[i];
  }
  __syncthreads();

  // Calculate global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_ops * C) return;

  // Decompose global index into operation index and channel index
  int op_idx = idx / C;
  int ch_idx = idx % C;

  // Get the indices for this operation
  int in_idx = in_indices[op_idx];
  int out_idx = out_indices[op_idx];

  // Bounds checking
  if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
    return;
  }

  // Perform the FMA operation using shared memory for B
  T a_val = a[in_idx * C + ch_idx];
  T b_val = shared_b[ch_idx];

  // Direct write since out_rows are unique (no conflicts)
  if constexpr (std::is_same_v<T, __half>) {
    c[out_idx * C + ch_idx] = __hadd(c[out_idx * C + ch_idx], __hmul(a_val, b_val));
  } else {
    c[out_idx * C + ch_idx] += a_val * b_val;
  }
}

/**
 * Float4 specialization of implicit FMA kernel for improved memory bandwidth
 * Uses vectorized loads/stores to process 4 floats at once
 * Each thread processes 4 consecutive channels (or remaining channels) for one operation
 */
template <>
__global__ void implicit_fma_kernel<float>(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           float* __restrict__ c,
                                           const int* __restrict__ in_indices,
                                           const int* __restrict__ out_indices,
                                           int num_ops,
                                           int C,
                                           int N_A,
                                           int N_C) {
  // Shared memory for vector B
  extern __shared__ char shared_mem_raw[];
  float* shared_b = reinterpret_cast<float*>(shared_mem_raw);

  // Load vector B into shared memory cooperatively using float4
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;
  int C_vec4 = C / 4;
  int C_remainder = C % 4;

  // Load 4 elements at a time with alignment checking
  for (int i = tid; i < C_vec4; i += threads_per_block) {
    const float* b_ptr = &b[i * 4];
    float* sb_ptr = &shared_b[i * 4];

    if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(sb_ptr) % 16 == 0) {
      // Aligned - use vectorized load
      float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
      *reinterpret_cast<float4*>(sb_ptr) = b_vec;
    } else {
      // Not aligned - use scalar loads
      for (int j = 0; j < 4; ++j) {
        shared_b[i * 4 + j] = b[i * 4 + j];
      }
    }
  }

  // Load remaining elements
  int start_remainder = C_vec4 * 4;
  for (int i = tid; i < C_remainder; i += threads_per_block) {
    shared_b[start_remainder + i] = b[start_remainder + i];
  }
  __syncthreads();

  // Modified thread mapping: each thread processes 4 consecutive channels
  int total_vec4_threads = num_ops * C_vec4;
  int total_remainder_threads = num_ops * (C_remainder > 0 ? 1 : 0);
  int total_threads = total_vec4_threads + total_remainder_threads;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_threads) return;

  if (idx < total_vec4_threads) {
    // Vectorized path: process 4 consecutive channels
    int op_idx = idx / C_vec4;
    int vec_idx = idx % C_vec4;

    // Get the indices for this operation
    int in_idx = in_indices[op_idx];
    int out_idx = out_indices[op_idx];

    // Bounds checking
    if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
      return;
    }

    // Check alignment for float4 operations
    const float* a_ptr = &a[in_idx * C + vec_idx * 4];
    float* c_ptr = &c[out_idx * C + vec_idx * 4];
    const float* b_ptr = &shared_b[vec_idx * 4];

    if (reinterpret_cast<uintptr_t>(a_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0) {
      // Aligned - use vectorized operations
      float4 a_vec = *reinterpret_cast<const float4*>(a_ptr);
      float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
      float4 c_vec = *reinterpret_cast<const float4*>(c_ptr);

      // Perform vectorized FMA: c += a * b
      c_vec.x += a_vec.x * b_vec.x;
      c_vec.y += a_vec.y * b_vec.y;
      c_vec.z += a_vec.z * b_vec.z;
      c_vec.w += a_vec.w * b_vec.w;

      // Store result back
      *reinterpret_cast<float4*>(c_ptr) = c_vec;
    } else {
      // Not aligned - fall back to scalar operations
      for (int i = 0; i < 4; ++i) {
        int ch_idx = vec_idx * 4 + i;
        float a_val = a[in_idx * C + ch_idx];
        float b_val = shared_b[ch_idx];
        c[out_idx * C + ch_idx] += a_val * b_val;
      }
    }
  } else if (C_remainder > 0) {
    // Scalar path: process remaining channels
    int remainder_idx = idx - total_vec4_threads;
    int op_idx = remainder_idx;  // One thread per operation for remainder

    // Get the indices for this operation
    int in_idx = in_indices[op_idx];
    int out_idx = out_indices[op_idx];

    // Bounds checking
    if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
      return;
    }

    // Process remaining channels
    int start_ch = C_vec4 * 4;
    for (int ch_offset = 0; ch_offset < C_remainder; ++ch_offset) {
      int ch_idx = start_ch + ch_offset;
      float a_val = a[in_idx * C + ch_idx];
      float b_val = shared_b[ch_idx];
      c[out_idx * C + ch_idx] += a_val * b_val;
    }
  }
}

/**
 * Half8 specialization of implicit FMA kernel for improved memory bandwidth
 * Uses vectorized loads/stores to process 8 halves at once (128 bits = 16 bytes)
 * Each thread processes 8 consecutive channels (or remaining channels) for one operation
 */
template <>
__global__ void implicit_fma_kernel<__half>(const __half* __restrict__ a,
                                            const __half* __restrict__ b,
                                            __half* __restrict__ c,
                                            const int* __restrict__ in_indices,
                                            const int* __restrict__ out_indices,
                                            int num_ops,
                                            int C,
                                            int N_A,
                                            int N_C) {
  // Shared memory for vector B
  extern __shared__ char shared_mem_raw[];
  __half* shared_b = reinterpret_cast<__half*>(shared_mem_raw);

  // Load vector B into shared memory cooperatively using half8 (uint4)
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;
  int C_vec8 = C / 8;
  int C_remainder = C % 8;

  // Load 8 elements at a time with alignment checking
  for (int i = tid; i < C_vec8; i += threads_per_block) {
    const __half* b_ptr = &b[i * 8];
    __half* sb_ptr = &shared_b[i * 8];

    if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(sb_ptr) % 16 == 0) {
      // Aligned - use vectorized load (8 halves as uint4)
      uint4 b_vec = *reinterpret_cast<const uint4*>(b_ptr);
      *reinterpret_cast<uint4*>(sb_ptr) = b_vec;
    } else {
      // Not aligned - use scalar loads
      for (int j = 0; j < 8; ++j) {
        shared_b[i * 8 + j] = b[i * 8 + j];
      }
    }
  }

  // Load remaining elements
  int start_remainder = C_vec8 * 8;
  for (int i = tid; i < C_remainder; i += threads_per_block) {
    shared_b[start_remainder + i] = b[start_remainder + i];
  }
  __syncthreads();

  // Modified thread mapping: each thread processes 8 consecutive channels
  int total_vec8_threads = num_ops * C_vec8;
  int total_remainder_threads = num_ops * (C_remainder > 0 ? 1 : 0);
  int total_threads = total_vec8_threads + total_remainder_threads;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_threads) return;

  if (idx < total_vec8_threads) {
    // Vectorized path: process 8 consecutive channels
    int op_idx = idx / C_vec8;
    int vec_idx = idx % C_vec8;

    // Get the indices for this operation
    int in_idx = in_indices[op_idx];
    int out_idx = out_indices[op_idx];

    // Bounds checking
    if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
      return;
    }

    // Check alignment for half8 operations
    const __half* a_ptr = &a[in_idx * C + vec_idx * 8];
    __half* c_ptr = &c[out_idx * C + vec_idx * 8];
    const __half* b_ptr = &shared_b[vec_idx * 8];

    if (reinterpret_cast<uintptr_t>(a_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0) {
      // Aligned - use vectorized operations
      uint4 a_vec = *reinterpret_cast<const uint4*>(a_ptr);
      uint4 b_vec = *reinterpret_cast<const uint4*>(b_ptr);
      uint4 c_vec = *reinterpret_cast<const uint4*>(c_ptr);

      // Extract halves from uint4 and perform vectorized FMA
      __half* a_halfs = reinterpret_cast<__half*>(&a_vec);
      __half* b_halfs = reinterpret_cast<__half*>(&b_vec);
      __half* c_halfs = reinterpret_cast<__half*>(&c_vec);

      // Perform vectorized FMA: c += a * b for all 8 halves
      for (int i = 0; i < 8; ++i) {
        c_halfs[i] = __hadd(c_halfs[i], __hmul(a_halfs[i], b_halfs[i]));
      }

      // Store result back
      *reinterpret_cast<uint4*>(c_ptr) = c_vec;
    } else {
      // Not aligned - fall back to scalar operations
      for (int i = 0; i < 8; ++i) {
        int ch_idx = vec_idx * 8 + i;
        __half a_val = a[in_idx * C + ch_idx];
        __half b_val = shared_b[ch_idx];
        c[out_idx * C + ch_idx] = __hadd(c[out_idx * C + ch_idx], __hmul(a_val, b_val));
      }
    }
  } else if (C_remainder > 0) {
    // Scalar path: process remaining channels
    int remainder_idx = idx - total_vec8_threads;
    int op_idx = remainder_idx;  // One thread per operation for remainder

    // Get the indices for this operation
    int in_idx = in_indices[op_idx];
    int out_idx = out_indices[op_idx];

    // Bounds checking
    if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
      return;
    }

    // Process remaining channels
    int start_ch = C_vec8 * 8;
    for (int ch_offset = 0; ch_offset < C_remainder; ++ch_offset) {
      int ch_idx = start_ch + ch_offset;
      __half a_val = a[in_idx * C + ch_idx];
      __half b_val = shared_b[ch_idx];
      c[out_idx * C + ch_idx] = __hadd(c[out_idx * C + ch_idx], __hmul(a_val, b_val));
    }
  }
}

/**
 * Row-wise processing version with shared memory for vector B (each thread processes one row)
 */
template <typename T>
__global__ void implicit_fma_kernel_rowwise(const T* __restrict__ a,
                                            const T* __restrict__ b,
                                            T* __restrict__ c,
                                            const int* __restrict__ in_indices,
                                            const int* __restrict__ out_indices,
                                            int num_ops,
                                            int C,
                                            int N_A,
                                            int N_C) {
  // Shared memory for vector B
  extern __shared__ char shared_mem_raw[];
  T* shared_b = reinterpret_cast<T*>(shared_mem_raw);

  // Load vector B into shared memory cooperatively
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Each thread loads multiple elements of B if necessary
  for (int i = tid; i < C; i += threads_per_block) {
    shared_b[i] = b[i];
  }
  __syncthreads();

  // Each thread processes one operation (one row)
  int op_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (op_idx >= num_ops) return;

  // Get the indices for this operation
  int in_idx = in_indices[op_idx];
  int out_idx = out_indices[op_idx];

  // Bounds checking
  if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
    return;
  }

  // Process all channels for this operation using shared memory
  for (int ch_idx = 0; ch_idx < C; ++ch_idx) {
    T a_val = a[in_idx * C + ch_idx];
    T b_val = shared_b[ch_idx];

    // Direct write since out_rows are unique (no conflicts)
    if constexpr (std::is_same_v<T, __half>) {
      c[out_idx * C + ch_idx] = __hadd(c[out_idx * C + ch_idx], __hmul(a_val, b_val));
    } else {
      c[out_idx * C + ch_idx] += a_val * b_val;
    }
  }
}

/**
 * Float4 specialization of rowwise kernel for improved memory bandwidth
 */
template <>
__global__ void implicit_fma_kernel_rowwise<float>(const float* __restrict__ a,
                                                   const float* __restrict__ b,
                                                   float* __restrict__ c,
                                                   const int* __restrict__ in_indices,
                                                   const int* __restrict__ out_indices,
                                                   int num_ops,
                                                   int C,
                                                   int N_A,
                                                   int N_C) {
  // Shared memory for vector B
  extern __shared__ char shared_mem_raw[];
  float* shared_b = reinterpret_cast<float*>(shared_mem_raw);

  // Load vector B into shared memory cooperatively using float4
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;
  int C_vec4 = C / 4;
  int C_remainder = C % 4;

  // Load 4 elements at a time with alignment checking
  for (int i = tid; i < C_vec4; i += threads_per_block) {
    const float* b_ptr = &b[i * 4];
    float* sb_ptr = &shared_b[i * 4];

    if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(sb_ptr) % 16 == 0) {
      // Aligned - use vectorized load
      float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
      *reinterpret_cast<float4*>(sb_ptr) = b_vec;
    } else {
      // Not aligned - use scalar loads
      for (int j = 0; j < 4; ++j) {
        shared_b[i * 4 + j] = b[i * 4 + j];
      }
    }
  }

  // Load remaining elements
  int start_remainder = C_vec4 * 4;
  for (int i = tid; i < C_remainder; i += threads_per_block) {
    shared_b[start_remainder + i] = b[start_remainder + i];
  }
  __syncthreads();

  // Each thread processes one operation (one row)
  int op_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (op_idx >= num_ops) return;

  // Get the indices for this operation
  int in_idx = in_indices[op_idx];
  int out_idx = out_indices[op_idx];

  // Bounds checking
  if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
    return;
  }

  // Process channels using float4 vectorization
  int ch_idx = 0;

  // Process 4 channels at a time
  for (int vec_idx = 0; vec_idx < C_vec4; ++vec_idx) {
    // Check alignment for float4 operations
    const float* a_ptr = &a[in_idx * C + vec_idx * 4];
    float* c_ptr = &c[out_idx * C + vec_idx * 4];
    const float* b_ptr = &shared_b[vec_idx * 4];

    if (reinterpret_cast<uintptr_t>(a_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0) {
      // Aligned - use vectorized operations
      float4 a_vec = *reinterpret_cast<const float4*>(a_ptr);
      float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
      float4 c_vec = *reinterpret_cast<const float4*>(c_ptr);

      // Perform vectorized FMA: c += a * b
      c_vec.x += a_vec.x * b_vec.x;
      c_vec.y += a_vec.y * b_vec.y;
      c_vec.z += a_vec.z * b_vec.z;
      c_vec.w += a_vec.w * b_vec.w;

      // Store result back
      *reinterpret_cast<float4*>(c_ptr) = c_vec;
    } else {
      // Not aligned - fall back to scalar operations
      for (int i = 0; i < 4; ++i) {
        int ch_idx = vec_idx * 4 + i;
        float a_val = a[in_idx * C + ch_idx];
        float b_val = shared_b[ch_idx];
        c[out_idx * C + ch_idx] += a_val * b_val;
      }
    }
  }

  // Process remaining channels
  int start_ch = C_vec4 * 4;
  for (int ch_offset = 0; ch_offset < C_remainder; ++ch_offset) {
    ch_idx = start_ch + ch_offset;
    float a_val = a[in_idx * C + ch_idx];
    float b_val = shared_b[ch_idx];
    c[out_idx * C + ch_idx] += a_val * b_val;
  }
}

/**
 * Half8 specialization of rowwise kernel for improved memory bandwidth
 * Uses vectorized loads/stores to process 8 halves at once (128 bits = 16 bytes)
 */
template <>
__global__ void implicit_fma_kernel_rowwise<__half>(const __half* __restrict__ a,
                                                    const __half* __restrict__ b,
                                                    __half* __restrict__ c,
                                                    const int* __restrict__ in_indices,
                                                    const int* __restrict__ out_indices,
                                                    int num_ops,
                                                    int C,
                                                    int N_A,
                                                    int N_C) {
  // Shared memory for vector B
  extern __shared__ char shared_mem_raw[];
  __half* shared_b = reinterpret_cast<__half*>(shared_mem_raw);

  // Load vector B into shared memory cooperatively using half8 (uint4)
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;
  int C_vec8 = C / 8;
  int C_remainder = C % 8;

  // Load 8 elements at a time with alignment checking
  for (int i = tid; i < C_vec8; i += threads_per_block) {
    const __half* b_ptr = &b[i * 8];
    __half* sb_ptr = &shared_b[i * 8];

    if (reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(sb_ptr) % 16 == 0) {
      // Aligned - use vectorized load (8 halves as uint4)
      uint4 b_vec = *reinterpret_cast<const uint4*>(b_ptr);
      *reinterpret_cast<uint4*>(sb_ptr) = b_vec;
    } else {
      // Not aligned - use scalar loads
      for (int j = 0; j < 8; ++j) {
        shared_b[i * 8 + j] = b[i * 8 + j];
      }
    }
  }

  // Load remaining elements
  int start_remainder = C_vec8 * 8;
  for (int i = tid; i < C_remainder; i += threads_per_block) {
    shared_b[start_remainder + i] = b[start_remainder + i];
  }
  __syncthreads();

  // Each thread processes one operation (one row)
  int op_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (op_idx >= num_ops) return;

  // Get the indices for this operation
  int in_idx = in_indices[op_idx];
  int out_idx = out_indices[op_idx];

  // Bounds checking
  if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) {
    return;
  }

  // Process channels using half8 vectorization
  int ch_idx = 0;

  // Process 8 channels at a time
  for (int vec_idx = 0; vec_idx < C_vec8; ++vec_idx) {
    // Check alignment for half8 operations
    const __half* a_ptr = &a[in_idx * C + vec_idx * 8];
    __half* c_ptr = &c[out_idx * C + vec_idx * 8];
    const __half* b_ptr = &shared_b[vec_idx * 8];

    if (reinterpret_cast<uintptr_t>(a_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(c_ptr) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(b_ptr) % 16 == 0) {
      // Aligned - use vectorized operations
      uint4 a_vec = *reinterpret_cast<const uint4*>(a_ptr);
      uint4 b_vec = *reinterpret_cast<const uint4*>(b_ptr);
      uint4 c_vec = *reinterpret_cast<const uint4*>(c_ptr);

      // Extract halves from uint4 and perform vectorized FMA
      __half* a_halfs = reinterpret_cast<__half*>(&a_vec);
      __half* b_halfs = reinterpret_cast<__half*>(&b_vec);
      __half* c_halfs = reinterpret_cast<__half*>(&c_vec);

      // Perform vectorized FMA: c += a * b for all 8 halves
      for (int i = 0; i < 8; ++i) {
        c_halfs[i] = __hadd(c_halfs[i], __hmul(a_halfs[i], b_halfs[i]));
      }

      // Store result back
      *reinterpret_cast<uint4*>(c_ptr) = c_vec;
    } else {
      // Not aligned - fall back to scalar operations
      for (int i = 0; i < 8; ++i) {
        int ch_idx = vec_idx * 8 + i;
        __half a_val = a[in_idx * C + ch_idx];
        __half b_val = shared_b[ch_idx];
        c[out_idx * C + ch_idx] = __hadd(c[out_idx * C + ch_idx], __hmul(a_val, b_val));
      }
    }
  }

  // Process remaining channels
  int start_ch = C_vec8 * 8;
  for (int ch_offset = 0; ch_offset < C_remainder; ++ch_offset) {
    ch_idx = start_ch + ch_offset;
    __half a_val = a[in_idx * C + ch_idx];
    __half b_val = shared_b[ch_idx];
    c[out_idx * C + ch_idx] = __hadd(c[out_idx * C + ch_idx], __hmul(a_val, b_val));
  }
}

// Main templated function implementation
namespace warpconvnet {
namespace implicit_fma {

/**
 * @brief Run an implicit FMA operation with gather/scatter support.
 *
 * @param tensor_a: Pointer to the A matrix.
 * @param tensor_b: Pointer to the B vector.
 * @param tensor_c: Pointer to the C matrix (modified in-place).
 * @param in_indices: Input indices for gathering from A.
 * @param out_indices: Output indices for scattering to C.
 * @param num_ops: Number of operations (length of indices).
 * @param C: Number of channels/columns.
 * @param N_A: Number of rows in A.
 * @param N_C: Number of rows in C.
 * @param kernel_type: Type of kernel to use ("basic", "rowwise").
 *
 * @return Status code indicating the success or failure of the operation.
 *
 * Operation: c[out_indices[i], :] += a[in_indices[i], :] * b
 */
template <typename ElementA, typename ElementB, typename ElementC>
int run_implicit_fma_templated(const void* tensor_a,
                               const void* tensor_b,
                               void* tensor_c,
                               const int* in_indices,
                               const int* out_indices,
                               int num_ops,
                               int C,
                               int N_A,
                               int N_C,
                               const std::string& kernel_type) {
  // Convert void pointers to appropriate types
  auto a_ptr = reinterpret_cast<const ElementA*>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementB*>(tensor_b);
  auto c_ptr = reinterpret_cast<ElementC*>(tensor_c);

  // Get CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Launch kernel configuration
  const int threads_per_block = 256;
  int shared_mem_size = C * sizeof(ElementB);

  // Dispatch based on kernel type
  if (kernel_type == "rowwise") {
    // Row-wise kernel: each thread processes one operation
    int blocks = (num_ops + threads_per_block - 1) / threads_per_block;

    if constexpr (std::is_same_v<ElementA, float>) {
      implicit_fma_kernel_rowwise<float><<<blocks, threads_per_block, shared_mem_size, stream>>>(
          a_ptr, b_ptr, c_ptr, in_indices, out_indices, num_ops, C, N_A, N_C);
    } else if constexpr (std::is_same_v<ElementA, cutlass::half_t>) {
      implicit_fma_kernel_rowwise<__half><<<blocks, threads_per_block, shared_mem_size, stream>>>(
          reinterpret_cast<const __half*>(a_ptr),
          reinterpret_cast<const __half*>(b_ptr),
          reinterpret_cast<__half*>(c_ptr),
          in_indices,
          out_indices,
          num_ops,
          C,
          N_A,
          N_C);
    } else if constexpr (std::is_same_v<ElementA, double>) {
      implicit_fma_kernel_rowwise<double><<<blocks, threads_per_block, shared_mem_size, stream>>>(
          a_ptr, b_ptr, c_ptr, in_indices, out_indices, num_ops, C, N_A, N_C);
    } else {
      return static_cast<int>(ImplicitFmaStatus::kErrorUnsupportedDataType);
    }
  } else if (kernel_type == "basic") {
    // Basic kernel: thread count depends on data type
    if constexpr (std::is_same_v<ElementA, float>) {
      // Float4 specialization: different thread mapping
      int C_vec4 = C / 4;
      int C_remainder = C % 4;
      int total_vec4_threads = num_ops * C_vec4;
      int total_remainder_threads = num_ops * (C_remainder > 0 ? 1 : 0);
      int total_threads = total_vec4_threads + total_remainder_threads;
      int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

      implicit_fma_kernel<float><<<blocks, threads_per_block, shared_mem_size, stream>>>(
          a_ptr, b_ptr, c_ptr, in_indices, out_indices, num_ops, C, N_A, N_C);
    } else if constexpr (std::is_same_v<ElementA, cutlass::half_t>) {
      // Half8 specialization: different thread mapping
      int C_vec8 = C / 8;
      int C_remainder = C % 8;
      int total_vec8_threads = num_ops * C_vec8;
      int total_remainder_threads = num_ops * (C_remainder > 0 ? 1 : 0);
      int total_threads = total_vec8_threads + total_remainder_threads;
      int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

      implicit_fma_kernel<__half><<<blocks, threads_per_block, shared_mem_size, stream>>>(
          reinterpret_cast<const __half*>(a_ptr),
          reinterpret_cast<const __half*>(b_ptr),
          reinterpret_cast<__half*>(c_ptr),
          in_indices,
          out_indices,
          num_ops,
          C,
          N_A,
          N_C);
    } else if constexpr (std::is_same_v<ElementA, double>) {
      // Generic version: each thread processes one element
      int total_threads = num_ops * C;
      int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

      implicit_fma_kernel<double><<<blocks, threads_per_block, shared_mem_size, stream>>>(
          a_ptr, b_ptr, c_ptr, in_indices, out_indices, num_ops, C, N_A, N_C);
    } else {
      return static_cast<int>(ImplicitFmaStatus::kErrorUnsupportedDataType);
    }
  } else {
    return static_cast<int>(ImplicitFmaStatus::kErrorInvalidKernelType);
  }

  // Check for CUDA errors
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(ImplicitFmaStatus::kErrorKernelExecution);
  }

  return static_cast<int>(ImplicitFmaStatus::kSuccess);
}

}  // namespace implicit_fma
}  // namespace warpconvnet

// Use the namespace for convenience in the rest of the file
using namespace warpconvnet::implicit_fma;

// Expose the template instantiations for use in pybind
template int warpconvnet::implicit_fma::run_implicit_fma_templated<float, float, float>(
    const void*,
    const void*,
    void*,
    const int*,
    const int*,
    int,
    int,
    int,
    int,
    const std::string&);

template int warpconvnet::implicit_fma::
    run_implicit_fma_templated<cutlass::half_t, cutlass::half_t, cutlass::half_t>(
        const void*,
        const void*,
        void*,
        const int*,
        const int*,
        int,
        int,
        int,
        int,
        const std::string&);

template int warpconvnet::implicit_fma::run_implicit_fma_templated<double, double, double>(
    const void*,
    const void*,
    void*,
    const int*,
    const int*,
    int,
    int,
    int,
    int,
    const std::string&);
