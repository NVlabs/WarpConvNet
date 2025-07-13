// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <string>
#include <type_traits>

/**
 * Templated Implicit Reduction CUDA Kernels with Optional B
 *
 * Clean template-based design following CUTLASS patterns:
 * - Single templated kernel implementation for all data types (float, half, bfloat16, double)
 * - Optional B parameter with compile-time optimization
 * - Organized in warpconvnet::implicit_reduction namespace
 * - Status-based error handling with clean propagation to Python
 * - Two-pass reduction for optimal performance
 *
 * Operation: result[c] = ∑_i A[a_indices[i], c] * B[b_indices[i], c]
 * Where:
 * - A: Input matrix A (N_A x C)
 * - B: Input matrix B (N_B x C) - optional, treated as ones if null
 * - a_indices: Indices into A (M,)
 * - b_indices: Indices into B (M,) - only used if B is not null
 * - result: Output vector (C,) - summed along the indexed dimension
 */

// Define error codes for implicit reduction operations
enum class ImplicitReductionStatus {
  kSuccess = 0,
  kErrorInvalidKernelType = 1,
  kErrorUnsupportedDataType = 2,
  kErrorKernelExecution = 3
};

/**
 * First pass reduction kernel with warp shuffles and shared memory optimization
 * Each thread block processes a subset of operations and channels
 */
template <typename T, bool B_is_null>
__global__ void implicit_reduction_pass1_kernel(
    const T* __restrict__ A,            // [N_A x C]
    const int* __restrict__ a_indices,  // [M]
    const T* __restrict__ B,            // [N_B x C] (ignored if B_is_null)
    const int* __restrict__ b_indices,  // [M] (ignored if B_is_null)
    T* __restrict__ partial_sums,       // [num_blocks x C]
    int M,                              // Number of operations
    int C,                              // Number of channels
    int N_A,                            // Number of rows in A
    int N_B,                            // Number of rows in B (ignored if B_is_null)
    int tile_C)                         // Channel tile size
{
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int block_threads = blockDim.x;

  int block_op_offset = blockIdx.x * block_threads;
  int block_c_offset = blockIdx.y * tile_C;
  int stride = gridDim.x * block_threads;

  // Per-thread local register accumulation
  T accum[32];  // assume tile_C ≤ 32
#pragma unroll
  for (int i = 0; i < tile_C; ++i) {
    if constexpr (std::is_same_v<T, __half>) {
      accum[i] = __float2half(0.0f);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      accum[i] = __float2bfloat16(0.0f);
    } else {
      accum[i] = T(0);
    }
  }

  // Step 1: Accumulate operations assigned to this thread
  for (int op = block_op_offset + tid; op < M; op += stride) {
    int a_idx = a_indices[op];

    // Bounds checking for A
    if (a_idx < 0 || a_idx >= N_A) continue;

    if constexpr (!B_is_null) {
      int b_idx = b_indices[op];
      // Bounds checking for B
      if (b_idx < 0 || b_idx >= N_B) continue;
    }

    for (int i = 0; i < tile_C; ++i) {
      int c = block_c_offset + i;
      if (c < C) {
        T a_val = A[a_idx * C + c];

        if constexpr (B_is_null) {
          // B is null - just add a_val directly (no need to multiply by 1)
          if constexpr (std::is_same_v<T, __half>) {
            accum[i] = __hadd(accum[i], a_val);
          } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            accum[i] = accum[i] + a_val;
          } else {
            accum[i] += a_val;
          }
        } else {
          // B is provided - multiply a_val with b_val
          int b_idx = b_indices[op];
          T b_val = B[b_idx * C + c];

          if constexpr (std::is_same_v<T, __half>) {
            accum[i] = __hadd(accum[i], __hmul(a_val, b_val));
          } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            accum[i] = accum[i] + a_val * b_val;
          } else {
            accum[i] += a_val * b_val;
          }
        }
      }
    }
  }

// Step 2: In-warp reduction with shuffles
#pragma unroll
  for (int i = 0; i < tile_C; ++i) {
    T val = accum[i];
    for (int offset = 16; offset > 0; offset /= 2) {
      if constexpr (std::is_same_v<T, __half>) {
        T other_val = __shfl_down_sync(0xffffffff, val, offset);
        val = __hadd(val, other_val);
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        T other_val = __shfl_down_sync(0xffffffff, val, offset);
        val = val + other_val;
      } else {
        val += __shfl_down_sync(0xffffffff, val, offset);
      }
    }
    accum[i] = val;  // lane 0 holds final value
  }

  // Step 3: Warp leader writes to global partial sums
  if (lane_id == 0) {
    for (int i = 0; i < tile_C; ++i) {
      int c = block_c_offset + i;
      if (c < C) {
        int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        partial_sums[block_idx * C + c] = accum[i];
      }
    }
  }
}

/**
 * Second pass kernel to combine partial sums into final result
 */
template <typename T>
__global__ void implicit_reduction_pass2_kernel(
    const T* __restrict__ partial_sums,  // [num_blocks x C]
    T* __restrict__ result,              // [C]
    int num_blocks_x,
    int num_blocks_y,
    int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;

  T sum;
  if constexpr (std::is_same_v<T, __half>) {
    sum = __float2half(0.0f);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    sum = __float2bfloat16(0.0f);
  } else {
    sum = T(0);
  }
  for (int by = 0; by < num_blocks_y; ++by) {
    for (int bx = 0; bx < num_blocks_x; ++bx) {
      int block_id = by * num_blocks_x + bx;
      if constexpr (std::is_same_v<T, __half>) {
        sum = __hadd(sum, partial_sums[block_id * C + c]);
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        sum = sum + partial_sums[block_id * C + c];
      } else {
        sum += partial_sums[block_id * C + c];
      }
    }
  }
  result[c] = sum;
}

// Main templated function implementation
namespace warpconvnet {
namespace implicit_reduction {

/**
 * @brief Run an implicit reduction operation with optional B.
 *
 * @param tensor_a: Pointer to the A matrix.
 * @param a_indices: Indices into A for gathering.
 * @param tensor_b: Pointer to the B matrix (can be nullptr).
 * @param b_indices: Indices into B for gathering (ignored if tensor_b is nullptr).
 * @param result: Pointer to the output vector.
 * @param M: Number of operations (length of indices).
 * @param C: Number of channels/columns.
 * @param N_A: Number of rows in A.
 * @param N_B: Number of rows in B (ignored if tensor_b is nullptr).
 * @param kernel_type: Type of kernel to use ("basic").
 *
 * @return Status code indicating the success or failure of the operation.
 *
 * Operation: result[c] = ∑_i A[a_indices[i], c] * B[b_indices[i], c]
 * If tensor_b is nullptr, B is treated as all ones.
 */
template <typename ElementA, typename ElementB, typename ElementOutput>
int run_implicit_reduction_templated(const void* tensor_a,
                                     const int* a_indices,
                                     const void* tensor_b,
                                     const int* b_indices,
                                     void* result,
                                     int M,
                                     int C,
                                     int N_A,
                                     int N_B,
                                     const std::string& kernel_type) {
  // Convert void pointers to appropriate types
  auto a_ptr = reinterpret_cast<const ElementA*>(tensor_a);
  auto b_ptr = tensor_b ? reinterpret_cast<const ElementB*>(tensor_b) : nullptr;
  auto result_ptr = reinterpret_cast<ElementOutput*>(result);

  // Get CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Kernel configuration
  const int threads_per_block = 256;
  const int TILE_C = 32;
  int num_tiles_C = (C + TILE_C - 1) / TILE_C;
  int num_blocks_M = std::min((M + threads_per_block - 1) / threads_per_block, 1024);

  // Allocate temporary storage for partial sums
  size_t partial_sums_size = num_blocks_M * num_tiles_C * C * sizeof(ElementOutput);
  ElementOutput* partial_sums;
  cudaMalloc(&partial_sums, partial_sums_size);
  cudaMemset(partial_sums, 0, partial_sums_size);

  // Launch configuration for pass 1
  dim3 grid_pass1(num_blocks_M, num_tiles_C);
  dim3 block_pass1(threads_per_block);

  // Dispatch based on kernel type and whether B is null
  if (kernel_type == "basic") {
    if (b_ptr == nullptr) {
      // B is null - treat as all ones
      if constexpr (std::is_same_v<ElementA, float>) {
        implicit_reduction_pass1_kernel<float, true><<<grid_pass1, block_pass1, 0, stream>>>(
            a_ptr, a_indices, nullptr, nullptr, partial_sums, M, C, N_A, N_B, TILE_C);
      } else if constexpr (std::is_same_v<ElementA, cutlass::half_t>) {
        implicit_reduction_pass1_kernel<__half, true>
            <<<grid_pass1, block_pass1, 0, stream>>>(reinterpret_cast<const __half*>(a_ptr),
                                                     a_indices,
                                                     nullptr,
                                                     nullptr,
                                                     reinterpret_cast<__half*>(partial_sums),
                                                     M,
                                                     C,
                                                     N_A,
                                                     N_B,
                                                     TILE_C);
      } else if constexpr (std::is_same_v<ElementA, cutlass::bfloat16_t>) {
        implicit_reduction_pass1_kernel<__nv_bfloat16, true>
            <<<grid_pass1, block_pass1, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(a_ptr),
                                                     a_indices,
                                                     nullptr,
                                                     nullptr,
                                                     reinterpret_cast<__nv_bfloat16*>(partial_sums),
                                                     M,
                                                     C,
                                                     N_A,
                                                     N_B,
                                                     TILE_C);
      } else if constexpr (std::is_same_v<ElementA, double>) {
        implicit_reduction_pass1_kernel<double, true><<<grid_pass1, block_pass1, 0, stream>>>(
            a_ptr, a_indices, nullptr, nullptr, partial_sums, M, C, N_A, N_B, TILE_C);
      } else {
        cudaFree(partial_sums);
        return static_cast<int>(ImplicitReductionStatus::kErrorUnsupportedDataType);
      }
    } else {
      // B is not null
      if constexpr (std::is_same_v<ElementA, float>) {
        implicit_reduction_pass1_kernel<float, false><<<grid_pass1, block_pass1, 0, stream>>>(
            a_ptr, a_indices, b_ptr, b_indices, partial_sums, M, C, N_A, N_B, TILE_C);
      } else if constexpr (std::is_same_v<ElementA, cutlass::half_t>) {
        implicit_reduction_pass1_kernel<__half, false>
            <<<grid_pass1, block_pass1, 0, stream>>>(reinterpret_cast<const __half*>(a_ptr),
                                                     a_indices,
                                                     reinterpret_cast<const __half*>(b_ptr),
                                                     b_indices,
                                                     reinterpret_cast<__half*>(partial_sums),
                                                     M,
                                                     C,
                                                     N_A,
                                                     N_B,
                                                     TILE_C);
      } else if constexpr (std::is_same_v<ElementA, cutlass::bfloat16_t>) {
        implicit_reduction_pass1_kernel<__nv_bfloat16, false>
            <<<grid_pass1, block_pass1, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(a_ptr),
                                                     a_indices,
                                                     reinterpret_cast<const __nv_bfloat16*>(b_ptr),
                                                     b_indices,
                                                     reinterpret_cast<__nv_bfloat16*>(partial_sums),
                                                     M,
                                                     C,
                                                     N_A,
                                                     N_B,
                                                     TILE_C);
      } else if constexpr (std::is_same_v<ElementA, double>) {
        implicit_reduction_pass1_kernel<double, false><<<grid_pass1, block_pass1, 0, stream>>>(
            a_ptr, a_indices, b_ptr, b_indices, partial_sums, M, C, N_A, N_B, TILE_C);
      } else {
        cudaFree(partial_sums);
        return static_cast<int>(ImplicitReductionStatus::kErrorUnsupportedDataType);
      }
    }

    // Launch pass 2 to combine partial sums.
    // This is common to both cases of B being null or not.
    int threads_pass2 = 256;
    int blocks_pass2 = (C + threads_pass2 - 1) / threads_pass2;

    if constexpr (std::is_same_v<ElementOutput, float>) {
      implicit_reduction_pass2_kernel<float><<<blocks_pass2, threads_pass2, 0, stream>>>(
          partial_sums, result_ptr, num_blocks_M, num_tiles_C, C);
    } else if constexpr (std::is_same_v<ElementOutput, cutlass::half_t>) {
      implicit_reduction_pass2_kernel<__half>
          <<<blocks_pass2, threads_pass2, 0, stream>>>(reinterpret_cast<__half*>(partial_sums),
                                                       reinterpret_cast<__half*>(result_ptr),
                                                       num_blocks_M,
                                                       num_tiles_C,
                                                       C);
    } else if constexpr (std::is_same_v<ElementOutput, cutlass::bfloat16_t>) {
      implicit_reduction_pass2_kernel<__nv_bfloat16><<<blocks_pass2, threads_pass2, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(partial_sums),
          reinterpret_cast<__nv_bfloat16*>(result_ptr),
          num_blocks_M,
          num_tiles_C,
          C);
    } else if constexpr (std::is_same_v<ElementOutput, double>) {
      implicit_reduction_pass2_kernel<double><<<blocks_pass2, threads_pass2, 0, stream>>>(
          partial_sums, result_ptr, num_blocks_M, num_tiles_C, C);
    }
  } else {
    cudaFree(partial_sums);
    return static_cast<int>(ImplicitReductionStatus::kErrorInvalidKernelType);
  }

  // Check for CUDA errors
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cudaFree(partial_sums);
    return static_cast<int>(ImplicitReductionStatus::kErrorKernelExecution);
  }

  // Clean up temporary storage
  cudaFree(partial_sums);

  return static_cast<int>(ImplicitReductionStatus::kSuccess);
}

}  // namespace implicit_reduction
}  // namespace warpconvnet

// Use the namespace for convenience in the rest of the file
using namespace warpconvnet::implicit_reduction;

// Expose the template instantiations for use in pybind
template int warpconvnet::implicit_reduction::run_implicit_reduction_templated<float, float, float>(
    const void*,
    const int*,
    const void*,
    const int*,
    void*,
    int,
    int,
    int,
    int,
    const std::string&);

template int warpconvnet::implicit_reduction::
    run_implicit_reduction_templated<cutlass::half_t, cutlass::half_t, cutlass::half_t>(
        const void*,
        const int*,
        const void*,
        const int*,
        void*,
        int,
        int,
        int,
        int,
        const std::string&);

template int warpconvnet::implicit_reduction::
    run_implicit_reduction_templated<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t>(
        const void*,
        const int*,
        const void*,
        const int*,
        void*,
        int,
        int,
        int,
        int,
        const std::string&);

template int
warpconvnet::implicit_reduction::run_implicit_reduction_templated<double, double, double>(
    const void*,
    const int*,
    const void*,
    const int*,
    void*,
    int,
    int,
    int,
    int,
    const std::string&);
