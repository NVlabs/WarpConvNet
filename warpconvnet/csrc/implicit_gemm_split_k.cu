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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <string>
#include <type_traits>

// Custom reduction operator for half-precision types
struct HalfAddOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    if constexpr (std::is_same_v<T, __half>) {
      return __hadd(a, b);
    } else {
      return a + b;
    }
  }
};

// Get current CUDA stream (simplified version)
__host__ __forceinline__ cudaStream_t getCurrentCUDAStream() { return cudaStreamDefault; }

// Define error codes for split-K implicit GEMM operations
enum class SplitKGemmStatus {
  kSuccess = 0,
  kErrorInvalidKernelType = 1,
  kErrorUnsupportedDataType = 2,
  kErrorKernelExecution = 3,
  kErrorInvalidDimensions = 4,
  kErrorInsufficientMemory = 5
};

/**
 * Split-K Implicit GEMM Kernel with CUB Block Reduction
 * Computes: C_partial += transpose(A[indices_a_chunk]) @ B[indices_b_chunk]
 *
 * Each block handles one output element (i, j) and threads cooperatively
 * reduce over the K dimension using CUB's efficient BlockReduce.
 *
 * Template parameters:
 * - Dtype: Data type (float, __half, __nv_bfloat16)
 * - Itype: Index type (int, int64_t)
 * - BLOCK_THREADS: Number of threads per block
 * - use_atomic: Whether to use atomic operations for final write
 */
template <typename Dtype, typename Itype, int BLOCK_THREADS, bool use_atomic = true>
__global__ void split_k_implicit_gemm_stage1(
    const Dtype *__restrict__ A,          // Input matrix A (N x C_a)
    const Dtype *__restrict__ B,          // Input matrix B (N x C_b)
    Dtype *__restrict__ C_partial,        // Partial output (C_a x C_b)
    const Itype *__restrict__ indices_a,  // Row indices for A
    const Itype *__restrict__ indices_b,  // Row indices for B
    const int C_a,                        // Number of columns in A
    const int C_b,                        // Number of columns in B
    const int chunk_start,                // Start index for this chunk
    const int chunk_size,                 // Number of indices in this chunk
    const int split_k_idx) {              // Split-K index for this kernel

  // Thread and block indices
  const int tid = threadIdx.x;

  // CUB block reduction setup
  using BlockReduce = cub::BlockReduce<Dtype, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Loop over output elements this block is responsible for
  for (int i = blockIdx.x; i < C_a; i += gridDim.x) {
    for (int j = blockIdx.y; j < C_b; j += gridDim.y) {
      // Initialize thread accumulator
      Dtype thread_sum;
      if constexpr (std::is_same_v<Dtype, __half>) {
        thread_sum = __float2half(0.0f);
      } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
        thread_sum = __float2bfloat16(0.0f);
      } else {
        thread_sum = Dtype(0);
      }

      // Each thread accumulates its part of the K range
      for (int k = chunk_start + tid; k < chunk_start + chunk_size; k += BLOCK_THREADS) {
        if (k < chunk_start + chunk_size) {
          const Itype ia = indices_a[k];
          const Itype ib = indices_b[k];

          // Bounds checking for indices
          if (ia >= 0 && ib >= 0) {
            const Dtype a_val = A[ia * C_a + i];  // transpose(A)[i][k] = A[ia][i]
            const Dtype b_val = B[ib * C_b + j];  // B[ib][j]

            if constexpr (std::is_same_v<Dtype, __half>) {
              thread_sum = __hadd(thread_sum, __hmul(a_val, b_val));
            } else {
              thread_sum += a_val * b_val;
            }
          }
        }
      }

      // CUB BlockReduce across all threads in the block using custom operator
      Dtype tile_sum = BlockReduce(temp_storage).Reduce(thread_sum, HalfAddOp{});
      __syncthreads();

      // One thread per block commits the result
      if (tid == 0) {
        const int c_offset = i * C_b + j;
        if constexpr (use_atomic) {
          if constexpr (std::is_same_v<Dtype, __half>) {
            atomicAdd(&C_partial[c_offset], tile_sum);
          } else {
            atomicAdd(&C_partial[c_offset], tile_sum);
          }
        } else {
          if constexpr (std::is_same_v<Dtype, __half>) {
            C_partial[c_offset] = __hadd(C_partial[c_offset], tile_sum);
          } else {
            C_partial[c_offset] += tile_sum;
          }
        }
      }
    }
  }
}

/**
 * Stage 2 - Reduction Kernel with CUB Block Reduction
 * Reduces multiple partial C matrices into final result
 */
template <typename Dtype, int BLOCK_THREADS>
__global__ void split_k_reduction_kernel(
    Dtype *__restrict__ C,                 // Final output matrix (C_a x C_b)
    const Dtype *__restrict__ C_partials,  // Partial matrices (num_splits x C_a x C_b)
    const int C_a,                         // C_a dimension
    const int C_b,                         // C_b dimension
    const int num_splits) {                // Number of split-K parts

  const int tid = threadIdx.x;

  // CUB block reduction setup
  using BlockReduce = cub::BlockReduce<Dtype, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // Loop over output elements this block is responsible for
  for (int i = blockIdx.x; i < C_a; i += gridDim.x) {
    for (int j = blockIdx.y; j < C_b; j += gridDim.y) {
      // Initialize thread accumulator for this output element
      Dtype thread_sum;
      if constexpr (std::is_same_v<Dtype, __half>) {
        thread_sum = __float2half(0.0f);
      } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
        thread_sum = __float2bfloat16(0.0f);
      } else {
        thread_sum = Dtype(0);
      }

      // Each thread processes its portion of splits
      for (int split = tid; split < num_splits; split += BLOCK_THREADS) {
        const int partial_idx = split * C_a * C_b + i * C_b + j;
        if constexpr (std::is_same_v<Dtype, __half>) {
          thread_sum = __hadd(thread_sum, C_partials[partial_idx]);
        } else {
          thread_sum += C_partials[partial_idx];
        }
      }

      // CUB BlockReduce across all threads in the block using custom operator
      Dtype block_sum = BlockReduce(temp_storage).Reduce(thread_sum, HalfAddOp{});
      __syncthreads();

      // One thread per block commits the result
      if (tid == 0) {
        const int output_idx = i * C_b + j;
        // Add to existing C (C += result)
        if constexpr (std::is_same_v<Dtype, __half>) {
          C[output_idx] = __hadd(C[output_idx], block_sum);
        } else {
          C[output_idx] += block_sum;
        }
      }
    }
  }
}

// Main namespace for split-K implicit GEMM
namespace warpconvnet {
namespace split_k_implicit_gemm {

/**
 * @brief Run split-K implicit GEMM: C += transpose(A[indices_a]) @ B[indices_b]
 *
 * @param tensor_a: Pointer to matrix A (N x C_a).
 * @param tensor_b: Pointer to matrix B (N x C_b).
 * @param tensor_c: Pointer to matrix C (C_a x C_b), modified in-place.
 * @param indices_a: Row indices for matrix A.
 * @param indices_b: Row indices for matrix B.
 * @param N: Number of rows in A and B.
 * @param C_a: Number of columns in A.
 * @param C_b: Number of columns in B.
 * @param K: Number of indices (length of indices_a and indices_b).
 * @param split_k_factor: Number of splits for the K dimension.
 * @param threads: CUDA threads per block.
 *
 * @return Status code indicating success or failure.
 */
template <typename ElementA, typename ElementB, typename ElementC, typename Itype>
int run_split_k_implicit_gemm_templated(const void *tensor_a,
                                        const void *tensor_b,
                                        void *tensor_c,
                                        const Itype *indices_a,
                                        const Itype *indices_b,
                                        int N,
                                        int C_a,
                                        int C_b,
                                        int K,
                                        int split_k_factor = 4,
                                        int block_threads = 256) {
  // Convert void pointers
  auto a_ptr = reinterpret_cast<const ElementA *>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementB *>(tensor_b);
  auto c_ptr = reinterpret_cast<ElementC *>(tensor_c);

  // Validate dimensions
  if (C_a <= 0 || C_b <= 0 || K <= 0 || N <= 0) {
    return static_cast<int>(SplitKGemmStatus::kErrorInvalidDimensions);
  }

  // Get CUDA stream
  cudaStream_t stream = getCurrentCUDAStream();

  // Calculate optimal split factor
  const int chunk_size = (K + split_k_factor - 1) / split_k_factor;
  const int actual_splits = (K + chunk_size - 1) / chunk_size;

  // Allocate temporary memory for partial results if needed
  ElementC *c_partials = nullptr;
  bool needs_reduction = (actual_splits > 1);

  if (needs_reduction) {
    size_t partial_size = actual_splits * C_a * C_b * sizeof(ElementC);
    cudaMalloc(&c_partials, partial_size);
    cudaMemset(c_partials, 0, partial_size);
  }

  // Stage 1: Launch split-K kernels
  // Use 2D grid where each block handles one output element
  // Grid size should cover the output matrix dimensions
  const int max_grid_x = std::min(C_a, 65535);  // CUDA grid limit
  const int max_grid_y = std::min(C_b, 65535);  // CUDA grid limit
  dim3 grid_2d(max_grid_x, max_grid_y);

  // Use 1D thread blocks for CUB reduction
  dim3 threads(block_threads);

  for (int split = 0; split < actual_splits; ++split) {
    const int chunk_start = split * chunk_size;
    const int current_chunk_size = std::min(chunk_size, K - chunk_start);

    ElementC *output_ptr = needs_reduction ? (c_partials + split * C_a * C_b) : c_ptr;

    const bool use_atomic = !needs_reduction;  // Only use atomic if writing directly to output

    // Launch kernel with CUB reduction
    if (block_threads == 128) {
      if (use_atomic) {
        split_k_implicit_gemm_stage1<ElementA, Itype, 128, true>
            <<<grid_2d, threads, 0, stream>>>(a_ptr,
                                              b_ptr,
                                              output_ptr,
                                              indices_a,
                                              indices_b,
                                              C_a,
                                              C_b,
                                              chunk_start,
                                              current_chunk_size,
                                              split);
      } else {
        split_k_implicit_gemm_stage1<ElementA, Itype, 128, false>
            <<<grid_2d, threads, 0, stream>>>(a_ptr,
                                              b_ptr,
                                              output_ptr,
                                              indices_a,
                                              indices_b,
                                              C_a,
                                              C_b,
                                              chunk_start,
                                              current_chunk_size,
                                              split);
      }
    } else if (block_threads == 256) {
      if (use_atomic) {
        split_k_implicit_gemm_stage1<ElementA, Itype, 256, true>
            <<<grid_2d, threads, 0, stream>>>(a_ptr,
                                              b_ptr,
                                              output_ptr,
                                              indices_a,
                                              indices_b,
                                              C_a,
                                              C_b,
                                              chunk_start,
                                              current_chunk_size,
                                              split);
      } else {
        split_k_implicit_gemm_stage1<ElementA, Itype, 256, false>
            <<<grid_2d, threads, 0, stream>>>(a_ptr,
                                              b_ptr,
                                              output_ptr,
                                              indices_a,
                                              indices_b,
                                              C_a,
                                              C_b,
                                              chunk_start,
                                              current_chunk_size,
                                              split);
      }
    } else if (block_threads == 512) {
      if (use_atomic) {
        split_k_implicit_gemm_stage1<ElementA, Itype, 512, true>
            <<<grid_2d, threads, 0, stream>>>(a_ptr,
                                              b_ptr,
                                              output_ptr,
                                              indices_a,
                                              indices_b,
                                              C_a,
                                              C_b,
                                              chunk_start,
                                              current_chunk_size,
                                              split);
      } else {
        split_k_implicit_gemm_stage1<ElementA, Itype, 512, false>
            <<<grid_2d, threads, 0, stream>>>(a_ptr,
                                              b_ptr,
                                              output_ptr,
                                              indices_a,
                                              indices_b,
                                              C_a,
                                              C_b,
                                              chunk_start,
                                              current_chunk_size,
                                              split);
      }
    } else {
      return static_cast<int>(SplitKGemmStatus::kErrorInvalidDimensions);
    }
  }

  // Stage 2: Reduction if needed
  if (needs_reduction) {
    // Use 2D grid for reduction kernel as well
    if (block_threads == 128) {
      split_k_reduction_kernel<ElementC, 128>
          <<<grid_2d, threads, 0, stream>>>(c_ptr, c_partials, C_a, C_b, actual_splits);
    } else if (block_threads == 256) {
      split_k_reduction_kernel<ElementC, 256>
          <<<grid_2d, threads, 0, stream>>>(c_ptr, c_partials, C_a, C_b, actual_splits);
    } else if (block_threads == 512) {
      split_k_reduction_kernel<ElementC, 512>
          <<<grid_2d, threads, 0, stream>>>(c_ptr, c_partials, C_a, C_b, actual_splits);
    } else {
      return static_cast<int>(SplitKGemmStatus::kErrorInvalidDimensions);
    }
  }

  // Synchronize and check for errors
  cudaStreamSynchronize(stream);
  cudaError_t cuda_status = cudaGetLastError();

  // Cleanup
  if (c_partials) {
    cudaFree(c_partials);
  }

  if (cuda_status != cudaSuccess) {
    return static_cast<int>(SplitKGemmStatus::kErrorKernelExecution);
  }

  return static_cast<int>(SplitKGemmStatus::kSuccess);
}

}  // namespace split_k_implicit_gemm
}  // namespace warpconvnet

// Explicit template instantiations
template int
warpconvnet::split_k_implicit_gemm::run_split_k_implicit_gemm_templated<float, float, float, int>(
    const void *, const void *, void *, const int *, const int *, int, int, int, int, int, int);

template int warpconvnet::split_k_implicit_gemm::
    run_split_k_implicit_gemm_templated<__half, __half, __half, int>(
        const void *, const void *, void *, const int *, const int *, int, int, int, int, int, int);

template int warpconvnet::split_k_implicit_gemm::
    run_split_k_implicit_gemm_templated<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int>(
        const void *, const void *, void *, const int *, const int *, int, int, int, int, int, int);
