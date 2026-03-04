
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

#include <string>
#include <type_traits>

#include "include/vectorized_types.h"

// Define error codes for implicit GEMM operations
enum class ImplicitGemmStatus {
  kSuccess = 0,
  kErrorInvalidKernelType = 1,
  kErrorUnsupportedDataType = 2,
  kErrorKernelExecution = 3,
  kErrorInvalidDimensions = 4
};

/**
 * Matrix multiplication (CUDA Kernel) on the device: C += A * B
 * Generic template for scalar types - 1D grid version for all dimensions
 * wA is A's width and wB is B's width
 * use_atomic: whether to use atomicAdd for output accumulation
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE, bool use_atomic = true>
__global__ void implicit_gemm(const Dtype *__restrict__ A,
                              const int wA,
                              const int hA,  //
                              const Dtype *__restrict__ B,
                              const int wB,
                              const int hB,           //
                              Dtype *__restrict__ C,  //
                              const Itype *__restrict__ in_map,
                              const Itype *__restrict__ out_map,
                              const int indices_size,
                              const int grid_x) {
  // Use in_feat as A and kernel as B

  // Block index in 1D grid
  const int block_idx = blockIdx.x;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Convert 1D block index to 2D coordinates
  const int bx = block_idx % grid_x;
  const int by = block_idx / grid_x;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  // Check if this thread should process a valid row
  const bool valid_thread = (y < indices_size);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  Dtype Csub;
  if constexpr (std::is_same_v<Dtype, __half>) {
    Csub = __float2half(0.0f);
  } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
    Csub = __float2bfloat16(0.0f);
  } else {
    Csub = Dtype(0);
  }

  // Only access in_map and out_map for valid threads to avoid out-of-bounds access
  Itype in_row = 0;
  Itype out_row = 0;
  if (valid_thread) {
    in_row = in_map[y];
    out_row = out_map[y];
  }

  // Declaration of the shared memory arrays used to
  // store the sub-matrices of A and B
  __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    if constexpr (std::is_same_v<Dtype, __half>) {
      As[ty][tx] = (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : __float2half(0.0f);
      Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B[wB * (s + ty) + x] : __float2half(0.0f);
    } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
      As[ty][tx] =
          (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : __float2bfloat16(0.0f);
      Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B[wB * (s + ty) + x] : __float2bfloat16(0.0f);
    } else {
      As[ty][tx] =
          (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : static_cast<Dtype>(0);
      Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B[wB * (s + ty) + x] : static_cast<Dtype>(0);
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      if constexpr (std::is_same_v<Dtype, __half>) {
        Csub = __hadd(Csub, __hmul(As[ty][k], Bs[k][tx]));
      } else {
        Csub += As[ty][k] * Bs[k][tx];
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (valid_thread && x < wB) {
    if constexpr (use_atomic) {
      if constexpr (std::is_same_v<Dtype, __half>) {
        atomicAdd(&C[wB * out_row + x], Csub);
      } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
        atomicAdd(&C[wB * out_row + x], Csub);
      } else {
        atomicAdd(&C[wB * out_row + x], Csub);
      }
    } else {
      if constexpr (std::is_same_v<Dtype, __half>) {
        C[wB * out_row + x] = __hadd(C[wB * out_row + x], Csub);
      } else {
        C[wB * out_row + x] += Csub;
      }
    }
  }
}

/**
 * Grouped implicit GEMM: C[out_map[y]] += A[in_map[y]] * B[weight_idx[y]]
 *
 * Each pair selects its weight matrix from a stacked weight tensor via weight_idx.
 * This enables processing multiple kernel offsets in a single launch with zero padding.
 *
 * Optimization: Since concatenated maps are sorted by weight_idx, most blocks have a
 * uniform weight_idx. For uniform blocks, B is tiled in shared memory (fast path).
 * For mixed blocks at weight-group boundaries, B is loaded per-thread from L2 (slow path).
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void implicit_gemm_grouped(const Dtype *__restrict__ A,
                                      const int wA,
                                      const int hA,
                                      const Dtype *__restrict__ B,  // [num_weights, hB, wB]
                                      const int wB,
                                      const int hB,
                                      Dtype *__restrict__ C,
                                      const Itype *__restrict__ in_map,
                                      const Itype *__restrict__ out_map,
                                      const Itype *__restrict__ weight_idx,
                                      const int indices_size,
                                      const int grid_x) {
  const int block_idx = blockIdx.x;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = block_idx % grid_x;
  const int by = block_idx / grid_x;
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  const bool valid_thread = (y < indices_size);

  Dtype Csub;
  if constexpr (std::is_same_v<Dtype, __half>) {
    Csub = __float2half(0.0f);
  } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
    Csub = __float2bfloat16(0.0f);
  } else {
    Csub = Dtype(0);
  }

  Itype in_row = 0;
  Itype out_row = 0;
  int my_widx = 0;
  if (valid_thread) {
    in_row = in_map[y];
    out_row = out_map[y];
    my_widx = weight_idx[y];
  }

  // Check if this block has uniform weight_idx (sorted maps: check first and last valid row)
  __shared__ int widx_first;
  __shared__ bool uniform_flag;
  if (ty == 0 && tx == 0) {
    int y_first = BLOCK_SIZE * by;
    int y_end = min(BLOCK_SIZE * by + BLOCK_SIZE - 1, indices_size - 1);
    if (y_first < indices_size) {
      widx_first = weight_idx[y_first];
      uniform_flag = (weight_idx[y_first] == weight_idx[y_end]);
    } else {
      widx_first = 0;
      uniform_flag = true;
    }
  }
  __syncthreads();
  const bool uniform_block = uniform_flag;

  // Pointer to this block's/thread's weight matrix
  const Dtype *B_uniform = B + widx_first * hB * wB;
  const Dtype *B_local = B + my_widx * hB * wB;

  __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

  if (uniform_block) {
    // Fast path: all rows use same weight matrix → tile both A and B in shared memory
    for (int s = 0; s < wA; s += BLOCK_SIZE) {
      if constexpr (std::is_same_v<Dtype, __half>) {
        As[ty][tx] = (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : __float2half(0.0f);
        Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B_uniform[wB * (s + ty) + x] : __float2half(0.0f);
      } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
        As[ty][tx] =
            (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : __float2bfloat16(0.0f);
        Bs[ty][tx] =
            ((s + ty) < hB && x < wB) ? B_uniform[wB * (s + ty) + x] : __float2bfloat16(0.0f);
      } else {
        As[ty][tx] =
            (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : static_cast<Dtype>(0);
        Bs[ty][tx] =
            ((s + ty) < hB && x < wB) ? B_uniform[wB * (s + ty) + x] : static_cast<Dtype>(0);
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        if constexpr (std::is_same_v<Dtype, __half>) {
          Csub = __hadd(Csub, __hmul(As[ty][k], Bs[k][tx]));
        } else {
          Csub += As[ty][k] * Bs[k][tx];
        }
      }
      __syncthreads();
    }
  } else {
    // Slow path: mixed weight indices → tile only A, load B per-thread from L2
    for (int s = 0; s < wA; s += BLOCK_SIZE) {
      if constexpr (std::is_same_v<Dtype, __half>) {
        As[ty][tx] = (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : __float2half(0.0f);
      } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
        As[ty][tx] =
            (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : __float2bfloat16(0.0f);
      } else {
        As[ty][tx] =
            (valid_thread && (s + tx) < wA) ? A[wA * in_row + s + tx] : static_cast<Dtype>(0);
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        if ((s + k) < hB && x < wB) {
          Dtype b_val = B_local[wB * (s + k) + x];
          if constexpr (std::is_same_v<Dtype, __half>) {
            Csub = __hadd(Csub, __hmul(As[ty][k], b_val));
          } else {
            Csub += As[ty][k] * b_val;
          }
        }
      }
      __syncthreads();
    }
  }

  if (valid_thread && x < wB) {
    atomicAdd(&C[wB * out_row + x], Csub);
  }
}

// Main templated function implementation
namespace warpconvnet {
namespace implicit_gemm {

/**
 * @brief Run a vectorized implicit GEMM operation with templated types.
 *
 * @param tensor_a: Pointer to the A matrix (hA x wA).
 * @param tensor_b: Pointer to the B matrix (hB x wB).
 * @param tensor_c: Pointer to the C matrix (hA x wB).
 * @param in_map: Input row mapping (hA,).
 * @param out_map: Output row mapping (hA,).
 * @param wA: Width of matrix A.
 * @param hA: Height of matrix A.
 * @param wB: Width of matrix B.
 * @param hB: Height of matrix B.
 * @param kernel_type: Type of kernel to use ("basic" or "vectorized").
 * @param block_size: CUDA block size (default 16).
 *
 * @return Status code indicating the success or failure of the operation.
 *
 * Operation: C = A * B (with sparse mapping via in_map and out_map)
 */
template <typename ElementA, typename ElementB, typename ElementC, typename Itype>
int run_implicit_gemm_templated(const void *tensor_a,
                                const void *tensor_b,
                                void *tensor_c,
                                const Itype *in_map,
                                const Itype *out_map,
                                int wA,
                                int hA,
                                int wB,
                                int hB,
                                int indices_size,
                                const std::string &kernel_type,
                                int block_size = 16) {
  // Convert void pointers to appropriate types
  auto a_ptr = reinterpret_cast<const ElementA *>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementB *>(tensor_b);
  auto c_ptr = reinterpret_cast<ElementC *>(tensor_c);

  // Validate dimensions
  if (wA != hB) {
    return static_cast<int>(ImplicitGemmStatus::kErrorInvalidDimensions);
  }

  // Get the current PyTorch CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // Determine if atomic operations are needed
  // When hB < block_size, each thread computes exactly one partial product,
  // potentially eliminating the need for atomic operations
  const bool use_atomic = (hB >= block_size);

  // Launch kernel configuration
  if (kernel_type == "basic") {
    // Use basic scalar kernel with 1D grid for all cases
    dim3 threads(block_size, block_size);

    // Calculate grid dimensions
    int grid_x = (wB + block_size - 1) / block_size;
    int grid_y = (indices_size + block_size - 1) / block_size;

    // Use 1D grid for all cases
    int total_blocks = grid_x * grid_y;
    dim3 grid_1d(total_blocks);

    if (use_atomic) {
      if (block_size == 4) {
        ::implicit_gemm<ElementA, Itype, 4, true><<<grid_1d, threads, 0, stream>>>(
            a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, indices_size, grid_x);
      } else if (block_size == 16) {
        ::implicit_gemm<ElementA, Itype, 16, true><<<grid_1d, threads, 0, stream>>>(
            a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, indices_size, grid_x);
      } else if (block_size == 32) {
        ::implicit_gemm<ElementA, Itype, 32, true><<<grid_1d, threads, 0, stream>>>(
            a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, indices_size, grid_x);
      } else {
        return static_cast<int>(ImplicitGemmStatus::kErrorInvalidKernelType);
      }
    } else {
      if (block_size == 4) {
        ::implicit_gemm<ElementA, Itype, 4, false><<<grid_1d, threads, 0, stream>>>(
            a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, indices_size, grid_x);
      } else if (block_size == 16) {
        ::implicit_gemm<ElementA, Itype, 16, false><<<grid_1d, threads, 0, stream>>>(
            a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, indices_size, grid_x);
      } else if (block_size == 32) {
        ::implicit_gemm<ElementA, Itype, 32, false><<<grid_1d, threads, 0, stream>>>(
            a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, indices_size, grid_x);
      } else {
        return static_cast<int>(ImplicitGemmStatus::kErrorInvalidKernelType);
      }
    }
  } else {
    return static_cast<int>(ImplicitGemmStatus::kErrorInvalidKernelType);
  }

  // Non-blocking error check: peek at the last error without synchronizing.
  // This catches launch configuration errors without blocking the pipeline.
  // Runtime errors are caught by PyTorch at the next synchronization point.
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(ImplicitGemmStatus::kErrorKernelExecution);
  }

  return static_cast<int>(ImplicitGemmStatus::kSuccess);
}

/**
 * @brief Run a grouped implicit GEMM operation.
 *
 * Processes multiple kernel offsets in one launch. Each pair (in_map[i], out_map[i])
 * uses weight matrix B[weight_idx[i]] from the stacked weight tensor.
 *
 * @param tensor_a: Input features (hA x wA)
 * @param tensor_b: Stacked weights (num_weights x hB x wB), contiguous
 * @param tensor_c: Output features, accumulated with atomicAdd
 * @param in_map: Concatenated input indices
 * @param out_map: Concatenated output indices
 * @param weight_idx: Per-pair weight selection index
 * @param wA, hA, wB, hB: Matrix dimensions
 * @param indices_size: Total number of pairs
 * @param kernel_type: "basic"
 * @param block_size: CUDA block size (4, 16, or 32)
 */
template <typename ElementA, typename ElementB, typename ElementC, typename Itype>
int run_implicit_gemm_grouped_templated(const void *tensor_a,
                                        const void *tensor_b,
                                        void *tensor_c,
                                        const Itype *in_map,
                                        const Itype *out_map,
                                        const Itype *weight_idx,
                                        int wA,
                                        int hA,
                                        int wB,
                                        int hB,
                                        int indices_size,
                                        const std::string &kernel_type,
                                        int block_size = 16) {
  auto a_ptr = reinterpret_cast<const ElementA *>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementB *>(tensor_b);
  auto c_ptr = reinterpret_cast<ElementC *>(tensor_c);

  if (wA != hB) {
    return static_cast<int>(ImplicitGemmStatus::kErrorInvalidDimensions);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (kernel_type == "basic") {
    dim3 threads(block_size, block_size);
    int grid_x = (wB + block_size - 1) / block_size;
    int grid_y = (indices_size + block_size - 1) / block_size;
    int total_blocks = grid_x * grid_y;
    dim3 grid_1d(total_blocks);

    if (block_size == 4) {
      ::implicit_gemm_grouped<ElementA, Itype, 4><<<grid_1d, threads, 0, stream>>>(
          a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, weight_idx, indices_size, grid_x);
    } else if (block_size == 16) {
      ::implicit_gemm_grouped<ElementA, Itype, 16><<<grid_1d, threads, 0, stream>>>(
          a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, weight_idx, indices_size, grid_x);
    } else if (block_size == 32) {
      ::implicit_gemm_grouped<ElementA, Itype, 32><<<grid_1d, threads, 0, stream>>>(
          a_ptr, wA, hA, b_ptr, wB, hB, c_ptr, in_map, out_map, weight_idx, indices_size, grid_x);
    } else {
      return static_cast<int>(ImplicitGemmStatus::kErrorInvalidKernelType);
    }
  } else {
    return static_cast<int>(ImplicitGemmStatus::kErrorInvalidKernelType);
  }

  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(ImplicitGemmStatus::kErrorKernelExecution);
  }

  return static_cast<int>(ImplicitGemmStatus::kSuccess);
}

}  // namespace implicit_gemm
}  // namespace warpconvnet

// Use the namespace for convenience in the rest of the file
using namespace warpconvnet::implicit_gemm;

// Expose the template instantiations for use in pybind
template int warpconvnet::implicit_gemm::run_implicit_gemm_templated<float, float, float, int>(
    const void *,
    const void *,
    void *,
    const int *,
    const int *,
    int,
    int,
    int,
    int,
    int,
    const std::string &,
    int);

template int warpconvnet::implicit_gemm::run_implicit_gemm_templated<__half, __half, __half, int>(
    const void *,
    const void *,
    void *,
    const int *,
    const int *,
    int,
    int,
    int,
    int,
    int,
    const std::string &,
    int);

template int warpconvnet::implicit_gemm::
    run_implicit_gemm_templated<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int>(
        const void *,
        const void *,
        void *,
        const int *,
        const int *,
        int,
        int,
        int,
        int,
        int,
        const std::string &,
        int);

// Grouped kernel template instantiations
template int
warpconvnet::implicit_gemm::run_implicit_gemm_grouped_templated<float, float, float, int>(
    const void *,
    const void *,
    void *,
    const int *,
    const int *,
    const int *,
    int,
    int,
    int,
    int,
    int,
    const std::string &,
    int);

template int
warpconvnet::implicit_gemm::run_implicit_gemm_grouped_templated<__half, __half, __half, int>(
    const void *,
    const void *,
    void *,
    const int *,
    const int *,
    const int *,
    int,
    int,
    int,
    int,
    int,
    const std::string &,
    int);

template int warpconvnet::implicit_gemm::
    run_implicit_gemm_grouped_templated<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int>(
        const void *,
        const void *,
        void *,
        const int *,
        const int *,
        const int *,
        int,
        int,
        int,
        int,
        int,
        const std::string &,
        int);
