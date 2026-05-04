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

// Template body header for run_split_k_implicit_gemm_templated. Included by
// warpconvnet/csrc/implicit_gemm_split_k.cu (existing instantiations) and by
// warpgemm-generated offset_gemm TUs that invoke INSTANTIATE_IMPLICIT_GEMM_SPLIT_K
// for the stable tier.

#pragma once

#include <c10/cuda/CUDAStream.h>
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

// Define error codes for split-K implicit GEMM operations
enum class SplitKGemmStatus {
  kSuccess = 0,
  kErrorInvalidKernelType = 1,
  kErrorUnsupportedDataType = 2,
  kErrorKernelExecution = 3,
  kErrorInvalidDimensions = 4,
  kErrorInsufficientMemory = 5
};

template <typename Dtype, typename Itype, int BLOCK_THREADS, bool use_atomic = true>
__global__ void split_k_implicit_gemm_stage1(const Dtype *__restrict__ A,
                                             const Dtype *__restrict__ B,
                                             Dtype *__restrict__ C_partial,
                                             const Itype *__restrict__ indices_a,
                                             const Itype *__restrict__ indices_b,
                                             const int C_a,
                                             const int C_b,
                                             const int chunk_start,
                                             const int chunk_size,
                                             const int split_k_idx) {
  const int tid = threadIdx.x;

  using BlockReduce = cub::BlockReduce<Dtype, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < C_a; i += gridDim.x) {
    for (int j = blockIdx.y; j < C_b; j += gridDim.y) {
      Dtype thread_sum;
      if constexpr (std::is_same_v<Dtype, __half>) {
        thread_sum = __float2half(0.0f);
      } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
        thread_sum = __float2bfloat16(0.0f);
      } else {
        thread_sum = Dtype(0);
      }

      for (int k = chunk_start + tid; k < chunk_start + chunk_size; k += BLOCK_THREADS) {
        if (k < chunk_start + chunk_size) {
          const Itype ia = indices_a[k];
          const Itype ib = indices_b[k];

          if (ia >= 0 && ib >= 0) {
            const Dtype a_val = A[ia * C_a + i];
            const Dtype b_val = B[ib * C_b + j];

            if constexpr (std::is_same_v<Dtype, __half>) {
              thread_sum = __hadd(thread_sum, __hmul(a_val, b_val));
            } else {
              thread_sum += a_val * b_val;
            }
          }
        }
      }

      Dtype tile_sum = BlockReduce(temp_storage).Reduce(thread_sum, HalfAddOp{});
      __syncthreads();

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

template <typename Dtype, int BLOCK_THREADS>
__global__ void split_k_reduction_kernel(Dtype *__restrict__ C,
                                         const Dtype *__restrict__ C_partials,
                                         const int C_a,
                                         const int C_b,
                                         const int num_splits) {
  const int tid = threadIdx.x;

  using BlockReduce = cub::BlockReduce<Dtype, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < C_a; i += gridDim.x) {
    for (int j = blockIdx.y; j < C_b; j += gridDim.y) {
      Dtype thread_sum;
      if constexpr (std::is_same_v<Dtype, __half>) {
        thread_sum = __float2half(0.0f);
      } else if constexpr (std::is_same_v<Dtype, __nv_bfloat16>) {
        thread_sum = __float2bfloat16(0.0f);
      } else {
        thread_sum = Dtype(0);
      }

      for (int split = tid; split < num_splits; split += BLOCK_THREADS) {
        const int partial_idx = split * C_a * C_b + i * C_b + j;
        if constexpr (std::is_same_v<Dtype, __half>) {
          thread_sum = __hadd(thread_sum, C_partials[partial_idx]);
        } else {
          thread_sum += C_partials[partial_idx];
        }
      }

      Dtype block_sum = BlockReduce(temp_storage).Reduce(thread_sum, HalfAddOp{});
      __syncthreads();

      if (tid == 0) {
        const int output_idx = i * C_b + j;
        if constexpr (std::is_same_v<Dtype, __half>) {
          C[output_idx] = __hadd(C[output_idx], block_sum);
        } else {
          C[output_idx] += block_sum;
        }
      }
    }
  }
}

namespace warpconvnet {
namespace split_k_implicit_gemm {

/**
 * @brief Run split-K implicit GEMM: C += transpose(A[indices_a]) @ B[indices_b]
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
                                        int block_threads = 256,
                                        void *scratch = nullptr) {
  auto a_ptr = reinterpret_cast<const ElementA *>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementB *>(tensor_b);
  auto c_ptr = reinterpret_cast<ElementC *>(tensor_c);

  if (C_a <= 0 || C_b <= 0 || K <= 0 || N <= 0) {
    return static_cast<int>(SplitKGemmStatus::kErrorInvalidDimensions);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const int chunk_size = (K + split_k_factor - 1) / split_k_factor;
  const int actual_splits = (K + chunk_size - 1) / chunk_size;

  ElementC *c_partials = nullptr;
  bool needs_reduction = (actual_splits > 1);

  if (needs_reduction) {
    c_partials = reinterpret_cast<ElementC *>(scratch);
    size_t partial_size = actual_splits * C_a * C_b * sizeof(ElementC);
    cudaMemsetAsync(c_partials, 0, partial_size, stream);
  }

  const int max_grid_x = std::min(C_a, 65535);
  const int max_grid_y = std::min(C_b, 65535);
  dim3 grid_2d(max_grid_x, max_grid_y);

  dim3 threads(block_threads);

  for (int split = 0; split < actual_splits; ++split) {
    const int chunk_start = split * chunk_size;
    const int current_chunk_size = std::min(chunk_size, K - chunk_start);

    ElementC *output_ptr = needs_reduction ? (c_partials + split * C_a * C_b) : c_ptr;

    const bool use_atomic = !needs_reduction;

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

  if (needs_reduction) {
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

  cudaError_t cuda_status = cudaGetLastError();

  if (cuda_status != cudaSuccess) {
    return static_cast<int>(SplitKGemmStatus::kErrorKernelExecution);
  }

  return static_cast<int>(SplitKGemmStatus::kSuccess);
}

}  // namespace split_k_implicit_gemm
}  // namespace warpconvnet
