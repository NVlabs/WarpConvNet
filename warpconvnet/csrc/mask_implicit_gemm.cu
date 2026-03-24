// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

// Helper: convert any type to float safely
template <typename T>
__device__ __forceinline__ float to_float(T val) {
  return float(val);
}
template <>
__device__ __forceinline__ float to_float(__half val) {
  return __half2float(val);
}
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

// Helper: convert float to any type
template <typename T>
__device__ __forceinline__ T from_float(float val) {
  return static_cast<T>(val);
}
template <>
__device__ __forceinline__ __half from_float(float val) {
  return __float2half(val);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float val) {
  return __float2bfloat16(val);
}

// Helper: zero value
template <typename T>
__device__ __forceinline__ T zero_val() {
  return static_cast<T>(0);
}
template <>
__device__ __forceinline__ __half zero_val() {
  return __float2half(0.0f);
}
template <>
__device__ __forceinline__ __nv_bfloat16 zero_val() {
  return __float2bfloat16(0.0f);
}

/**
 * Forward: output[argsort[i]] = sum_k { input[pair_table[k,argsort[i]]] @ weight[k] }
 * Grid: (C_out_tiles, N_out_tiles)
 */
template <typename Dtype, int BLOCK_SIZE>
__global__ void mask_implicit_gemm_fwd_kernel(
    const Dtype *__restrict__ A,        // [N_in, C_in]
    const Dtype *__restrict__ B,        // [K, C_in, C_out]
    Dtype *__restrict__ C,              // [N_out, C_out]
    const int *__restrict__ pair_table, // [K * N_out]
    const uint32_t *__restrict__ pair_mask,
    const int *__restrict__ mask_argsort,
    const int N_in,
    const int N_out,
    const int C_in,
    const int C_out,
    const int K) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int out_col = blockIdx.x * BLOCK_SIZE + tx;
  const int sorted_row = blockIdx.y * BLOCK_SIZE + ty;

  // Look up real output row via argsort
  int real_row = 0;
  uint32_t my_mask = 0;
  if (sorted_row < N_out) {
    real_row = mask_argsort[sorted_row];
    if (real_row >= 0 && real_row < N_out) {
      my_mask = pair_mask[real_row];
    } else {
      real_row = 0;
    }
  }

  // Block-level mask union
  __shared__ uint32_t block_mask;
  if (tx == 0 && ty == 0) block_mask = 0;
  __syncthreads();
  if (sorted_row < N_out) atomicOr(&block_mask, my_mask);
  __syncthreads();
  const uint32_t active_offsets = block_mask;

  float accum = 0.0f;
  __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int k = 0; k < K; k++) {
    if (K <= 32 && !(active_offsets & (1u << k))) continue;

    int in_row = -1;
    bool thread_active = false;
    if (sorted_row < N_out && (K > 32 || (my_mask & (1u << k)))) {
      int pt = pair_table[k * N_out + real_row];
      if (pt >= 0 && pt < N_in) {
        in_row = pt;
        thread_active = true;
      }
    }

    const Dtype *Bk = B + k * C_in * C_out;

    for (int s = 0; s < C_in; s += BLOCK_SIZE) {
      As[ty][tx] = (thread_active && (s + tx) < C_in)
                       ? A[in_row * C_in + s + tx]
                       : zero_val<Dtype>();
      Bs[ty][tx] = ((s + ty) < C_in && out_col < C_out)
                       ? Bk[(s + ty) * C_out + out_col]
                       : zero_val<Dtype>();
      __syncthreads();

#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; i++) {
        accum += to_float(As[ty][i]) * to_float(Bs[i][tx]);
      }
      __syncthreads();
    }
  }

  if (sorted_row < N_out && out_col < C_out && real_row >= 0 && real_row < N_out) {
    C[real_row * C_out + out_col] = from_float<Dtype>(accum);
  }
}

/**
 * Backward dgrad: grad_input[in_row] += grad_output[real_row] @ weight[k]^T
 * Grid: (C_in_tiles, N_out_tiles)
 */
template <typename Dtype, int BLOCK_SIZE>
__global__ void mask_implicit_gemm_bwd_dgrad_kernel(
    const Dtype *__restrict__ grad_output,
    const Dtype *__restrict__ B,
    Dtype *__restrict__ grad_input,
    const int *__restrict__ pair_table,
    const uint32_t *__restrict__ pair_mask,
    const int *__restrict__ mask_argsort,
    const int N_in,
    const int N_out,
    const int C_in,
    const int C_out,
    const int K) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int in_col = blockIdx.x * BLOCK_SIZE + tx;
  const int sorted_row = blockIdx.y * BLOCK_SIZE + ty;

  int real_row = 0;
  uint32_t my_mask = 0;
  if (sorted_row < N_out) {
    real_row = mask_argsort[sorted_row];
    if (real_row >= 0 && real_row < N_out) {
      my_mask = pair_mask[real_row];
    } else {
      real_row = 0;
    }
  }

  __shared__ uint32_t block_mask;
  if (tx == 0 && ty == 0) block_mask = 0;
  __syncthreads();
  if (sorted_row < N_out) atomicOr(&block_mask, my_mask);
  __syncthreads();
  const uint32_t active_offsets = block_mask;

  __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int k = 0; k < K; k++) {
    if (K <= 32 && !(active_offsets & (1u << k))) continue;

    int in_row = -1;
    bool thread_active = false;
    if (sorted_row < N_out && (K > 32 || (my_mask & (1u << k)))) {
      int pt = pair_table[k * N_out + real_row];
      if (pt >= 0 && pt < N_in) {
        in_row = pt;
        thread_active = true;
      }
    }

    const Dtype *Bk = B + k * C_in * C_out;

    float accum = 0.0f;
    for (int s = 0; s < C_out; s += BLOCK_SIZE) {
      // As = grad_output[real_row, s:s+BS]
      As[ty][tx] = (thread_active && (s + tx) < C_out)
                       ? grad_output[real_row * C_out + s + tx]
                       : zero_val<Dtype>();
      // Bs = weight[k, in_col, s:s+BS] (transposed)
      Bs[ty][tx] = (in_col < C_in && (s + ty) < C_out)
                       ? Bk[in_col * C_out + s + ty]
                       : zero_val<Dtype>();
      __syncthreads();

#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; i++) {
        accum += to_float(As[ty][i]) * to_float(Bs[i][tx]);
      }
      __syncthreads();
    }

    // atomicAdd to grad_input — bounds-checked
    if (thread_active && in_row >= 0 && in_row < N_in && in_col < C_in) {
      atomicAdd(&grad_input[in_row * C_in + in_col], from_float<Dtype>(accum));
    }
  }
}

/**
 * Backward wgrad: grad_weight[k] = sum_i { input[pair[k,i]]^T @ grad_output[i] }
 * Grid: (C_out_tiles, C_in_tiles, K)
 */
template <typename Dtype, int BLOCK_SIZE>
__global__ void mask_implicit_gemm_bwd_wgrad_kernel(
    const Dtype *__restrict__ input,
    const Dtype *__restrict__ grad_output,
    Dtype *__restrict__ grad_weight,
    const int *__restrict__ pair_table,
    const uint32_t *__restrict__ pair_mask,
    const int N_in,
    const int N_out,
    const int C_in,
    const int C_out,
    const int K) {

  const int k = blockIdx.z;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int w_col = blockIdx.x * BLOCK_SIZE + tx;  // C_out
  const int w_row = blockIdx.y * BLOCK_SIZE + ty;  // C_in

  float accum = 0.0f;
  const int *pt_k = pair_table + k * N_out;

  for (int i = 0; i < N_out; i++) {
    bool active;
    if (K <= 32) {
      active = (pair_mask[i] & (1u << k)) != 0;
    } else {
      active = (pt_k[i] >= 0);
    }
    if (!active) continue;

    int in_row = pt_k[i];
    // Bounds check on in_row
    if (in_row < 0 || in_row >= N_in) continue;

    if (w_row < C_in && w_col < C_out) {
      accum += to_float(input[in_row * C_in + w_row]) *
               to_float(grad_output[i * C_out + w_col]);
    }
  }

  if (w_row < C_in && w_col < C_out) {
    grad_weight[k * C_in * C_out + w_row * C_out + w_col] = from_float<Dtype>(accum);
  }
}

// ============================================================================
// Launch wrappers
// ============================================================================

namespace warpconvnet {
namespace mask_implicit_gemm {

template <typename ElementA, typename ElementC>
int run_mask_implicit_gemm_fwd(
    const void *input, const void *weight, void *output,
    const int *pair_table, const uint32_t *pair_mask, const int *mask_argsort,
    int N_in, int N_out, int C_in, int C_out, int K, int block_size) {

  auto a = reinterpret_cast<const ElementA *>(input);
  auto b = reinterpret_cast<const ElementA *>(weight);
  auto c = reinterpret_cast<ElementC *>(output);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 threads(block_size, block_size);
  dim3 grid((C_out + block_size - 1) / block_size,
            (N_out + block_size - 1) / block_size);

  if (block_size == 16)
    mask_implicit_gemm_fwd_kernel<ElementA, 16><<<grid, threads, 0, stream>>>(
        a, b, c, pair_table, pair_mask, mask_argsort, N_in, N_out, C_in, C_out, K);
  else if (block_size == 32)
    mask_implicit_gemm_fwd_kernel<ElementA, 32><<<grid, threads, 0, stream>>>(
        a, b, c, pair_table, pair_mask, mask_argsort, N_in, N_out, C_in, C_out, K);
  else
    return 1;

  return (cudaGetLastError() != cudaSuccess) ? 3 : 0;
}

template <typename ElementA, typename ElementC>
int run_mask_implicit_gemm_bwd_dgrad(
    const void *grad_output, const void *weight, void *grad_input,
    const int *pair_table, const uint32_t *pair_mask, const int *mask_argsort,
    int N_in, int N_out, int C_in, int C_out, int K, int block_size) {

  auto go = reinterpret_cast<const ElementA *>(grad_output);
  auto b = reinterpret_cast<const ElementA *>(weight);
  auto gi = reinterpret_cast<ElementC *>(grad_input);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 threads(block_size, block_size);
  dim3 grid((C_in + block_size - 1) / block_size,
            (N_out + block_size - 1) / block_size);

  if (block_size == 16)
    mask_implicit_gemm_bwd_dgrad_kernel<ElementA, 16><<<grid, threads, 0, stream>>>(
        go, b, gi, pair_table, pair_mask, mask_argsort, N_in, N_out, C_in, C_out, K);
  else if (block_size == 32)
    mask_implicit_gemm_bwd_dgrad_kernel<ElementA, 32><<<grid, threads, 0, stream>>>(
        go, b, gi, pair_table, pair_mask, mask_argsort, N_in, N_out, C_in, C_out, K);
  else
    return 1;

  return (cudaGetLastError() != cudaSuccess) ? 3 : 0;
}

template <typename ElementA, typename ElementC>
int run_mask_implicit_gemm_bwd_wgrad(
    const void *input, const void *grad_output, void *grad_weight,
    const int *pair_table, const uint32_t *pair_mask,
    int N_in, int N_out, int C_in, int C_out, int K, int block_size) {

  auto inp = reinterpret_cast<const ElementA *>(input);
  auto go = reinterpret_cast<const ElementA *>(grad_output);
  auto gw = reinterpret_cast<ElementC *>(grad_weight);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 threads(block_size, block_size);
  dim3 grid((C_out + block_size - 1) / block_size,
            (C_in + block_size - 1) / block_size, K);

  if (block_size == 16)
    mask_implicit_gemm_bwd_wgrad_kernel<ElementA, 16><<<grid, threads, 0, stream>>>(
        inp, go, gw, pair_table, pair_mask, N_in, N_out, C_in, C_out, K);
  else if (block_size == 32)
    mask_implicit_gemm_bwd_wgrad_kernel<ElementA, 32><<<grid, threads, 0, stream>>>(
        inp, go, gw, pair_table, pair_mask, N_in, N_out, C_in, C_out, K);
  else
    return 1;

  return (cudaGetLastError() != cudaSuccess) ? 3 : 0;
}

// Explicit instantiations
#define INSTANTIATE_MASK_GEMM(T) \
  template int run_mask_implicit_gemm_fwd<T, T>( \
      const void*, const void*, void*, const int*, const uint32_t*, const int*, \
      int, int, int, int, int, int); \
  template int run_mask_implicit_gemm_bwd_dgrad<T, T>( \
      const void*, const void*, void*, const int*, const uint32_t*, const int*, \
      int, int, int, int, int, int); \
  template int run_mask_implicit_gemm_bwd_wgrad<T, T>( \
      const void*, const void*, void*, const int*, const uint32_t*, \
      int, int, int, int, int, int);

INSTANTIATE_MASK_GEMM(__half)
INSTANTIATE_MASK_GEMM(float)
INSTANTIATE_MASK_GEMM(__nv_bfloat16)
#undef INSTANTIATE_MASK_GEMM

}  // namespace mask_implicit_gemm
}  // namespace warpconvnet
