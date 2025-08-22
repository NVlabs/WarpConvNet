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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <type_traits>

using namespace nvcuda;

// -------------------- Stream helper --------------------
__host__ __forceinline__ cudaStream_t getCurrentCUDAStream() { return cudaStreamDefault; }

// -------------------- Type traits & converters --------------------
template <typename T>
struct IsInputSupported : std::false_type {};
template <>
struct IsInputSupported<half> : std::true_type {};
template <>
struct IsInputSupported<nv_bfloat16> : std::true_type {};

template <typename T>
struct IsOutputSupported : std::false_type {};
template <>
struct IsOutputSupported<float> : std::true_type {};
template <>
struct IsOutputSupported<half> : std::true_type {};
template <>
struct IsOutputSupported<nv_bfloat16> : std::true_type {};

template <typename T>
struct WmmaInputTag;
template <>
struct WmmaInputTag<half> {
  using type = half;
};
template <>
struct WmmaInputTag<nv_bfloat16> {
  using type = nv_bfloat16;
};

__device__ __forceinline__ float to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float *) { return x; }
__device__ __forceinline__ half from_float(float x, half *) { return __float2half_rn(x); }
__device__ __forceinline__ nv_bfloat16 from_float(float x, nv_bfloat16 *) {
  return __float2bfloat16(x);
}

// -------------------- Ampere cp.async helpers (16B) --------------------
static __device__ __forceinline__ void cp_async_16B(void *smem_ptr,
                                                    const void *gmem_ptr,
                                                    bool pred) {
#if __CUDA_ARCH__ >= 800
  if (pred) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr), "n"(16));
  } else {
    uint4 z = {0, 0, 0, 0};
    *reinterpret_cast<uint4 *>(smem_ptr) = z;
  }
#else
  if (pred) {
    uint4 v = *reinterpret_cast<const uint4 *>(gmem_ptr);
    *reinterpret_cast<uint4 *>(smem_ptr) = v;
  } else {
    uint4 z = {0, 0, 0, 0};
    *reinterpret_cast<uint4 *>(smem_ptr) = z;
  }
#endif
}
static __device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}
static __device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n");
#endif
}

// -------------------- Stage 1: WMMA split-K (produces float partials) --------------------
template <typename InT, typename Itype>
__global__ void wmma_splitk_stage1(
    const InT *__restrict__ A,        // [N, C_a] row-major
    const InT *__restrict__ B,        // [N, C_b] row-major
    float *__restrict__ C_partial,    // [C_a, C_b] row-major (per split)
    const Itype *__restrict__ idx_a,  // [chunk_size]
    const Itype *__restrict__ idx_b,  // [chunk_size]
    int N,                            // rows in A,B
    int C_a,                          // cols in A
    int C_b,                          // cols in B
    int chunk_start,                  // start into idx arrays
    int chunk_size) {                 // size of this chunk (K-splits)
  static_assert(IsInputSupported<InT>::value, "InT must be half or nv_bfloat16");

  if (blockDim.x != 32) return;  // one warp per block

  constexpr int TILE_M = 16, TILE_N = 16, TILE_K = 16;

  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x;  // along C_b
  const int tile_m = blockIdx.y;  // along C_a
  const int i_base = tile_m * TILE_M;
  const int j_base = tile_n * TILE_N;

  // Double-buffered smem tiles (each 16x16 of InT)
  __shared__ __align__(16) InT Asmem[2][TILE_K * TILE_M];  // stored as [K, M] row-major
  __shared__ __align__(16) InT Bsmem[2][TILE_K * TILE_N];  // stored as [K, N] row-major
  __shared__ float OutTile[TILE_M * TILE_N];               // f32 accumulation dump buffer

  using WmmaT = typename WmmaInputTag<InT>::type;
  wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, WmmaT, wmma::col_major>
      a_frag;  // Asmem seen as [M,K] col-major
  wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, WmmaT, wmma::row_major>
      b_frag;  // Bsmem seen as [K,N] row-major
  wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  auto stage_load_tiles = [&](int k0, int stage) {
    const int row = lane & 15;  // 0..15 along K-tile
    const int seg = lane >> 4;  // 0..1, 8 elements per 16B

    const int kc = k0 + row;
    bool k_in = (kc < chunk_size);

    // A: gather rows idx_a[kc], contiguous across i (columns)
    long arow = (k_in ? static_cast<long>(idx_a[chunk_start + kc]) : -1);
    const InT *gA = (arow >= 0 && arow < N) ? (A + (size_t)arow * C_a + i_base + seg * 8) : nullptr;
    bool predA = (arow >= 0 && arow < N) && (i_base + seg * 8 + 7 < C_a);
    InT *sA = &Asmem[stage][row * TILE_M + seg * 8];  // [K,M]
    cp_async_16B((void *)sA, (const void *)gA, predA);
    // Fallback for partial segments: scalar masked loads when columns remain but < 8
    if (!predA) {
      const int col_start = i_base + seg * 8;
      const int remaining = C_a - col_start;
      if (arow >= 0 && arow < N && remaining > 0) {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          sA[t] = (t < remaining) ? gA[t] : from_float(0.0f, (InT *)nullptr);
        }
      }
    }

    // B: gather rows idx_b[kc], contiguous across j (columns)
    long brow = (k_in ? static_cast<long>(idx_b[chunk_start + kc]) : -1);
    const InT *gB = (brow >= 0 && brow < N) ? (B + (size_t)brow * C_b + j_base + seg * 8) : nullptr;
    bool predB = (brow >= 0 && brow < N) && (j_base + seg * 8 + 7 < C_b);
    InT *sB = &Bsmem[stage][row * TILE_N + seg * 8];  // [K,N]
    cp_async_16B((void *)sB, (const void *)gB, predB);
    // Fallback for partial segments on B
    if (!predB) {
      const int col_start = j_base + seg * 8;
      const int remaining = C_b - col_start;
      if (brow >= 0 && brow < N && remaining > 0) {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          sB[t] = (t < remaining) ? gB[t] : from_float(0.0f, (InT *)nullptr);
        }
      }
    }
  };

  // Preload first K-slice
  int stage = 0;
  stage_load_tiles(/*k0=*/0, /*stage=*/stage);
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  // K loop (in TILE_K steps)
  for (int k0 = 0; k0 < chunk_size; k0 += TILE_K) {
    const int next_stage = stage ^ 1;

    if (k0 + TILE_K < chunk_size) {
      stage_load_tiles(k0 + TILE_K, next_stage);
      cp_async_commit();
    }

    // Compute for current stage
    wmma::load_matrix_sync(a_frag, &Asmem[stage][0], /*ldm (col-major M)*/ TILE_K);
    wmma::load_matrix_sync(b_frag, &Bsmem[stage][0], /*ldm (row-major N)*/ TILE_N);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    if (k0 + TILE_K < chunk_size) cp_async_wait_all();
    __syncthreads();
    stage = next_stage;
  }

  // Dump accumulator to smem (row-major MxN)
  wmma::store_matrix_sync(OutTile, acc_frag, TILE_N, wmma::mem_row_major);
  __syncthreads();

  // Store to C_partial without atomics (one split writes its own buffer)
  // 2x loops over 32 lanes => 64 chunks of 4 floats -> 256 elements
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int chunk_id = lane + i * 32;      // 0..63
    int r = (chunk_id * 4) / TILE_N;   // row within tile
    int c4 = (chunk_id * 4) % TILE_N;  // col start (multiple of 4)
    int m = i_base + r;
    int n = j_base + c4;

    if (m < C_a) {
      float v0 = OutTile[r * TILE_N + c4 + 0];
      float v1 = OutTile[r * TILE_N + c4 + 1];
      float v2 = OutTile[r * TILE_N + c4 + 2];
      float v3 = OutTile[r * TILE_N + c4 + 3];
      float *__restrict__ Dst = &C_partial[(size_t)m * C_b + n];
      const bool full = (n + 3 < C_b);
      const bool aligned16 = ((reinterpret_cast<uintptr_t>(Dst) & 0xF) == 0);
      if (full && aligned16) {
        float4 v = make_float4(v0, v1, v2, v3);
        *reinterpret_cast<float4 *>(Dst) = v;
      } else {
        if (n + 0 < C_b) Dst[0] = v0;
        if (n + 1 < C_b) Dst[1] = v1;
        if (n + 2 < C_b) Dst[2] = v2;
        if (n + 3 < C_b) Dst[3] = v3;
      }
    }
  }
}

// -------------------- Stage 2: CUB reduction across splits (float partials -> ElementC)
// --------------------
template <typename ElementC, int BLOCK_THREADS>
__global__ void splitk_reduce_partials(ElementC *__restrict__ C,            // [C_a, C_b]
                                       const float *__restrict__ Partials,  // [S, C_a, C_b]
                                       int C_a,
                                       int C_b,
                                       int num_splits) {
  const int tid = threadIdx.x;

  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.y; i < C_a; i += gridDim.y) {
    for (int j = blockIdx.x; j < C_b; j += gridDim.x) {
      float thread_sum = 0.0f;
      for (int split = tid; split < num_splits; split += BLOCK_THREADS) {
        const size_t idx = (size_t)split * C_a * C_b + (size_t)i * C_b + j;
        thread_sum += Partials[idx];
      }

      float block_sum = BlockReduce(temp_storage).Reduce(thread_sum, cub::Sum());
      __syncthreads();

      if (tid == 0) {
        const size_t out_idx = (size_t)i * C_b + j;
        if constexpr (std::is_same<ElementC, float>::value) {
          C[out_idx] += block_sum;
        } else if constexpr (std::is_same<ElementC, half>::value) {
          C[out_idx] = __hadd(C[out_idx], from_float(block_sum, (half *)nullptr));
        } else if constexpr (std::is_same<ElementC, nv_bfloat16>::value) {
          C[out_idx] = from_float(__bfloat162float(C[out_idx]) + block_sum, (nv_bfloat16 *)nullptr);
        } else {
          // Fallback: convert via float add
          C[out_idx] = static_cast<ElementC>(static_cast<float>(C[out_idx]) + block_sum);
        }
      }
      __syncthreads();
    }
  }
}

// -------------------- Host-side runner --------------------
namespace warpconvnet {
namespace wmma_split_k_sm80 {

template <typename ElementIn, typename ElementOut, typename Itype>
int run_split_k_wmma_templated(const void *tensor_a,    // [N, C_a]
                               const void *tensor_b,    // [N, C_b]
                               void *tensor_c,          // [C_a, C_b]
                               const Itype *indices_a,  // [K]
                               const Itype *indices_b,  // [K]
                               int N,
                               int C_a,
                               int C_b,
                               int K,
                               int split_k_factor = 4) {
  static_assert(IsInputSupported<ElementIn>::value, "ElementIn must be half or nv_bfloat16");
  static_assert(IsOutputSupported<ElementOut>::value,
                "ElementOut must be float, half, or nv_bfloat16");

  auto a_ptr = reinterpret_cast<const ElementIn *>(tensor_a);
  auto b_ptr = reinterpret_cast<const ElementIn *>(tensor_b);
  auto c_ptr = reinterpret_cast<ElementOut *>(tensor_c);

  if (C_a <= 0 || C_b <= 0 || K <= 0 || N <= 0) {
    return 4;  // invalid dims
  }

  cudaStream_t stream = getCurrentCUDAStream();

  const int chunk_size = (K + split_k_factor - 1) / split_k_factor;
  const int actual_splits = (K + chunk_size - 1) / chunk_size;

  // Allocate float partials [S, C_a, C_b]
  float *c_partials = nullptr;
  {
    size_t partial_size_bytes = (size_t)actual_splits * C_a * C_b * sizeof(float);
    cudaError_t err = cudaMalloc(&c_partials, partial_size_bytes);
    if (err != cudaSuccess) return 5;  // insufficient memory
  }

  // Grid / block config
  dim3 block(32, 1, 1);
  dim3 grid((C_b + 16 - 1) / 16, (C_a + 16 - 1) / 16, 1);

  // Stage 1: launch per split
  for (int split = 0; split < actual_splits; ++split) {
    const int cs = split * chunk_size;
    const int cur_sz = std::min(chunk_size, K - cs);
    float *out_ptr = c_partials + (size_t)split * C_a * C_b;

    // Each split writes its own buffer (no atomics)
    wmma_splitk_stage1<ElementIn, Itype><<<grid, block, 0, stream>>>(
        a_ptr, b_ptr, out_ptr, indices_a, indices_b, N, C_a, C_b, cs, cur_sz);
  }

  // Stage 2: reduction over splits -> add into C
  {
    dim3 red_grid((C_b + 16 - 1) / 16, (C_a + 16 - 1) / 16, 1);
    constexpr int RED_THREADS = 256;
    splitk_reduce_partials<ElementOut, RED_THREADS>
        <<<red_grid, RED_THREADS, 0, stream>>>(c_ptr, c_partials, C_a, C_b, actual_splits);
  }

  cudaStreamSynchronize(stream);
  cudaError_t cuda_status = cudaGetLastError();

  if (c_partials) cudaFree(c_partials);

  if (cuda_status != cudaSuccess) {
    return 3;  // kernel execution error
  }

  return 0;
}

}  // namespace wmma_split_k_sm80
}  // namespace warpconvnet

// -------------------- Explicit instantiations --------------------
namespace warpconvnet {
namespace wmma_split_k_sm80 {
template int run_split_k_wmma_templated<half, float, int>(
    const void *, const void *, void *, const int *, const int *, int, int, int, int, int);
template int run_split_k_wmma_templated<half, half, int>(
    const void *, const void *, void *, const int *, const int *, int, int, int, int, int);
template int run_split_k_wmma_templated<nv_bfloat16, float, int>(
    const void *, const void *, void *, const int *, const int *, int, int, int, int, int);
template int run_split_k_wmma_templated<nv_bfloat16, nv_bfloat16, int>(
    const void *, const void *, void *, const int *, const int *, int, int, int, int, int);
}  // namespace wmma_split_k_sm80
}  // namespace warpconvnet
