// Copyright 2025 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
//
// Segmented/grouped implicit GEMM with per-segment B matrices and per-row gates.
// SM80 WMMA kernel, one warp per block. See README_grouped_implicit_gemm_moe.md

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <type_traits>
using namespace nvcuda;

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

// Map a global tile index t (over all segments) to (segment e, tile_in_e)
__device__ __forceinline__ void map_tile_to_segment(int t,
                                                    const int *seg_offsets,
                                                    int E,
                                                    int &e_out,
                                                    int &tile_in_e_out,
                                                    int &seg_start_out,
                                                    int &seg_end_out) {
  constexpr int TILE_M = 16;
  int t_remaining = t;
  for (int e = 0; e < E; ++e) {
    int seg_start = seg_offsets[e];
    int seg_end = seg_offsets[e + 1];
    int len = seg_end - seg_start;
    int tiles_e = (len + TILE_M - 1) / TILE_M;
    if (t_remaining < tiles_e) {
      e_out = e;
      tile_in_e_out = t_remaining;
      seg_start_out = seg_start;
      seg_end_out = seg_end;
      return;
    }
    t_remaining -= tiles_e;
  }
  e_out = -1;
  tile_in_e_out = 0;
  seg_start_out = 0;
  seg_end_out = 0;
}

template <typename InT, typename OutT, bool UseAtomic = true>
__global__ void segmented_implicit_gemm_wmma(
    const InT *__restrict__ A,            // [M,K], row-major
    const InT *__restrict__ B_multi,      // [E,K,N], row-major per segment
    const InT *__restrict__ Cbias,        // [M,N] optional (may be nullptr)
    const long *__restrict__ a_inds,      // [P]
    const long *__restrict__ d_inds,      // [P]
    const float *__restrict__ gates,      // [P] per-row gates
    const int *__restrict__ seg_offsets,  // [E+1] exclusive-scan
    OutT *__restrict__ D,                 // [Q,N], row-major
    int M,
    int K,
    int N,
    int P,
    int Q,
    int E,
    float alpha,
    float beta) {
  static_assert(IsInputSupported<InT>::value, "InT must be half or nv_bfloat16");
  static_assert(IsOutputSupported<OutT>::value, "OutT must be float, half, or nv_bfloat16");

  if (blockDim.x != 32) return;  // one warp per block

  constexpr int TILE_M = 16, TILE_N = 16, TILE_K = 16;

  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x;  // 16-wide tile index along N
  const int n_base = tile_n * TILE_N;

  __shared__ long a_row_idx[TILE_M];
  __shared__ long d_row_idx[TILE_M];
  __shared__ float row_gate[TILE_M];
  __shared__ __align__(16) InT Asmem[2][TILE_M * TILE_K];
  __shared__ __align__(16) InT Bsmem[2][TILE_K * TILE_N];
  __shared__ float OutTile[TILE_M * TILE_N];

  using WmmaT = typename WmmaInputTag<InT>::type;
  wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, WmmaT, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, WmmaT, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_frag;

  for (int t = blockIdx.y;;) {
    int e, tile_in_e, seg_start, seg_end;
    map_tile_to_segment(t, seg_offsets, E, e, tile_in_e, seg_start, seg_end);
    if (e < 0) break;
    const int p0 = seg_start + tile_in_e * TILE_M;

    if (lane < TILE_M) {
      const int p = p0 + lane;
      const bool in_seg = (p < seg_end);
      a_row_idx[lane] = in_seg && (p < P) ? a_inds[p] : -1;
      d_row_idx[lane] = in_seg && (p < P) ? d_inds[p] : -1;
      row_gate[lane] = in_seg && (p < P) ? gates[p] : 0.0f;
    }
    __syncthreads();

    const InT *B = B_multi + (size_t)e * (size_t)K * (size_t)N;
    wmma::fill_fragment(acc_frag, 0.0f);

    auto stage_load_tiles = [&](int k0, int stage) {
      const int row = lane & 15;  // 0..15
      const int seg = lane >> 4;  // 0..1 (8 elements => 16B per seg)

      long arow = a_row_idx[row];
      const InT *gA = (arow >= 0 && arow < M) ? (A + (size_t)arow * K + k0 + seg * 8) : nullptr;
      bool predA = (arow >= 0 && arow < M) && (k0 + seg * 8 + 7 < K);
      InT *sA = &Asmem[stage][row * TILE_K + seg * 8];
      cp_async_16B((void *)sA, (const void *)gA, predA);

      const int kb = k0 + row;
      const int nb = n_base + seg * 8;
      const InT *gB = (kb < K && nb < N) ? (B + (size_t)kb * N + nb) : nullptr;
      bool predB = (kb < K) && (nb + 7 < N);
      InT *sB = &Bsmem[stage][row * TILE_N + seg * 8];
      cp_async_16B((void *)sB, (const void *)gB, predB);
    };

    int stage = 0;
    stage_load_tiles(0, stage);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
      const int next_stage = stage ^ 1;
      if (k0 + TILE_K < K) {
        stage_load_tiles(k0 + TILE_K, next_stage);
        cp_async_commit();
      }
      wmma::load_matrix_sync(a_frag, &Asmem[stage][0], TILE_K);
      wmma::load_matrix_sync(b_frag, &Bsmem[stage][0], TILE_N);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      if (k0 + TILE_K < K) cp_async_wait_all();
      __syncthreads();
      stage = next_stage;
    }

    wmma::store_matrix_sync(OutTile, acc_frag, TILE_N, wmma::mem_row_major);
    __syncthreads();

    if constexpr (std::is_same<OutT, float>::value) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int chunk_id = lane + i * 32;      // 0..63
        int r = (chunk_id * 4) / TILE_N;   // row
        int c4 = (chunk_id * 4) % TILE_N;  // col start
        int n = n_base + c4;
        int p = p0 + r;

        long arow = a_row_idx[r];
        long drow = d_row_idx[r];
        if (!(p < P && arow >= 0 && arow < M && drow >= 0 && drow < Q)) continue;
        const float g = row_gate[r];

        const float f0 = OutTile[r * TILE_N + c4 + 0] * alpha * g;
        const float f1 = OutTile[r * TILE_N + c4 + 1] * alpha * g;
        const float f2 = OutTile[r * TILE_N + c4 + 2] * alpha * g;
        const float f3 = OutTile[r * TILE_N + c4 + 3] * alpha * g;

        if constexpr (UseAtomic) {
          float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
          if (Cbias && beta != 0.0f) {
            if (n + 3 < N) {
              b0 = beta * to_float(Cbias[(size_t)arow * N + (n + 0)]);
              b1 = beta * to_float(Cbias[(size_t)arow * N + (n + 1)]);
              b2 = beta * to_float(Cbias[(size_t)arow * N + (n + 2)]);
              b3 = beta * to_float(Cbias[(size_t)arow * N + (n + 3)]);
            } else {
              if (n + 0 < N) b0 = beta * to_float(Cbias[(size_t)arow * N + (n + 0)]);
              if (n + 1 < N) b1 = beta * to_float(Cbias[(size_t)arow * N + (n + 1)]);
              if (n + 2 < N) b2 = beta * to_float(Cbias[(size_t)arow * N + (n + 2)]);
              if (n + 3 < N) b3 = beta * to_float(Cbias[(size_t)arow * N + (n + 3)]);
            }
          }
          float *__restrict__ Drow = &D[(size_t)drow * N + n];
          if (n + 3 < N) {
            atomicAdd(&Drow[0], f0 + b0);
            atomicAdd(&Drow[1], f1 + b1);
            atomicAdd(&Drow[2], f2 + b2);
            atomicAdd(&Drow[3], f3 + b3);
          } else {
            if (n + 0 < N) atomicAdd(&Drow[0], f0 + b0);
            if (n + 1 < N) atomicAdd(&Drow[1], f1 + b1);
            if (n + 2 < N) atomicAdd(&Drow[2], f2 + b2);
            if (n + 3 < N) atomicAdd(&Drow[3], f3 + b3);
          }
        } else {
          float v0 = f0, v1 = f1, v2 = f2, v3 = f3;
          if (Cbias && beta != 0.0f) {
            if (n + 3 < N) {
              v0 += beta * to_float(Cbias[(size_t)arow * N + (n + 0)]);
              v1 += beta * to_float(Cbias[(size_t)arow * N + (n + 1)]);
              v2 += beta * to_float(Cbias[(size_t)arow * N + (n + 2)]);
              v3 += beta * to_float(Cbias[(size_t)arow * N + (n + 3)]);
            } else {
              if (n + 0 < N) v0 += beta * to_float(Cbias[(size_t)arow * N + (n + 0)]);
              if (n + 1 < N) v1 += beta * to_float(Cbias[(size_t)arow * N + (n + 1)]);
              if (n + 2 < N) v2 += beta * to_float(Cbias[(size_t)arow * N + (n + 2)]);
              if (n + 3 < N) v3 += beta * to_float(Cbias[(size_t)arow * N + (n + 3)]);
            }
          }
          float *__restrict__ Dst = &D[(size_t)drow * N + n];
          const bool aligned16 = ((reinterpret_cast<uintptr_t>(Dst) & 0xF) == 0);
          if ((n + 3 < N) && aligned16) {
            float4 v = make_float4(v0, v1, v2, v3);
            *reinterpret_cast<float4 *>(Dst) = v;
          } else {
            if (n + 0 < N) D[(size_t)drow * N + (n + 0)] = v0;
            if (n + 1 < N) D[(size_t)drow * N + (n + 1)] = v1;
            if (n + 2 < N) D[(size_t)drow * N + (n + 2)] = v2;
            if (n + 3 < N) D[(size_t)drow * N + (n + 3)] = v3;
          }
        }
      }
    } else {
      int chunk_id = lane;               // 0..31
      int r = (chunk_id * 8) / TILE_N;   // row
      int c8 = (chunk_id * 8) % TILE_N;  // col start (0 or 8)
      int n = n_base + c8;
      int p = p0 + r;

      long arow = a_row_idx[r];
      long drow = d_row_idx[r];

      if (p < P && arow >= 0 && arow < M && drow >= 0 && drow < Q) {
        const float g = row_gate[r];
        if (n + 7 < N) {
          if constexpr (UseAtomic) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              float val = OutTile[r * TILE_N + c8 + j] * alpha * g;
              if (Cbias && beta != 0.0f) val += beta * to_float(Cbias[(size_t)arow * N + (n + j)]);
              atomicAdd(&D[(size_t)drow * N + (n + j)], from_float(val, (OutT *)nullptr));
            }
          } else {
            alignas(16) OutT tmp8[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              float val = OutTile[r * TILE_N + c8 + j] * alpha * g;
              if (Cbias && beta != 0.0f) val += beta * to_float(Cbias[(size_t)arow * N + (n + j)]);
              tmp8[j] = from_float(val, (OutT *)nullptr);
            }
            OutT *__restrict__ Dst = &D[(size_t)drow * N + n];
            const bool aligned16 = ((reinterpret_cast<uintptr_t>(Dst) & 0xF) == 0);
            if (aligned16) {
              *reinterpret_cast<uint4 *>(Dst) = *reinterpret_cast<const uint4 *>(tmp8);
            } else {
              for (int j = 0; j < 8; ++j) {
                Dst[j] = tmp8[j];
              }
            }
          }
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            int nc = n + j;
            if (nc >= N) break;
            float val = OutTile[r * TILE_N + c8 + j] * alpha * g;
            if (Cbias && beta != 0.0f) val += beta * to_float(Cbias[(size_t)arow * N + nc]);
            if constexpr (UseAtomic) {
              atomicAdd(&D[(size_t)drow * N + nc], from_float(val, (OutT *)nullptr));
            } else {
              D[(size_t)drow * N + nc] = from_float(val, (OutT *)nullptr);
            }
          }
        }
      }
    }

    t += gridDim.y;
    __syncthreads();
  }
}

template <typename InT, typename OutT, bool UseAtomic>
void launch_segmented_implicit_gemm(dim3 grid,
                                    dim3 block,
                                    cudaStream_t stream,
                                    const InT *A,
                                    const InT *B_multi,
                                    const InT *Cbias,
                                    const long *a_inds,
                                    const long *d_inds,
                                    const float *gates,
                                    const int *seg_offsets,
                                    OutT *D,
                                    int M,
                                    int K,
                                    int N,
                                    int P,
                                    int Q,
                                    int E,
                                    float alpha,
                                    float beta) {
  segmented_implicit_gemm_wmma<InT, OutT, UseAtomic><<<grid, block, 0, stream>>>(
      A, B_multi, Cbias, a_inds, d_inds, gates, seg_offsets, D, M, K, N, P, Q, E, alpha, beta);
}

namespace warpconvnet {
namespace wmma_segmented_gemm_sm80 {

template <typename ElementInput, typename ElementOutput>
int run_segmented(const void *A,
                  const void *B_multi,
                  const void *C,
                  void *D,
                  const long *a_inds,
                  const long *d_inds,
                  const float *gates,
                  const int *seg_offsets_device,
                  int M,
                  int K,
                  int N,
                  int P,
                  int Q,
                  int E,
                  float alpha,
                  float beta,
                  cudaStream_t stream) {
  // Copy offsets back to host to count tiles
  int total_tiles = 0;
  int *h_offsets = (int *)malloc((size_t)(E + 1) * sizeof(int));
  if (!h_offsets) return 1;
  cudaError_t err = cudaMemcpy(
      h_offsets, seg_offsets_device, (size_t)(E + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    free(h_offsets);
    return 1;
  }
  for (int e = 0; e < E; ++e) {
    int len = h_offsets[e + 1] - h_offsets[e];
    total_tiles += (len + 16 - 1) / 16;
  }
  free(h_offsets);

  dim3 block(32, 1, 1);
  int num_n_tiles = (N + 16 - 1) / 16;
  int grid_y = total_tiles > 1024 ? 1024 : (total_tiles < 1 ? 1 : total_tiles);
  dim3 grid(num_n_tiles, grid_y, 1);

  if constexpr (std::is_same<ElementInput, half>::value &&
                std::is_same<ElementOutput, float>::value) {
    launch_segmented_implicit_gemm<half, float, true>(grid,
                                                      block,
                                                      stream,
                                                      reinterpret_cast<const half *>(A),
                                                      reinterpret_cast<const half *>(B_multi),
                                                      reinterpret_cast<const half *>(C),
                                                      a_inds,
                                                      d_inds,
                                                      gates,
                                                      seg_offsets_device,
                                                      reinterpret_cast<float *>(D),
                                                      M,
                                                      K,
                                                      N,
                                                      P,
                                                      Q,
                                                      E,
                                                      alpha,
                                                      beta);
    return 0;
  } else if constexpr (std::is_same<ElementInput, half>::value &&
                       std::is_same<ElementOutput, half>::value) {
    launch_segmented_implicit_gemm<half, half, true>(grid,
                                                     block,
                                                     stream,
                                                     reinterpret_cast<const half *>(A),
                                                     reinterpret_cast<const half *>(B_multi),
                                                     reinterpret_cast<const half *>(C),
                                                     a_inds,
                                                     d_inds,
                                                     gates,
                                                     seg_offsets_device,
                                                     reinterpret_cast<half *>(D),
                                                     M,
                                                     K,
                                                     N,
                                                     P,
                                                     Q,
                                                     E,
                                                     alpha,
                                                     beta);
    return 0;
  } else if constexpr (std::is_same<ElementInput, nv_bfloat16>::value &&
                       std::is_same<ElementOutput, float>::value) {
    launch_segmented_implicit_gemm<nv_bfloat16, float, true>(
        grid,
        block,
        stream,
        reinterpret_cast<const nv_bfloat16 *>(A),
        reinterpret_cast<const nv_bfloat16 *>(B_multi),
        reinterpret_cast<const nv_bfloat16 *>(C),
        a_inds,
        d_inds,
        gates,
        seg_offsets_device,
        reinterpret_cast<float *>(D),
        M,
        K,
        N,
        P,
        Q,
        E,
        alpha,
        beta);
    return 0;
  } else if constexpr (std::is_same<ElementInput, nv_bfloat16>::value &&
                       std::is_same<ElementOutput, nv_bfloat16>::value) {
    launch_segmented_implicit_gemm<nv_bfloat16, nv_bfloat16, true>(
        grid,
        block,
        stream,
        reinterpret_cast<const nv_bfloat16 *>(A),
        reinterpret_cast<const nv_bfloat16 *>(B_multi),
        reinterpret_cast<const nv_bfloat16 *>(C),
        a_inds,
        d_inds,
        gates,
        seg_offsets_device,
        reinterpret_cast<nv_bfloat16 *>(D),
        M,
        K,
        N,
        P,
        Q,
        E,
        alpha,
        beta);
    return 0;
  } else {
    return 1;
  }
}

}  // namespace wmma_segmented_gemm_sm80
}  // namespace warpconvnet

namespace warpconvnet {
namespace wmma_segmented_gemm_sm80 {
template int run_segmented<half, float>(const void *,
                                        const void *,
                                        const void *,
                                        void *,
                                        const long *,
                                        const long *,
                                        const float *,
                                        const int *,
                                        int,
                                        int,
                                        int,
                                        int,
                                        int,
                                        int,
                                        float,
                                        float,
                                        cudaStream_t);
template int run_segmented<half, half>(const void *,
                                       const void *,
                                       const void *,
                                       void *,
                                       const long *,
                                       const long *,
                                       const float *,
                                       const int *,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int,
                                       float,
                                       float,
                                       cudaStream_t);
template int run_segmented<nv_bfloat16, float>(const void *,
                                               const void *,
                                               const void *,
                                               void *,
                                               const long *,
                                               const long *,
                                               const float *,
                                               const int *,
                                               int,
                                               int,
                                               int,
                                               int,
                                               int,
                                               int,
                                               float,
                                               float,
                                               cudaStream_t);
template int run_segmented<nv_bfloat16, nv_bfloat16>(const void *,
                                                     const void *,
                                                     const void *,
                                                     void *,
                                                     const long *,
                                                     const long *,
                                                     const float *,
                                                     const int *,
                                                     int,
                                                     int,
                                                     int,
                                                     int,
                                                     int,
                                                     int,
                                                     float,
                                                     float,
                                                     cudaStream_t);
}  // namespace wmma_segmented_gemm_sm80
}  // namespace warpconvnet
