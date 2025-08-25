// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Segmented tr(A)B gather-scatter for batch processing
// Computes, per segment s, dW[e] += A_s^T @ (B_s \odot gates_s)
// where A_s := X[a_inds[p0:p1), :], B_s := DY[d_inds[p0:p1), :].
//
// SM80 WMMA kernel, one warp per block. TILE_{M,N,K} = 16.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <type_traits>

using namespace nvcuda;

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

// -------------------- Kernel --------------------
template <typename InT, typename OutT>
__global__ void segmented_trab_gather_wmma(
    const InT *__restrict__ X,            // [N_rows, C_in], row-major
    const InT *__restrict__ DY,           // [M_rows, C_out], row-major
    OutT *__restrict__ dW_all,            // [E, C_in, C_out], row-major contiguous
    const long *__restrict__ a_inds,      // [P]
    const long *__restrict__ d_inds,      // [P]
    const float *__restrict__ gates,      // [P]
    const int *__restrict__ seg_offsets,  // [S+1], S = num_segments
    int N_rows,
    int M_rows,
    int C_in,
    int C_out,
    int P,
    int num_segments,
    float alpha,
    float beta) {
  static_assert(IsInputSupported<InT>::value, "InT must be half or nv_bfloat16");
  static_assert(IsOutputSupported<OutT>::value, "OutT must be float, half, or nv_bfloat16");

  if (blockDim.x != 32) return;  // one warp per block

  constexpr int TILE_M = 16, TILE_N = 16, TILE_K = 16;  // M=C_in, N=C_out, K=segment rows

  const int lane = threadIdx.x & 31;
  const int tile_n = blockIdx.x;  // along C_out
  const int tile_m = blockIdx.z;  // along C_in
  const int n_base = tile_n * TILE_N;
  const int m_base = tile_m * TILE_M;

  // Segment index this block processes (one segment per blockIdx.y step)
  for (int s = blockIdx.y; s < num_segments; s += gridDim.y) {
    const int seg_start = seg_offsets[s];
    const int seg_end = seg_offsets[s + 1];
    const int seg_len = seg_end - seg_start;
    if (seg_len <= 0) continue;

    // Shared buffers
    __shared__ long a_row_idx[TILE_K];
    __shared__ long d_row_idx[TILE_K];
    __shared__ float row_gate[TILE_K];
    __shared__ __align__(16) InT Asmem[TILE_K * TILE_M];  // stored as [K,M] row-major
    __shared__ __align__(16) InT Bsmem[TILE_K * TILE_N];  // stored as [K,N] row-major
    __shared__ float OutTile[TILE_M * TILE_N];

    using WmmaT = typename WmmaInputTag<InT>::type;
    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, WmmaT, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, WmmaT, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // K-loop over segment rows in chunks of 16
    for (int p0 = seg_start; p0 < seg_end; p0 += TILE_K) {
      // Gather indices and gates for current 16-row tile
      if (lane < TILE_K) {
        const int p = p0 + lane;
        const bool valid = (p < seg_end);
        a_row_idx[lane] = (valid && p < P) ? a_inds[p] : -1;
        d_row_idx[lane] = (valid && p < P) ? d_inds[p] : -1;
        row_gate[lane] = (valid && p < P) ? gates[p] : 0.0f;
      }
      __syncthreads();

      // Stage loads: each lane loads 16B from X and DY into shared
      const int row = lane & 15;  // 0..15 (K dimension)
      const int seg = lane >> 4;  // 0..1 (two 8-element vecs per row)

      long arow = a_row_idx[row];
      const InT *gA =
          (arow >= 0 && arow < N_rows) ? (X + (size_t)arow * C_in + m_base + seg * 8) : nullptr;
      bool predA = (arow >= 0 && arow < N_rows) && (m_base + seg * 8 + 7 < C_in);
      InT *sA = &Asmem[row * TILE_M + seg * 8];
      cp_async_16B((void *)sA, (const void *)gA, predA);

      long drow = d_row_idx[row];
      const InT *gB =
          (drow >= 0 && drow < M_rows) ? (DY + (size_t)drow * C_out + n_base + seg * 8) : nullptr;
      bool predB = (drow >= 0 && drow < M_rows) && (n_base + seg * 8 + 7 < C_out);
      InT *sB = &Bsmem[row * TILE_N + seg * 8];
      cp_async_16B((void *)sB, (const void *)gB, predB);

#if __CUDA_ARCH__ >= 800
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 0;\n");
#endif
      __syncthreads();

      // Apply per-row gating to Bsmem rows in-place
      for (int idx = lane; idx < TILE_K * TILE_N; idx += 32) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        float g = row_gate[r];
        InT v = Bsmem[r * TILE_N + c];
        float vf = to_float(v) * g;
        Bsmem[r * TILE_N + c] = from_float(vf, (InT *)nullptr);
      }
      __syncthreads();

      // MMA accumulate
      wmma::load_matrix_sync(a_frag, &Asmem[0], /*ldm (col-major M)*/ TILE_K);
      wmma::load_matrix_sync(b_frag, &Bsmem[0], /*ldm (row-major N)*/ TILE_N);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      __syncthreads();
    }

    // Store accumulator tile
    wmma::store_matrix_sync(OutTile, acc_frag, TILE_N, wmma::mem_row_major);
    __syncthreads();

    // Write to dW[s, m_base: , n_base: ] with epilogue alpha/beta
    const size_t slice_stride = (size_t)C_in * (size_t)C_out;
    OutT *__restrict__ DstBase =
        dW_all + (size_t)s * slice_stride + (size_t)m_base * C_out + n_base;

    if constexpr (std::is_same<OutT, float>::value) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int chunk_id = lane + i * 32;      // 0..63
        int r = (chunk_id * 4) / TILE_N;   // row within tile
        int c4 = (chunk_id * 4) % TILE_N;  // col start
        int n = n_base + c4;
        int m = m_base + r;
        if (m < C_in) {
          float v0 = OutTile[r * TILE_N + c4 + 0] * alpha;
          float v1 = OutTile[r * TILE_N + c4 + 1] * alpha;
          float v2 = OutTile[r * TILE_N + c4 + 2] * alpha;
          float v3 = OutTile[r * TILE_N + c4 + 3] * alpha;
          float *Dst = DstBase + (size_t)r * C_out + c4;
          const bool full = (n + 3 < C_out);
          const bool aligned16 = ((reinterpret_cast<uintptr_t>(Dst) & 0xF) == 0);
          if (beta != 0.0f) {
            if (full && aligned16) {
              float4 prev = *reinterpret_cast<const float4 *>(Dst);
              v0 += beta * prev.x;
              v1 += beta * prev.y;
              v2 += beta * prev.z;
              v3 += beta * prev.w;
            } else {
              if (n + 0 < C_out) v0 += beta * Dst[0];
              if (n + 1 < C_out) v1 += beta * Dst[1];
              if (n + 2 < C_out) v2 += beta * Dst[2];
              if (n + 3 < C_out) v3 += beta * Dst[3];
            }
          }
          if (full && aligned16) {
            float4 vv = make_float4(v0, v1, v2, v3);
            *reinterpret_cast<float4 *>(Dst) = vv;
          } else {
            if (n + 0 < C_out) Dst[0] = v0;
            if (n + 1 < C_out) Dst[1] = v1;
            if (n + 2 < C_out) Dst[2] = v2;
            if (n + 3 < C_out) Dst[3] = v3;
          }
        }
      }
    } else {
      // Half/BF16 path: convert, optionally blend beta
      int chunk_id = lane;               // 0..31
      int r = (chunk_id * 8) / TILE_N;   // row within tile
      int c8 = (chunk_id * 8) % TILE_N;  // col start (0 or 8)
      int n = n_base + c8;
      if (m_base + r < C_in) {
        OutT *Dst = DstBase + (size_t)r * C_out + c8;
        alignas(16) OutT tmp8[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          int nc = n + j;
          float v = (nc < C_out) ? (OutTile[r * TILE_N + c8 + j] * alpha) : 0.0f;
          if (beta != 0.0f && nc < C_out) {
            if constexpr (std::is_same<OutT, half>::value) {
              v += beta * __half2float(reinterpret_cast<const half *>(Dst)[j]);
            } else if constexpr (std::is_same<OutT, nv_bfloat16>::value) {
              v += beta * __bfloat162float(reinterpret_cast<const nv_bfloat16 *>(Dst)[j]);
            }
          }
          tmp8[j] = from_float(v, (OutT *)nullptr);
        }
        const bool aligned16 = ((reinterpret_cast<uintptr_t>(Dst) & 0xF) == 0);
        if (aligned16) {
          *reinterpret_cast<uint4 *>(Dst) = *reinterpret_cast<const uint4 *>(tmp8);
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            if (n + j < C_out) Dst[j] = tmp8[j];
          }
        }
      }
    }
    __syncthreads();
  }
}

// -------------------- Host launcher --------------------
template <typename InT, typename OutT>
static void launch_segmented_trab(dim3 grid,
                                  dim3 block,
                                  cudaStream_t stream,
                                  const InT *X,
                                  const InT *DY,
                                  OutT *D_all,
                                  const long *a_inds,
                                  const long *d_inds,
                                  const float *gates,
                                  const int *seg_offsets,
                                  int N_rows,
                                  int M_rows,
                                  int C_in,
                                  int C_out,
                                  int P,
                                  int num_segments,
                                  float alpha,
                                  float beta) {
  segmented_trab_gather_wmma<InT, OutT><<<grid, block, 0, stream>>>(X,
                                                                    DY,
                                                                    D_all,
                                                                    a_inds,
                                                                    d_inds,
                                                                    gates,
                                                                    seg_offsets,
                                                                    N_rows,
                                                                    M_rows,
                                                                    C_in,
                                                                    C_out,
                                                                    P,
                                                                    num_segments,
                                                                    alpha,
                                                                    beta);
}

namespace warpconvnet {
namespace wmma_segmented_trab_sm80 {

template <typename ElementInput, typename ElementOutput>
int run_segmented_trab(const void *X,
                       const void *DY,
                       void *D_all,
                       const long *a_inds,
                       const long *d_inds,
                       const float *gates,
                       const int *seg_offsets_device,
                       int N_rows,
                       int M_rows,
                       int C_in,
                       int C_out,
                       int P,
                       int num_segments,
                       float alpha,
                       float beta,
                       cudaStream_t stream) {
  // grid configuration
  dim3 block(32, 1, 1);
  int grid_x = (C_out + 16 - 1) / 16;
  int grid_z = (C_in + 16 - 1) / 16;
  int grid_y = num_segments > 1024 ? 1024 : (num_segments < 1 ? 1 : num_segments);
  dim3 grid(grid_x, grid_y, grid_z);

  if constexpr (std::is_same<ElementInput, half>::value &&
                std::is_same<ElementOutput, float>::value) {
    launch_segmented_trab<half, float>(grid,
                                       block,
                                       stream,
                                       reinterpret_cast<const half *>(X),
                                       reinterpret_cast<const half *>(DY),
                                       reinterpret_cast<float *>(D_all),
                                       a_inds,
                                       d_inds,
                                       gates,
                                       seg_offsets_device,
                                       N_rows,
                                       M_rows,
                                       C_in,
                                       C_out,
                                       P,
                                       num_segments,
                                       alpha,
                                       beta);
    return 0;
  } else if constexpr (std::is_same<ElementInput, half>::value &&
                       std::is_same<ElementOutput, half>::value) {
    launch_segmented_trab<half, half>(grid,
                                      block,
                                      stream,
                                      reinterpret_cast<const half *>(X),
                                      reinterpret_cast<const half *>(DY),
                                      reinterpret_cast<half *>(D_all),
                                      a_inds,
                                      d_inds,
                                      gates,
                                      seg_offsets_device,
                                      N_rows,
                                      M_rows,
                                      C_in,
                                      C_out,
                                      P,
                                      num_segments,
                                      alpha,
                                      beta);
    return 0;
  } else if constexpr (std::is_same<ElementInput, nv_bfloat16>::value &&
                       std::is_same<ElementOutput, float>::value) {
    launch_segmented_trab<nv_bfloat16, float>(grid,
                                              block,
                                              stream,
                                              reinterpret_cast<const nv_bfloat16 *>(X),
                                              reinterpret_cast<const nv_bfloat16 *>(DY),
                                              reinterpret_cast<float *>(D_all),
                                              a_inds,
                                              d_inds,
                                              gates,
                                              seg_offsets_device,
                                              N_rows,
                                              M_rows,
                                              C_in,
                                              C_out,
                                              P,
                                              num_segments,
                                              alpha,
                                              beta);
    return 0;
  } else if constexpr (std::is_same<ElementInput, nv_bfloat16>::value &&
                       std::is_same<ElementOutput, nv_bfloat16>::value) {
    launch_segmented_trab<nv_bfloat16, nv_bfloat16>(grid,
                                                    block,
                                                    stream,
                                                    reinterpret_cast<const nv_bfloat16 *>(X),
                                                    reinterpret_cast<const nv_bfloat16 *>(DY),
                                                    reinterpret_cast<nv_bfloat16 *>(D_all),
                                                    a_inds,
                                                    d_inds,
                                                    gates,
                                                    seg_offsets_device,
                                                    N_rows,
                                                    M_rows,
                                                    C_in,
                                                    C_out,
                                                    P,
                                                    num_segments,
                                                    alpha,
                                                    beta);
    return 0;
  } else {
    return 1;
  }
}

}  // namespace wmma_segmented_trab_sm80
}  // namespace warpconvnet

// -------------------- Explicit instantiations --------------------
namespace warpconvnet {
namespace wmma_segmented_trab_sm80 {
template int run_segmented_trab<half, float>(const void *,
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
template int run_segmented_trab<half, half>(const void *,
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
template int run_segmented_trab<nv_bfloat16, float>(const void *,
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
template int run_segmented_trab<nv_bfloat16, nv_bfloat16>(const void *,
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
}  // namespace wmma_segmented_trab_sm80
}  // namespace warpconvnet
