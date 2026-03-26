// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Mask-based fused CuTe wgrad kernel with split-K parallelization.
//
// Grid: (C_in_tiles * C_out_tiles, K, split_k)
//   blockIdx.x → (C_in_tile, C_out_tile) output weight tile
//   blockIdx.y → kernel offset k
//   blockIdx.z → split-K shard (each shard reduces over N_out/split_k pairs)
//
// Each block reduces over its shard of output rows for one offset k,
// then atomicAdd partial results to grad_weight[k]. For split_k=1,
// uses direct store (no atomicAdd).
//
// The output grad_weight must be zero-initialized before launch.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cute_gemm_config.h"

namespace warpconvnet {
namespace cute_gemm {

/// atomicAdd wrapper for wgrad epilogue
template <typename T>
__device__ __forceinline__ void wgrad_atomic_add(T *addr, T val) {
  atomicAdd(addr, val);
}
template <>
__device__ __forceinline__ void wgrad_atomic_add<cutlass::half_t>(cutlass::half_t *addr,
                                                                  cutlass::half_t val) {
  atomicAdd(reinterpret_cast<__half *>(addr), __float2half(float(val)));
}
template <>
__device__ __forceinline__ void wgrad_atomic_add<cutlass::bfloat16_t>(cutlass::bfloat16_t *addr,
                                                                      cutlass::bfloat16_t val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16 *>(addr), __float2bfloat16(float(val)));
}

template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmMaskWgradKernel {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;

  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr int tM = cute::size<0>(TileShape{});  // C_in tile
  static constexpr int tN = cute::size<1>(TileShape{});  // C_out tile
  static constexpr int tK = cute::size<2>(TileShape{});  // reduction tile (pairs)
  static constexpr int NumStages = TileConfig::NumStages;

  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::Int<tM>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::Int<tN>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);
  static constexpr int kVec = 16 / sizeof(ElementInput);

  __device__ void operator()(const ElementInput *ptr_A,  // input [N_in, C_in]
                             const ElementInput *ptr_B,  // grad_output [N_out, C_out]
                             ElementOutput *ptr_D,       // grad_weight [K, C_in, C_out] (zero-init)
                             const int *pair_table,      // [K * N_out]
                             const uint32_t *pair_mask,  // [N_out]
                             const int *mask_argsort,    // [N_out]
                             int N_in,
                             int N_out,
                             int C_in,
                             int C_out,
                             int K,
                             float alpha,
                             bool use_atomic,
                             char *smem_buf) const {
    using namespace cute;

    // Grid dispatch
    int grid_n = (C_out + tN - 1) / tN;
    int m_tile = int(blockIdx.x) / grid_n;
    int n_tile = int(blockIdx.x) % grid_n;
    int m_start = m_tile * tM;  // C_in start
    int n_start = n_tile * tN;  // C_out start
    int k_off = int(blockIdx.y);
    int split_idx = int(blockIdx.z);
    int total_splits = int(gridDim.z);

    // Compute this shard's pair range
    int pairs_per_split = (N_out + total_splits - 1) / total_splits;
    int pair_begin = split_idx * pairs_per_split;
    int pair_end = min(pair_begin + pairs_per_split, N_out);
    int shard_size = pair_end - pair_begin;

    if (shard_size <= 0) return;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    // MMA setup
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<tM>{}, Int<tN>{}));
    clear(accum);

    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_slice(threadIdx.x);
    Tensor tCsA = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_slice(threadIdx.x);
    Tensor tCsB = smem_thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

    auto K_BLOCK_MAX = size<2>(tCrA);

    int num_pair_tiles = (shard_size + tK - 1) / tK;
    if (num_pair_tiles == 0) return;

    // Prolog
    _load_A_transposed(ptr_A,
                       pair_table,
                       pair_mask,
                       mask_argsort,
                       sA(_, _, 0),
                       m_start,
                       pair_begin,
                       N_in,
                       N_out,
                       C_in,
                       k_off,
                       K);
    _load_B_gathered(
        ptr_B, pair_mask, mask_argsort, sB(_, _, 0), n_start, pair_begin, N_out, C_out, k_off, K);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // Pipelined mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for (int ptile = 1; ptile < num_pair_tiles; ++ptile) {
      int curr_stage = (ptile - 1) % NumStages;
      int next_stage = ptile % NumStages;
      int pair_start = pair_begin + ptile * tK;

      _load_A_transposed(ptr_A,
                         pair_table,
                         pair_mask,
                         mask_argsort,
                         sA(_, _, next_stage),
                         m_start,
                         pair_start,
                         N_in,
                         N_out,
                         C_in,
                         k_off,
                         K);
      _load_B_gathered(ptr_B,
                       pair_mask,
                       mask_argsort,
                       sB(_, _, next_stage),
                       n_start,
                       pair_start,
                       N_out,
                       C_out,
                       k_off,
                       K);
      cute::cp_async_fence();

      CUTLASS_PRAGMA_UNROLL
      for (int kb = 0; kb < K_BLOCK_MAX; ++kb) {
        copy(smem_tiled_copy_A, tCsA(_, _, kb, curr_stage), tCrA_copy_view(_, _, kb));
        copy(smem_tiled_copy_B, tCsB(_, _, kb, curr_stage), tCrB_copy_view(_, _, kb));
        cute::gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), accum);
      }

      cute::cp_async_wait<NumStages - 2>();
      __syncthreads();
    }

    // Epilog: last pair tile
    {
      int last_stage = (num_pair_tiles - 1) % NumStages;
      CUTLASS_PRAGMA_UNROLL
      for (int kb = 0; kb < K_BLOCK_MAX; ++kb) {
        copy(smem_tiled_copy_A, tCsA(_, _, kb, last_stage), tCrA_copy_view(_, _, kb));
        copy(smem_tiled_copy_B, tCsB(_, _, kb, last_stage), tCrB_copy_view(_, _, kb));
        cute::gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), accum);
      }
    }

    // Write result
    if (use_atomic) {
      _epilogue_atomic(accum, ptr_D, k_off, m_start, n_start, C_in, C_out, alpha, tiled_mma);
    } else {
      _epilogue_direct(accum, ptr_D, k_off, m_start, n_start, C_in, C_out, alpha, tiled_mma);
    }
  }

private:
  /// Load A^T tile: input[pair_table[k, mask_argsort[row]], c_in] transposed to smem
  template <class SmemTensor>
  __device__ void _load_A_transposed(const ElementInput *ptr_A,
                                     const int *pair_table,
                                     const uint32_t *pair_mask,
                                     const int *mask_argsort,
                                     SmemTensor smem_tile,
                                     int m_start,
                                     int pair_start,
                                     int N_in,
                                     int N_out,
                                     int C_in,
                                     int offset_k,
                                     int K) const {
    static_assert(tM % kVec == 0, "tM must be a multiple of vector width");
    constexpr int m_vecs = tM / kVec;
    constexpr int total_vecs = tK * m_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int pair_local = idx / m_vecs;
      int mv = idx % m_vecs;
      int m_local = mv * kVec;
      int pair_global = pair_start + pair_local;
      int m_global = m_start + m_local;

      uint4 vec_data = make_uint4(0, 0, 0, 0);

      if (pair_global < N_out) {
        int real_row = mask_argsort[pair_global];
        bool has_offset = false;
        if (real_row >= 0 && real_row < N_out) {
          has_offset = (K <= 32) ? ((pair_mask[real_row] & (1u << offset_k)) != 0)
                                 : (pair_table[offset_k * N_out + real_row] >= 0);
        }
        if (has_offset) {
          int in_row = pair_table[offset_k * N_out + real_row];
          if (in_row >= 0 && in_row < N_in && m_global + kVec <= C_in) {
            vec_data = *reinterpret_cast<const uint4 *>(&ptr_A[in_row * C_in + m_global]);
          } else if (in_row >= 0 && in_row < N_in) {
            auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
            for (int v = 0; v < kVec; ++v)
              if (m_global + v < C_in) elems[v] = ptr_A[in_row * C_in + m_global + v];
          }
        }
      }
      // Transpose store: smem[m_local, pair_local]
      auto *elems = reinterpret_cast<const ElementInput *>(&vec_data);
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kVec; ++v) {
        smem_tile(m_local + v, pair_local) = elems[v];
      }
    }
  }

  /// Load B tile: grad_output rows
  template <class SmemTensor>
  __device__ void _load_B_gathered(const ElementInput *ptr_B,
                                   const uint32_t *pair_mask,
                                   const int *mask_argsort,
                                   SmemTensor smem_tile,
                                   int n_start,
                                   int pair_start,
                                   int N_out,
                                   int C_out,
                                   int offset_k,
                                   int K) const {
    static_assert(tN % kVec == 0, "tN must be a multiple of vector width");
    constexpr int n_vecs = tN / kVec;
    constexpr int total_vecs = tK * n_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int pair_local = idx / n_vecs;
      int nv = idx % n_vecs;
      int n_local = nv * kVec;
      int pair_global = pair_start + pair_local;
      int n_global = n_start + n_local;

      uint4 vec_data = make_uint4(0, 0, 0, 0);

      if (pair_global < N_out) {
        int real_row = mask_argsort[pair_global];
        bool has_offset = false;
        if (real_row >= 0 && real_row < N_out) {
          has_offset = (K <= 32) ? ((pair_mask[real_row] & (1u << offset_k)) != 0) : true;
        }
        if (has_offset && n_global + kVec <= C_out) {
          vec_data = *reinterpret_cast<const uint4 *>(&ptr_B[real_row * C_out + n_global]);
        } else if (has_offset) {
          auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
          for (int v = 0; v < kVec; ++v)
            if (n_global + v < C_out) elems[v] = ptr_B[real_row * C_out + n_global + v];
        }
      }
      *reinterpret_cast<uint4 *>(&smem_tile(n_local, pair_local)) = vec_data;
    }
  }

  /// Direct store epilogue (split_k == 1)
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_direct(Accumulator &accum,
                                   ElementOutput *ptr_D,
                                   int k_off,
                                   int m_start,
                                   int n_start,
                                   int C_in,
                                   int C_out,
                                   float alpha,
                                   TiledMma_ &tiled_mma) const {
    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));
    ElementOutput *ptr_Dk = ptr_D + k_off * C_in * C_out;

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int m_global = m_start + get<0>(coord);
      int n_global = n_start + get<1>(coord);
      if (m_global < C_in && n_global < C_out) {
        float result = alpha * float(accum(i));
        ptr_Dk[m_global * C_out + n_global] = static_cast<ElementOutput>(result);
      }
    }
  }

  /// atomicAdd epilogue (split_k > 1)
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_atomic(Accumulator &accum,
                                   ElementOutput *ptr_D,
                                   int k_off,
                                   int m_start,
                                   int n_start,
                                   int C_in,
                                   int C_out,
                                   float alpha,
                                   TiledMma_ &tiled_mma) const {
    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));
    ElementOutput *ptr_Dk = ptr_D + k_off * C_in * C_out;

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int m_global = m_start + get<0>(coord);
      int n_global = n_start + get<1>(coord);
      if (m_global < C_in && n_global < C_out) {
        float result = alpha * float(accum(i));
        wgrad_atomic_add(&ptr_Dk[m_global * C_out + n_global], static_cast<ElementOutput>(result));
      }
    }
  }
};

/// Global kernel entry point
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void cute_gemm_mask_wgrad_kernel_entry(
    const typename Kernel::ElementInput *ptr_A,
    const typename Kernel::ElementInput *ptr_B,
    typename Kernel::ElementOutput *ptr_D,
    const int *pair_table,
    const uint32_t *pair_mask,
    const int *mask_argsort,
    int N_in,
    int N_out,
    int C_in,
    int C_out,
    int K,
    float alpha,
    bool use_atomic) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A,
           ptr_B,
           ptr_D,
           pair_table,
           pair_mask,
           mask_argsort,
           N_in,
           N_out,
           C_in,
           C_out,
           K,
           alpha,
           use_atomic,
           smem);
}

}  // namespace cute_gemm
}  // namespace warpconvnet
