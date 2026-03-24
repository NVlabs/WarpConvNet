// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Mask-based fused CuTe GEMM kernel for sparse convolution.
//
// Processes ALL kernel offsets in a single CUDA launch using bitmask-based
// offset skipping with tensor core MMA. Output voxels are sorted by mask
// pattern (mask_argsort) for warp-coherent execution and atomicAdd-free output.
//
// For each output tile:
//   1. Look up real output row indices via mask_argsort
//   2. Compute block-level mask union (which offsets are needed by ANY row)
//   3. For each active offset k:
//      a. Load gathered input A[pair_table[k, row], :] into shared memory
//      b. Load weight B[k, :, :] into shared memory
//      c. Run CuTe MMA mainloop, accumulating across offsets
//   4. Write output (no atomicAdd — sorted output ensures exclusive rows)
//
// Grid: (total_m_tiles * n_tiles, 1, 1)
// Block: 128 threads (4 warps for SM80 MMA)

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cute/tensor.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute_gemm_config.h"

namespace warpconvnet {
namespace cute_gemm {

/// atomicAdd wrapper for cutlass types (needed by dgrad epilogue)
template <typename T>
__device__ __forceinline__ void mask_atomic_add(T *addr, T val) {
  atomicAdd(addr, val);
}
template <>
__device__ __forceinline__ void mask_atomic_add<cutlass::half_t>(cutlass::half_t *addr,
                                                                 cutlass::half_t val) {
  atomicAdd(reinterpret_cast<__half *>(addr), __float2half(float(val)));
}
template <>
__device__ __forceinline__ void mask_atomic_add<cutlass::bfloat16_t>(cutlass::bfloat16_t *addr,
                                                                     cutlass::bfloat16_t val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16 *>(addr), __float2bfloat16(float(val)));
}

/// Mask-based fused sparse conv forward kernel
template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmMaskKernel {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;

  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;

  static constexpr int tM = cute::size<0>(TileShape{});
  static constexpr int tN = cute::size<1>(TileShape{});
  static constexpr int tK = cute::size<2>(TileShape{});
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
    uint32_t block_mask;  // Union of all masks in this block's M-tile
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);
  static constexpr int kVec = 16 / sizeof(ElementInput);  // 8 for fp16/bf16

  /// Main kernel operator
  __device__ void operator()(
      const ElementInput *ptr_A,    // input features [N_in, C_in]
      const ElementInput *ptr_B,    // weight [K, C_in, C_out] — K weights stacked
      ElementOutput *ptr_D,         // output features [N_out, C_out]
      const int *pair_table,        // [K * N_out] — pair_table[k*N_out + i] = input idx
      const uint32_t *pair_mask,    // [N_out] — bitmask per output voxel
      const int *mask_argsort,      // [N_out] — sorted permutation
      int N_in,
      int N_out,
      int C_in,
      int C_out,
      int K,
      float alpha,
      char *smem_buf) const {

    using namespace cute;

    // --- Step 1: Determine which output tile this block handles ---
    int grid_n = (C_out + tN - 1) / tN;
    int global_block = int(blockIdx.x);
    int m_tile = global_block / grid_n;
    int n_tile = global_block % grid_n;
    int m_start = m_tile * tM;
    int n_start = n_tile * tN;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    // --- Step 2: Compute block-level mask union ---
    if (threadIdx.x == 0) storage.block_mask = 0;
    __syncthreads();

    // Each thread checks its row's mask and contributes to block union
    for (int m_local = threadIdx.x; m_local < tM; m_local += MaxThreadsPerBlock) {
      int sorted_row = m_start + m_local;
      if (sorted_row < N_out) {
        int real_row = mask_argsort[sorted_row];
        uint32_t row_mask = pair_mask[real_row];
        atomicOr(&storage.block_mask, row_mask);
      }
    }
    __syncthreads();
    uint32_t active_offsets = storage.block_mask;

    // --- Step 3: Setup CuTe MMA ---
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

    // --- Step 4: Iterate over kernel offsets ---
    for (int k = 0; k < K; ++k) {
      // Skip offset if no row in this block needs it
      if (K <= 32 && !(active_offsets & (1u << k))) continue;

      // Weight pointer for offset k: B[k] = ptr_B + k * C_in * C_out
      const ElementInput *ptr_Bk = ptr_B + k * C_in * C_out;

      int num_k_tiles = (C_in + tK - 1) / tK;
      if (num_k_tiles == 0) continue;

      // Prolog: load first k-tile
      _load_A_masked(ptr_A, pair_table, pair_mask, mask_argsort,
                     sA(_, _, 0), m_start, 0, N_in, N_out, C_in, k, K);
      _load_dense_B_cpasync(ptr_Bk, sB(_, _, 0), n_start, 0, C_out, C_in);
      cute::cp_async_fence();
      cute::cp_async_wait<0>();
      __syncthreads();

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for (int ktile = 1; ktile < num_k_tiles; ++ktile) {
        int curr_stage = (ktile - 1) % NumStages;
        int next_stage = ktile % NumStages;
        int k_start_cin = ktile * tK;

        _load_A_masked(ptr_A, pair_table, pair_mask, mask_argsort,
                       sA(_, _, next_stage), m_start, k_start_cin, N_in, N_out, C_in, k, K);
        _load_dense_B_cpasync(ptr_Bk, sB(_, _, next_stage), n_start, k_start_cin, C_out, C_in);
        cute::cp_async_fence();

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
          copy(smem_tiled_copy_A, tCsA(_, _, k_block, curr_stage), tCrA_copy_view(_, _, k_block));
          copy(smem_tiled_copy_B, tCsB(_, _, k_block, curr_stage), tCrB_copy_view(_, _, k_block));
          cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
        }

        cute::cp_async_wait<NumStages - 2>();
        __syncthreads();
      }

      // Epilog: last k-tile
      {
        int last_stage = (num_k_tiles - 1) % NumStages;
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
          copy(smem_tiled_copy_A, tCsA(_, _, k_block, last_stage), tCrA_copy_view(_, _, k_block));
          copy(smem_tiled_copy_B, tCsB(_, _, k_block, last_stage), tCrB_copy_view(_, _, k_block));
          cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
        }
      }
      __syncthreads();  // Ensure all threads done before next offset loads
    }

    // --- Step 5: Write output (no atomicAdd) ---
    _epilogue_direct(accum, ptr_D, mask_argsort, m_start, n_start, N_out, C_out, alpha, tiled_mma);
  }

private:
  /// Load A tile with mask-based gather from pair_table
  template <class SmemTensor>
  __device__ void _load_A_masked(
      const ElementInput *ptr_A,
      const int *pair_table,
      const uint32_t *pair_mask,
      const int *mask_argsort,
      SmemTensor smem_tile,
      int m_start,
      int k_start,
      int N_in,
      int N_out,
      int C_in,
      int offset_k,
      int K) const {

    static_assert(tK % kVec == 0, "tK must be a multiple of vector width");
    constexpr int k_vecs = tK / kVec;
    constexpr int total_vecs = tM * k_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int m_local = idx / k_vecs;
      int kv = idx % k_vecs;
      int k_local = kv * kVec;
      int sorted_row = m_start + m_local;
      int k_global = k_start + k_local;

      uint4 vec_data = make_uint4(0, 0, 0, 0);

      if (sorted_row < N_out) {
        int real_row = mask_argsort[sorted_row];
        bool has_offset = false;
        if (real_row >= 0 && real_row < N_out) {
          if (K <= 32) {
            has_offset = (pair_mask[real_row] & (1u << offset_k)) != 0;
          } else {
            has_offset = (pair_table[offset_k * N_out + real_row] >= 0);
          }
        }

        if (has_offset) {
          int in_row = pair_table[offset_k * N_out + real_row];
          // Bounds check on input row
          if (in_row < 0 || in_row >= N_in) {
            has_offset = false;
          }
        }

        if (has_offset) {
          int in_row = pair_table[offset_k * N_out + real_row];
          if (k_global + kVec <= C_in) {
            vec_data = *reinterpret_cast<const uint4 *>(&ptr_A[in_row * C_in + k_global]);
          } else {
            auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
            for (int v = 0; v < kVec; ++v) {
              if (k_global + v < C_in) elems[v] = ptr_A[in_row * C_in + k_global + v];
            }
          }
        }
      }
      *reinterpret_cast<uint4 *>(&smem_tile(m_local, k_local)) = vec_data;
    }
  }

  /// Load B (weight) tile — dense, no gather
  template <class SmemTensor>
  __device__ void _load_dense_B_cpasync(
      const ElementInput *ptr_B,
      SmemTensor smem_tile,
      int n_start,
      int k_start,
      int N,
      int K_dim) const {

    static_assert(tN % kVec == 0, "tN must be a multiple of vector width");
    constexpr int n_vecs = tN / kVec;
    constexpr int total_vecs = tK * n_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int k_local = idx / n_vecs;
      int nv = idx % n_vecs;
      int n_local = nv * kVec;
      int n_global = n_start + n_local;
      int k_global = k_start + k_local;

      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));
      bool pred = (k_global < K_dim) && (n_global + kVec <= N);

      if (pred) {
        const void *gmem_src = &ptr_B[k_global * N + n_global];
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
            ::"r"(smem_addr), "l"(gmem_src), "n"(16));
      } else {
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
            ::"r"(smem_addr), "l"(ptr_B), "n"(16), "r"(0));
      }
    }
  }

  /// Write output without atomicAdd (mask_argsort ensures exclusive rows)
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_direct(
      Accumulator &accum,
      ElementOutput *ptr_D,
      const int *mask_argsort,
      int m_start,
      int n_start,
      int N_out,
      int C_out,
      float alpha,
      TiledMma_ &tiled_mma) const {

    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int m_local = get<0>(coord);
      int n_local = get<1>(coord);
      int sorted_row = m_start + m_local;
      int n_global = n_start + n_local;

      if (sorted_row < N_out && n_global < C_out) {
        int real_row = mask_argsort[sorted_row];
        float result = alpha * float(accum(i));
        ptr_D[real_row * C_out + n_global] = static_cast<ElementOutput>(result);
      }
    }
  }
};

// =====================================================================
// Backward dgrad: grad_input[pair[k,i]] += grad_output[i] @ weight[k]^T
//
// Same structure as forward but:
//   A = grad_output (gathered via mask_argsort, reduction over C_out)
//   B = weight[k] pre-transposed to [C_out, C_in] by Python dispatch
//   The MMA computes A[M,K] @ B[N,K]^T where K=C_out, N=C_in
//   With pre-transposed weight, B loads are contiguous (128-bit cp.async)
//   Output: atomicAdd scatter to grad_input via pair_table
// =====================================================================

template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmMaskDgradKernel {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;

  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr int tM = cute::size<0>(TileShape{});  // output rows (sorted)
  static constexpr int tN = cute::size<1>(TileShape{});  // C_in columns
  static constexpr int tK = cute::size<2>(TileShape{});  // C_out reduction
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
    uint32_t block_mask;
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);
  static constexpr int kVec = 16 / sizeof(ElementInput);

  __device__ void operator()(
      const ElementInput *ptr_GO,   // grad_output [N_out, C_out]
      const ElementInput *ptr_B,    // weight pre-transposed [K, C_out, C_in]
      ElementOutput *ptr_GI,        // grad_input [N_in, C_in]
      const int *pair_table,
      const uint32_t *pair_mask,
      const int *mask_argsort,
      int N_in, int N_out, int C_in, int C_out, int K,
      float alpha, char *smem_buf) const {

    using namespace cute;

    int grid_n = (C_in + tN - 1) / tN;
    int global_block = int(blockIdx.x);
    int m_tile = global_block / grid_n;
    int n_tile = global_block % grid_n;
    int m_start = m_tile * tM;
    int n_start = n_tile * tN;  // C_in start

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    // Block-level mask union
    if (threadIdx.x == 0) storage.block_mask = 0;
    __syncthreads();
    for (int m_local = threadIdx.x; m_local < tM; m_local += MaxThreadsPerBlock) {
      int sorted_row = m_start + m_local;
      if (sorted_row < N_out) {
        int real_row = mask_argsort[sorted_row];
        if (real_row >= 0 && real_row < N_out)
          atomicOr(&storage.block_mask, pair_mask[real_row]);
      }
    }
    __syncthreads();
    uint32_t active_offsets = storage.block_mask;

    // MMA setup
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<tM>{}, Int<tN>{}));

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

    // Iterate over kernel offsets, accumulating grad for each
    for (int k_off = 0; k_off < K; ++k_off) {
      if (K <= 32 && !(active_offsets & (1u << k_off))) continue;

      // For dgrad we accumulate per-offset then scatter
      // Reset accumulator per offset (each offset scatters to different input rows)
      clear(accum);

      // Weight pointer: B[k] is pre-transposed to [C_out, C_in].
      // MMA B operand tiles [N=C_in, K=C_out].
      // Memory layout [C_out, C_in] => ptr_Bk[k_global * C_in + n_global]
      // which matches the forward kernel's dense B load pattern.
      const ElementInput *ptr_Bk = ptr_B + k_off * C_in * C_out;

      int num_k_tiles = (C_out + tK - 1) / tK;
      if (num_k_tiles == 0) continue;

      // Prolog: load first k-tile using vectorized cp.async
      _load_GO_masked(ptr_GO, pair_mask, mask_argsort,
                      sA(_, _, 0), m_start, 0, N_out, C_out, k_off, K);
      _load_dense_B_cpasync(ptr_Bk, sB(_, _, 0), n_start, 0, C_in, C_out);
      cute::cp_async_fence();
      cute::cp_async_wait<0>();
      __syncthreads();

      // Pipelined mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for (int ktile = 1; ktile < num_k_tiles; ++ktile) {
        int curr_stage = (ktile - 1) % NumStages;
        int next_stage = ktile % NumStages;
        int k_start_cout = ktile * tK;

        _load_GO_masked(ptr_GO, pair_mask, mask_argsort,
                        sA(_, _, next_stage), m_start, k_start_cout, N_out, C_out, k_off, K);
        _load_dense_B_cpasync(ptr_Bk, sB(_, _, next_stage), n_start, k_start_cout, C_in, C_out);
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

      // Epilog: last k-tile
      {
        int last_stage = (num_k_tiles - 1) % NumStages;
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < K_BLOCK_MAX; ++kb) {
          copy(smem_tiled_copy_A, tCsA(_, _, kb, last_stage), tCrA_copy_view(_, _, kb));
          copy(smem_tiled_copy_B, tCsB(_, _, kb, last_stage), tCrB_copy_view(_, _, kb));
          cute::gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), accum);
        }
      }
      __syncthreads();

      // Scatter: atomicAdd to grad_input at pair_table[k_off, real_row]
      _epilogue_scatter(accum, ptr_GI, pair_table, pair_mask, mask_argsort,
                        m_start, n_start, N_in, N_out, C_in, k_off, K, alpha, tiled_mma);
    }
  }

private:
  /// Load grad_output tile — mask-based gather, same structure as forward A load
  template <class SmemTensor>
  __device__ void _load_GO_masked(
      const ElementInput *ptr_GO,
      const uint32_t *pair_mask,
      const int *mask_argsort,
      SmemTensor smem_tile,
      int m_start, int k_start,
      int N_out, int C_out,
      int offset_k, int K) const {

    constexpr int k_vecs = tK / kVec;
    constexpr int total_vecs = tM * k_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int m_local = idx / k_vecs;
      int kv = idx % k_vecs;
      int k_local = kv * kVec;
      int sorted_row = m_start + m_local;
      int k_global = k_start + k_local;

      uint4 vec_data = make_uint4(0, 0, 0, 0);

      if (sorted_row < N_out) {
        int real_row = mask_argsort[sorted_row];
        bool has_offset = false;
        if (real_row >= 0 && real_row < N_out) {
          has_offset = (K <= 32) ? ((pair_mask[real_row] & (1u << offset_k)) != 0)
                                 : true;  // For K>32, all rows loaded
        }
        if (has_offset && k_global + kVec <= C_out) {
          vec_data = *reinterpret_cast<const uint4 *>(&ptr_GO[real_row * C_out + k_global]);
        } else if (has_offset) {
          auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
          for (int v = 0; v < kVec; ++v)
            if (k_global + v < C_out) elems[v] = ptr_GO[real_row * C_out + k_global + v];
        }
      }
      *reinterpret_cast<uint4 *>(&smem_tile(m_local, k_local)) = vec_data;
    }
  }

  /// Load B (weight) tile — dense, vectorized 128-bit cp.async loads.
  /// Weight is pre-transposed to [C_out, C_in], so memory layout is
  /// ptr_Bk[k_global * N + n_global] with N=C_in, K_dim=C_out.
  template <class SmemTensor>
  __device__ void _load_dense_B_cpasync(
      const ElementInput *ptr_B,
      SmemTensor smem_tile,
      int n_start,
      int k_start,
      int N,
      int K_dim) const {

    static_assert(tN % kVec == 0, "tN must be a multiple of vector width");
    constexpr int n_vecs = tN / kVec;
    constexpr int total_vecs = tK * n_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int k_local = idx / n_vecs;
      int nv = idx % n_vecs;
      int n_local = nv * kVec;
      int n_global = n_start + n_local;
      int k_global = k_start + k_local;

      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));
      bool pred = (k_global < K_dim) && (n_global + kVec <= N);

      if (pred) {
        const void *gmem_src = &ptr_B[k_global * N + n_global];
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
            ::"r"(smem_addr), "l"(gmem_src), "n"(16));
      } else {
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
            ::"r"(smem_addr), "l"(ptr_B), "n"(16), "r"(0));
      }
    }
  }

  /// Scatter epilogue: atomicAdd to grad_input via pair_table
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_scatter(
      Accumulator &accum,
      ElementOutput *ptr_GI,
      const int *pair_table,
      const uint32_t *pair_mask,
      const int *mask_argsort,
      int m_start, int n_start,
      int N_in, int N_out, int C_in,
      int offset_k, int K,
      float alpha,
      TiledMma_ &tiled_mma) const {

    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int m_local = get<0>(coord);
      int n_local = get<1>(coord);
      int sorted_row = m_start + m_local;
      int n_global = n_start + n_local;  // C_in index

      if (sorted_row < N_out && n_global < C_in) {
        int real_row = mask_argsort[sorted_row];
        if (real_row >= 0 && real_row < N_out) {
          bool has_offset = (K <= 32) ? ((pair_mask[real_row] & (1u << offset_k)) != 0)
                                      : true;
          if (has_offset) {
            int in_row = pair_table[offset_k * N_out + real_row];
            if (in_row >= 0 && in_row < N_in) {
              float result = alpha * float(accum(i));
              mask_atomic_add(&ptr_GI[in_row * C_in + n_global],
                         static_cast<ElementOutput>(result));
            }
          }
        }
      }
    }
  }
};

/// dgrad kernel entry point
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock)
void cute_gemm_mask_dgrad_kernel_entry(
    const typename Kernel::ElementInput *ptr_GO,
    const typename Kernel::ElementInput *ptr_B,
    typename Kernel::ElementOutput *ptr_GI,
    const int *pair_table,
    const uint32_t *pair_mask,
    const int *mask_argsort,
    int N_in, int N_out, int C_in, int C_out, int K,
    float alpha) {
  extern __shared__ char smem[];
  Kernel{}(ptr_GO, ptr_B, ptr_GI, pair_table, pair_mask, mask_argsort,
           N_in, N_out, C_in, C_out, K, alpha, smem);
}

/// Global kernel entry point
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock)
void cute_gemm_mask_kernel_entry(
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
    float alpha) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A, ptr_B, ptr_D, pair_table, pair_mask, mask_argsort,
           N_in, N_out, C_in, C_out, K, alpha, smem);
}

}  // namespace cute_gemm
}  // namespace warpconvnet
