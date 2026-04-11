// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// clang-format off
#include "cute/tensor.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
// clang-format on
#include "cute_gemm_config.h"
#include "mma_macros.h"

namespace warpconvnet {
namespace cute_gemm {

// Wgrad kernel: dW[k] = Gather_k(X)^T @ dY_sorted
// Grid: (C_in_tiles * C_out_tiles, K, split_k)
// Contraction: over N_out (voxel dimension) in tK-sized chunks
// Accumulator: f32
// Epilogue: atomic
template <class TileConfig, typename ElementOutput_ = float>
struct MaskGemm_wgrad_64x64x32_2s_f32_atomic {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;
  // Wgrad: gathered loaders write M/N-vectors (not K-vectors) via cp.async.
  // Both A and B need their non-K dimension contiguous in smem for aligned stores.
  // Use SmemLayoutAtomB (N/M-contiguous) for both, with LDSM_T for register loads.
  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomB;  // M-contiguous
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;  // N-contiguous
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomB;      // LDSM_T
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;      // LDSM_T

  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int tM = cute::size<0>(TileShape{});  // C_in tile
  static constexpr int tN = cute::size<1>(TileShape{});  // C_out tile
  static constexpr int tK = cute::size<2>(TileShape{});  // N_out tile (voxel contraction)
  static constexpr int NumStages = 2;
  static constexpr int NumWarps = MaxThreadsPerBlock / 32;
  static constexpr int kVec = 16 / sizeof(ElementInput);
  static constexpr int kMmaK = cute::size<2>(typename TiledMma::AtomShape_MNK{});
  static constexpr int K_BLOCK_MAX_STATIC = tK / kMmaK;

  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::Int<tM>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::Int<tN>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
    // Pre-cached voxel info: avoids redundant __ldg in A and B loaders
    int cached_real_rows[tK];          // mask_argsort[v] for each voxel in tile
    int cached_in_rows[tK];            // pair_table[k_off, real_row] for each voxel
    uint32_t cached_active[NumWarps];  // per-warp ballot for block reduce
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);

  __device__ void operator()(
      const ElementInput *ptr_A,     // input [N_in, C_in]
      const ElementInput *ptr_B,     // grad_output [N_out, C_out]
      ElementOutput *ptr_D,          // grad_weight [K, C_in, C_out] or workspace
      const int *pair_table,         // [K * N_out]
      const uint32_t *pair_mask,     // [N_out]
      const int *mask_argsort,       // [N_out]
      const uint32_t *reduced_mask,  // [ceil(N_out/tK)] OR-reduced pair_masks
      int N_in,
      int N_out,
      int C_in,
      int C_out,
      int K,
      float alpha,
      char *smem_buf) const {
    using namespace cute;

    // Grid: (C_in_tiles * C_out_tiles, K, split_k)
    int grid_n = (C_out + tN - 1) / tN;
    int mn_idx = int(blockIdx.x);
    int m_tile = mn_idx / grid_n;
    int n_tile = mn_idx % grid_n;
    int k_off = int(blockIdx.y);
    int split_idx = int(blockIdx.z);
    int split_k = int(gridDim.z);

    int m_start = m_tile * tM;
    int n_start = n_tile * tN;

    // Split-K shard assignment
    // Contiguous split-K: shard handles [shard_start, shard_end)
    // Align to tK boundaries for reduced mask compatibility.
    int shard_size_raw = (N_out + split_k - 1) / split_k;
    int shard_size = ((shard_size_raw + tK - 1) / tK) * tK;
    int shard_start = split_idx * shard_size;
    int shard_end = shard_start + shard_size;
    if (shard_end > N_out) shard_end = N_out;
    if (shard_start >= N_out) return;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    // MMA setup with f32 accumulator
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<tM>{}, Int<tN>{}));
    clear(accum);

    Tensor tCrA_0 = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrA_1 = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB_0 = thr_mma.partition_fragment_B(sB(_, _, 0));
    Tensor tCrB_1 = thr_mma.partition_fragment_B(sB(_, _, 0));

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_slice(threadIdx.x);
    Tensor tCsA = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_0 = smem_thr_copy_A.retile_D(tCrA_0);
    Tensor tCrA_copy_1 = smem_thr_copy_A.retile_D(tCrA_1);

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_slice(threadIdx.x);
    Tensor tCsB = smem_thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_0 = smem_thr_copy_B.retile_D(tCrB_0);
    Tensor tCrB_copy_1 = smem_thr_copy_B.retile_D(tCrB_1);

    auto K_BLOCK_MAX = size<2>(tCrA_0);

    // --- Pipelined voxel-tile loop with cached voxel info ---
    // For each tile: pre-scan mask_argsort + pair_table into smem cache (1 __ldg each),
    // then A and B loaders read from cache instead of doing redundant __ldg.
    // Saves ~480 __ldg per tile (16x redundancy eliminated).
    const int *pt_k = pair_table + k_off * N_out;
    int mask_words = (K + 31) / 32;
    uint32_t k_mask_bit = 1u << (k_off % 32);
    int k_mask_word = k_off / 32;
    int num_voxel_tiles = (shard_end - shard_start + tK - 1) / tK;

// Pre-scan helper: populates smem cache and returns true if any voxel is active.
// Inlined instead of lambda to avoid compiler issues with __syncthreads.
#define PRESCAN_TILE(v_start_expr, pair_bound_expr, result_var)                        \
  {                                                                                    \
    int _ps_v_start = (v_start_expr);                                                  \
    int _ps_pair_bound = (pair_bound_expr);                                            \
    uint32_t _ps_any = 0;                                                              \
    for (int _ps_i = threadIdx.x; _ps_i < tK; _ps_i += MaxThreadsPerBlock) {           \
      int _ps_pg = _ps_v_start + _ps_i;                                                \
      int _ps_rr = -1, _ps_ir = -1;                                                    \
      if (_ps_pg < _ps_pair_bound) {                                                   \
        _ps_rr = __ldg(&mask_argsort[_ps_pg]);                                         \
        uint32_t _ps_rm = __ldg(&pair_mask[_ps_rr * mask_words + k_mask_word]);        \
        if (_ps_rm & k_mask_bit) {                                                     \
          _ps_ir = __ldg(&pt_k[_ps_rr]);                                               \
          _ps_any = 1;                                                                 \
        } else {                                                                       \
          _ps_rr = -1;                                                                 \
        }                                                                              \
      }                                                                                \
      storage.cached_real_rows[_ps_i] = _ps_rr;                                        \
      storage.cached_in_rows[_ps_i] = _ps_ir;                                          \
    }                                                                                  \
    uint32_t _ps_ballot = __ballot_sync(0xffffffff, _ps_any != 0);                     \
    __syncthreads(); /* Warp 0 collects from all warps via smem */                     \
    if ((threadIdx.x & 31) == 0) storage.cached_active[threadIdx.x / 32] = _ps_ballot; \
    __syncthreads();                                                                   \
    uint32_t _ps_block_active = 0;                                                     \
    for (int _ps_w = 0; _ps_w < NumWarps; ++_ps_w)                                     \
      _ps_block_active |= storage.cached_active[_ps_w];                                \
    (result_var) = (_ps_block_active != 0);                                            \
  }

    if (num_voxel_tiles > 0) {
      int smem_stage = 0;

      // Find first active tile via reduced mask (O(1) per tile)
      int rm_offset = shard_start / tK;
      int vt = 0;
      for (; vt < num_voxel_tiles; ++vt) {
        if (__ldg(&reduced_mask[(rm_offset + vt) * mask_words + k_mask_word]) & k_mask_bit) break;
      }
      if (vt >= num_voxel_tiles) goto wgrad_epilogue;

      // Prescan first active tile to populate cache
      {
        int vs = shard_start + vt * tK;
        int pb = (vs + tK < N_out) ? (vs + tK) : N_out;
        bool _dummy;
        PRESCAN_TILE(vs, pb, _dummy)
      }

      // Prolog: load first active tile using cached info
      _load_A_cached(ptr_A, storage, sA(_, _, 0), m_start, N_in, C_in);
      _load_B_cached(ptr_B, storage, sB(_, _, 0), n_start, C_out);
      cute::cp_async_fence();
      cute::cp_async_wait<0>();
      __syncthreads();
      ++vt;

      // Main loop (vt already advanced past prolog tile)
      for (; vt < num_voxel_tiles; ++vt) {
        int v_next = shard_start + vt * tK;
        int pb_next = (v_next + tK < N_out) ? (v_next + tK) : N_out;
        if (!(__ldg(&reduced_mask[(rm_offset + vt) * mask_words + k_mask_word]) & k_mask_bit))
          continue;

        bool _dummy;
        PRESCAN_TILE(v_next, pb_next, _dummy)

        int write_stage = (smem_stage + 1) % NumStages;

        // Pipelined: launch async loads FIRST, then MMA on current stage
        _load_A_cached(ptr_A, storage, sA(_, _, write_stage), m_start, N_in, C_in);
        _load_B_cached(ptr_B, storage, sB(_, _, write_stage), n_start, C_out);
        cute::cp_async_fence();

        MMA_DOUBLE_BUFFERED(smem_stage)

        cute::cp_async_wait<0>();
        __syncthreads();
        smem_stage = write_stage;
      }

      MMA_DOUBLE_BUFFERED(smem_stage)
      __syncthreads();
    }
  wgrad_epilogue:

    // Epilogue: dense store to dW[k_off]
    {
      auto thr_mma_epi = tiled_mma.get_slice(threadIdx.x);
      Tensor tCrC = thr_mma_epi.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

      ElementOutput *out_ptr = ptr_D + k_off * C_in * C_out;

      CUTE_UNROLL
      for (int i = 0; i < size(accum); ++i) {
        auto coord = tCrC(i);
        int m_global = m_start + get<0>(coord);
        int n_global = n_start + get<1>(coord);
        if (m_global < C_in && n_global < C_out) {
          float val = float(accum(i));
          if constexpr (sizeof(ElementOutput) == 4) {
            atomicAdd(reinterpret_cast<float *>(&out_ptr[m_global * C_out + n_global]), val);
          } else {
            atomicAdd(reinterpret_cast<__half *>(&out_ptr[m_global * C_out + n_global]),
                      __float2half(val));
          }
        }
      }
    }
  }

private:
  // Cached A loader: reads pre-resolved in_row from smem instead of __ldg.
  // prescan_tile() already resolved mask_argsort → pair_table → in_row.
  template <class SmemTensor>
  __device__ void _load_A_cached(const ElementInput *ptr_A,
                                 const SharedStorage &storage,
                                 SmemTensor smem_tile,
                                 int m_start,
                                 int N_in,
                                 int C_in) const {
    constexpr int m_vecs = tM / kVec;
    constexpr int total_work = tK * m_vecs;
    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_work; idx += MaxThreadsPerBlock) {
      int pair_local = idx / m_vecs;
      int mv = idx % m_vecs;
      int m_local = mv * kVec;
      int m_global = m_start + m_local;
      int in_row = storage.cached_in_rows[pair_local];
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(m_local, pair_local));
      bool valid = (in_row >= 0) && (in_row < N_in) && (m_global + kVec <= C_in);
      if (valid) {
        const void *src = &ptr_A[in_row * C_in + m_global];
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                     "l"(src),
                     "n"(16));
      } else {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                     "l"(ptr_A),
                     "n"(16),
                     "r"(0));
      }
    }
  }

  // Cached B loader: reads pre-resolved real_row from smem.
  template <class SmemTensor>
  __device__ void _load_B_cached(const ElementInput *ptr_B,
                                 const SharedStorage &storage,
                                 SmemTensor smem_tile,
                                 int n_start,
                                 int C_out) const {
    constexpr int n_vecs = tN / kVec;
    constexpr int total_vecs = tK * n_vecs;
    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int pair_local = idx / n_vecs;
      int nv = idx % n_vecs;
      int n_local = nv * kVec;
      int n_global = n_start + n_local;
      int real_row = storage.cached_real_rows[pair_local];
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, pair_local));
      bool valid = (real_row >= 0) && (n_global + kVec <= C_out);
      if (valid) {
        const void *src = &ptr_B[real_row * C_out + n_global];
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                     "l"(src),
                     "n"(16));
      } else {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                     "l"(ptr_B),
                     "n"(16),
                     "r"(0));
      }
    }
  }
};
}  // namespace cute_gemm
}  // namespace warpconvnet
