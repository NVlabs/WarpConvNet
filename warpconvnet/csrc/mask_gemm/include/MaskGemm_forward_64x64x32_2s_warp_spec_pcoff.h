// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cassert>

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

// Debug-only entry-gate: in debug builds, block if K exceeds the MaskWords
// capacity — this is the silent-zero condition from the 2026-04-17 SILENT_ZERO
// bug (see notes/2026_04_17_SILENT_ZERO_BUG_STORY.md). One thread per launch
// prints. Zero cost in release builds (compiles to ((void)0)).
// Guarded because every generated header emits this block; without the guard
// we'd get redefinition warnings when a .cu includes multiple kernel headers.
#ifndef WARPGEMM_MW_ASSERT_ENTRY
#if !defined(NDEBUG)
#define WARPGEMM_MW_ASSERT_ENTRY(K_runtime_)                                         \
  do {                                                                               \
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { \
      assert((K_runtime_) <= int(MaskWords) * 32 &&                                  \
             "K exceeds MaskWords*32 - kernel will silently skip offsets");          \
    }                                                                                \
  } while (0)
#else
#define WARPGEMM_MW_ASSERT_ENTRY(K_runtime_) ((void)0)
#endif
#endif  // WARPGEMM_MW_ASSERT_ENTRY

template <class TileConfig, typename ElementOutput_ = float, int MaskWords_ = 1>
struct MaskGemm_forward_64x64x32_2s_warp_spec_pcoff {
  static constexpr int MaskWords = MaskWords_;
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;
  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;

  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr int tM = cute::size<0>(TileShape{});
  static constexpr int tN = cute::size<1>(TileShape{});
  static constexpr int tK = cute::size<2>(TileShape{});
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int NumStages = 2;
  static constexpr int NumWarps = MaxThreadsPerBlock / 32;
  static constexpr int kVec = 16 / sizeof(ElementInput);
  static constexpr int kMmaK = cute::size<2>(typename TiledMma::AtomShape_MNK{});
  static constexpr int K_BLOCK_MAX_STATIC = tK / kMmaK;
  static constexpr bool UseSmemEpilogue = true;
  static constexpr bool UseScalarEpilogue = false;

  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::Int<tM>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::Int<tN>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));

  static constexpr int CopyBNThreads = tN / kVec;
  static constexpr int CopyBKThreads = MaxThreadsPerBlock / CopyBNThreads;
  using GmemTiledCopyB = decltype(cute::make_tiled_copy(
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementInput>{},
      cute::Layout<cute::Shape<cute::Int<CopyBNThreads>, cute::Int<CopyBKThreads>>,
                   cute::Stride<cute::_1, cute::Int<CopyBNThreads>>>{},
      cute::Layout<cute::Shape<cute::Int<kVec>, cute::_1>>{}));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
    uint32_t warp_masks[NumWarps * MaskWords];
    int real_rows[tM];
    uint32_t row_masks[tM * MaskWords];
    int precomputed_in_rows[MaskWords * 32 * tM];
  };

  static constexpr size_t EpilogueSmemSize = UseSmemEpilogue ? tM * (tN + 8) * 2 : 0;
  static constexpr size_t SharedStorageSize = sizeof(SharedStorage) > EpilogueSmemSize
                                                  ? sizeof(SharedStorage)
                                                  : EpilogueSmemSize;

  __device__ void operator()(const ElementInput *ptr_A_base,
                             const ElementInput *ptr_B_base,
                             ElementOutput *ptr_D_base,
                             const int *pair_table,
                             const uint32_t *pair_mask,
                             const int *mask_argsort,
                             int N_in,
                             int N_out,
                             int C_in,
                             int C_out,
                             int K,
                             float alpha,
                             int stride_A,
                             int stride_D,
                             int identity_offset,
                             char *smem_buf) const {
    using namespace cute;
    // Entry-gate: in debug builds, assert K fits in MaskWords*32.
    // Release: no-op (see templates.py WARPGEMM_MW_ASSERT_ENTRY macro).
    WARPGEMM_MW_ASSERT_ENTRY(K);
    // Group conv: blockIdx.z selects group, offset A/B/D pointers
    int group_id = int(blockIdx.z);
    const ElementInput *ptr_A = ptr_A_base + group_id * C_in;
    const ElementInput *ptr_B = ptr_B_base + group_id * C_in * C_out;
    ElementOutput *ptr_D = ptr_D_base + group_id * C_out;
    // Weight K-stride: for [K, G, Cig, Cog] layout, stride between k and k+1 is G*Cig*Cog
    int stride_B_K = int(gridDim.z) * C_in * C_out;

    int grid_n = (C_out + tN - 1) / tN;
    int m_tile = int(blockIdx.x) / grid_n;
    int n_tile = int(blockIdx.x) % grid_n;
    int m_start = m_tile * tM;
    int n_start = n_tile * tN;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    // Warp-shuffle mask union + precompute row info
    // MaskWords=1: single uint32 active_offsets (original fast path).
    // MaskWords>1: array of uint32 words for K>32.
    uint32_t active_offsets_arr[MaskWords];
    if (K == 1) {
      for (int _w = 0; _w < MaskWords; ++_w) active_offsets_arr[_w] = 0;
      active_offsets_arr[0] = 1;
      CUTLASS_PRAGMA_UNROLL
      for (int m_local = threadIdx.x; m_local < tM; m_local += MaxThreadsPerBlock) {
        int sorted_row = m_start + m_local;
        if (sorted_row < N_out) {
          int rr = __ldg(&mask_argsort[sorted_row]);
          storage.real_rows[m_local] = rr;
          storage.row_masks[m_local * MaskWords] = 1;
          for (int _w = 1; _w < MaskWords; ++_w) storage.row_masks[m_local * MaskWords + _w] = 0;
        } else {
          storage.real_rows[m_local] = -1;
          for (int _w = 0; _w < MaskWords; ++_w) storage.row_masks[m_local * MaskWords + _w] = 0;
        }
      }
      __syncthreads();
    } else {
      int warp_id = threadIdx.x / 32;
      int lane_id = threadIdx.x % 32;
      uint32_t my_mask[MaskWords];
      for (int _w = 0; _w < MaskWords; ++_w) my_mask[_w] = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int m_local = threadIdx.x; m_local < tM; m_local += MaxThreadsPerBlock) {
        int sorted_row = m_start + m_local;
        if (sorted_row < N_out) {
          int rr = __ldg(&mask_argsort[sorted_row]);
          storage.real_rows[m_local] = rr;
          for (int _w = 0; _w < MaskWords; ++_w) {
            uint32_t rm = __ldg(&pair_mask[rr * MaskWords + _w]);
            storage.row_masks[m_local * MaskWords + _w] = rm;
            my_mask[_w] |= rm;
          }
        } else {
          storage.real_rows[m_local] = -1;
          for (int _w = 0; _w < MaskWords; ++_w) storage.row_masks[m_local * MaskWords + _w] = 0;
        }
      }
      for (int _w = 0; _w < MaskWords; ++_w) {
        CUTLASS_PRAGMA_UNROLL
        for (int s = 16; s >= 1; s >>= 1) {
          my_mask[_w] |= __shfl_xor_sync(0xffffffff, my_mask[_w], s);
        }
        if (lane_id == 0) storage.warp_masks[warp_id * MaskWords + _w] = my_mask[_w];
      }
      __syncthreads();
      for (int _w = 0; _w < MaskWords; ++_w) {
        active_offsets_arr[_w] = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int _w2 = 0; _w2 < NumWarps; ++_w2)
          active_offsets_arr[_w] |= storage.warp_masks[_w2 * MaskWords + _w];
      }
    }
    // Alias for backward compatibility with existing mainloop code
    uint32_t &active_offsets = active_offsets_arr[0];

    // Multi-word offset iterator: extracts offsets one at a time across words.
    int _mw_iter_word = 0;
    uint32_t _mw_iter_bits = active_offsets_arr[0];
    auto _mw_has_next = [&]() {
      while (!_mw_iter_bits && _mw_iter_word + 1 < MaskWords) {
        ++_mw_iter_word;
        _mw_iter_bits = active_offsets_arr[_mw_iter_word];
      }
      return _mw_iter_bits != 0;
    };
    auto _mw_next = [&]() {
      int bit = __ffs(_mw_iter_bits) - 1;
      _mw_iter_bits &= _mw_iter_bits - 1;
      return _mw_iter_word * 32 + bit;
    };

    // E1 offset-index precomputation — fill storage.precomputed_in_rows[k][m]
    // with pair_table[k * N_out + real_rows[m]] (or -1 if inactive / invalid).
    // Collaborative: MaxThreadsPerBlock threads split K*tM entries.
    {
      const int _km_total = MaskWords * 32 * tM;
      CUTLASS_PRAGMA_UNROLL
      for (int _i = threadIdx.x; _i < _km_total; _i += MaxThreadsPerBlock) {
        int _k_pc = _i / tM;
        int _m_pc = _i % tM;
        int _in_row = -1;
        if (_k_pc < K) {
          int _rr_pc = storage.real_rows[_m_pc];
          bool _active =
              (storage.row_masks[_m_pc * MaskWords + _k_pc / 32] & (1u << (_k_pc % 32))) != 0;
          if (_rr_pc >= 0 && _active) {
            _in_row = __ldg(&pair_table[_k_pc * N_out + _rr_pc]);
          }
        }
        storage.precomputed_in_rows[_i] = _in_row;
      }
      __syncthreads();
    }

    // --- MMA setup ---
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
    GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(threadIdx.x);
    bool n_full_tile = (n_start + tN <= C_out);

    AIteratorState a_state;
    _init_A_iterator(a_state);

    // --- Offset loop (iterate active offsets via __ffs, multi-word mask) ---
    int num_k_tiles = (C_in + tK - 1) / tK;

    // --- Submanifold identity shortcut ---
    // For odd submanifold conv (3x3x3, 5x5x5), offset K/2 maps each voxel to itself.
    // Process as dense A load (real_rows[m] IS the input row) to avoid pair_table.
    // identity_offset < 0 disables (non-submanifold or even kernel sizes).
    if (identity_offset >= 0) {
      int iden_k = identity_offset;
      int iden_word = iden_k / 32;
      uint32_t iden_bit = 1u << (iden_k % 32);
      if (active_offsets_arr[iden_word] & iden_bit) {
        active_offsets_arr[iden_word] &= ~iden_bit;  // remove from offset loop
        const ElementInput *ptr_Bk = ptr_B + iden_k * stride_B_K;
        // Dense A load: input row = real_rows[m] (identity mapping, no pair_table)
        _load_A_identity(ptr_A, storage.real_rows, sA(_, _, 0), 0, N_in, C_in, stride_A);
        _load_B_tile(ptr_Bk, sB(_, _, 0), gmem_thr_copy_B, n_start, 0, C_out, C_in, n_full_tile);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();
        MMA_DOUBLE_BUFFERED(0)
        __syncthreads();
        // Multi k-tile identity (C_in > tK)
        for (int kt = 1; kt < num_k_tiles; ++kt) {
          int ks = kt * tK;
          _load_A_identity(ptr_A, storage.real_rows, sA(_, _, 0), ks, N_in, C_in, stride_A);
          _load_B_tile(ptr_Bk, sB(_, _, 0), gmem_thr_copy_B, n_start, ks, C_out, C_in, n_full_tile);
          cute::cp_async_fence();
          cute::cp_async_wait<0>();
          __syncthreads();
          MMA_DOUBLE_BUFFERED(0)
          __syncthreads();
        }
      }
    }  // identity_offset guard

    int _offset_idx = 0;
    for (int _mw = 0; _mw < MaskWords; ++_mw) {
      uint32_t _word = active_offsets_arr[_mw];
      while (_word) {
        int k = _mw * 32 + __ffs(_word) - 1;
        _word &= _word - 1;

        const ElementInput *ptr_Bk = ptr_B + k * stride_B_K;

        _update_A_indices_from_smem(a_state, storage.precomputed_in_rows, N_in, C_in, k, stride_A);

        // ---- Overlapped pipeline: all warps load + compute, 2-stage smem ----
        if (num_k_tiles == 1) {
          // Single k-tile: simple load-wait-compute
          _load_A_with_offsets(ptr_A, a_state, sA(_, _, 0), 0, C_in, stride_A);
          _load_B_tile(ptr_Bk, sB(_, _, 0), gmem_thr_copy_B, n_start, 0, C_out, C_in, n_full_tile);
          cute::cp_async_fence();
          cute::cp_async_wait<0>();
          __syncthreads();
          MMA_DOUBLE_BUFFERED(0)
          __syncthreads();
        } else {
          // Multi k-tile: 2-stage pipeline
          // Prolog: load k-tile 0 into stage 0
          _load_A_with_offsets(ptr_A, a_state, sA(_, _, 0), 0, C_in, stride_A);
          _load_B_tile(ptr_Bk, sB(_, _, 0), gmem_thr_copy_B, n_start, 0, C_out, C_in, n_full_tile);
          cute::cp_async_fence();
          cute::cp_async_wait<0>();
          __syncthreads();

          int smem_read = 0;
          for (int kt = 1; kt < num_k_tiles; ++kt) {
            int smem_write = 1 - smem_read;
            int k_next = kt * tK;

            // Launch async load for NEXT k-tile FIRST (overlaps with MMA below)
            _load_A_with_offsets(ptr_A, a_state, sA(_, _, smem_write), k_next, C_in, stride_A);
            _load_B_tile(ptr_Bk,
                         sB(_, _, smem_write),
                         gmem_thr_copy_B,
                         n_start,
                         k_next,
                         C_out,
                         C_in,
                         n_full_tile);
            cute::cp_async_fence();

            // MMA on current stage (all 4 warps)
            MMA_DOUBLE_BUFFERED(smem_read)

            // Wait for next stage to be ready
            cute::cp_async_wait<0>();
            __syncthreads();
            smem_read = smem_write;
          }

          // Epilog: MMA on last loaded k-tile
          MMA_DOUBLE_BUFFERED(smem_read)
          __syncthreads();
        }
        ++_offset_idx;
      }
    }

    // --- Epilogue ---
    if constexpr (UseScalarEpilogue) {
      _epilogue_scalar(
          accum, ptr_D, mask_argsort, m_start, n_start, N_out, C_out, alpha, tiled_mma, stride_D);
    } else {
      _epilogue_direct(accum,
                       ptr_D,
                       mask_argsort,
                       m_start,
                       n_start,
                       N_out,
                       C_out,
                       alpha,
                       tiled_mma,
                       smem_buf,
                       stride_D);
    }
  }

private:
#include "warpgemm_fwd_helpers.cuh"
};
}  // namespace cute_gemm
}  // namespace warpconvnet
