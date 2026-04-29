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
struct MaskGemm_forward_64x128x32_2s_fused {
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

    // --- Fused offset loop: cp.async between MMA k-blocks ---
    int num_k_tiles = (C_in + tK - 1) / tK;
    if (!_mw_has_next()) goto epilogue;

    {
      int smem_stage = 0;

      // Prolog: load first offset
      int k_cur = _mw_next();
      const ElementInput *ptr_Bk_cur = ptr_B + k_cur * stride_B_K;
      _update_A_indices(a_state,
                        pair_table,
                        storage.real_rows,
                        storage.row_masks,
                        N_in,
                        N_out,
                        C_in,
                        k_cur,
                        K,
                        stride_A);
      _load_A_with_offsets(ptr_A, a_state, sA(_, _, 0), 0, C_in, stride_A);
      _load_B_tile(ptr_Bk_cur, sB(_, _, 0), gmem_thr_copy_B, n_start, 0, C_out, C_in, n_full_tile);
      cute::cp_async_fence();
      cute::cp_async_wait<0>();
      __syncthreads();

      // For multi-k-tile offsets: process all but last k-tile with flat pattern
      if (num_k_tiles > 1) {
        for (int kt = 0; kt < num_k_tiles - 1; ++kt) {
          MMA_DOUBLE_BUFFERED(0)
          __syncthreads();
          int next_k = (kt + 1) * tK;
          _load_A_with_offsets(ptr_A, a_state, sA(_, _, 0), next_k, C_in, stride_A);
          _load_B_tile(
              ptr_Bk_cur, sB(_, _, 0), gmem_thr_copy_B, n_start, next_k, C_out, C_in, n_full_tile);
          cute::cp_async_fence();
          cute::cp_async_wait<0>();
          __syncthreads();
        }
      }

      // Now smem stage 0 has the LAST k-tile of offset 0, ready for fused MMA+load
      while (true) {
        if (_mw_has_next()) {
          // Fused: MMA on current stage with cp.async for next offset interleaved
          int k_next = _mw_next();
          int write_stage = 1 - smem_stage;

          // Resolve indices for next offset
          _update_A_indices(a_state,
                            pair_table,
                            storage.real_rows,
                            storage.row_masks,
                            N_in,
                            N_out,
                            C_in,
                            k_next,
                            K,
                            stride_A);
          const ElementInput *ptr_Bk_next = ptr_B + k_next * stride_B_K;

          // Fused MMA + interleaved cp.async for next offset
          // Issue cp.async loads for NEXT offset's A and B data BETWEEN k-blocks
          // of current MMA. Both A and B loads are chunked per kblock so memory
          // traffic is distributed evenly across tensor core execution.
          //
          // Schedule (K_BLOCK_MAX=4 / k=8 example):
          //   smem→reg(kb=0,1)  gemm(kb=0)  A_kblock(0) + B_kblock(0)
          //   smem→reg(kb=2,3)  gemm(kb=1)  A_kblock(1) + B_kblock(1)
          //   gemm(kb=2)  A_kblock(2) + B_kblock(2)
          //   gemm(kb=3)  A_kblock(3) + B_kblock(3) + fence
          //   wait + sync
          {
            int ws = 1 - smem_stage;  // write stage for next offset

            // Initial smem→reg prefetch for kb=0 and kb=1
            copy(smem_tiled_copy_A, tCsA(_, _, 0, smem_stage), tCrA_copy_0(_, _, 0));
            copy(smem_tiled_copy_B, tCsB(_, _, 0, smem_stage), tCrB_copy_0(_, _, 0));
            if (K_BLOCK_MAX > 1) {
              copy(smem_tiled_copy_A, tCsA(_, _, 1, smem_stage), tCrA_copy_1(_, _, 1));
              copy(smem_tiled_copy_B, tCsB(_, _, 1, smem_stage), tCrB_copy_1(_, _, 1));
            }

            // --- kb=0: MMA + A_kblock(0) + B load ---
            cute::gemm(tiled_mma, tCrA_0(_, _, 0), tCrB_0(_, _, 0), accum);
            _load_A_kblock<decltype(sA(_, _, 0)), 0>(
                ptr_A, a_state, sA(_, _, ws), 0, C_in, stride_A);
            _load_B_tile(
                ptr_Bk_next, sB(_, _, ws), gmem_thr_copy_B, n_start, 0, C_out, C_in, n_full_tile);

            // --- kb=1: MMA + A_kblock(1) ---
            if (K_BLOCK_MAX > 1) {
              if (K_BLOCK_MAX > 2) {
                copy(smem_tiled_copy_A, tCsA(_, _, 2, smem_stage), tCrA_copy_0(_, _, 2));
                copy(smem_tiled_copy_B, tCsB(_, _, 2, smem_stage), tCrB_copy_0(_, _, 2));
              }
              cute::gemm(tiled_mma, tCrA_1(_, _, 1), tCrB_1(_, _, 1), accum);
              if (K_BLOCK_MAX_STATIC > 1) {
                _load_A_kblock<decltype(sA(_, _, 0)), 1>(
                    ptr_A, a_state, sA(_, _, ws), 0, C_in, stride_A);
              }
            }
            // --- kb=2: MMA + A_kblock(2) ---
            if (K_BLOCK_MAX > 2) {
              if (K_BLOCK_MAX > 3) {
                copy(smem_tiled_copy_A, tCsA(_, _, 3, smem_stage), tCrA_copy_1(_, _, 3));
                copy(smem_tiled_copy_B, tCsB(_, _, 3, smem_stage), tCrB_copy_1(_, _, 3));
              }
              cute::gemm(tiled_mma, tCrA_0(_, _, 2), tCrB_0(_, _, 2), accum);
              if (K_BLOCK_MAX_STATIC > 2) {
                _load_A_kblock<decltype(sA(_, _, 0)), 2>(
                    ptr_A, a_state, sA(_, _, ws), 0, C_in, stride_A);
              }
            }
            // --- kb=3: MMA + A_kblock(3) ---
            if (K_BLOCK_MAX > 3) {
              cute::gemm(tiled_mma, tCrA_1(_, _, 3), tCrB_1(_, _, 3), accum);
              if (K_BLOCK_MAX_STATIC > 3) {
                _load_A_kblock<decltype(sA(_, _, 0)), 3>(
                    ptr_A, a_state, sA(_, _, ws), 0, C_in, stride_A);
              }
            }

            cute::cp_async_fence();
            cute::cp_async_wait<0>();
            __syncthreads();
          }
          smem_stage = 1 - smem_stage;
          ptr_Bk_cur = ptr_Bk_next;

          // Process remaining k-tiles of this offset (flat, no interleaving)
          if (num_k_tiles > 1) {
            for (int kt = 0; kt < num_k_tiles - 1; ++kt) {
              MMA_DOUBLE_BUFFERED(smem_stage)
              __syncthreads();
              int next_k = (kt + 1) * tK;
              _load_A_with_offsets(ptr_A, a_state, sA(_, _, smem_stage), next_k, C_in, stride_A);
              _load_B_tile(ptr_Bk_cur,
                           sB(_, _, smem_stage),
                           gmem_thr_copy_B,
                           n_start,
                           next_k,
                           C_out,
                           C_in,
                           n_full_tile);
              cute::cp_async_fence();
              cute::cp_async_wait<0>();
              __syncthreads();
            }
          }
        } else {
          // Last offset: just MMA, no more loads
          MMA_DOUBLE_BUFFERED(smem_stage)
          break;
        }
      }
    }
  epilogue:

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
