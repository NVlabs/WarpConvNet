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

// Dgrad kernel: dX[j,:] = sum_k dY[rev_pt[k,j],:] @ W[k,:,:]^T
// C_in, C_out in CONV semantics. Contraction over C_out, output dim C_in.
template <class TileConfig, typename ElementOutput_ = float, int MaskWords_ = 1>
struct MaskGemm_dgrad_64x128x32_1s_flat_direpi {
  static constexpr int MaskWords = MaskWords_;
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;
  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  // Dgrad B: weight read with transposed strides -> K(C_out) contiguous.
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomA;

  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr int tM = cute::size<0>(TileShape{});  // N_in rows
  static constexpr int tN = cute::size<1>(TileShape{});  // C_in channels (output)
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int tK = cute::size<2>(TileShape{});  // C_out contraction
  static constexpr int NumStages = 1;
  static constexpr int NumWarps = MaxThreadsPerBlock / 32;
  static constexpr int kVec = 16 / sizeof(ElementInput);
  static constexpr int kMmaK = cute::size<2>(typename TiledMma::AtomShape_MNK{});
  static constexpr int K_BLOCK_MAX_STATIC = tK / kMmaK;
  static constexpr bool UseSmemEpilogue = false;
  static constexpr bool UseScalarEpilogue = false;
  static constexpr int SplitOffsets = 1;

  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::Int<tM>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::Int<tN>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));

  // GmemTiledCopyB — same as forward (N-outer, N-vec values).
  // The dgrad B load bypasses this and uses manual cpasync + CuTe-addressed smem.
  // This declaration only exists so the mainloop template parameter compiles.
  using GmemTiledCopyB = decltype(cute::make_tiled_copy(
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementInput>{},
      cute::Layout<cute::Shape<cute::Int<tN / kVec>, cute::Int<MaxThreadsPerBlock / (tN / kVec)>>,
                   cute::Stride<cute::_1, cute::Int<tN / kVec>>>{},
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

  // ============================================================================
  // INTEGRATOR CONTRACT — read this before calling.
  // ============================================================================
  //
  // Pass weight in its NATIVE layout [K, G, C_in, C_out]. Do NOT transpose.
  //
  //   ✓ correct:    weight = torch.randn(K, G, C_in, C_out)
  //                 _C.mask_gemm_dgrad(grad_output, weight.contiguous(), ...)
  //
  //   ✗ wrong:      weight_T = weight.transpose(-2, -1).contiguous()
  //                 _C.mask_gemm_dgrad(grad_output, weight_T, ...)
  //
  // The wrong call gives output of the right MAGNITUDE but ~zero correlation
  // with the true gradient (rdiff ~1.4 in practice). The kernel header field
  // `BIsTransposed = true` describes the SMEM layout, not the gmem layout
  // expected from the caller — the loader (_load_B_tile) walks K_dim=C_out
  // and treats N_dim=C_in as the row index, which matches the un-transposed
  // [K, G, C_in, C_out] memory layout exactly.
  //
  // History: this contract was previously stated only in a single inline
  // parameter comment ("weight [K, G, C_in, C_out] — NOT transposed") which
  // an integrator missed. The result was a numerical regression that broke
  // ScanNet training (warpconvnet val/miou collapse from 53% to 5%, run
  // y26vckgf). Fixed in warpconvnet commit 0e98dd7d. See
  // notes/2026_04_17_DGRAD_NUMERICAL_BUG.md and
  // tests/test_dgrad_correctness.py for the regression guard.
  //
  // ============================================================================
  __device__ void operator()(
      const ElementInput *ptr_A_base,  // grad_output [N_out, C_out_total]
      // PASS WEIGHT UN-TRANSPOSED — see contract above.
      const ElementInput *ptr_B_base,  // weight [K, G, C_in, C_out] — NOT transposed
      ElementOutput *ptr_D_base,       // grad_input [N_in, C_in_total]
      const int *pair_table,           // reverse_pair_table [K * N_in]
      const uint32_t *pair_mask,       // reverse_pair_mask [N_in]
      const int *mask_argsort,         // reverse_mask_argsort [N_in]
      int N_in,
      int N_out,
      int C_in,   // per-group output dim (grad_input channels)
      int C_out,  // per-group contraction dim (grad_output channels)
      int K,
      float alpha,
      int stride_A,             // full row stride of grad_output (C_out_total)
      int stride_D,             // full row stride of grad_input (C_in_total)
      int /*identity_offset*/,  // unused in dgrad (shared launch wrapper with fwd)
      char *smem_buf) const {
    using namespace cute;
    // Entry-gate: in debug builds, assert K fits in MaskWords*32.
    // Release: no-op (see templates.py WARPGEMM_MW_ASSERT_ENTRY macro).
    WARPGEMM_MW_ASSERT_ENTRY(K);
    // Group conv: blockIdx.z selects group, offset A/B/D pointers
    int group_id = int(blockIdx.z);
    const ElementInput *ptr_A = ptr_A_base + group_id * C_out;
    const ElementInput *ptr_B = ptr_B_base + group_id * C_in * C_out;
    ElementOutput *ptr_D = ptr_D_base + group_id * C_in;
    // Weight K-stride: for [K, G, Cig, Cog] layout, stride between k and k+1 is G*Cig*Cog
    int stride_B_K = int(gridDim.z) * C_in * C_out;

    // Grid: ceil(N_in/tM) * ceil(C_in/tN) blocks
    int grid_n = (C_in + tN - 1) / tN;
    int m_tile = int(blockIdx.x) / grid_n;
    int n_tile = int(blockIdx.x) % grid_n;
    int m_start = m_tile * tM;
    int n_start = n_tile * tN;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    // Mask union: N_out here = N_in (the dgrad output rows, mask array size)
    // The mask_union code uses N_out as the bound — rename for clarity.
    int N_out_mask = N_in;  // mask arrays are sized N_in
#define N_out N_out_mask

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

#undef N_out

    // MMA setup
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
    bool n_full_tile = (n_start + tN <= C_in);

    AIteratorState a_state;
    _init_A_iterator(a_state);

    // --- Dgrad offset loop: contract over C_out (multi-word mask) ---
    int num_k_tiles = (C_out + tK - 1) / tK;
    int _offset_idx = 0;
    int split_y = (SplitOffsets > 1) ? int(blockIdx.y) : 0;
    while (_mw_has_next()) {
      int k = _mw_next();

      if (SplitOffsets > 1 && (_offset_idx % SplitOffsets) != split_y) {
        ++_offset_idx;
        continue;
      }

      const ElementInput *ptr_Bk = ptr_B + k * stride_B_K;

      _update_A_indices(a_state,
                        pair_table,
                        storage.real_rows,
                        storage.row_masks,
                        N_out,
                        N_in,
                        C_out,
                        k,
                        K,
                        stride_A);

      // ---- Flat mainloop ----
      if (num_k_tiles == 1) {
        int smem_stage = _offset_idx & 1;
        _load_A_with_offsets(ptr_A, a_state, sA(_, _, smem_stage), 0, C_out, stride_A);
        _load_B_tile(
            ptr_Bk, sB(_, _, smem_stage), gmem_thr_copy_B, n_start, 0, C_in, C_out, n_full_tile);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();
        MMA_DOUBLE_BUFFERED(smem_stage)
      } else {
        for (int kt = 0; kt < num_k_tiles; ++kt) {
          int k_start = kt * tK;
          int smem_stage = kt & 1;
          _load_A_with_offsets(ptr_A, a_state, sA(_, _, smem_stage), k_start, C_out, stride_A);
          _load_B_tile(ptr_Bk,
                       sB(_, _, smem_stage),
                       gmem_thr_copy_B,
                       n_start,
                       k_start,
                       C_in,
                       C_out,
                       n_full_tile);
          cute::cp_async_fence();
          cute::cp_async_wait<0>();
          __syncthreads();
          MMA_DOUBLE_BUFFERED(smem_stage)
          __syncthreads();
        }
      }
      ++_offset_idx;
    }

    // Epilogue: scatter to grad_input [N_in, C_in]
    if constexpr (UseScalarEpilogue) {
      _epilogue_scalar(
          accum, ptr_D, mask_argsort, m_start, n_start, N_in, C_in, alpha, tiled_mma, stride_D);
    } else if constexpr (SplitOffsets > 1) {
      _epilogue_atomic(
          accum, ptr_D, mask_argsort, m_start, n_start, N_in, C_in, alpha, tiled_mma, stride_D);
    } else {
      _epilogue_direct(accum,
                       ptr_D,
                       mask_argsort,
                       m_start,
                       n_start,
                       N_in,
                       C_in,
                       alpha,
                       tiled_mma,
                       smem_buf,
                       stride_D);
    }
  }

private:
#include "warpgemm_dgrad_helpers.cuh"
};
}  // namespace cute_gemm
}  // namespace warpconvnet
