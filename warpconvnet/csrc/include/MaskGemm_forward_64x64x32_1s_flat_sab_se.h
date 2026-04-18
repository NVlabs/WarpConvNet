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
struct MaskGemm_forward_64x64x32_1s_flat_sab_se {
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
  static constexpr int NumStages = 1;
  static constexpr int NumWarps = MaxThreadsPerBlock / 32;
  static constexpr int kVec = 16 / sizeof(ElementInput);
  static constexpr int kMmaK = cute::size<2>(typename TiledMma::AtomShape_MNK{});
  static constexpr int K_BLOCK_MAX_STATIC = tK / kMmaK;
  static constexpr bool UseSmemEpilogue = true;
  static constexpr bool UseScalarEpilogue = true;

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

        _update_A_indices(a_state,
                          pair_table,
                          storage.real_rows,
                          storage.row_masks,
                          N_in,
                          N_out,
                          C_in,
                          k,
                          K,
                          stride_A);

        // ---- Pipelined flat with offset-alternating stages ----
        // Stage alternates per offset (not per k-tile) so consecutive offsets
        // use different smem buffers. This eliminates 1 __syncthreads per offset
        // for single-k-tile cases (common at small C).
        if (num_k_tiles == 1) {
          // Single k-tile: alternate stage per offset, skip post-MMA sync
          int smem_stage = _offset_idx & 1;
          _load_A_with_offsets(ptr_A, a_state, sA(_, _, smem_stage), 0, C_in, stride_A);
          _load_B_tile(
              ptr_Bk, sB(_, _, smem_stage), gmem_thr_copy_B, n_start, 0, C_out, C_in, n_full_tile);
          cute::cp_async_fence();
          cute::cp_async_wait<0>();
          __syncthreads();
          MMA_DOUBLE_BUFFERED(smem_stage)
          // No __syncthreads here — next offset uses the OTHER stage
        } else {
          // Multi k-tile: stage alternates per k-tile within offset
          for (int kt = 0; kt < num_k_tiles; ++kt) {
            int k_start_cin = kt * tK;
            int smem_stage = kt & 1;
            _load_A_with_offsets(ptr_A, a_state, sA(_, _, smem_stage), k_start_cin, C_in, stride_A);
            _load_B_tile(ptr_Bk,
                         sB(_, _, smem_stage),
                         gmem_thr_copy_B,
                         n_start,
                         k_start_cin,
                         C_out,
                         C_in,
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
  // ---- Scalar A loader (for non-kVec-aligned C_in) ----
  static constexpr int k_vecs_A = tK / kVec;
  static constexpr int total_vecs_A = tM * k_vecs_A;
  static constexpr int kItersPerThread =
      (total_vecs_A + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;

  static constexpr int kWarpKVecs = k_vecs_A;
  static constexpr int kWarpRows = 32 / kWarpKVecs;  // unique M-rows per warp per iter
  static constexpr int kRowDelta = kWarpRows;
  static constexpr int kRowsPerWarp = kWarpRows * kItersPerThread;

  struct AIteratorState {
    int row[kItersPerThread];
    int k_col;
    int gmem_byte_offsets[kItersPerThread];
    bool valid[kItersPerThread];
    int in_rows[kItersPerThread];
  };

  __device__ void _init_A_iterator(AIteratorState &state) const {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int lane_row = lane_id / kWarpKVecs;
    int lane_kvec = lane_id % kWarpKVecs;
    state.k_col = lane_kvec * kVec;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kItersPerThread; ++s) {
      state.row[s] = warp_id * kRowsPerWarp + lane_row + s * kRowDelta;
      state.valid[s] = false;
      state.in_rows[s] = -1;
    }
  }

  __device__ void _update_A_indices(AIteratorState &state,
                                    const int *pair_table,
                                    int *real_rows,
                                    uint32_t *row_masks,
                                    int N_in,
                                    int N_out,
                                    int C_in,
                                    int offset_k,
                                    int K,
                                    int stride_A = 0) const {
    const int *pt_base = pair_table + offset_k * N_out;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kItersPerThread; ++s) {
      int m = state.row[s];
      if (m < tM) {
        int real_row = real_rows[m];
        uint32_t row_mask = row_masks[m];
        int in_row = -1;
        if (real_row >= 0 && (K > 32 || (row_mask & (1u << offset_k))))
          in_row = __ldg(&pt_base[real_row]);
        if (in_row >= 0 && in_row < N_in) {
          state.valid[s] = true;
          state.in_rows[s] = in_row;
        } else {
          state.valid[s] = false;
          state.in_rows[s] = -1;
        }
      } else {
        state.valid[s] = false;
        state.in_rows[s] = -1;
      }
    }
  }

  /// Scalar A tile load: element-by-element gmem reads, 16-byte smem stores.
  template <class SmemTensor>
  __device__ void _load_A_with_offsets(const ElementInput *ptr_A,
                                       const AIteratorState &state,
                                       SmemTensor smem_tile,
                                       int k_start,
                                       int C_in,
                                       int stride_A = 0) const {
    if (stride_A == 0) stride_A = C_in;
    int k_global = k_start + state.k_col;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kItersPerThread; ++s) {
      int m = state.row[s];
      if (m < tM) {
        int4 frag = make_int4(0, 0, 0, 0);
        if (state.valid[s]) {
          const ElementInput *row_ptr = ptr_A + state.in_rows[s] * stride_A;
          ElementInput *dst = reinterpret_cast<ElementInput *>(&frag);
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVec; ++v) {
            int k = k_global + v;
            if (k < C_in) dst[v] = row_ptr[k];
          }
        }
        *reinterpret_cast<int4 *>(&smem_tile(m, state.k_col)) = frag;
      }
    }
    // Synchronous stores — issue empty cp.async fence for mainloop compat
    cute::cp_async_fence();
  }

  /// Per-kblock split
  template <class SmemTensor, int KB>
  __device__ void _load_A_kblock(const ElementInput *ptr_A,
                                 const AIteratorState &state,
                                 SmemTensor smem_tile,
                                 int k_start,
                                 int C_in,
                                 int stride_A = 0) const {
    if (stride_A == 0) stride_A = C_in;
    if (KB < kItersPerThread) {
      int k_global = k_start + state.k_col;
      int m = state.row[KB];
      if (m < tM) {
        int4 frag = make_int4(0, 0, 0, 0);
        if (state.valid[KB]) {
          const ElementInput *row_ptr = ptr_A + state.in_rows[KB] * stride_A;
          ElementInput *dst = reinterpret_cast<ElementInput *>(&frag);
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVec; ++v) {
            int k = k_global + v;
            if (k < C_in) dst[v] = row_ptr[k];
          }
        }
        *reinterpret_cast<int4 *>(&smem_tile(m, state.k_col)) = frag;
      }
    }
    cute::cp_async_fence();
  }

  /// Warp-raked B load with pre-computed stride increments.
  template <class SmemTensor>
  __device__ void _load_B_warp_raked(const ElementInput *ptr_Bk,
                                     SmemTensor smem_stage,
                                     int n_start,
                                     int k_start,
                                     int C_out,
                                     int C_in) const {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    constexpr int kWarpDilN = tN / (MaxThreadsPerBlock / 32);
    constexpr int kWarpDilK = tK / kVec;

    int n_base = warp_id * kWarpDilN + (lane_id / kWarpDilK);
    int k_base = (lane_id % kWarpDilK) * kVec;

    constexpr int n_stride = 32 / kWarpDilK;
    constexpr int num_iters = tN / (n_stride * (MaxThreadsPerBlock / 32));

    int n_global = n_start + n_base;
    int k_global = k_start + k_base;
    int inc_strided_bytes = n_stride * sizeof(ElementInput);

    const char *gptr = reinterpret_cast<const char *>(&ptr_Bk[k_global * C_out + n_global]);
    // Group-conv alignment: cp.async.v4 needs 16-byte source. Misaligned when
    // ptr_Bk or C_out * sizeof(half) isn't 16-byte aligned (e.g. groups>1
    // with per-group C_out not multiple of kVec).
    bool b_aligned =
        ((reinterpret_cast<uintptr_t>(ptr_Bk) | (uintptr_t)(C_out * (int)sizeof(ElementInput))) &
         15u) == 0;

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < num_iters; ++s) {
      int n_cur = n_global + s * n_stride;
      uint32_t sa = cute::cast_smem_ptr_to_uint(&smem_stage(n_cur - n_start, k_base));
      bool ok =
          (n_cur + kVec <= n_start + tN) && (n_cur + kVec <= C_out) && (k_global + kVec <= C_in);

      if (ok && b_aligned) {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(sa),
                     "l"(gptr + s * inc_strided_bytes),
                     "n"(16));
      } else if (b_aligned) {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(sa),
                     "l"(gptr),
                     "n"(16),
                     "r"(0));
      } else {
        // Scalar fallback for misaligned source.
        int4 frag = make_int4(0, 0, 0, 0);
        if ((n_cur < C_out) && (k_global < C_in)) {
          const ElementInput *row_ptr = &ptr_Bk[k_global * C_out];
          ElementInput *dst = reinterpret_cast<ElementInput *>(&frag);
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVec; ++v) {
            int n = n_cur + v;
            if (n < C_out) dst[v] = row_ptr[n];
          }
        }
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(sa),
                     "r"(frag.x),
                     "r"(frag.y),
                     "r"(frag.z),
                     "r"(frag.w));
      }
    }
  }

  static constexpr bool BIsTransposed = false;

  template <class SmemTensor, class GmemThrCopyB>
  __device__ void _load_B_tile(const ElementInput *ptr_Bk,
                               SmemTensor smem_stage,
                               GmemThrCopyB & /*unused*/,
                               int n_start,
                               int k_start,
                               int C_out,
                               int C_in,
                               bool /*n_full_tile*/) const {
    // Scalar B load: N(C_out) contiguous. Element-by-element reads.
    constexpr int n_vecs = tN / kVec;
    constexpr int total_vecs = tK * n_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int k_local = idx / n_vecs;
      int n_local = (idx % n_vecs) * kVec;
      int n_global = n_start + n_local;
      int k_global = k_start + k_local;

      int4 frag = make_int4(0, 0, 0, 0);
      if (k_global < C_in) {
        const ElementInput *row_ptr = &ptr_Bk[k_global * C_out];
        ElementInput *dst = reinterpret_cast<ElementInput *>(&frag);
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kVec; ++v) {
          int n = n_global + v;
          if (n < C_out) dst[v] = row_ptr[n];
        }
      }
      *reinterpret_cast<int4 *>(&smem_stage(n_local, k_local)) = frag;
    }
  }

  /// Unified dense B load via cp.async, parameterized by layout.
  template <class SmemTensor, bool IsTransposed>
  __device__ void _load_dense_B_generic(const ElementInput *ptr_B,
                                        SmemTensor smem_tile,
                                        int n_start,
                                        int k_start,
                                        int N,
                                        int K_dim) const {
    static_assert(tN % kVec == 0);
    static_assert(tK % kVec == 0);
    constexpr int n_vecs = tN / kVec;
    constexpr int k_vecs = tK / kVec;
    // For N-contiguous: cp.async loads kVec elements along N. Iterate all K rows × N vectors.
    // For K-contiguous: cp.async loads kVec elements along K. Iterate all N rows × K vectors.
    constexpr int total_vecs = IsTransposed ? (tN * k_vecs) : (tK * n_vecs);
    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int k_local, n_local;
      if constexpr (!IsTransposed) {
        // Forward (N-contiguous): K outer (step=1), N-vec inner (step=kVec)
        k_local = idx / n_vecs;
        n_local = (idx % n_vecs) * kVec;
      } else {
        // Transposed (K-contiguous): N outer (step=1), K-vec inner (step=kVec)
        n_local = idx / k_vecs;
        k_local = (idx % k_vecs) * kVec;
      }
      int n_global = n_start + n_local;
      int k_global = k_start + k_local;
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));
      const void *src;
      int src_bytes;
      if constexpr (!IsTransposed) {
        // N-contiguous: vector along N, partial at N boundary
        int valid_n = (k_global < K_dim && n_global < N) ? min(kVec, N - n_global) : 0;
        src_bytes = valid_n * (int)sizeof(ElementInput);
        src = &ptr_B[k_global * N + n_global];
      } else {
        // K-contiguous: vector along K, partial at K boundary
        int valid_k = (n_global < N && k_global < K_dim) ? min(kVec, K_dim - k_global) : 0;
        src_bytes = valid_k * (int)sizeof(ElementInput);
        src = &ptr_B[n_global * K_dim + k_global];
      }
      if (src_bytes > 0) {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                     "l"(src),
                     "n"(16),
                     "r"(src_bytes));
      } else {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                     "l"(ptr_B),
                     "n"(16),
                     "r"(0));
      }
    }
  }

  /// Unified per-kblock B load for fused mainloop interleaving.
  /// Warp-raked for both N-contiguous and K-contiguous cases.
  template <class SmemTensor, int KB, bool IsTransposed = BIsTransposed>
  __device__ void _load_B_kblock(const ElementInput *ptr_B,
                                 SmemTensor smem_tile,
                                 int n_start,
                                 int k_start,
                                 int N,
                                 int K_dim) const {
    static_assert(tN % kVec == 0);
    constexpr int k_offset = KB * kMmaK;

    if constexpr (!IsTransposed) {
      // Forward (N-contiguous): each cp.async loads kVec N-elements at 1 K-row.
      // Warp-raked: K outer (step=1), N-vec inner (step=kVec).
      constexpr int n_vecs = tN / kVec;
      constexpr int total_vecs = kMmaK * n_vecs;
      CUTLASS_PRAGMA_UNROLL
      for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
        int k_local = k_offset + idx / n_vecs;
        int n_local = (idx % n_vecs) * kVec;
        int n_global = n_start + n_local;
        int k_global = k_start + k_local;
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));
        int valid_n = (k_global < K_dim && n_global < N) ? min(kVec, N - n_global) : 0;
        int src_bytes = valid_n * (int)sizeof(ElementInput);
        if (src_bytes > 0) {
          const void *src = &ptr_B[k_global * N + n_global];
          asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                       "l"(src),
                       "n"(16),
                       "r"(src_bytes));
        } else {
          asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                       "l"(ptr_B),
                       "n"(16),
                       "r"(0));
        }
      }
    } else {
      // Transposed (K-contiguous): each cp.async loads kVec K-elements at 1 N-row.
      // Warp-raked: threads in a warp cover consecutive K-vectors at same N-row.
      constexpr int k_vecs_per_block = kMmaK / kVec;
      constexpr int k_threads = k_vecs_per_block;  // threads per N-row along K
      constexpr int n_per_wave = MaxThreadsPerBlock / k_threads;
      constexpr int num_waves = (tN + n_per_wave - 1) / n_per_wave;

      int k_tid = threadIdx.x % k_threads;
      int n_tid = threadIdx.x / k_threads;

      CUTLASS_PRAGMA_UNROLL
      for (int wave = 0; wave < num_waves; ++wave) {
        int n_local = wave * n_per_wave + n_tid;
        int k_local = k_offset + k_tid * kVec;
        int n_global = n_start + n_local;
        int k_global = k_start + k_local;
        if (n_local < tN) {
          uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));
          int valid_k = (n_global < N && k_global < K_dim) ? min(kVec, K_dim - k_global) : 0;
          int src_bytes = valid_k * (int)sizeof(ElementInput);
          if (src_bytes > 0) {
            const void *src = &ptr_B[n_global * K_dim + k_global];
            asm volatile(
                "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                "l"(src),
                "n"(16),
                "r"(src_bytes));
          } else {
            asm volatile(
                "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                "l"(ptr_B),
                "n"(16),
                "r"(0));
          }
        }
      }
    }
  }

  // Legacy generic per-kblock B load (round-robin, non-raked — kept for reference).
  template <class SmemTensor, int KB, bool IsTransposed_unused = BIsTransposed>
  __device__ void _load_B_kblock_generic(const ElementInput *ptr_B,
                                         SmemTensor smem_tile,
                                         int n_start,
                                         int k_start,
                                         int N,
                                         int K_dim) const {
    static_assert(tN % kVec == 0);
    constexpr int n_vecs = tN / kVec;
    constexpr int k_offset = KB * kMmaK;
    constexpr int k_vecs = kMmaK / kVec;
    constexpr int total_vecs = IsTransposed_unused ? (tN * k_vecs) : (kMmaK * n_vecs);

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int k_local, n_local;
      if constexpr (!IsTransposed_unused) {
        k_local = k_offset + idx / n_vecs;
        n_local = (idx % n_vecs) * kVec;
      } else {
        n_local = idx / k_vecs;
        k_local = k_offset + (idx % k_vecs) * kVec;
      }
      int n_global = n_start + n_local;
      int k_global = k_start + k_local;
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));
      bool pred;
      const void *src;
      if constexpr (!IsTransposed_unused) {
        pred = (k_global < K_dim) && (n_global + kVec <= N);
        src = &ptr_B[k_global * N + n_global];
      } else {
        pred = (n_global < N) && (k_global + kVec <= K_dim);
        src = &ptr_B[n_global * K_dim + k_global];
      }
      if (pred) {
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

  // Identity A loader: for submanifold center offset (real_rows[m] IS the input row).
  // No pair_table lookup needed — simpler and faster than gathered A.
  // Same smem layout as gathered A: smem_tile(m_row, k_col) with kVec-sized cp.async.
  template <class SmemTensor>
  __device__ void _load_A_identity(const ElementInput *ptr_A,
                                   const int *real_rows,
                                   SmemTensor smem_tile,
                                   int k_start,
                                   int N_in,
                                   int C_in,
                                   int stride_A) const {
    // Thread mapping: same as gathered_a_precomputed but row = real_rows[m]
    // Each cp.async loads kVec elements along K at one M-row.
    // total_vecs = tK (K-rows) × (tM / kVec) (M-vectors)
    // Actually, A smem is (tM, tK) with K contiguous per kVec load.
    // We iterate: K-outer (step 1), M-vec-inner (step kVec)
    constexpr int k_iters = tK;        // one K-row per iteration
    constexpr int m_vecs = tM / kVec;  // kVec M-elements per vector
    constexpr int total_vecs = k_iters * m_vecs;
    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int k_local = idx / m_vecs;
      int mv = idx % m_vecs;
      int m_local = mv * kVec;
      int k_global = k_start + k_local;
      int in_row = real_rows[mv];  // identity: one real_row per kVec M-rows
      // Actually each cp.async loads kVec consecutive M-elements at one K-row.
      // A layout in gmem: A[row, C_in], contiguous along C_in (K-direction in GEMM).
      // cp.async loads kVec elements = 16 bytes from A[in_row, k_global..k_global+kVec-1]
      // But this only works if K is contiguous... For A, the "K" in the GEMM is C_in,
      // which IS contiguous in gmem. And smem stores (m_row, k_col).
      // Wait — the gathered A loader iterates M-outer, K-vec-inner with:
      //   m_local = pair_local (which M-row in the tile)
      //   k_local = byte offset along C_in
      // The smem tile is (tM, tK) but loaded as:
      //   for each thread's (pair_local, k_vec): load kVec at A[in_row, k_start + k_vec*kVec]
      // So the identity loader should use the same pattern: each thread handles one
      // (m_row, k_vec) pair and loads kVec C_in elements.
    }
    // Simplified: reuse the exact same iteration as gathered_a but with in_row = real_rows[m]
    {
      // Group-conv alignment: cp.async.v4 needs 16-byte aligned source.
      // Misaligned when groups>1 AND per-group C_in not multiple of kVec.
      int _stride_bytes = stride_A * (int)sizeof(ElementInput);
      bool ptr_A_aligned =
          ((reinterpret_cast<uintptr_t>(ptr_A) | (uintptr_t)_stride_bytes) & 15u) == 0;
      constexpr int kItersPerThread =
          (tK / kVec * tM + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kItersPerThread; ++iter) {
        int idx = threadIdx.x + iter * MaxThreadsPerBlock;
        if (idx >= tK / kVec * tM) break;
        int pair_local = idx / (tK / kVec);
        int kv = idx % (tK / kVec);
        int k_local = kv * kVec;
        int k_global = k_start + k_local;
        int in_row = real_rows[pair_local];
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(pair_local, k_local));
        // Vectorized fast path needs C_in >= kVec AND (k_global + kVec) <= C_in.
        // Unaligned tail, C_in < kVec, or group-conv misaligned base:
        // per-element scalar load with bounds check.
        bool row_valid = (in_row >= 0) && (in_row < N_in);
        if (row_valid && (k_global + kVec <= C_in) && ptr_A_aligned) {
          const void *src = &ptr_A[in_row * stride_A + k_global];
          asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                       "l"(src),
                       "n"(16));
        } else {
          int4 frag = make_int4(0, 0, 0, 0);
          if (row_valid) {
            const ElementInput *row_ptr = ptr_A + in_row * stride_A;
            ElementInput *dst = reinterpret_cast<ElementInput *>(&frag);
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kVec; ++v) {
              int k = k_global + v;
              if (k < C_in) dst[v] = row_ptr[k];
            }
          }
          asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(smem_addr),
                       "r"(frag.x),
                       "r"(frag.y),
                       "r"(frag.z),
                       "r"(frag.w));
        }
      }
    }
  }

  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_direct(Accumulator &accum,
                                   ElementOutput *ptr_D,
                                   const int *mask_argsort,
                                   int m_start,
                                   int n_start,
                                   int N_out,
                                   int C_out,
                                   float alpha,
                                   TiledMma_ &tiled_mma,
                                   char *smem_buf,
                                   int stride_out = 0) const {
    if (stride_out == 0) stride_out = C_out;
    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    if constexpr (sizeof(ElementOutput) == 2 && UseSmemEpilogue) {
      // Shared memory staging epilogue (CUTLASS pattern)
      constexpr int EPL_PAD = 8;
      constexpr int EPL_STRIDE = tN + EPL_PAD;
      __half *epi_smem = reinterpret_cast<__half *>(smem_buf);

      CUTE_UNROLL
      for (int i = 0; i < size(accum); ++i) {
        auto coord = tCrC(i);
        float val = (alpha == 1.0f) ? float(accum(i)) : alpha * float(accum(i));
        epi_smem[get<0>(coord) * EPL_STRIDE + get<1>(coord)] = __float2half(val);
      }
      __syncthreads();

      constexpr int VEC = 4;
      constexpr int COLS_PER_THREAD = tN / VEC;
      constexpr int ROWS_PER_ITER = MaxThreadsPerBlock / COLS_PER_THREAD;
      int col_group = threadIdx.x % COLS_PER_THREAD;
      int row_in_iter = threadIdx.x / COLS_PER_THREAD;
      int col_start = col_group * VEC;

      // Group-conv alignment: uint64 stores need 8-byte alignment of the
      // target. ptr_D = ptr_D_base + group_id * C_per_group; if that shift
      // (plus stride_out*2) isn't 8-byte aligned, uint64 stores misalign.
      // Fall back to scalar writes in that case.
      bool epi_aligned = ((reinterpret_cast<uintptr_t>(ptr_D) |
                           (uintptr_t)(stride_out * (int)sizeof(ElementOutput))) &
                          7u) == 0;

#pragma unroll
      for (int iter = 0; iter < tM; iter += ROWS_PER_ITER) {
        int row = iter + row_in_iter;
        if (row < tM) {
          int sorted_row = m_start + row;
          if (sorted_row < N_out && col_start + n_start + VEC <= C_out && epi_aligned) {
            int out_row = mask_argsort[sorted_row];
            uint64_t packed;
            memcpy(&packed, &epi_smem[row * EPL_STRIDE + col_start], sizeof(uint64_t));
            char *dst = reinterpret_cast<char *>(ptr_D) +
                        ((size_t)out_row * stride_out + n_start + col_start) * 2;
            *reinterpret_cast<uint64_t *>(dst) = packed;
          } else if (sorted_row < N_out) {
            int out_row = mask_argsort[sorted_row];
            for (int v = 0; v < VEC; ++v) {
              int n_global = n_start + col_start + v;
              if (n_global < C_out) {
                __half h = epi_smem[row * EPL_STRIDE + col_start + v];
                memcpy(&ptr_D[out_row * stride_out + n_global], &h, 2);
              }
            }
          }
        }
      }
    } else if constexpr (sizeof(ElementOutput) == 2) {
      // Direct half2 epilogue (no smem staging — lower overhead for small tiles).
      // Half2 stores require 4-byte alignment: ptr_D (after per-group shift),
      // the column index (n_global), and the row stride (stride_out) must all
      // yield 4-byte-aligned target addresses.
      bool ptr_D_aligned = (reinterpret_cast<uintptr_t>(ptr_D) & 3u) == 0;
      bool stride_aligned = (stride_out & 1) == 0;
      CUTE_UNROLL
      for (int frag = 0; frag < size(accum); frag += 4) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          int base = frag + pair * 2;
          auto coord = tCrC(base);
          int sorted_row = m_start + get<0>(coord);
          int n_global = n_start + get<1>(coord);
          if (sorted_row < N_out && n_global + 1 < C_out && stride_aligned && ptr_D_aligned &&
              ((n_global & 1) == 0)) {
            int out_row = mask_argsort[sorted_row];
            float v0 = (alpha == 1.0f) ? float(accum(base)) : alpha * float(accum(base));
            float v1 = (alpha == 1.0f) ? float(accum(base + 1)) : alpha * float(accum(base + 1));
            __half h0 = __float2half(v0), h1 = __float2half(v1);
            unsigned short s0, s1;
            memcpy(&s0, &h0, 2);
            memcpy(&s1, &h1, 2);
            uint32_t packed = (uint32_t)s0 | ((uint32_t)s1 << 16);
            char *base_ptr = reinterpret_cast<char *>(ptr_D);
            *reinterpret_cast<uint32_t *>(base_ptr +
                                          ((size_t)out_row * stride_out + n_global) * 2) = packed;
          } else if (sorted_row < N_out) {
            // Scalar fallback: either the write would straddle C (n_global+1 >= C)
            // or the half2 store would be misaligned (stride or col odd).
            int out_row = mask_argsort[sorted_row];
            if (n_global < C_out) {
              float val = (alpha == 1.0f) ? float(accum(base)) : alpha * float(accum(base));
              ptr_D[out_row * stride_out + n_global] = static_cast<ElementOutput>(val);
            }
            if (n_global + 1 < C_out) {
              float val1 =
                  (alpha == 1.0f) ? float(accum(base + 1)) : alpha * float(accum(base + 1));
              ptr_D[out_row * stride_out + n_global + 1] = static_cast<ElementOutput>(val1);
            }
          }
        }
      }
    } else {
      // f32 output path: scalar stores
      CUTE_UNROLL
      for (int i = 0; i < size(accum); ++i) {
        auto coord = tCrC(i);
        int sorted_row = m_start + get<0>(coord);
        int n_global = n_start + get<1>(coord);
        if (sorted_row < N_out && n_global < C_out) {
          int out_row = mask_argsort[sorted_row];
          float val = float(accum(i));
          ElementOutput result = (alpha == 1.0f) ? static_cast<ElementOutput>(val)
                                                 : static_cast<ElementOutput>(alpha * val);
          ptr_D[out_row * stride_out + n_global] = result;
        }
      }
    }
  }

  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_scalar(Accumulator &accum,
                                   ElementOutput *ptr_D,
                                   const int *mask_argsort,
                                   int m_start,
                                   int n_start,
                                   int N_out,
                                   int C_out,
                                   float alpha,
                                   TiledMma_ &tiled_mma,
                                   int stride_out = 0) const {
    if (stride_out == 0) stride_out = C_out;
    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    // Scalar epilogue: write one element at a time (no alignment requirement)
    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int sorted_row = m_start + get<0>(coord);
      int n_global = n_start + get<1>(coord);
      if (sorted_row < N_out && n_global < C_out) {
        int out_row = mask_argsort[sorted_row];
        float val = (alpha == 1.0f) ? float(accum(i)) : alpha * float(accum(i));
        ptr_D[out_row * stride_out + n_global] = static_cast<ElementOutput>(val);
      }
    }
  }
};
}  // namespace cute_gemm
}  // namespace warpconvnet
