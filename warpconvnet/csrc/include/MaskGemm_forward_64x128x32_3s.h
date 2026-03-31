// Auto-generated — DO NOT EDIT
// Config: MaskGemm_forward_64x128x32_3s
//   op_type=forward, tile=(64x128x32), stages=3
//   epilogue=direct, warp_shuffle=True, precomp_rows=True

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

template <class TileConfig, typename ElementOutput_ = float>
struct MaskGemm_forward_64x128x32_3s {
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
  static constexpr int NumStages = 3;
  static constexpr int NumWarps = MaxThreadsPerBlock / 32;
  static constexpr int kVec = 16 / sizeof(ElementInput);
  static constexpr int kMmaK = cute::size<2>(typename TiledMma::AtomShape_MNK{});
  static constexpr int K_BLOCK_MAX_STATIC = tK / kMmaK;
  static constexpr bool UseSmemEpilogue = true;

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
    uint32_t warp_masks[NumWarps];
    int real_rows[tM];
    uint32_t row_masks[tM];
  };

  static constexpr size_t EpilogueSmemSize = UseSmemEpilogue ? tM * (tN + 8) * 2 : 0;
  static constexpr size_t SharedStorageSize = sizeof(SharedStorage) > EpilogueSmemSize
                                                  ? sizeof(SharedStorage)
                                                  : EpilogueSmemSize;

  __device__ void operator()(const ElementInput *ptr_A,
                             const ElementInput *ptr_B,
                             ElementOutput *ptr_D,
                             const int *pair_table,
                             const uint32_t *pair_mask,
                             const int *mask_argsort,
                             int N_in,
                             int N_out,
                             int C_in,
                             int C_out,
                             int K,
                             float alpha,
                             char *smem_buf) const {
    using namespace cute;

    int grid_n = (C_out + tN - 1) / tN;
    int m_tile = int(blockIdx.x) / grid_n;
    int n_tile = int(blockIdx.x) % grid_n;
    int m_start = m_tile * tM;
    int n_start = n_tile * tN;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    // Warp-shuffle mask union + precompute row info
    uint32_t active_offsets;
    if (K == 1) {
      // K=1 fast path: single offset always active, skip mask union
      active_offsets = 1;
      CUTLASS_PRAGMA_UNROLL
      for (int m_local = threadIdx.x; m_local < tM; m_local += MaxThreadsPerBlock) {
        int sorted_row = m_start + m_local;
        if (sorted_row < N_out) {
          int rr = __ldg(&mask_argsort[sorted_row]);
          storage.real_rows[m_local] = rr;
          storage.row_masks[m_local] = 1;
        } else {
          storage.real_rows[m_local] = -1;
          storage.row_masks[m_local] = 0;
        }
      }
      __syncthreads();
    } else {
      int warp_id = threadIdx.x / 32;
      int lane_id = threadIdx.x % 32;
      uint32_t my_mask = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int m_local = threadIdx.x; m_local < tM; m_local += MaxThreadsPerBlock) {
        int sorted_row = m_start + m_local;
        if (sorted_row < N_out) {
          int rr = __ldg(&mask_argsort[sorted_row]);
          uint32_t rm = __ldg(&pair_mask[rr]);
          storage.real_rows[m_local] = rr;
          storage.row_masks[m_local] = rm;
          my_mask |= rm;
        } else {
          storage.real_rows[m_local] = -1;
          storage.row_masks[m_local] = 0;
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int s = 16; s >= 1; s >>= 1) {
        my_mask |= __shfl_xor_sync(0xffffffff, my_mask, s);
      }
      if (lane_id == 0) storage.warp_masks[warp_id] = my_mask;
      __syncthreads();
      active_offsets = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < NumWarps; ++w) active_offsets |= storage.warp_masks[w];
    }

    // --- MMA setup with register double-buffering ---
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

    // --- Offset loop (iterate only active offsets via __ffs) ---
    int num_k_tiles = (C_in + tK - 1) / tK;
    uint32_t offsets_remaining = active_offsets;
    int _offset_idx = 0;  // for stage alternation across offsets
    while (offsets_remaining) {
      int k = __ffs(offsets_remaining) - 1;
      offsets_remaining &= offsets_remaining - 1;

      const ElementInput *ptr_Bk = ptr_B + k * C_in * C_out;

      _update_A_indices(
          a_state, pair_table, storage.real_rows, storage.row_masks, N_in, N_out, C_in, k, K);

      if (num_k_tiles == 1) {
        // ---- Single k-tile: skip pipeline overhead ----
        _load_A_with_offsets(ptr_A, a_state, sA(_, _, 0), 0, C_in);
        _load_B_tile(ptr_Bk, sB(_, _, 0), gmem_thr_copy_B, n_start, 0, C_out, C_in, n_full_tile);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();
        MMA_DOUBLE_BUFFERED(0)
        __syncthreads();
      } else {
        // ---- Multi k-tile: pipelined with interleaved A loads ----
        {
          int prolog_tiles = (num_k_tiles < NumStages) ? num_k_tiles : (NumStages - 1);
          CUTLASS_PRAGMA_UNROLL
          for (int s = 0; s < NumStages - 1; ++s) {
            if (s < prolog_tiles) {
              _load_A_with_offsets(ptr_A, a_state, sA(_, _, s), s * tK, C_in);
              _load_B_tile(
                  ptr_Bk, sB(_, _, s), gmem_thr_copy_B, n_start, s * tK, C_out, C_in, n_full_tile);
            }
            cute::cp_async_fence();
          }
        }
        cute::cp_async_wait<NumStages - 2>();
        __syncthreads();

        int smem_pipe_read = 0;
        int smem_pipe_write = NumStages - 1;
        int prolog_tiles = (num_k_tiles < NumStages) ? num_k_tiles : (NumStages - 1);

        CUTLASS_PRAGMA_NO_UNROLL
        for (int ktile = prolog_tiles; ktile < num_k_tiles; ++ktile) {
          int k_start_cin = ktile * tK;
          copy(smem_tiled_copy_A, tCsA(_, _, 0, smem_pipe_read), tCrA_copy_0(_, _, 0));
          copy(smem_tiled_copy_B, tCsB(_, _, 0, smem_pipe_read), tCrB_copy_0(_, _, 0));
          _Pragma("unroll") for (int kb = 0; kb < K_BLOCK_MAX; ++kb) {
            // Prefetch next k-block registers
            if (kb + 1 < K_BLOCK_MAX) {
              int nkb = kb + 1;
              if (kb % 2 == 0) {
                copy(smem_tiled_copy_A, tCsA(_, _, nkb, smem_pipe_read), tCrA_copy_1(_, _, nkb));
                copy(smem_tiled_copy_B, tCsB(_, _, nkb, smem_pipe_read), tCrB_copy_1(_, _, nkb));
              } else {
                copy(smem_tiled_copy_A, tCsA(_, _, nkb, smem_pipe_read), tCrA_copy_0(_, _, nkb));
                copy(smem_tiled_copy_B, tCsB(_, _, nkb, smem_pipe_read), tCrB_copy_0(_, _, nkb));
              }
            }
            // Interleave A kblock + B loads with MMA.
            switch (kb) {
              case 0:
                _load_A_kblock<decltype(sA(_, _, 0)), 0>(
                    ptr_A, a_state, sA(_, _, smem_pipe_write), k_start_cin, C_in);
                _load_B_tile(ptr_Bk,
                             sB(_, _, smem_pipe_write),
                             gmem_thr_copy_B,
                             n_start,
                             k_start_cin,
                             C_out,
                             C_in,
                             n_full_tile);
                break;
              case 1:
                if (K_BLOCK_MAX_STATIC > 1)
                  _load_A_kblock<decltype(sA(_, _, 0)), 1>(
                      ptr_A, a_state, sA(_, _, smem_pipe_write), k_start_cin, C_in);

                break;
              case 2:
                if (K_BLOCK_MAX_STATIC > 2)
                  _load_A_kblock<decltype(sA(_, _, 0)), 2>(
                      ptr_A, a_state, sA(_, _, smem_pipe_write), k_start_cin, C_in);

                break;
              case 3:
                if (K_BLOCK_MAX_STATIC > 3)
                  _load_A_kblock<decltype(sA(_, _, 0)), 3>(
                      ptr_A, a_state, sA(_, _, smem_pipe_write), k_start_cin, C_in);

                break;
            }
            if (kb % 2 == 0)
              cute::gemm(tiled_mma, tCrA_0(_, _, kb), tCrB_0(_, _, kb), accum);
            else
              cute::gemm(tiled_mma, tCrA_1(_, _, kb), tCrB_1(_, _, kb), accum);
            if (kb == K_BLOCK_MAX - 1) {
              cute::cp_async_fence();
            }
          }
          cute::cp_async_wait<NumStages - 2>();
          __syncthreads();
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          if (smem_pipe_read >= NumStages) smem_pipe_read = 0;
        }

        {
          int remaining = (num_k_tiles < NumStages) ? num_k_tiles : (NumStages - 1);
          CUTLASS_PRAGMA_UNROLL
          for (int ep = 0; ep < NumStages - 1; ++ep) {
            if (ep < remaining) {
              MMA_DOUBLE_BUFFERED(smem_pipe_read)
              ++smem_pipe_read;
              if (smem_pipe_read >= NumStages) smem_pipe_read = 0;
            }
          }
        }
        __syncthreads();
      }
      ++_offset_idx;
    }

    // --- Epilogue ---
    _epilogue_direct(
        accum, ptr_D, mask_argsort, m_start, n_start, N_out, C_out, alpha, tiled_mma, smem_buf);
  }

private:
  // ---- Gathered A iterator with pre-resolved byte offsets ----

  // Work distribution: total_vecs = tM * k_vecs_A, distributed round-robin across threads
  static constexpr int k_vecs_A = tK / kVec;
  static constexpr int total_vecs_A = tM * k_vecs_A;
  // Number of vector iterations each thread performs (NOT rows — vectors!)
  static constexpr int kRowsPerThread =
      (total_vecs_A + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;

  // Pre-resolved byte offsets for gathered A rows (one per vector this thread touches)
  // Resolved once per K-offset in _update_A_indices, reused for all K-tiles
  struct AIteratorState {
    int byte_offsets[kRowsPerThread > 0 ? kRowsPerThread : 1];  // byte offset into ptr_A
    bool valid[kRowsPerThread > 0 ? kRowsPerThread : 1];        // bounds check
  };

  /// Resolve pair_table indices for the current K-offset.
  /// Called once per K-offset change (not per K-tile).
  /// Converts pair_table[k, real_row] → byte offset into ptr_A.
  __device__ void _update_A_indices(AIteratorState &state,
                                    const int *pair_table,
                                    const int *real_rows,
                                    const uint32_t *row_masks,
                                    int N_in,
                                    int N_out,
                                    int C_in,
                                    int offset_k,
                                    int K) const {
    const int *pt_base = pair_table + offset_k * N_out;
    int elem_bytes = sizeof(ElementInput);

    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kRowsPerThread; ++r) {
      int vec_idx = threadIdx.x + r * MaxThreadsPerBlock;
      int m_local = vec_idx / k_vecs_A;

      if (m_local < tM) {
        int real_row = real_rows[m_local];
        uint32_t row_mask = row_masks[m_local];
        int in_row = -1;
        if (real_row >= 0 && (K > 32 || (row_mask & (1u << offset_k)))) {
          in_row = __ldg(&pt_base[real_row]);
        }
        if (in_row >= 0 && in_row < N_in) {
          state.byte_offsets[r] = in_row * C_in * elem_bytes;
          state.valid[r] = true;
        } else {
          state.byte_offsets[r] = 0;
          state.valid[r] = false;
        }
      } else {
        state.byte_offsets[r] = 0;
        state.valid[r] = false;
      }
    }
  }

  /// Load A tile — full tile (calls all per-kblock loads sequentially).
  template <class SmemTensor>
  __device__ void _load_A_with_offsets(const ElementInput *ptr_A,
                                       const AIteratorState &state,
                                       SmemTensor smem_tile,
                                       int k_start,
                                       int C_in) const {
    _load_A_kblock<SmemTensor, 0>(ptr_A, state, smem_tile, k_start, C_in);
    if (K_BLOCK_MAX_STATIC > 1)
      _load_A_kblock<SmemTensor, 1>(ptr_A, state, smem_tile, k_start, C_in);
  }

  /// Per-k-block A load — template on KB for zero-branch compiled dispatch.
  template <class SmemTensor, int KB>
  __device__ void _load_A_kblock(const ElementInput *ptr_A,
                                 const AIteratorState &state,
                                 SmemTensor smem_tile,
                                 int k_start,
                                 int C_in) const {
    const char *base_ptr = reinterpret_cast<const char *>(ptr_A);
    int elem_bytes = sizeof(ElementInput);
    constexpr int iters_per_kb = (kRowsPerThread + K_BLOCK_MAX_STATIC - 1) / K_BLOCK_MAX_STATIC;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < iters_per_kb; ++i) {
      int r = KB * iters_per_kb + i;
      if (r < kRowsPerThread) {
        int vec_idx = threadIdx.x + r * MaxThreadsPerBlock;
        if (vec_idx < total_vecs_A) {
          int m_local = vec_idx / k_vecs_A;
          int kv = vec_idx % k_vecs_A;
          int k_local = kv * kVec;
          int k_global = k_start + k_local;
          uint32_t sa = cute::cast_smem_ptr_to_uint(&smem_tile(m_local, k_local));
          bool ok = state.valid[r] && (k_global + kVec <= C_in);
          if (ok) {
            const void *src = base_ptr + state.byte_offsets[r] + k_global * elem_bytes;
            asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(sa),
                         "l"(src),
                         "n"(16));
          } else {
            asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(sa),
                         "l"(ptr_A),
                         "n"(16),
                         "r"(0));
          }
        }
      }
    }
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

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < num_iters; ++s) {
      int n_cur = n_global + s * n_stride;
      uint32_t sa = cute::cast_smem_ptr_to_uint(&smem_stage(n_cur - n_start, k_base));
      bool ok =
          (n_cur + kVec <= n_start + tN) && (n_cur + kVec <= C_out) && (k_global + kVec <= C_in);

      if (ok) {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(sa),
                     "l"(gptr + s * inc_strided_bytes),
                     "n"(16));
      } else {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(sa),
                     "l"(gptr),
                     "n"(16),
                     "r"(0));
      }
    }
  }

  static constexpr bool BIsTransposed = false;

  // Dense B load: CuTe TiledCopy for full tiles, fallback for boundary
  template <class SmemTensor, class GmemThrCopyB>
  __device__ void _load_B_tile(const ElementInput *ptr_Bk,
                               SmemTensor smem_stage,
                               GmemThrCopyB &gmem_thr_copy_B,
                               int n_start,
                               int k_start,
                               int C_out,
                               int C_in,
                               bool n_full_tile) const {
    using namespace cute;
    bool k_full_tile = (k_start + tK <= C_in);
    if (n_full_tile && k_full_tile) {
      // Full tile: use CuTe TiledCopy (correct and fast)
      Tensor gB_tile = make_tensor(make_gmem_ptr(ptr_Bk + n_start + k_start * C_out),
                                   make_shape(Int<tN>{}, Int<tK>{}),
                                   make_stride(Int<1>{}, C_out));
      auto thr_src_B = gmem_thr_copy_B.partition_S(gB_tile);
      auto thr_dst_B = gmem_thr_copy_B.partition_D(smem_stage);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(thr_src_B); ++k) {
        copy(GmemTiledCopyB{}, thr_src_B(_, _, k), thr_dst_B(_, _, k));
      }
    } else {
      // Partial tile: use per-element cp.async with explicit bounds checking.
      // Write 16B (kVec elements) per thread along the N-contiguous dimension.
      static_assert(tN % kVec == 0);
      constexpr int n_vecs = tN / kVec;
      constexpr int total_vecs = tK * n_vecs;
      CUTLASS_PRAGMA_UNROLL
      for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
        int k_local = idx / n_vecs;
        int nv = idx % n_vecs;
        int n_local = nv * kVec;
        int n_global = n_start + n_local;
        int k_global = k_start + k_local;
        // Use CuTe indexing for correct swizzled smem address
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_stage(n_local, k_local));
        bool pred = (k_global < C_in) && (n_global + kVec <= C_out);
        if (pred) {
          const void *src = &ptr_Bk[k_global * C_out + n_global];
          asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                       "l"(src),
                       "n"(16));
        } else {
          asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                       "l"(ptr_Bk),
                       "n"(16),
                       "r"(0));
        }
      }
    }
  }

  /// Fallback: delegates to generic N-contiguous load.
  template <class SmemTensor>
  __device__ void _load_dense_B_cpasync(const ElementInput *ptr_B,
                                        SmemTensor smem_tile,
                                        int n_start,
                                        int k_start,
                                        int N,
                                        int K_dim) const {
    _load_dense_B_generic<SmemTensor, false>(ptr_B, smem_tile, n_start, k_start, N, K_dim);
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
      bool pred;
      const void *src;
      if constexpr (!IsTransposed) {
        pred = (k_global < K_dim) && (n_global + kVec <= N);
        src = &ptr_B[k_global * N + n_global];  // N-contiguous
      } else {
        pred = (n_global < N) && (k_global + kVec <= K_dim);
        src = &ptr_B[n_global * K_dim + k_global];  // K-contiguous
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
        bool pred = (k_global < K_dim) && (n_global + kVec <= N);
        const void *src = &ptr_B[k_global * N + n_global];
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
          bool pred = (n_global < N) && (k_global + kVec <= K_dim);
          if (pred) {
            const void *src = &ptr_B[n_global * K_dim + k_global];
            asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                         "l"(src),
                         "n"(16));
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
                                   char *smem_buf) const {
    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    if constexpr (sizeof(ElementOutput) == 2 && UseSmemEpilogue) {
      // Shared memory staging epilogue
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

#pragma unroll
      for (int iter = 0; iter < tM; iter += ROWS_PER_ITER) {
        int row = iter + row_in_iter;
        if (row < tM) {
          int sorted_row = m_start + row;
          if (sorted_row < N_out && col_start + n_start + VEC <= C_out) {
            int out_row = mask_argsort[sorted_row];
            uint64_t packed;
            memcpy(&packed, &epi_smem[row * EPL_STRIDE + col_start], sizeof(uint64_t));
            char *dst = reinterpret_cast<char *>(ptr_D) +
                        ((size_t)out_row * C_out + n_start + col_start) * 2;
            *reinterpret_cast<uint64_t *>(dst) = packed;
          } else if (sorted_row < N_out) {
            int out_row = mask_argsort[sorted_row];
            for (int v = 0; v < VEC; ++v) {
              int n_global = n_start + col_start + v;
              if (n_global < C_out) {
                __half h = epi_smem[row * EPL_STRIDE + col_start + v];
                memcpy(&ptr_D[out_row * C_out + n_global], &h, 2);
              }
            }
          }
        }
      }
    } else if constexpr (sizeof(ElementOutput) == 2) {
      // Direct half2 epilogue (no smem staging — lower overhead for small tiles)
      CUTE_UNROLL
      for (int frag = 0; frag < size(accum); frag += 4) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          int base = frag + pair * 2;
          auto coord = tCrC(base);
          int sorted_row = m_start + get<0>(coord);
          int n_global = n_start + get<1>(coord);
          if (sorted_row < N_out && n_global + 1 < C_out) {
            int out_row = mask_argsort[sorted_row];
            float v0 = (alpha == 1.0f) ? float(accum(base)) : alpha * float(accum(base));
            float v1 = (alpha == 1.0f) ? float(accum(base + 1)) : alpha * float(accum(base + 1));
            __half h0 = __float2half(v0), h1 = __float2half(v1);
            unsigned short s0, s1;
            memcpy(&s0, &h0, 2);
            memcpy(&s1, &h1, 2);
            uint32_t packed = (uint32_t)s0 | ((uint32_t)s1 << 16);
            char *base_ptr = reinterpret_cast<char *>(ptr_D);
            *reinterpret_cast<uint32_t *>(base_ptr + ((size_t)out_row * C_out + n_global) * 2) =
                packed;
          } else if (sorted_row < N_out && n_global < C_out) {
            int out_row = mask_argsort[sorted_row];
            float val = (alpha == 1.0f) ? float(accum(base)) : alpha * float(accum(base));
            ptr_D[out_row * C_out + n_global] = static_cast<ElementOutput>(val);
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
          ptr_D[out_row * C_out + n_global] = result;
        }
      }
    }
  }
};
}  // namespace cute_gemm
}  // namespace warpconvnet
