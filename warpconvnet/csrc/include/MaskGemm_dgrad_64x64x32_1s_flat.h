// Auto-generated — DO NOT EDIT
// Config: MaskGemm_dgrad_64x64x32_1s_flat
//   op_type=dgrad, tile=(64x64x32), stages=1
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

// Dgrad kernel: dX[j,:] = sum_k dY[rev_pt[k,j],:] @ W[k,:,:]^T
// C_in, C_out in CONV semantics. Contraction over C_out, output dim C_in.
template <class TileConfig, typename ElementOutput_ = float>
struct MaskGemm_dgrad_64x64x32_1s_flat {
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
  static constexpr bool UseSmemEpilogue = true;
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
    uint32_t warp_masks[NumWarps];
    int real_rows[tM];
    uint32_t row_masks[tM];
  };

  static constexpr size_t EpilogueSmemSize = UseSmemEpilogue ? tM * (tN + 8) * 2 : 0;
  static constexpr size_t SharedStorageSize = sizeof(SharedStorage) > EpilogueSmemSize
                                                  ? sizeof(SharedStorage)
                                                  : EpilogueSmemSize;

  __device__ void operator()(const ElementInput *ptr_A,  // grad_output [N_out, C_out]
                             const ElementInput *ptr_B,  // weight [K, C_in, C_out] — NOT transposed
                             ElementOutput *ptr_D,       // grad_input [N_in, C_in]
                             const int *pair_table,      // reverse_pair_table [K * N_in]
                             const uint32_t *pair_mask,  // reverse_pair_mask [N_in]
                             const int *mask_argsort,    // reverse_mask_argsort [N_in]
                             int N_in,
                             int N_out,
                             int C_in,   // output dim (grad_input channels)
                             int C_out,  // contraction dim (grad_output channels)
                             int K,
                             float alpha,
                             char *smem_buf) const {
    using namespace cute;

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

    // --- Dgrad offset loop: contract over C_out ---
    int num_k_tiles = (C_out + tK - 1) / tK;
    uint32_t offsets_remaining = active_offsets;
    int _offset_idx = 0;
    // Offset splitting: blockIdx.y selects which offsets this block processes.
    // Each block handles offsets where (_offset_idx % SplitOffsets == blockIdx.y).
    int split_y = (SplitOffsets > 1) ? int(blockIdx.y) : 0;
    while (offsets_remaining) {
      int k = __ffs(offsets_remaining) - 1;
      offsets_remaining &= offsets_remaining - 1;

      if (SplitOffsets > 1 && (_offset_idx % SplitOffsets) != split_y) {
        ++_offset_idx;
        continue;
      }

      const ElementInput *ptr_Bk = ptr_B + k * C_in * C_out;

      _update_A_indices(
          a_state, pair_table, storage.real_rows, storage.row_masks, N_out, N_in, C_out, k, K);

      // ---- Flat mainloop ----
      if (num_k_tiles == 1) {
        int smem_stage = _offset_idx & 1;
        _load_A_with_offsets(ptr_A, a_state, sA(_, _, smem_stage), 0, C_out);
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
          _load_A_with_offsets(ptr_A, a_state, sA(_, _, smem_stage), k_start, C_out);
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
    if constexpr (SplitOffsets > 1) {
      // Atomic epilogue: multiple Y-blocks accumulate to the same output
      _epilogue_atomic(accum, ptr_D, mask_argsort, m_start, n_start, N_in, C_in, alpha, tiled_mma);
    } else {
      _epilogue_direct(
          accum, ptr_D, mask_argsort, m_start, n_start, N_in, C_in, alpha, tiled_mma, smem_buf);
    }
  }

private:
  // ---- Pre-computed A loader (bypasses CuTe in hot path) ----

  static constexpr int k_vecs_A = tK / kVec;
  static constexpr int total_vecs_A = tM * k_vecs_A;
  static constexpr int kItersPerThread =
      (total_vecs_A + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;

  struct AIteratorState {
    int gmem_byte_offsets[kItersPerThread];  // in_row * C * sizeof(half), per K-offset
    bool valid[kItersPerThread];             // per K-offset
    // Pre-computed per-thread constants (set once, reused across all offsets + k-tiles):
    int m_local[kItersPerThread];        // M-row for each iteration
    int k_byte_offset[kItersPerThread];  // k_local * sizeof(half)
    // Cached from smem (read once in init, avoid repeated LDS per offset):
    int cached_real_row[kItersPerThread];       // from storage.real_rows[m]
    uint32_t cached_row_mask[kItersPerThread];  // from storage.row_masks[m]
  };

  /// Pre-compute thread's iteration constants + cache smem row info.
  /// Called ONCE after mask_union populates storage.real_rows/row_masks.
  /// Caches real_row and row_mask in registers to avoid LDS per offset.
  __device__ void _init_A_iterator(AIteratorState &state,
                                   const int *real_rows,
                                   const uint32_t *row_masks) const {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kItersPerThread; ++r) {
      int vec_idx = threadIdx.x + r * MaxThreadsPerBlock;
      if (vec_idx < total_vecs_A) {
        int m = vec_idx / k_vecs_A;
        state.m_local[r] = m;
        state.k_byte_offset[r] = (vec_idx % k_vecs_A) * kVec * sizeof(ElementInput);
        // Cache smem reads in registers (avoid LDS per offset iteration)
        if (m < tM) {
          state.cached_real_row[r] = real_rows[m];
          state.cached_row_mask[r] = row_masks[m];
        } else {
          state.cached_real_row[r] = -1;
          state.cached_row_mask[r] = 0;
        }
      } else {
        state.m_local[r] = tM;
        state.k_byte_offset[r] = 0;
        state.cached_real_row[r] = -1;
        state.cached_row_mask[r] = 0;
      }
    }
  }

  /// Overload for backward compat (no smem args).
  __device__ void _init_A_iterator(AIteratorState &state) const {
    // Fallback: will read from smem in _update_A_indices (slower)
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kItersPerThread; ++r) {
      int vec_idx = threadIdx.x + r * MaxThreadsPerBlock;
      if (vec_idx < total_vecs_A) {
        state.m_local[r] = vec_idx / k_vecs_A;
        state.k_byte_offset[r] = (vec_idx % k_vecs_A) * kVec * sizeof(ElementInput);
        state.cached_real_row[r] = -2;  // sentinel: not cached
        state.cached_row_mask[r] = 0;
      } else {
        state.m_local[r] = tM;
        state.k_byte_offset[r] = 0;
        state.cached_real_row[r] = -2;
        state.cached_row_mask[r] = 0;
      }
    }
  }

  /// Resolve pair_table for current K-offset using cached row info.
  /// No shared memory reads in the hot path — all from registers.
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
    for (int r = 0; r < kItersPerThread; ++r) {
      int m = state.m_local[r];
      if (m < tM) {
        // Use cached values (registers) instead of smem reads
        int real_row = (state.cached_real_row[r] != -2) ? state.cached_real_row[r] : real_rows[m];
        uint32_t row_mask =
            (state.cached_real_row[r] != -2) ? state.cached_row_mask[r] : row_masks[m];
        int in_row = -1;
        if (real_row >= 0 && (K > 32 || (row_mask & (1u << offset_k)))) {
          in_row = __ldg(&pt_base[real_row]);
        }
        if (in_row >= 0 && in_row < N_in) {
          state.gmem_byte_offsets[r] = in_row * C_in * elem_bytes;
          state.valid[r] = true;
        } else {
          state.gmem_byte_offsets[r] = 0;
          state.valid[r] = false;
        }
      } else {
        state.gmem_byte_offsets[r] = 0;
        state.valid[r] = false;
      }
    }
  }

  /// Load full A tile using pre-computed addresses.
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

  /// Per-kblock A load with pre-computed thread positions.
  /// Eliminates vec_idx division. Uses SmemTensor for stage-correct smem addressing.
  template <class SmemTensor, int KB>
  __device__ void _load_A_kblock(const ElementInput *ptr_A,
                                 const AIteratorState &state,
                                 SmemTensor smem_tile,
                                 int k_start,
                                 int C_in) const {
    const char *base_ptr = reinterpret_cast<const char *>(ptr_A);
    int k_start_bytes = k_start * sizeof(ElementInput);
    constexpr int iters_per_kb = (kItersPerThread + K_BLOCK_MAX_STATIC - 1) / K_BLOCK_MAX_STATIC;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < iters_per_kb; ++i) {
      int r = KB * iters_per_kb + i;
      if (r < kItersPerThread && state.m_local[r] < tM) {
        int m = state.m_local[r];
        int k_local = state.k_byte_offset[r] / int(sizeof(ElementInput));
        uint32_t sa = cute::cast_smem_ptr_to_uint(&smem_tile(m, k_local));
        bool ok = state.valid[r] && (k_start + k_local + kVec <= C_in);
        if (ok) {
          const void *src =
              base_ptr + state.gmem_byte_offsets[r] + k_start_bytes + state.k_byte_offset[r];
          asm volatile(
              "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(sa), "l"(src), "n"(16));
        } else {
          asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(sa),
                       "l"(ptr_A),
                       "n"(16),
                       "r"(0));
        }
      }
    }
  }

  static constexpr bool BIsTransposed = true;

  // _load_B_tile: K-contiguous B load for dgrad.
  // Optimized: precompute smem base address once, use compile-time offsets
  // for each wave to minimize instruction count.
  template <class SmemTensor, class GmemThrCopyB>
  __device__ void _load_B_tile(const ElementInput *ptr_Bk,
                               SmemTensor smem_stage,
                               GmemThrCopyB & /*unused*/,
                               int n_start,
                               int k_start,
                               int N_dim,
                               int K_dim,
                               bool n_full_tile) const {
    static_assert(tK % kVec == 0);
    constexpr int k_vecs = tK / kVec;
    constexpr int k_threads = k_vecs;
    constexpr int n_per_wave = MaxThreadsPerBlock / k_threads;
    constexpr int num_waves = (tN + n_per_wave - 1) / n_per_wave;

    // Precompute this thread's K-position and N-position within wave
    int k_tid = threadIdx.x % k_threads;
    int n_tid = threadIdx.x / k_threads;
    int k_local = k_tid * kVec;
    int k_global = k_start + k_local;
    bool k_valid = (k_global + kVec <= K_dim);

    // Precompute smem base for this thread's K-position at N=0
    // smem_stage(0, k_local) gives the base; each wave adds n_per_wave to N.
    uint32_t smem_base = cute::cast_smem_ptr_to_uint(&smem_stage(0, k_local));

    // Precompute gmem base for this thread
    const ElementInput *gmem_row_base = ptr_Bk + k_global;  // + n * K_dim per row

    if (n_full_tile && k_valid) {
      // Fast path: no bounds checking, all waves valid
      CUTLASS_PRAGMA_UNROLL
      for (int wave = 0; wave < num_waves; ++wave) {
        int n_local = wave * n_per_wave + n_tid;
        // Smem offset from base: difference between smem_stage(n_local, k_local) and smem_stage(0,
        // k_local)
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_stage(n_local, k_local));
        const void *src = gmem_row_base + (int64_t)(n_start + n_local) * K_dim;
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                     "l"(src),
                     "n"(16));
      }
    } else {
      // Partial tile: bounds checking
      CUTLASS_PRAGMA_UNROLL
      for (int wave = 0; wave < num_waves; ++wave) {
        int n_local = wave * n_per_wave + n_tid;
        if (n_local < tN) {
          int n_global = n_start + n_local;
          uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_stage(n_local, k_local));
          bool pred = (n_global < N_dim) && k_valid;
          if (pred) {
            const void *src = gmem_row_base + (int64_t)n_global * K_dim;
            asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                         "l"(src),
                         "n"(16));
          } else {
            asm volatile(
                "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                "l"(ptr_Bk),
                "n"(16),
                "r"(0));
          }
        }
      }
    }
  }

  // Legacy transposed API (backward compat)
  template <class SmemTensor>
  __device__ void _load_B_tile_transposed(const ElementInput *ptr_Bk,
                                          SmemTensor smem_stage,
                                          int n_start,
                                          int k_start,
                                          int N_dim,
                                          int K_dim) const {
    _load_dense_B_generic<SmemTensor, true>(ptr_Bk, smem_stage, n_start, k_start, N_dim, K_dim);
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
                                   int N_in,
                                   int C_in,
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
          if (sorted_row < N_in && col_start + n_start + VEC <= C_in) {
            int out_row = mask_argsort[sorted_row];
            uint64_t packed;
            memcpy(&packed, &epi_smem[row * EPL_STRIDE + col_start], sizeof(uint64_t));
            char *dst = reinterpret_cast<char *>(ptr_D) +
                        ((size_t)out_row * C_in + n_start + col_start) * 2;
            *reinterpret_cast<uint64_t *>(dst) = packed;
          } else if (sorted_row < N_in) {
            int out_row = mask_argsort[sorted_row];
            for (int v = 0; v < VEC; ++v) {
              int n_global = n_start + col_start + v;
              if (n_global < C_in) {
                __half h = epi_smem[row * EPL_STRIDE + col_start + v];
                memcpy(&ptr_D[out_row * C_in + n_global], &h, 2);
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
          if (sorted_row < N_in && n_global + 1 < C_in) {
            int out_row = mask_argsort[sorted_row];
            float v0 = (alpha == 1.0f) ? float(accum(base)) : alpha * float(accum(base));
            float v1 = (alpha == 1.0f) ? float(accum(base + 1)) : alpha * float(accum(base + 1));
            __half h0 = __float2half(v0), h1 = __float2half(v1);
            unsigned short s0, s1;
            memcpy(&s0, &h0, 2);
            memcpy(&s1, &h1, 2);
            uint32_t packed = (uint32_t)s0 | ((uint32_t)s1 << 16);
            char *base_ptr = reinterpret_cast<char *>(ptr_D);
            *reinterpret_cast<uint32_t *>(base_ptr + ((size_t)out_row * C_in + n_global) * 2) =
                packed;
          } else if (sorted_row < N_in && n_global < C_in) {
            int out_row = mask_argsort[sorted_row];
            float val = (alpha == 1.0f) ? float(accum(base)) : alpha * float(accum(base));
            ptr_D[out_row * C_in + n_global] = static_cast<ElementOutput>(val);
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
        if (sorted_row < N_in && n_global < C_in) {
          int out_row = mask_argsort[sorted_row];
          float val = float(accum(i));
          ElementOutput result = (alpha == 1.0f) ? static_cast<ElementOutput>(val)
                                                 : static_cast<ElementOutput>(alpha * val);
          ptr_D[out_row * C_in + n_global] = result;
        }
      }
    }
  }

  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_atomic(Accumulator &accum,
                                   ElementOutput *ptr_D,
                                   const int *mask_argsort,
                                   int m_start,
                                   int n_start,
                                   int N_in,
                                   int C_in,
                                   float alpha,
                                   TiledMma_ &tiled_mma) const {
    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    // Atomic epilogue: each element uses atomicAdd to accumulate across Y-blocks.
    // For fp16 output, use atomicAdd(__half*, __half) which is supported on SM70+.
    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int sorted_row = m_start + get<0>(coord);
      int n_global = n_start + get<1>(coord);
      if (sorted_row < N_in && n_global < C_in) {
        int out_row = mask_argsort[sorted_row];
        float val = (alpha == 1.0f) ? float(accum(i)) : alpha * float(accum(i));
        if constexpr (sizeof(ElementOutput) == 2) {
          atomicAdd(reinterpret_cast<__half *>(ptr_D + out_row * C_in + n_global),
                    __float2half(val));
        } else {
          atomicAdd(ptr_D + out_row * C_in + n_global, static_cast<ElementOutput>(val));
        }
      }
    }
  }
};
}  // namespace cute_gemm
}  // namespace warpconvnet
