// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CuTe GEMM kernel with manual gather/scatter and 2-stage pipelining.
//
// Operand A: gathered via indices → cp.async 128-bit gmem→smem along K
//            (K is contiguous within each gathered row) to K-contiguous
//            smem (Swizzle<2,3,3>) + LDSM_N (non-transposing)
// Operand B: dense → cp.async 128-bit gmem→smem (bypasses registers) to
//            N-contiguous smem (Swizzle<3,3,3>) + LDSM_T (transposing)
//
// The mainloop uses 2-stage double-buffering: while tensor cores compute on
// the current K-tile (stage[curr]), the next K-tile is being loaded into
// stage[next]. Both A and B use cp.async for truly async gmem→smem transfers.

#pragma once

#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm80.hpp"  // cp_async_fence, cp_async_wait

#include "cute_gemm_config.h"

namespace warpconvnet {
namespace cute_gemm {

using namespace cute;

/// Device GEMM kernel with manual gather/scatter and pipelined mainloop
template <class TileConfig>
struct CuteGemmKernel {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;

  static constexpr int tM = size<0>(TileShape{});
  static constexpr int tN = size<1>(TileShape{});
  static constexpr int tK = size<2>(TileShape{});
  static constexpr int NumStages = TileConfig::NumStages;

  // 3D smem layouts: (M/N, K, Stages) — third dimension indexes pipeline stages
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(Int<tM>{}, Int<tK>{}, Int<NumStages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(Int<tN>{}, Int<tK>{}, Int<NumStages>{})));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);

  /// Main kernel with 2-stage pipelined mainloop
  __device__ void operator()(const ElementInput *ptr_A,
                             const ElementInput *ptr_B,
                             const float *ptr_C,
                             float *ptr_D,
                             const int *in_map,
                             const int *out_map,
                             int M,
                             int N,
                             int K_dim,
                             float alpha,
                             float beta,
                             char *smem_buf) const {
    int m_tile = int(blockIdx.x);
    int n_tile = int(blockIdx.y);
    int m_start = m_tile * tM;
    int n_start = n_tile * tN;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});  // (tM, tK, NumStages)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});  // (tN, tK, NumStages)

    // MMA setup
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<tM>{}, Int<tN>{}));
    clear(accum);

    // MMA register fragments — partitioned from a 2D stage slice
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));  // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));  // (MMA,MMA_N,MMA_K)

    // Smem → register copy with LDSM — partition the 3D smem tensors
    // producing 4D results: (CPY, CPY_M/N, CPY_K, NumStages)
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_slice(threadIdx.x);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);       // (CPY,CPY_M,CPY_K,NumStages)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);       // (CPY,CPY_M,CPY_K)

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_slice(threadIdx.x);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);       // (CPY,CPY_N,CPY_K,NumStages)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);       // (CPY,CPY_N,CPY_K)

    int num_k_tiles = (K_dim + tK - 1) / tK;
    auto K_BLOCK_MAX = size<2>(tCrA);

    // Early exit for zero-K GEMM
    if (num_k_tiles == 0) {
      _epilogue(accum, ptr_C, ptr_D, out_map,
                m_start, n_start, M, N, alpha, beta, tiled_mma);
      return;
    }

    // ==================== PROLOG: load k_tile=0 into stage[0] ====================
    _load_gathered_tile_cpasync(ptr_A, in_map, sA(_, _, 0),
                                m_start, 0, M, K_dim);
    _load_dense_B_tile_cpasync(ptr_B, sB(_, _, 0),
                               n_start, 0, N, K_dim);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // ==================== MAINLOOP: overlap load(next) with compute(curr) ====================
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 1; k_tile < num_k_tiles; ++k_tile) {
      int curr_stage = (k_tile - 1) & 1;
      int next_stage = k_tile & 1;
      int k_start = k_tile * tK;

      // Issue loads for NEXT k_tile into next_stage (both A and B are cp.async)
      _load_gathered_tile_cpasync(ptr_A, in_map, sA(_, _, next_stage),
                                  m_start, k_start, M, K_dim);
      _load_dense_B_tile_cpasync(ptr_B, sB(_, _, next_stage),
                                 n_start, k_start, N, K_dim);
      cute::cp_async_fence();

      // Compute CURRENT k_tile from curr_stage
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, curr_stage), tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, curr_stage), tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      }

      // Wait for next_stage loads to complete
      cute::cp_async_wait<0>();
      __syncthreads();
    }

    // ==================== EPILOG: compute last k_tile ====================
    {
      int last_stage = (num_k_tiles - 1) & 1;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, last_stage), tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, last_stage), tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      }
    }

    // ==================== EPILOGUE ====================
    _epilogue(accum, ptr_C, ptr_D, out_map,
              m_start, n_start, M, N, alpha, beta, tiled_mma);
  }

 private:
  // Number of ElementInput values per 128-bit vector load/store
  static constexpr int kVec = 16 / sizeof(ElementInput);  // 8 for fp16/bf16

  /// Load a gathered A tile into smem using cp.async (128-bit async gmem→smem).
  ///
  /// A is (M_phys, K_dim) row-major, gathered by gather_map.
  /// K is contiguous in both gmem (within each gathered row) and smem.
  /// The Swizzle<2,3,3> layout preserves 8-element contiguity along K,
  /// so cp.async 128-bit transfers are aligned for both gmem source and smem dest.
  ///
  /// Each thread resolves the gather index (synchronous LDG for the int32 index),
  /// then issues cp.async for the 128-bit K-contiguous data (async gmem→smem,
  /// bypasses registers). Out-of-bounds accesses use zero-fill (src_size=0).
  /// Caller must issue cp_async_fence() after this function returns.
  template <class SmemTensor>
  __device__ void _load_gathered_tile_cpasync(const ElementInput *ptr,
                                              const int *gather_map,
                                              SmemTensor smem_tile,
                                              int m_start,
                                              int k_start,
                                              int M_phys,
                                              int K_dim) const {
    static_assert(tK % kVec == 0, "tK must be a multiple of vector width");
    constexpr int k_vecs = tK / kVec;
    constexpr int total_vecs = tM * k_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int m_local = idx / k_vecs;
      int kv = idx % k_vecs;
      int k_local = kv * kVec;
      int m_global = m_start + m_local;
      int k_global = k_start + k_local;

      // Smem destination: 16-byte aligned (Swizzle<2,3,3> preserves 8-elem contiguity along K)
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(m_local, k_local));

      bool pred = (m_global < M_phys) && (k_global + kVec <= K_dim);

      if (pred) {
        // Resolve gather index (synchronous 4-byte LDG for the int32 index)
        int phys_row = gather_map[m_global];
        // 128-bit cp.async: gmem → smem, bypasses registers
        const void *gmem_src = &ptr[phys_row * K_dim + k_global];
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
            :: "r"(smem_addr), "l"(gmem_src), "n"(16));
      } else {
        // Zero-fill: src_size=0 writes zeros to smem without reading gmem
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
            :: "r"(smem_addr), "l"(ptr), "n"(16), "r"(0));
      }
    }
  }

  /// Load a dense B tile into smem using cp.async (128-bit async gmem→smem).
  ///
  /// B is (K, N) row-major: B[k, n] = ptr[k * N + n]. N is contiguous in gmem.
  /// Smem B stores (tN, tK) with N-contiguous layout (Swizzle<3,3,3>).
  /// cp.async bypasses registers — the SM80 copy engine transfers data directly
  /// from gmem to smem. Out-of-bounds accesses use zero-fill (src_size=0).
  /// Caller must issue cp_async_fence() after this function returns.
  template <class SmemTensor>
  __device__ void _load_dense_B_tile_cpasync(const ElementInput *ptr_B,
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

      // Smem destination: 16-byte aligned (Swizzle<3,3,3> preserves 8-elem contiguity)
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));

      bool pred = (k_global < K_dim) && (n_global + kVec <= N);

      if (pred) {
        // 128-bit cp.async: gmem → smem, bypasses registers
        const void *gmem_src = &ptr_B[k_global * N + n_global];
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
            :: "r"(smem_addr), "l"(gmem_src), "n"(16));
      } else {
        // Zero-fill: src_size=0 writes zeros to smem without reading gmem
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
            :: "r"(smem_addr), "l"(ptr_B), "n"(16), "r"(0));
      }
    }
  }

  /// Epilogue: D[out_map[i], j] = alpha * accum(i, j) + beta * C[out_map[i], j]
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue(Accumulator &accum,
                            const float *ptr_C,
                            float *ptr_D,
                            const int *out_map,
                            int m_start,
                            int n_start,
                            int M,
                            int N,
                            float alpha,
                            float beta,
                            TiledMma_ &tiled_mma) const {
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(
        make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int m_local = get<0>(coord);
      int n_local = get<1>(coord);
      int m_global = m_start + m_local;
      int n_global = n_start + n_local;

      if (m_global < M && n_global < N) {
        int phys_row = out_map[m_global];
        float acc_val = accum(i);
        float result = alpha * acc_val;
        if (beta != 0.0f) {
          result += beta * ptr_C[phys_row * N + n_global];
        }
        ptr_D[phys_row * N + n_global] = result;
      }
    }
  }
};

/// Global kernel entry point
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock)
    void cute_gemm_kernel_entry(const typename Kernel::ElementInput *ptr_A,
                                const typename Kernel::ElementInput *ptr_B,
                                const float *ptr_C,
                                float *ptr_D,
                                const int *in_map,
                                const int *out_map,
                                int M,
                                int N,
                                int K,
                                float alpha,
                                float beta) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A, ptr_B, ptr_C, ptr_D, in_map, out_map, M, N, K, alpha, beta, smem);
}

}  // namespace cute_gemm
}  // namespace warpconvnet
