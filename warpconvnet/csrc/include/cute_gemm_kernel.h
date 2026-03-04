// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CuTe GEMM kernel with manual gather/scatter.
// Loads gathered A rows and dense B into shared memory via cooperative
// 128-bit vectorized loads, then uses CuTe TiledMMA for tensor core computation.
//
// Operand A: K-contiguous smem (Swizzle<2,3,3>) + LDSM_N (non-transposing)
// Operand B: N-contiguous smem (Swizzle<3,3,3>) + LDSM_T (transposing)
// The LDSM_T instruction transposes N-contiguous smem data to K-contiguous
// registers matching the MMA fragment layout.

#pragma once

#include "cute/tensor.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/copy.hpp"

#include "cute_gemm_config.h"

namespace warpconvnet {
namespace cute_gemm {

using namespace cute;

/// Device GEMM kernel with manual gather/scatter
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

  // Single-stage smem layout (no cp_async pipelining — manual element-wise
  // loads from global memory don't benefit from multi-stage buffering)
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(Int<tM>{}, Int<tK>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(Int<tN>{}, Int<tK>{})));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);

  /// Main kernel
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
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});  // (tM, tK)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});  // (tN, tK)

    // MMA setup
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<tM>{}, Int<tN>{}));
    clear(accum);

    // MMA register fragments — partitioned from smem tile shape
    Tensor tCrA = thr_mma.partition_fragment_A(sA);  // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB);  // (MMA,MMA_N,MMA_K)

    // Smem → register copy with LDSM
    // Key: use retile_D() to create a view of the register fragment
    // that matches the copy atom's destination layout
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_slice(threadIdx.x);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);       // (CPY,CPY_M,CPY_K)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);       // (CPY,CPY_M,CPY_K)

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_slice(threadIdx.x);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);       // (CPY,CPY_N,CPY_K)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);       // (CPY,CPY_N,CPY_K)

    int num_k_tiles = (K_dim + tK - 1) / tK;

    // Number of MMA K-steps within one smem tile
    // (e.g., tK=32 / MMA_K=16 = 2 steps)
    auto K_BLOCK_MAX = size<2>(tCrA);

    // ==================== MAINLOOP ====================
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
      int k_start = k_tile * tK;

      // Load gathered A and dense B tiles into smem (element-wise)
      _load_gathered_tile(ptr_A, in_map, sA,
                          m_start, k_start, M, K_dim, true);
      _load_dense_B_tile(ptr_B, sB,
                         n_start, k_start, N, K_dim);
      __syncthreads();

      // Copy smem → registers and compute, iterating over K blocks
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        // LDSM: smem → registers (using retiled destination view)
        copy(smem_tiled_copy_A, tCsA(_, _, k_block), tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block), tCrB_copy_view(_, _, k_block));
        // MMA: register compute for this K block
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      }

      __syncthreads();
    }

    // ==================== EPILOGUE ====================
    _epilogue(accum, ptr_C, ptr_D, out_map,
              m_start, n_start, M, N, alpha, beta, tiled_mma);
  }

 private:
  // Number of ElementInput values per 128-bit vector load/store
  static constexpr int kVec = 16 / sizeof(ElementInput);  // 8 for fp16/bf16

  /// Load a gathered A tile into smem with vectorized 128-bit loads along K.
  ///
  /// A is (M_phys, K_dim) row-major, gathered by gather_map.
  /// K is contiguous in both gmem (within each row) and smem (after swizzle).
  /// The Swizzle<2,3,3> layout preserves 8-element contiguity along K,
  /// so we can do 128-bit vectorized loads from gmem AND stores to smem.
  template <class SmemTensor>
  __device__ void _load_gathered_tile(const ElementInput *ptr,
                                      const int *gather_map,
                                      SmemTensor smem_tile,
                                      int m_start,
                                      int k_start,
                                      int M_phys,
                                      int K_dim,
                                      bool is_gathered) const {
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

      uint4 vec_data = make_uint4(0, 0, 0, 0);
      if (m_global < M_phys) {
        int phys_row = is_gathered ? gather_map[m_global] : m_global;
        if (k_global + kVec <= K_dim) {
          // Vectorized 128-bit load (K-contiguous in gmem)
          vec_data = *reinterpret_cast<const uint4 *>(
              &ptr[phys_row * K_dim + k_global]);
        } else {
          // Boundary: element-wise fallback
          auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
          for (int v = 0; v < kVec; ++v) {
            if (k_global + v < K_dim)
              elems[v] = ptr[phys_row * K_dim + k_global + v];
          }
        }
      }
      // Vectorized 128-bit store to smem (K-contiguous after swizzle)
      *reinterpret_cast<uint4 *>(&smem_tile(m_local, k_local)) = vec_data;
    }
  }

  /// Load a dense B tile into smem with vectorized 128-bit loads along N.
  ///
  /// B is (K, N) row-major: B[k, n] = ptr[k * N + n]. N is contiguous in gmem.
  /// Smem B stores (tN, tK) with N-contiguous layout (SmemLayoutAtomB_FP16).
  /// Swizzle<3,3,3> preserves 8-element contiguity along N, enabling
  /// 128-bit vectorized stores to smem.
  /// Iteration order: K outer, N-vec inner — adjacent threads load
  /// consecutive 128-bit chunks along N (coalesced gmem access).
  template <class SmemTensor>
  __device__ void _load_dense_B_tile(const ElementInput *ptr_B,
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

      uint4 vec_data = make_uint4(0, 0, 0, 0);
      if (k_global < K_dim) {
        if (n_global + kVec <= N) {
          // Vectorized 128-bit load (N-contiguous in gmem)
          vec_data = *reinterpret_cast<const uint4 *>(
              &ptr_B[k_global * N + n_global]);
        } else {
          // Boundary: element-wise fallback
          auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
          for (int v = 0; v < kVec; ++v) {
            if (n_global + v < N)
              elems[v] = ptr_B[k_global * N + n_global + v];
          }
        }
      }
      // Vectorized 128-bit store to smem (N-contiguous after swizzle)
      *reinterpret_cast<uint4 *>(&smem_tile(n_local, k_local)) = vec_data;
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
