// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CuTe GEMM kernel with manual gather/scatter and 2-stage pipelining.
//
// Operand A: gathered via indices → either:
//   - LDG.128 + STS.128 (synchronous, default) — UseCpAsyncGatherA = false
//   - cp.async 128-bit gmem→smem (async)        — UseCpAsyncGatherA = true
//            K-contiguous smem (Swizzle<2,3,3>) + LDSM_N (non-transposing)
// Operand B: dense → cp.async 128-bit gmem→smem (bypasses registers) to
//            N-contiguous smem (Swizzle<3,3,3>) + LDSM_T (transposing)
//
// The mainloop uses 2-stage double-buffering: while tensor cores compute on
// the current K-tile (stage[curr]), the next K-tile is being loaded into
// stage[next]. B always uses cp.async; A's strategy is configurable.

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm80.hpp"  // cp_async_fence, cp_async_wait
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cute_gemm_config.h"

namespace warpconvnet {
namespace cute_gemm {

using namespace cute;

/// Device GEMM kernel with manual gather/scatter and pipelined mainloop
template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmKernel {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

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

  // When true, gathered A uses cp.async (async gmem→smem, register bypass).
  // When false (default), gathered A uses synchronous LDG.128 + STS.128.
  // cp.async for A helps on large compute-heavy problems but can hurt on
  // smaller configs due to the serial gather-index dependency.
  static constexpr bool UseCpAsyncGatherA = TileConfig::UseCpAsyncGatherA;

  // 3D smem layouts: (M/N, K, Stages) — third dimension indexes pipeline stages
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
                                             make_shape(Int<tM>{}, Int<tK>{}, Int<NumStages>{})));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
                                             make_shape(Int<tN>{}, Int<tK>{}, Int<NumStages>{})));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);

  /// Main kernel with 2-stage pipelined mainloop
  __device__ void operator()(const ElementInput *ptr_A,
                             const ElementInput *ptr_B,
                             const ElementOutput *ptr_C,
                             ElementOutput *ptr_D,
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
    Tensor sA =
        make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});  // (tM, tK, NumStages)
    Tensor sB =
        make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});  // (tN, tK, NumStages)

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
    auto smem_thr_copy_A = smem_tiled_copy_A.get_slice(threadIdx.x);
    Tensor tCsA = smem_thr_copy_A.partition_S(sA);           // (CPY,CPY_M,CPY_K,NumStages)
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);  // (CPY,CPY_M,CPY_K)

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_slice(threadIdx.x);
    Tensor tCsB = smem_thr_copy_B.partition_S(sB);           // (CPY,CPY_N,CPY_K,NumStages)
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);  // (CPY,CPY_N,CPY_K)

    int num_k_tiles = (K_dim + tK - 1) / tK;
    auto K_BLOCK_MAX = size<2>(tCrA);

    // Early exit for zero-K GEMM
    if (num_k_tiles == 0) {
      _epilogue(accum, ptr_C, ptr_D, out_map, m_start, n_start, M, N, alpha, beta, tiled_mma);
      return;
    }

    // ==================== PROLOG: load k_tile=0 into stage[0] ====================
    _load_A(ptr_A, in_map, sA(_, _, 0), m_start, 0, M, K_dim);
    _load_dense_B_tile_cpasync(ptr_B, sB(_, _, 0), n_start, 0, N, K_dim);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // ==================== MAINLOOP: overlap load(next) with compute(curr) ====================
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 1; k_tile < num_k_tiles; ++k_tile) {
      int curr_stage = (k_tile - 1) % NumStages;
      int next_stage = k_tile % NumStages;
      int k_start = k_tile * tK;

      // Issue loads for NEXT k_tile into next_stage
      _load_A(ptr_A, in_map, sA(_, _, next_stage), m_start, k_start, M, K_dim);
      _load_dense_B_tile_cpasync(ptr_B, sB(_, _, next_stage), n_start, k_start, N, K_dim);
      cute::cp_async_fence();

      // Compute CURRENT k_tile from curr_stage
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, curr_stage), tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, curr_stage), tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      }

      // Wait for oldest in-flight stage to complete
      cute::cp_async_wait<NumStages - 2>();
      __syncthreads();
    }

    // ==================== EPILOG: compute last k_tile ====================
    {
      int last_stage = (num_k_tiles - 1) % NumStages;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, last_stage), tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, last_stage), tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      }
    }

    // ==================== EPILOGUE ====================
    _epilogue(accum, ptr_C, ptr_D, out_map, m_start, n_start, M, N, alpha, beta, tiled_mma);
  }

private:
  // Number of ElementInput values per 128-bit vector load/store
  static constexpr int kVec = 16 / sizeof(ElementInput);  // 8 for fp16/bf16

  /// Dispatch A load based on UseCpAsyncGatherA flag.
  /// Both paths are cp_async_fence-compatible: the sync path's STS completes
  /// before the fence, the async path's cp.async is committed by the fence.
  template <class SmemTensor>
  __device__ void _load_A(const ElementInput *ptr,
                          const int *gather_map,
                          SmemTensor smem_tile,
                          int m_start,
                          int k_start,
                          int M_phys,
                          int K_dim) const {
    if constexpr (UseCpAsyncGatherA) {
      _load_gathered_tile_cpasync(ptr, gather_map, smem_tile, m_start, k_start, M_phys, K_dim);
    } else {
      _load_gathered_tile_sync(ptr, gather_map, smem_tile, m_start, k_start, M_phys, K_dim);
    }
  }

  /// Load a gathered A tile into smem with synchronous LDG.128 + STS.128 along K.
  ///
  /// A is (M_phys, K_dim) row-major, gathered by gather_map.
  /// K is contiguous in both gmem (within each row) and smem (after swizzle).
  /// The Swizzle<2,3,3> layout preserves 8-element contiguity along K,
  /// so we can do 128-bit vectorized loads from gmem AND stores to smem.
  template <class SmemTensor>
  __device__ void _load_gathered_tile_sync(const ElementInput *ptr,
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

      uint4 vec_data = make_uint4(0, 0, 0, 0);
      if (m_global < M_phys) {
        int phys_row = gather_map[m_global];
        if (k_global + kVec <= K_dim) {
          vec_data = *reinterpret_cast<const uint4 *>(&ptr[phys_row * K_dim + k_global]);
        } else {
          auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
          for (int v = 0; v < kVec; ++v) {
            if (k_global + v < K_dim) elems[v] = ptr[phys_row * K_dim + k_global + v];
          }
        }
      }
      *reinterpret_cast<uint4 *>(&smem_tile(m_local, k_local)) = vec_data;
    }
  }

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
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                     "l"(gmem_src),
                     "n"(16));
      } else {
        // Zero-fill: src_size=0 writes zeros to smem without reading gmem
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                     "l"(ptr),
                     "n"(16),
                     "r"(0));
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
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                     "l"(gmem_src),
                     "n"(16));
      } else {
        // Zero-fill: src_size=0 writes zeros to smem without reading gmem
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                     "l"(ptr_B),
                     "n"(16),
                     "r"(0));
      }
    }
  }

  /// Epilogue: D[out_map[i], j] = alpha * accum(i, j) + beta * C[out_map[i], j]
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue(Accumulator &accum,
                            const ElementOutput *ptr_C,
                            ElementOutput *ptr_D,
                            const int *out_map,
                            int m_start,
                            int n_start,
                            int M,
                            int N,
                            float alpha,
                            float beta,
                            TiledMma_ &tiled_mma) const {
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

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
          result += beta * static_cast<float>(ptr_C[phys_row * N + n_global]);
        }
        ptr_D[phys_row * N + n_global] = static_cast<ElementOutput>(result);
      }
    }
  }
};

/// Global kernel entry point (AD gather-scatter)
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void cute_gemm_kernel_entry(
    const typename Kernel::ElementInput *ptr_A,
    const typename Kernel::ElementInput *ptr_B,
    const typename Kernel::ElementOutput *ptr_C,
    typename Kernel::ElementOutput *ptr_D,
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

// ===========================================================================
// TrAB Gather Kernel: D[k,n] = alpha * A[idx_a]^T @ B[idx_b] + beta * C[k,n]
//
// MMA dimension mapping:
//   M axis → K (input channels), tiled by tM
//   N axis → N (output channels), tiled by tN
//   K axis → gathered indices (reduction), chunked by tK
//
// A is (M_A, K_dim) row-major, gathered by idx_a. K contiguous in gmem.
// B is (M_B, N) row-major, gathered by idx_b. N contiguous in gmem.
// D is (K_dim, N) — dense output (no scatter).
//
// sA: (tM, tK) = (K_tile, gather_tile). SmemLayoutAtomA: gather contiguous.
//   Load: vectorized LDG along K, scatter-store (transpose) to smem.
// sB: (tN, tK) = (N_tile, gather_tile). SmemLayoutAtomB: N contiguous.
//   Load: vectorized LDG along N, vectorized STS (N contiguous in both).
// ===========================================================================

template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmTrABKernel {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;

  static constexpr int tM = size<0>(TileShape{});  // tiles K_dim
  static constexpr int tN = size<1>(TileShape{});  // tiles N
  static constexpr int tK = size<2>(TileShape{});  // tiles gathered indices
  static constexpr int NumStages = TileConfig::NumStages;

  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
                                             make_shape(Int<tM>{}, Int<tK>{}, Int<NumStages>{})));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
                                             make_shape(Int<tN>{}, Int<tK>{}, Int<NumStages>{})));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);

  // Number of ElementInput values per 128-bit vector load/store
  static constexpr int kVec = 16 / sizeof(ElementInput);  // 8 for fp16/bf16

  /// Main TrAB kernel with 2-stage pipelined mainloop
  __device__ void operator()(const ElementInput *ptr_A,
                             const ElementInput *ptr_B,
                             const ElementOutput *ptr_C,
                             ElementOutput *ptr_D,
                             const int *idx_a,
                             const int *idx_b,
                             int K_dim,        // input channels (MMA M)
                             int N,            // output channels (MMA N)
                             int gather_size,  // num gathered pairs (MMA K reduction)
                             float alpha,
                             float beta,
                             char *smem_buf) const {
    int k_tile = int(blockIdx.x);  // tiles K_dim
    int n_tile = int(blockIdx.y);  // tiles N
    int k_start = k_tile * tM;
    int n_start = n_tile * tN;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

    // MMA setup
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

    int num_g_tiles = (gather_size + tK - 1) / tK;
    auto K_BLOCK_MAX = size<2>(tCrA);

    if (num_g_tiles == 0) {
      _epilogue_trAB(accum, ptr_C, ptr_D, k_start, n_start, K_dim, N, alpha, beta, tiled_mma);
      return;
    }

    // ==================== PROLOG ====================
    _load_A_trAB(ptr_A, idx_a, sA(_, _, 0), k_start, 0, K_dim, gather_size);
    _load_B_trAB(ptr_B, idx_b, sB(_, _, 0), n_start, 0, N, gather_size);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // ==================== MAINLOOP ====================
    CUTLASS_PRAGMA_NO_UNROLL
    for (int g_tile = 1; g_tile < num_g_tiles; ++g_tile) {
      int curr_stage = (g_tile - 1) % NumStages;
      int next_stage = g_tile % NumStages;
      int g_start = g_tile * tK;

      _load_A_trAB(ptr_A, idx_a, sA(_, _, next_stage), k_start, g_start, K_dim, gather_size);
      _load_B_trAB(ptr_B, idx_b, sB(_, _, next_stage), n_start, g_start, N, gather_size);
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

    // ==================== EPILOG ====================
    {
      int last_stage = (num_g_tiles - 1) % NumStages;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, last_stage), tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, last_stage), tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      }
    }

    _epilogue_trAB(accum, ptr_C, ptr_D, k_start, n_start, K_dim, N, alpha, beta, tiled_mma);
  }

private:
  /// Load A^T tile: vectorized LDG along K, scatter-store (transpose) to smem.
  /// A is (M_A, K_dim), gathered by idx_a. K is contiguous in gmem.
  /// sA is (tM=K_tile, tK=gather_tile) with gather contiguous (SmemLayoutAtomA).
  /// The transpose: K-contiguous LDG → scatter individual elements to sA(k, g).
  template <class SmemTensor>
  __device__ void _load_A_trAB(const ElementInput *ptr_A,
                               const int *idx_a,
                               SmemTensor smem_tile,
                               int k_start,
                               int g_start,
                               int K_dim,
                               int gather_size) const {
    static_assert(tM % kVec == 0, "tM must be a multiple of vector width");
    constexpr int k_vecs = tM / kVec;  // tM tiles K_dim
    constexpr int total_vecs = tK * k_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int g_local = idx / k_vecs;
      int kv = idx % k_vecs;
      int k_local = kv * kVec;
      int g_global = g_start + g_local;
      int k_global = k_start + k_local;

      uint4 vec_data = make_uint4(0, 0, 0, 0);
      if (g_global < gather_size) {
        int phys_row = idx_a[g_global];
        if (k_global + kVec <= K_dim) {
          vec_data = *reinterpret_cast<const uint4 *>(&ptr_A[phys_row * K_dim + k_global]);
        } else {
          // Partial K boundary
          auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
          for (int v = 0; v < kVec; ++v) {
            if (k_global + v < K_dim) elems[v] = ptr_A[phys_row * K_dim + k_global + v];
          }
        }
      }
      // Scatter-store: each element to sA(k_local + v, g_local)
      // K is the first (non-contiguous) dimension, g is contiguous.
      auto *elems = reinterpret_cast<const ElementInput *>(&vec_data);
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kVec; ++v) {
        smem_tile(k_local + v, g_local) = elems[v];
      }
    }
  }

  /// Load B tile: vectorized LDG along N, vectorized STS to N-contiguous smem.
  /// B is (M_B, N), gathered by idx_b. N is contiguous in gmem.
  /// sB is (tN=N_tile, tK=gather_tile) with N contiguous (SmemLayoutAtomB).
  template <class SmemTensor>
  __device__ void _load_B_trAB(const ElementInput *ptr_B,
                               const int *idx_b,
                               SmemTensor smem_tile,
                               int n_start,
                               int g_start,
                               int N,
                               int gather_size) const {
    static_assert(tN % kVec == 0, "tN must be a multiple of vector width");
    constexpr int n_vecs = tN / kVec;
    constexpr int total_vecs = tK * n_vecs;

    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < total_vecs; idx += MaxThreadsPerBlock) {
      int g_local = idx / n_vecs;
      int nv = idx % n_vecs;
      int n_local = nv * kVec;
      int g_global = g_start + g_local;
      int n_global = n_start + n_local;

      uint4 vec_data = make_uint4(0, 0, 0, 0);
      if (g_global < gather_size) {
        int phys_row = idx_b[g_global];
        if (n_global + kVec <= N) {
          vec_data = *reinterpret_cast<const uint4 *>(&ptr_B[phys_row * N + n_global]);
        } else {
          auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
          for (int v = 0; v < kVec; ++v) {
            if (n_global + v < N) elems[v] = ptr_B[phys_row * N + n_global + v];
          }
        }
      }
      // N-contiguous in both gmem and smem → vectorized STS
      *reinterpret_cast<uint4 *>(&smem_tile(n_local, g_local)) = vec_data;
    }
  }

  /// Dense epilogue: D[k, n] = alpha * accum + beta * C[k, n]
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_trAB(Accumulator &accum,
                                 const ElementOutput *ptr_C,
                                 ElementOutput *ptr_D,
                                 int k_start,
                                 int n_start,
                                 int K_dim,
                                 int N,
                                 float alpha,
                                 float beta,
                                 TiledMma_ &tiled_mma) const {
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int k_local = get<0>(coord);
      int n_local = get<1>(coord);
      int k_global = k_start + k_local;
      int n_global = n_start + n_local;

      if (k_global < K_dim && n_global < N) {
        float acc_val = accum(i);
        float result = alpha * acc_val;
        if (beta != 0.0f) {
          result += beta * static_cast<float>(ptr_C[k_global * N + n_global]);
        }
        ptr_D[k_global * N + n_global] = static_cast<ElementOutput>(result);
      }
    }
  }
};

/// Global kernel entry point (TrAB gather)
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void cute_gemm_trAB_kernel_entry(
    const typename Kernel::ElementInput *ptr_A,
    const typename Kernel::ElementInput *ptr_B,
    const typename Kernel::ElementOutput *ptr_C,
    typename Kernel::ElementOutput *ptr_D,
    const int *idx_a,
    const int *idx_b,
    int K_dim,
    int N,
    int gather_size,
    float alpha,
    float beta) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A, ptr_B, ptr_C, ptr_D, idx_a, idx_b, K_dim, N, gather_size, alpha, beta, smem);
}

}  // namespace cute_gemm
}  // namespace warpconvnet
