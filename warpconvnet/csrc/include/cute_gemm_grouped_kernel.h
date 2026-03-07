// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused multi-offset CuTe GEMM kernel for sparse convolution.
//
// Instead of launching K separate GEMM kernels (one per kernel offset),
// this kernel processes all offsets in a single launch.  Each threadblock
// determines its group (offset) via binary search on a prefix-sum array
// of M-tiles, then runs the standard CuTe GEMM mainloop with that
// group's weight pointer and gather/scatter indices.
//
// All groups share:
//   ptr_A  — input features  [N_in, K_dim]
//   ptr_D  — output features [N_out, N] (zero-initialized, accumulated via atomicAdd)
//   in_map / out_map — concatenated gather/scatter indices [L]
//
// Per-group:
//   ptr_B_array[g]  — weight pointer for group g  [K_dim, N]
//   map_offsets[g]   — start offset into in_map / out_map
//   group_sizes[g]   — number of pairs (M_g) for group g

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cute_gemm_config.h"
#include "grouped_gemm_params.h"

namespace warpconvnet {
namespace cute_gemm {

/// atomicAdd wrapper that handles cutlass types by casting to native CUDA types.
template <typename T>
__device__ __forceinline__ void atomic_add(T *addr, T val) {
  atomicAdd(addr, val);
}

template <>
__device__ __forceinline__ void atomic_add<cutlass::half_t>(cutlass::half_t *addr,
                                                            cutlass::half_t val) {
  atomicAdd(reinterpret_cast<__half *>(addr), __float2half(float(val)));
}

template <>
__device__ __forceinline__ void atomic_add<cutlass::bfloat16_t>(cutlass::bfloat16_t *addr,
                                                                cutlass::bfloat16_t val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16 *>(addr), __float2bfloat16(float(val)));
}

/// Binary search: find g such that tile_offsets[g] <= tile_idx < tile_offsets[g+1]
__device__ __forceinline__ int find_group(const int *tile_offsets, int num_groups, int tile_idx) {
  int lo = 0, hi = num_groups;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (tile_offsets[mid + 1] <= tile_idx)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

/// Fused grouped GEMM kernel with gather on A and atomicAdd scatter on D.
template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmGroupedKernel {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;

  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;

  static constexpr int tM = cute::size<0>(TileShape{});
  static constexpr int tN = cute::size<1>(TileShape{});
  static constexpr int tK = cute::size<2>(TileShape{});
  static constexpr int NumStages = TileConfig::NumStages;
  static constexpr bool UseCpAsyncGatherA = TileConfig::UseCpAsyncGatherA;

  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::Int<tM>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::Int<tN>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);

  static constexpr int kVec = 16 / sizeof(ElementInput);  // 8 for fp16/bf16

  __device__ void operator()(const ElementInput *ptr_A,
                             ElementOutput *ptr_D,
                             const int *in_map,
                             const int *out_map,
                             GroupedGemmParams params,
                             int N,
                             int K_dim,
                             float alpha,
                             char *smem_buf) const {
    using namespace cute;
    // --- Step 1: Determine which group this block belongs to ---
    int global_m_tile = int(blockIdx.x);
    int n_tile = int(blockIdx.y);

    int g = find_group(params.tile_offsets, params.num_groups, global_m_tile);
    int local_m_tile = global_m_tile - params.tile_offsets[g];
    int m_start = local_m_tile * tM;
    int M_g = params.group_sizes[g];
    int map_offset_g = params.map_offsets[g];
    int n_start = n_tile * tN;

    // Per-group pointers
    const ElementInput *ptr_B_g = reinterpret_cast<const ElementInput *>(params.ptr_B_array[g]);
    const int *in_map_g = in_map + map_offset_g;
    const int *out_map_g = out_map + map_offset_g;

    // --- Step 2: Standard CuTe GEMM mainloop ---
    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

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

    int num_k_tiles = (K_dim + tK - 1) / tK;
    auto K_BLOCK_MAX = size<2>(tCrA);

    if (num_k_tiles == 0) {
      _epilogue_atomic(accum, ptr_D, out_map_g, m_start, n_start, M_g, N, alpha, tiled_mma);
      return;
    }

    // Prolog: load k_tile=0 into stage[0]
    _load_A(ptr_A, in_map_g, sA(_, _, 0), m_start, 0, M_g, K_dim);
    _load_dense_B_tile_cpasync(ptr_B_g, sB(_, _, 0), n_start, 0, N, K_dim);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // Mainloop: overlap load(next) with compute(curr)
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 1; k_tile < num_k_tiles; ++k_tile) {
      int curr_stage = (k_tile - 1) % NumStages;
      int next_stage = k_tile % NumStages;
      int k_start = k_tile * tK;

      _load_A(ptr_A, in_map_g, sA(_, _, next_stage), m_start, k_start, M_g, K_dim);
      _load_dense_B_tile_cpasync(ptr_B_g, sB(_, _, next_stage), n_start, k_start, N, K_dim);
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

    // Epilog: compute last k_tile
    {
      int last_stage = (num_k_tiles - 1) % NumStages;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, last_stage), tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, last_stage), tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      }
    }

    _epilogue_atomic(accum, ptr_D, out_map_g, m_start, n_start, M_g, N, alpha, tiled_mma);
  }

private:
  // ---- A load (gathered) ----
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

      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(m_local, k_local));
      bool pred = (m_global < M_phys) && (k_global + kVec <= K_dim);

      if (pred) {
        int phys_row = gather_map[m_global];
        const void *gmem_src = &ptr[phys_row * K_dim + k_global];
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                     "l"(gmem_src),
                     "n"(16));
      } else {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                     "l"(ptr),
                     "n"(16),
                     "r"(0));
      }
    }
  }

  // ---- B load (dense, per-group pointer) ----
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

      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));
      bool pred = (k_global < K_dim) && (n_global + kVec <= N);

      if (pred) {
        const void *gmem_src = &ptr_B[k_global * N + n_global];
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                     "l"(gmem_src),
                     "n"(16));
      } else {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_addr),
                     "l"(ptr_B),
                     "n"(16),
                     "r"(0));
      }
    }
  }

  // ---- Epilogue with atomicAdd for concurrent group writes ----
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_atomic(Accumulator &accum,
                                   ElementOutput *ptr_D,
                                   const int *out_map,
                                   int m_start,
                                   int n_start,
                                   int M_g,
                                   int N,
                                   float alpha,
                                   TiledMma_ &tiled_mma) const {
    using namespace cute;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int m_local = get<0>(coord);
      int n_local = get<1>(coord);
      int m_global = m_start + m_local;
      int n_global = n_start + n_local;

      if (m_global < M_g && n_global < N) {
        int phys_row = out_map[m_global];
        float result = alpha * float(accum(i));
        atomic_add(&ptr_D[phys_row * N + n_global], static_cast<ElementOutput>(result));
      }
    }
  }
};

/// Global kernel entry point for grouped AD gather-scatter
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void cute_gemm_grouped_kernel_entry(
    const typename Kernel::ElementInput *ptr_A,
    typename Kernel::ElementOutput *ptr_D,
    const int *in_map,
    const int *out_map,
    GroupedGemmParams params,
    int N,
    int K_dim,
    float alpha) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A, ptr_D, in_map, out_map, params, N, K_dim, alpha, smem);
}

// ===========================================================================
// Grouped TrAB Gather Kernel
//
// Fuses all kernel offsets into a single launch for the weight gradient:
//   D_g[k,n] = alpha * A[idx_a_g]^T @ B[idx_b_g]    for each group g
//
// Grid: (K_tiles, N_tiles, num_groups)
// Each threadblock picks its group from blockIdx.z and runs a standard
// TrAB mainloop with that group's gather indices and output pointer.
// No atomicAdd — each group writes to its own output matrix.
// ===========================================================================

template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmGroupedTrABKernel {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;
  using SmemCopyAtomA = typename TileConfig::SmemCopyAtomA;
  using SmemCopyAtomB = typename TileConfig::SmemCopyAtomB;

  static constexpr int MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;

  static constexpr int tM = cute::size<0>(TileShape{});  // tiles K_dim
  static constexpr int tN = cute::size<1>(TileShape{});  // tiles N
  static constexpr int tK = cute::size<2>(TileShape{});  // tiles gathered indices
  static constexpr int NumStages = TileConfig::NumStages;

  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::Int<tM>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::Int<tN>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));

  struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);
  static constexpr int kVec = 16 / sizeof(ElementInput);  // 8 for fp16/bf16

  __device__ void operator()(const ElementInput *ptr_A,
                             const ElementInput *ptr_B,
                             const int *idx_a,
                             const int *idx_b,
                             GroupedTrABGemmParams params,
                             int K_dim,
                             int N,
                             float alpha,
                             char *smem_buf) const {
    using namespace cute;
    // --- Group dispatch ---
    int g = int(blockIdx.z);
    int k_tile = int(blockIdx.x);  // tiles K_dim
    int n_tile = int(blockIdx.y);  // tiles N
    int k_start = k_tile * tM;
    int n_start = n_tile * tN;

    int gather_size_g = params.gather_sizes[g];
    int map_offset_g = params.map_offsets[g];
    ElementOutput *ptr_D_g = reinterpret_cast<ElementOutput *>(params.ptr_D_array[g]);
    const int *idx_a_g = idx_a + map_offset_g;
    const int *idx_b_g = idx_b + map_offset_g;

    // --- Standard TrAB mainloop ---
    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

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

    int num_g_tiles = (gather_size_g + tK - 1) / tK;
    auto K_BLOCK_MAX = size<2>(tCrA);

    if (num_g_tiles == 0) {
      _epilogue_trAB(accum, ptr_D_g, k_start, n_start, K_dim, N, alpha, tiled_mma);
      return;
    }

    // Prolog: load g_tile=0 into stage[0]
    _load_A_trAB(ptr_A, idx_a_g, sA(_, _, 0), k_start, 0, K_dim, gather_size_g);
    _load_B_trAB(ptr_B, idx_b_g, sB(_, _, 0), n_start, 0, N, gather_size_g);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for (int g_tile = 1; g_tile < num_g_tiles; ++g_tile) {
      int curr_stage = (g_tile - 1) % NumStages;
      int next_stage = g_tile % NumStages;
      int g_start = g_tile * tK;

      _load_A_trAB(ptr_A, idx_a_g, sA(_, _, next_stage), k_start, g_start, K_dim, gather_size_g);
      _load_B_trAB(ptr_B, idx_b_g, sB(_, _, next_stage), n_start, g_start, N, gather_size_g);
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

    // Epilog: compute last g_tile
    {
      int last_stage = (num_g_tiles - 1) % NumStages;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, last_stage), tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, last_stage), tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
      }
    }

    _epilogue_trAB(accum, ptr_D_g, k_start, n_start, K_dim, N, alpha, tiled_mma);
  }

private:
  /// Load A^T tile: vectorized LDG along K, scatter-store (transpose) to smem.
  template <class SmemTensor>
  __device__ void _load_A_trAB(const ElementInput *ptr_A,
                               const int *idx_a,
                               SmemTensor smem_tile,
                               int k_start,
                               int g_start,
                               int K_dim,
                               int gather_size) const {
    static_assert(tM % kVec == 0, "tM must be a multiple of vector width");
    constexpr int k_vecs = tM / kVec;
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
          auto *elems = reinterpret_cast<ElementInput *>(&vec_data);
          for (int v = 0; v < kVec; ++v) {
            if (k_global + v < K_dim) elems[v] = ptr_A[phys_row * K_dim + k_global + v];
          }
        }
      }
      auto *elems = reinterpret_cast<const ElementInput *>(&vec_data);
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < kVec; ++v) {
        smem_tile(k_local + v, g_local) = elems[v];
      }
    }
  }

  /// Load B tile: vectorized LDG along N, vectorized STS to N-contiguous smem.
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
      *reinterpret_cast<uint4 *>(&smem_tile(n_local, g_local)) = vec_data;
    }
  }

  /// Dense epilogue: D_g[k, n] = alpha * accum (no beta, no C)
  template <class Accumulator, class TiledMma_>
  __device__ void _epilogue_trAB(Accumulator &accum,
                                 ElementOutput *ptr_D,
                                 int k_start,
                                 int n_start,
                                 int K_dim,
                                 int N,
                                 float alpha,
                                 TiledMma_ &tiled_mma) const {
    using namespace cute;
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
        float result = alpha * float(accum(i));
        ptr_D[k_global * N + n_global] = static_cast<ElementOutput>(result);
      }
    }
  }
};

/// Global kernel entry point for grouped TrAB gather
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void cute_gemm_grouped_trAB_kernel_entry(
    const typename Kernel::ElementInput *ptr_A,
    const typename Kernel::ElementInput *ptr_B,
    const int *idx_a,
    const int *idx_b,
    GroupedTrABGemmParams params,
    int K_dim,
    int N,
    float alpha) {
  extern __shared__ char smem[];
  Kernel{}(ptr_A, ptr_B, idx_a, idx_b, params, K_dim, N, alpha, smem);
}

}  // namespace cute_gemm
}  // namespace warpconvnet
