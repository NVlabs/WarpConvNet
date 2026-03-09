// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SM90 (Hopper) WGMMA-based fused multi-offset grouped CuTe GEMM kernel
// for sparse convolution.
//
// This is the SM90 adaptation of cute_gemm_grouped_kernel.h. It uses:
// - WGMMA (SM90_64xNx16_F32F16F16_SS) — both operands from shared memory
// - GMMA-compatible smem layouts with 128-byte swizzle
// - warpgroup_arrive/commit_batch/wait synchronization
// - Multi-stage pipelining (NumStages from TileConfig, typically 4)
// - Element-wise gather for A (gmem->smem), dense copy for B (gmem->smem)
// - atomicAdd epilogue for output accumulation
//
// The binary search and gather/scatter logic is identical to SM80.

#pragma once

#if defined(WARPCONVNET_SM90_ENABLED)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/tensor.hpp"  // MUST come first for CUDA 12.9 compat
#include "cute_gemm_config_sm90.h"
#include "grouped_gemm_params.h"

namespace warpconvnet {
namespace cute_gemm {

// Reuse the atomic_add helpers and find_group from SM80 kernel
// (they are in cute_gemm_grouped_kernel.h, included transitively).
// If building this file standalone, the helpers are redefined here
// guarded to avoid ODR violations.

#ifndef WARPCONVNET_ATOMIC_ADD_DEFINED
#define WARPCONVNET_ATOMIC_ADD_DEFINED

/// atomicAdd wrapper that handles cutlass types by casting to native CUDA types.
template <typename T>
__device__ __forceinline__ void atomic_add_sm90(T *addr, T val) {
  atomicAdd(addr, val);
}

template <>
__device__ __forceinline__ void atomic_add_sm90<cutlass::half_t>(cutlass::half_t *addr,
                                                                 cutlass::half_t val) {
  atomicAdd(reinterpret_cast<__half *>(addr), __float2half(float(val)));
}

template <>
__device__ __forceinline__ void atomic_add_sm90<cutlass::bfloat16_t>(cutlass::bfloat16_t *addr,
                                                                     cutlass::bfloat16_t val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16 *>(addr), __float2bfloat16(float(val)));
}

#endif  // WARPCONVNET_ATOMIC_ADD_DEFINED

/// Binary search: find g such that tile_offsets[g] <= tile_idx < tile_offsets[g+1]
__device__ __forceinline__ int find_group_sm90(const int *tile_offsets,
                                               int num_groups,
                                               int tile_idx) {
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

/// SM90 WGMMA-based fused grouped GEMM kernel with gather on A and atomicAdd scatter on D.
template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmGroupedKernelSm90 {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;

  // SM90 WGMMA uses 128 threads (1 warp group)
  static constexpr int MaxThreadsPerBlock = 128;
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
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>, 128> smem_a;
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>, 128> smem_b;
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

    int g = find_group_sm90(params.tile_offsets, params.num_groups, global_m_tile);
    int local_m_tile = global_m_tile - params.tile_offsets[g];
    int m_start = local_m_tile * tM;
    int M_g = params.group_sizes[g];
    int map_offset_g = params.map_offsets[g];
    int n_start = n_tile * tN;

    // Per-group pointers
    const ElementInput *ptr_B_g = reinterpret_cast<const ElementInput *>(params.ptr_B_array[g]);
    const int *in_map_g = in_map + map_offset_g;
    const int *out_map_g = out_map + map_offset_g;

    // --- Step 2: SM90 WGMMA mainloop ---
    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

    // WGMMA setup: partition smem tensors and create descriptors
    TiledMma tiled_mma;

    // For SS WGMMA, all 128 threads in the warp group participate
    auto thread_mma = tiled_mma.get_slice(threadIdx.x);

    // Partition smem for A and B
    Tensor tCsA = thread_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)

    // Create GMMA descriptors from smem partitions
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)

    // Accumulator in registers
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<tM>{}, Int<tN>{}));
    clear(accum);

    int num_k_tiles = (K_dim + tK - 1) / tK;

    if (num_k_tiles == 0) {
      _epilogue_atomic(accum, ptr_D, out_map_g, m_start, n_start, M_g, N, alpha, tiled_mma);
      return;
    }

    // ==================== PROLOG: load k_tile=0 into stage[0] ====================
    _load_A(ptr_A, in_map_g, sA(_, _, 0), m_start, 0, M_g, K_dim);
    _load_dense_B_tile_cpasync(ptr_B_g, sB(_, _, 0), n_start, 0, N, K_dim);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // ==================== MAINLOOP with WGMMA ====================
    // Use ScaleOut::Zero for first iteration, then One for accumulation
    tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;

    auto K_BLOCK_MAX = size<2>(tCrA);

    if (num_k_tiles == 1) {
      // Single k-tile: compute and epilogue
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        cute::gemm(tiled_mma, tCrA(_, _, k_block, 0), tCrB(_, _, k_block, 0), accum);
        tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(accum);

      _epilogue_atomic(accum, ptr_D, out_map_g, m_start, n_start, M_g, N, alpha, tiled_mma);
      return;
    }

    // Multi k-tile: pipelined loads overlapped with WGMMA compute.
    // Structure matches cute_gemm_kernel_sm90.h: load(next) then compute(curr),
    // with cp_async_wait<NumStages-2> to keep older loads in flight and
    // warpgroup_wait<1> to overlap GMMA compute with next iteration's loads.
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 1; k_tile < num_k_tiles; ++k_tile) {
      int curr_stage = (k_tile - 1) % NumStages;
      int next_stage = k_tile % NumStages;
      int k_start = k_tile * tK;

      // Issue loads for NEXT k_tile into next_stage
      _load_A(ptr_A, in_map_g, sA(_, _, next_stage), m_start, k_start, M_g, K_dim);
      _load_dense_B_tile_cpasync(ptr_B_g, sB(_, _, next_stage), n_start, k_start, N, K_dim);
      cute::cp_async_fence();

      // Compute CURRENT k_tile from curr_stage via WGMMA
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        cute::gemm(
            tiled_mma, tCrA(_, _, k_block, curr_stage), tCrB(_, _, k_block, curr_stage), accum);
        tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<1>();  // Allow 1 GMMA batch in flight for compute/load overlap
      warpgroup_fence_operand(accum);

      // Wait for oldest in-flight loads to complete before next iteration
      // overwrites that stage
      cute::cp_async_wait<NumStages - 2>();
      __syncthreads();
    }

    // Epilog: compute last k_tile, drain all pipelines before reading accum
    {
      int last_stage = (num_k_tiles - 1) % NumStages;
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        cute::gemm(
            tiled_mma, tCrA(_, _, k_block, last_stage), tCrB(_, _, k_block, last_stage), accum);
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();  // Drain all GMMA batches before epilogue reads accum
      warpgroup_fence_operand(accum);
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

    // For WGMMA, the accumulator is distributed across the warp group.
    // We use partition_C on an identity tensor to get the (m,n) coordinates
    // that each thread owns in the accumulator.
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
        atomic_add_sm90(&ptr_D[phys_row * N + n_global], static_cast<ElementOutput>(result));
      }
    }
  }
};

/// Global kernel entry point for SM90 grouped AD gather-scatter
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void cute_gemm_grouped_kernel_sm90_entry(
    const typename Kernel::ElementInput *ptr_A,
    typename Kernel::ElementOutput *ptr_D,
    const int *in_map,
    const int *out_map,
    GroupedGemmParams params,
    int N,
    int K_dim,
    float alpha) {
  // WGMMA instructions are only available on SM90+. On older arches this kernel
  // exists as a stub so that host-side launcher code compiles for fat binaries.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ char smem[];
  Kernel{}(ptr_A, ptr_D, in_map, out_map, params, N, K_dim, alpha, smem);
#endif  // __CUDA_ARCH__ >= 900
}

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
