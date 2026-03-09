// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SM90 (Hopper) WGMMA GEMM kernel with manual gather/scatter and multi-stage
// pipelining.
//
// Operand A: gathered via indices -> element-wise LDG + STS to GMMA-compatible
//            K-major smem (Layout_K_SW128_Atom swizzle).
// Operand B: dense -> cp.async 128-bit gmem->smem to GMMA-compatible
//            MN-major smem (Layout_MN_SW128_Atom swizzle).
//
// The mainloop uses NumStages-deep double/multi-buffering: while WGMMA computes
// on the current K-tile (stage[curr]), the next K-tile is being loaded into
// stage[next]. Both operands are read from shared memory by the WGMMA
// instruction (SS variant) -- no smem->register copy is needed.
//
// WGMMA synchronization pattern (per K-tile):
//   warpgroup_fence_operand(accum);
//   warpgroup_arrive();
//   cute::gemm(tiled_mma, tCrA(...), tCrB(...), accum);
//   warpgroup_commit_batch();
//   warpgroup_wait<1>();       // mainloop: allow 1 batch in flight
//   warpgroup_wait<0>();       // epilog: drain all before reading accum
//   warpgroup_fence_operand(accum);

#pragma once

#if defined(WARPCONVNET_SM90_ENABLED)

#include "cute/tensor.hpp"  // MUST come before cute/algorithm headers for CUDA 12.8+ compat
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/copy_sm80.hpp"      // cp_async_fence, cp_async_wait
#include "cute/arch/mma_sm90_gmma.hpp"  // warpgroup_arrive, warpgroup_commit_batch, etc.
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute_gemm_config_sm90.h"

// This header requires <cuda.h> for CUtensorMap (TMA descriptors).
// Include it in the .cu translation unit before this header.

namespace warpconvnet {
namespace cute_gemm {

/// SM90 WGMMA GEMM kernel with manual gather/scatter and pipelined mainloop.
///
/// Both operands A and B are read from shared memory by the WGMMA instruction
/// (SS variant). The gather for operand A is done element-wise during the
/// gmem->smem copy phase, writing into GMMA-compatible swizzled smem layout.
template <class TileConfig, typename ElementOutput_ = float>
struct CuteGemmKernelSm90 {
  using TileShape = typename TileConfig::TileShape;
  using TiledMma = typename TileConfig::TiledMma;
  using ElementInput = typename TileConfig::ElementInput;
  using ElementOutput = ElementOutput_;

  using SmemLayoutAtomA = typename TileConfig::SmemLayoutAtomA;
  using SmemLayoutAtomB = typename TileConfig::SmemLayoutAtomB;

  static constexpr int tM = cute::size<0>(TileShape{});
  static constexpr int tN = cute::size<1>(TileShape{});
  static constexpr int tK = cute::size<2>(TileShape{});

  // WGMMA SS: 128 threads per warp group. Tiles with tM > 64 use multiple
  // warp groups (tM/64), each requiring 128 threads.
  static constexpr int NumWarpGroups = tM / 64;
  static constexpr int MaxThreadsPerBlock = 128 * NumWarpGroups;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int NumStages = TileConfig::NumStages;

  // When true, gathered A uses cp.async (async gmem->smem, register bypass).
  // When false (default), gathered A uses synchronous LDG.128 + STS.128.
  static constexpr bool UseCpAsyncGatherA = TileConfig::UseCpAsyncGatherA;

  // TMA box width in elements (128-byte swizzle requires exactly 128 bytes in fast dim).
  static constexpr int kTmaBoxN = 128 / sizeof(ElementInput);  // 64 for fp16/bf16

  // TMA for dense B is only used when tN equals the TMA box width (single sub-tile).
  // For wider tiles (tN > kTmaBoxN), multi-sub-tile TMA loads would require matching
  // the CuTe tile_to_shape swizzle pattern exactly, which is non-trivial.
  // Fall back to cp.async for those cases.
  static constexpr bool UseTmaLoadB = TileConfig::UseTmaLoadB && (tN == kTmaBoxN);

  // 3D smem layouts: (M/N, K, Stages) -- third dimension indexes pipeline stages
  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::Int<tM>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::Int<tN>{}, cute::Int<tK>{}, cute::Int<NumStages>{})));

  struct SharedStorage {
    // 128-byte aligned for GMMA descriptor compatibility
    alignas(128) cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>, 128> smem_a;
    alignas(128) cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>, 128> smem_b;
    // mbarrier for TMA completion tracking (one per pipeline stage)
    // Only used when UseTmaLoadB = true; occupies 8*NumStages bytes.
    alignas(8) uint64_t tma_barriers[NumStages];
  };

  static constexpr size_t SharedStorageSize = sizeof(SharedStorage);

  /// Main kernel with multi-stage pipelined mainloop using WGMMA.
  /// When UseTmaLoadB is true, tma_desc_B must point to a valid CUtensorMap for B.
  /// When UseTmaLoadB is false, tma_desc_B is ignored and B is loaded via cp.async.
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
                             char *smem_buf,
                             const CUtensorMap *tma_desc_B = nullptr) const {
    using namespace cute;
    int m_tile = int(blockIdx.x);
    int n_tile = int(blockIdx.y);
    int m_start = m_tile * tM;
    int n_start = n_tile * tN;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    Tensor sA =
        make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});  // (tM, tK, NumStages)
    Tensor sB =
        make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});  // (tN, tK, NumStages)

    // MMA setup -- WGMMA SS uses 128 threads (1 warp group)
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // Accumulator in registers
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<tM>{}, Int<tN>{}));
    clear(accum);

    // Partition smem tensors for WGMMA
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);

    // Create smem descriptor fragments (GMMA descriptors for SS variant)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA, MMA_M, MMA_K, NumStages)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA, MMA_N, MMA_K, NumStages)

    int num_k_tiles = (K_dim + tK - 1) / tK;
    auto K_BLOCK_MAX = size<2>(tCrA);

    // Early exit for zero-K GEMM
    if (num_k_tiles == 0) {
      _epilogue(accum, ptr_C, ptr_D, out_map, m_start, n_start, M, N, alpha, beta, tiled_mma);
      return;
    }

    // TMA: initialize mbarrier pipeline and phase tracking
    [[maybe_unused]] int tma_phases[NumStages] = {};
    if constexpr (UseTmaLoadB) {
      _init_tma_barriers(storage.tma_barriers);
    }

    // ==================== MAINLOOP: load then compute each K-tile ====================
    // Uses double-buffering (2 stages) with full drain between iterations.
    // The load/compute overlap comes from WGMMA's asynchronous execution —
    // warpgroup_wait<1> allows GMMA to overlap with the next load phase.
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
      int stage = k_tile % NumStages;
      int k_start = k_tile * tK;

      // Load A and B for this K-tile
      _load_A(ptr_A, in_map, sA(_, _, stage), m_start, k_start, M, K_dim);
      if constexpr (UseTmaLoadB) {
        cute::cp_async_fence();
        _load_dense_B_tile_tma(*tma_desc_B, sB(_, _, stage), &storage.tma_barriers[stage], n_start,
                               k_start);
        cute::cp_async_wait<0>();
        _wait_tma_barrier(&storage.tma_barriers[stage], tma_phases[stage]);
        tma_phases[stage] ^= 1;
      } else {
        _load_dense_B_tile_cpasync(ptr_B, sB(_, _, stage), n_start, k_start, N, K_dim);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
      }
      __syncthreads();

      // Compute this K-tile via WGMMA
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        cute::gemm(tiled_mma, tCrA(_, _, k_block, stage), tCrB(_, _, k_block, stage), accum);
      }
      warpgroup_commit_batch();
      if (k_tile < num_k_tiles - 1) {
        warpgroup_wait<1>();  // Allow 1 batch in flight for load overlap
      } else {
        warpgroup_wait<0>();  // Drain all before epilogue
      }
      warpgroup_fence_operand(accum);
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
  /// K is contiguous in gmem (within each row). The GMMA K-major swizzle
  /// (Layout_K_SW128_Atom) preserves 8-element contiguity along K,
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

  /// Load a gathered A tile into smem using cp.async (128-bit async gmem->smem).
  ///
  /// A is (M_phys, K_dim) row-major, gathered by gather_map.
  /// K is contiguous in both gmem (within each gathered row) and smem.
  /// The GMMA K-major swizzle preserves 8-element contiguity along K,
  /// so cp.async 128-bit transfers are aligned for both gmem source and smem dest.
  ///
  /// Each thread resolves the gather index (synchronous LDG for the int32 index),
  /// then issues cp.async for the 128-bit K-contiguous data (async gmem->smem,
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

      // Smem destination: 16-byte aligned (GMMA K-major swizzle preserves 8-elem contiguity)
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(m_local, k_local));

      bool pred = (m_global < M_phys) && (k_global + kVec <= K_dim);

      if (pred) {
        // Resolve gather index (synchronous 4-byte LDG for the int32 index)
        int phys_row = gather_map[m_global];
        // 128-bit cp.async: gmem -> smem, bypasses registers
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

  /// Load a dense B tile into smem using cp.async (128-bit async gmem->smem).
  ///
  /// B is (K, N) row-major: B[k, n] = ptr[k * N + n]. N is contiguous in gmem.
  /// Smem B stores (tN, tK) with N-contiguous layout (Layout_MN_SW128_Atom).
  /// cp.async bypasses registers -- the copy engine transfers data directly
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

      // Smem destination: 16-byte aligned (GMMA MN-major swizzle preserves 8-elem contiguity)
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));

      bool pred = (k_global < K_dim) && (n_global + kVec <= N);

      if (pred) {
        // 128-bit cp.async: gmem -> smem, bypasses registers
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

  // ---- TMA B load helpers ----

  /// Initialize mbarriers for TMA pipeline stages. Must be called once at kernel start.
  __device__ void _init_tma_barriers(uint64_t *barriers) const {
    if (threadIdx.x == 0) {
      for (int s = 0; s < NumStages; ++s) {
        // Initialize barrier with expected arrival count = 1 (single TMA operation)
        uint32_t bar_addr = _smem_uint32(&barriers[s]);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(bar_addr), "r"(1));
      }
    }
    __syncthreads();
  }

  /// Convert a shared memory pointer to a uint32_t smem address for PTX instructions.
  __device__ static uint32_t _smem_uint32(const void *smem_ptr) {
    uint32_t addr;
    asm("{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr)
        : "l"(smem_ptr));
    return addr;
  }

  /// Load a dense B tile into smem using TMA (Tensor Memory Accelerator).
  /// Only thread 0 issues the TMA instruction(s); the mbarrier tracks completion.
  /// The TMA descriptor encodes a (box_n, tK) sub-tile with 128-byte swizzle.
  /// For tN > box_n, we issue tN/box_n TMA loads targeting the same mbarrier.
  ///
  /// Since UseTmaLoadB is only true when tN == kTmaBoxN, a single TMA instruction
  /// loads the entire (tN, tK) tile. The mbarrier tracks completion.
  template <class SmemTensor>
  __device__ void _load_dense_B_tile_tma(const CUtensorMap &tma_desc_B,
                                         SmemTensor smem_tile,
                                         uint64_t *barrier,
                                         int n_start,
                                         int k_start) const {
    uint32_t smem_addr = _smem_uint32(&smem_tile(0, 0));
    uint32_t barrier_addr = _smem_uint32(barrier);

    if (threadIdx.x == 0) {
      // Set expected transaction bytes
      constexpr uint32_t tma_bytes = tN * tK * sizeof(ElementInput);
      asm volatile(
          "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(barrier_addr),
          "r"(tma_bytes));

      // Single TMA load for the entire tile (tN == kTmaBoxN guaranteed by UseTmaLoadB)
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_addr),
          "l"(&tma_desc_B), "r"(n_start), "r"(k_start), "r"(barrier_addr));
    }
  }

  /// Wait for TMA load completion on a barrier using try_wait with phase tracking.
  __device__ void _wait_tma_barrier(uint64_t *barrier, int phase) const {
    uint32_t barrier_addr = _smem_uint32(barrier);

    // Spin-wait on mbarrier phase
    asm volatile(
        "{\n"
        "  .reg .pred P1;\n"
        "WAIT_LOOP:\n"
        "  mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
        "  @!P1 bra WAIT_LOOP;\n"
        "}\n" ::"r"(barrier_addr),
        "r"(phase));
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

/// Global kernel entry point (AD gather-scatter, SM90) — cp.async B path
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void cute_gemm_sm90_kernel_entry(
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ char smem[];
  Kernel{}(ptr_A, ptr_B, ptr_C, ptr_D, in_map, out_map, M, N, K, alpha, beta, smem);
#endif
}

/// Global kernel entry point (AD gather-scatter, SM90) — TMA B path
/// The TMA descriptor for B is passed by value (copied to constant memory by the driver).
template <class Kernel>
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock) void cute_gemm_sm90_kernel_entry_tma(
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
    float beta,
    const __grid_constant__ CUtensorMap tma_desc_B) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ char smem[];
  Kernel{}(ptr_A, ptr_B, ptr_C, ptr_D, in_map, out_map, M, N, K, alpha, beta, smem, &tma_desc_B);
#endif
}

}  // namespace cute_gemm
}  // namespace warpconvnet

#endif  // WARPCONVNET_SM90_ENABLED
