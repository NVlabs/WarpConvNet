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

// Strided sparse_conv forward kernel - neighbor_map driven gather.
// Sister of MaskGemm_forward_128x64x32_2s_fused but for N_in != N_out
// downsample layers. Bond #24 - FlexGEMM strided port.
template <class TileConfig, typename ElementOutput_ = float, int MaskWords_ = 1>
struct MaskGemm_forward_128x64x32_2s_fused_strided {
  static constexpr int MaskWords = MaskWords_;  // unused; kept for ABI symmetry
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
  // Path A multi-stage: 2 smem stages for cp.async double-buffered pipeline.
  static constexpr int NumStages = 2;
  static constexpr int NumWarps = MaxThreadsPerBlock / 32;
  static constexpr int kVec = 16 / sizeof(ElementInput);
  static constexpr int kMmaK = cute::size<2>(typename TiledMma::AtomShape_MNK{});
  static constexpr int K_BLOCK_MAX_STATIC = tK / kMmaK;
  static constexpr bool UseSmemEpilogue = true;
  static constexpr bool UseScalarEpilogue = false;
  static constexpr bool IsStrided = true;

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

  __device__ void operator()(const ElementInput *ptr_A_base,
                             const ElementInput *ptr_B_base,
                             ElementOutput *ptr_D_base,
                             const int *neighbor_map,
                             int N_in,
                             int N_out,
                             int C_in,
                             int C_out,
                             int V,
                             float alpha,
                             int stride_A,
                             int stride_D,
                             char *smem_buf) const {
    using namespace cute;

    int group_id = int(blockIdx.z);
    int groups = int(gridDim.z);
    const ElementInput *ptr_A = ptr_A_base + group_id * C_in;
    const ElementInput *ptr_B = ptr_B_base + group_id * C_in * C_out;
    ElementOutput *ptr_D = ptr_D_base + group_id * C_out;
    int stride_B_V = groups * C_in * C_out;

    int grid_n = (C_out + tN - 1) / tN;
    int m_tile = int(blockIdx.x) / grid_n;
    int n_tile = int(blockIdx.x) % grid_n;
    int m_start = m_tile * tM;
    int n_start = n_tile * tN;

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<tM>{}, Int<tN>{}));
    clear(accum);

    // Double-buffered MMA fragments — required by MMA_DOUBLE_BUFFERED macro.
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

    // Step counting: (V, num_kc) outer × inner = total contractions.
    int num_kc = (C_in + tK - 1) / tK;
    int total_steps = V * num_kc;
    if (total_steps == 0) goto fwd_strided_epilogue;

    {
      // Prolog: load step 0 into smem stage 0.
      int v0 = 0, kc0 = 0;
      _load_A_strided(
          ptr_A, neighbor_map, v0, m_start, sA(_, _, 0), kc0 * tK, N_in, N_out, C_in, stride_A);
      _load_B_strided(ptr_B + (size_t)v0 * stride_B_V, sB(_, _, 0), n_start, kc0 * tK, C_out, C_in);
      cute::cp_async_fence();
      cute::cp_async_wait<0>();
      __syncthreads();

      int smem_stage = 0;
      // Main loop: alternate load + MMA across stages.
      for (int step = 1; step < total_steps; ++step) {
        int v = step / num_kc;
        int kc = (step % num_kc) * tK;
        int write_stage = (smem_stage + 1) % NumStages;

        _load_A_strided(ptr_A,
                        neighbor_map,
                        v,
                        m_start,
                        sA(_, _, write_stage),
                        kc,
                        N_in,
                        N_out,
                        C_in,
                        stride_A);
        _load_B_strided(
            ptr_B + (size_t)v * stride_B_V, sB(_, _, write_stage), n_start, kc, C_out, C_in);
        cute::cp_async_fence();

        MMA_DOUBLE_BUFFERED(smem_stage)

        cute::cp_async_wait<0>();
        __syncthreads();
        smem_stage = write_stage;
      }

      // Drain final stage.
      MMA_DOUBLE_BUFFERED(smem_stage)
      __syncthreads();
    }

  fwd_strided_epilogue:

    auto thr_mma_epi = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrC = thr_mma_epi.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));
    CUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      auto coord = tCrC(i);
      int m_global = m_start + get<0>(coord);
      int n_global = n_start + get<1>(coord);
      if (m_global < N_out && n_global < C_out) {
        float val = (alpha == 1.0f) ? float(accum(i)) : alpha * float(accum(i));
        ptr_D[m_global * stride_D + n_global] = static_cast<ElementOutput>(val);
      }
    }
  }

private:
  // Strided fwd A loader: gather X rows via neighbor_map[v, m].
  // Sentinel in_row = -1 -> zero-fill smem (masked).
  template <class SmemTensor>
  __device__ void _load_A_strided(const ElementInput *ptr_A,
                                  const int *neighbor_map,
                                  int v,
                                  int m_start,
                                  SmemTensor smem_tile,
                                  int k_start,
                                  int N_in,
                                  int N_out,
                                  int C_in,
                                  int stride_A) const {
    int _stride_bytes = stride_A * (int)sizeof(ElementInput);
    bool ptr_A_aligned =
        ((reinterpret_cast<uintptr_t>(ptr_A) | (uintptr_t)_stride_bytes) & 15u) == 0;
    constexpr int kItersPerThread = (tK / kVec * tM + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;
    const int *nm_row = neighbor_map + v * N_out;
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < kItersPerThread; ++iter) {
      int idx = threadIdx.x + iter * MaxThreadsPerBlock;
      if (idx >= tK / kVec * tM) break;
      int m_local = idx / (tK / kVec);
      int kv = idx % (tK / kVec);
      int k_local = kv * kVec;
      int k_global = k_start + k_local;
      int m_global = m_start + m_local;
      int in_row = (m_global < N_out) ? __ldg(&nm_row[m_global]) : -1;
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(m_local, k_local));
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
          for (int v_e = 0; v_e < kVec; ++v_e) {
            int k = k_global + v_e;
            if (k < C_in) dst[v_e] = row_ptr[k];
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

  // Strided fwd B loader: dense gather of W[v, k_chunk:tK, n_start:tN].
  // Weight gmem layout (per-group): [V, Cig, Cog] (group offset already
  // applied by caller, v offset applied here via stride_B_V).
  // Smem layout: (tN, tK) — N is the contiguous axis.
  template <class SmemTensor>
  __device__ void _load_B_strided(const ElementInput *ptr_B_v,
                                  SmemTensor smem_tile,
                                  int n_start,
                                  int k_start,
                                  int C_out,
                                  int C_in) const {
    constexpr int kVecsPerRow = tN / kVec;
    constexpr int kTotalVecs = tK * kVecsPerRow;
    // 128-bit cp.async requires a 16B-aligned source. Row stride is C_out
    // elements; the weight base ptr_B_v can be 2/4/8B-aligned (DeepSpeed
    // ZeRO contiguous-view weights are NOT 16B-aligned). Guard the vector
    // path on (base | C_out*sizeof) — mirror _load_A_strided / _load_B_tile.
    // Without this, a misaligned weight base faults "misaligned address".
    int _b_stride_bytes = C_out * (int)sizeof(ElementInput);
    bool ptr_B_aligned =
        ((reinterpret_cast<uintptr_t>(ptr_B_v) | (uintptr_t)_b_stride_bytes) & 15u) == 0;
    CUTLASS_PRAGMA_UNROLL
    for (int idx = threadIdx.x; idx < kTotalVecs; idx += MaxThreadsPerBlock) {
      int ki = idx / kVecsPerRow;
      int ni_v = idx % kVecsPerRow;
      int n_local = ni_v * kVec;
      int n_global = n_start + n_local;
      int k_global = k_start + ki;
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, ki));
      bool k_valid = (k_global < C_in);
      bool n_full = (n_global + kVec <= C_out);
      if (k_valid && n_full && ptr_B_aligned) {
        const void *src = &ptr_B_v[(size_t)k_global * C_out + n_global];
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr),
                     "l"(src),
                     "n"(16));
      } else {
        int4 frag = make_int4(0, 0, 0, 0);
        if (k_valid) {
          const ElementInput *row = ptr_B_v + (size_t)k_global * C_out;
          ElementInput *dst = reinterpret_cast<ElementInput *>(&frag);
          CUTLASS_PRAGMA_UNROLL
          for (int v_e = 0; v_e < kVec; ++v_e) {
            int n = n_global + v_e;
            if (n < C_out) dst[v_e] = row[n];
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
};
}  // namespace cute_gemm
}  // namespace warpconvnet
