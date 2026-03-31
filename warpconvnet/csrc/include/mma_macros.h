// Common MMA macros for generated kernels
// Created: 2026-04-02 17:00:00
// Edited: 2026-04-02 19:30:00
#pragma once

// Double-buffered MMA: prefetch kb+1 while computing kb
// Requires: tCrA_0, tCrA_1, tCrB_0, tCrB_1, tCrA_copy_0, tCrA_copy_1,
//           tCrB_copy_0, tCrB_copy_1, tCsA, tCsB, smem_tiled_copy_A/B,
//           tiled_mma, K_BLOCK_MAX, accum
#define MMA_DOUBLE_BUFFERED(STAGE)                                               \
  copy(smem_tiled_copy_A, tCsA(_, _, 0, STAGE), tCrA_copy_0(_, _, 0));           \
  copy(smem_tiled_copy_B, tCsB(_, _, 0, STAGE), tCrB_copy_0(_, _, 0));           \
  _Pragma("unroll") for (int kb = 0; kb < K_BLOCK_MAX; ++kb) {                   \
    if (kb + 1 < K_BLOCK_MAX) {                                                  \
      int nkb = kb + 1;                                                          \
      if (kb % 2 == 0) {                                                         \
        copy(smem_tiled_copy_A, tCsA(_, _, nkb, STAGE), tCrA_copy_1(_, _, nkb)); \
        copy(smem_tiled_copy_B, tCsB(_, _, nkb, STAGE), tCrB_copy_1(_, _, nkb)); \
      } else {                                                                   \
        copy(smem_tiled_copy_A, tCsA(_, _, nkb, STAGE), tCrA_copy_0(_, _, nkb)); \
        copy(smem_tiled_copy_B, tCsB(_, _, nkb, STAGE), tCrB_copy_0(_, _, nkb)); \
      }                                                                          \
    }                                                                            \
    if (kb % 2 == 0)                                                             \
      cute::gemm(tiled_mma, tCrA_0(_, _, kb), tCrB_0(_, _, kb), accum);          \
    else                                                                         \
      cute::gemm(tiled_mma, tCrA_1(_, _, kb), tCrB_1(_, _, kb), accum);          \
  }
