# CuTe 3.x GEMM Kernel with Gather/Scatter — Implementation Reference

## Overview

This document describes the CuTe 3.x GEMM kernel for sparse convolution in WarpConvNet. The kernel computes:

```
D[out_map[i], j] = alpha * sum_k(A[in_map[i], k] * B[k, j]) + beta * C[out_map[i], j]
```

where `in_map` and `out_map` are integer gather/scatter index arrays that select which rows of A to read and which rows of D to write. This operation is the core of explicit-GEMM sparse convolution, where the kernel map (produced by coordinate hashing) determines how input features are gathered for each convolution kernel offset, and how outputs are scattered back to the output feature tensor.

The implementation uses CUTLASS 3.x CuTe abstractions for tensor core operations (TiledMMA, LDSM copy atoms, swizzled shared memory layouts) while handling the irregular gather/scatter pattern through manual global memory loads with vectorized 128-bit transfers where possible.

### Files

| File | Purpose |
|------|---------|
| `csrc/include/cute_gemm_config.h` | Tile configurations: MMA atoms, smem layouts, copy atoms for 8 (dtype x tile) combos |
| `csrc/include/cute_gemm_kernel.h` | Device kernel: gmem load, smem-to-reg LDSM copy, MMA compute, epilogue |
| `csrc/include/cute_gemm_launch.h` | Host launcher: grid/block dims, smem allocation, kernel dispatch |
| `csrc/include/cute_gather_tensor.hpp` | Re-exports CUTLASS `gather_tensor.hpp` utilities (currently unused at runtime, kept for future ComposedLayout work) |
| `csrc/cutlass_cute_gemm_gather_scatter.cu` | Explicit template instantiations for all 8 (dtype x tile) combinations |
| `csrc/bindings/gemm_bindings.cpp` | pybind11 dispatch: dtype routing, tensor validation, Python-facing API |
| `tests/csrc/test_cutlass_cute_gemm.py` | 25 correctness tests (tiles, dtypes, beta accumulation, edge cases, cross-validation vs 2.x) |
| `tests/csrc/benchmark_cute_gemm.py` | Performance benchmark vs CUTLASS 2.x and PyTorch |

---

## Architecture

### Data Flow

```
Global Memory                  Shared Memory                Registers
============                  =============                =========

A[in_map[i], k] ──cp.async──>  sA(m, k, stage)  ──LDSM_N──>  tCrA(MMA,MMA_M,MMA_K)
                  128-bit       Swizzle<2,3,3>     retile_D
                  gmem→smem     K-contiguous
                  (gather idx
                   via LDG)                              │
                                                                         │
B[k, n]         ──cp.async──>  sB(n, k, stage)  ──LDSM_T──>  tCrB(MMA,MMA_N,MMA_K)
                  128-bit       Swizzle<3,3,3>     retile_D              │
                  gmem→smem     N-contiguous       (transpose)           │
                                                                    ┌────▼────┐
                                                                    │ TiledMMA │
                                                                    │ (TC ops) │
                                                                    └────┬────┘
                                                                         │
D[out_map[i], j] <──element── accum(i)                                   │
                    wise store  = alpha * accum + beta * C[out_map[i], j] ┘

Pipeline: 2-stage double-buffering (load stage[next] while computing stage[curr])
```

### Why Manual Loads for Gathered A

CUTLASS 3.x `CollectiveMma` with `MainloopSm80CpAsyncUnpredicated` uses vectorized 128-bit `cp.async` for gmem-to-smem. The gather operation (`A[in_map[i], :]`) introduces indirection that breaks contiguity in the M dimension — each row may come from an arbitrary physical location in A.

Several approaches were attempted to express gather as a CuTe layout property:

1. **ComposedLayout with cp.async**: The `upcast<N>` specialization for `ComposedLayout` (needed by vectorized copies) requires a `ScaledBasis` stride in the inner layout. After CuTe's `partition_S` / `tile2thrfrg` / `compose`, the `ComposedLayout` gets flattened to a regular `Layout` with dynamic strides, losing `ScaledBasis`. The vectorized copy atom then fails with "src failed to vectorize into registers."

2. **Flat Layout with CustomStride**: Placing `CustomStride<IndexedGather, Stride>` in a flat layout causes compilation errors because CuTe's internal operations don't support `CustomStride` as a stride element in tuples.

3. **Manual vectorized loads (current approach)**: Each thread computes its gathered source address, then issues vectorized 128-bit LDG.128 along K (which is contiguous within each gathered row) and 128-bit STS.128 to swizzled smem. Dense B uses 128-bit `cp.async` (gmem→smem bypassing registers) with N-contiguous layout and transposing LDSM for smem→register.

### Kernel Structure

```
CuteGemmKernel<TileConfig>::operator()
│
├── Setup: 3D smem tensors sA(tM,tK,NumStages), sB(tN,tK,NumStages)
├── MMA setup: TiledMma, accum = partition_fragment_C(...)
├── Register fragments: tCrA = partition_fragment_A(sA(_,_,0))
├── LDSM copy setup:
│   ├── smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA, tiled_mma)  // LDSM_N
│   ├── smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB, tiled_mma)  // LDSM_T
│   ├── tCsA = partition_S(sA)  // (CPY,CPY_M,CPY_K,NumStages) — 4D
│   └── tCrA_copy_view = retile_D(tCrA)
│
├── PROLOG: load k_tile=0 into stage[0]
│   ├── _load_gathered_tile(A, in_map, sA(_,_,0), ...)   // LDG.128 + STS.128
│   ├── _load_dense_B_tile_cpasync(B, sB(_,_,0), ...)    // cp.async 128-bit
│   ├── cp_async_fence() → cp_async_wait<0>()
│   └── __syncthreads()
│
├── MAINLOOP (k_tile = 1..num_k_tiles-1):
│   ├── Load NEXT into stage[next]:
│   │   ├── _load_gathered_tile(A, in_map, sA(_,_,next), ...)
│   │   ├── _load_dense_B_tile_cpasync(B, sB(_,_,next), ...)
│   │   └── cp_async_fence()
│   ├── Compute CURRENT from stage[curr]:
│   │   ├── copy(smem_tiled_copy_A, tCsA(_,_,k,curr), ...)  // LDSM_N
│   │   ├── copy(smem_tiled_copy_B, tCsB(_,_,k,curr), ...)  // LDSM_T
│   │   └── cute::gemm(tiled_mma, tCrA(k), tCrB(k), accum)
│   ├── cp_async_wait<0>()
│   └── __syncthreads()
│
├── EPILOG: compute last k_tile from stage[last]
│
└── _epilogue: D[out_map[i], j] = alpha * accum(i,j) + beta * C[out_map[i], j]
```

---

## Tile Configurations

All configurations use the same warp layout, matching CUTLASS's `DefaultGemmConfigurationToCutlass3Types` for SM80:

| Property | Value |
|----------|-------|
| Warp layout | `Layout<Shape<_2, _2, _1>>` (2x2 = 4 warps, 128 threads) |
| MMA Tile | `Tile<_32, _32, _16>` (32x32 output per step, K=16 per MMA) |
| MMA Atom (FP16) | `SM80_16x8x16_F32F16F16F32_TN` |
| MMA Atom (BF16) | `SM80_16x8x16_F32BF16BF16F32_TN` |
| SmemLayoutAtomA | `Swizzle<2,3,3>` + `(8,32) Stride<32,1>` — K-contiguous |
| SmemLayoutAtomB | `Swizzle<3,3,3>` + `(64,8) Stride<1,64>` — N-contiguous |
| SmemCopyAtomA | `Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>` (non-transposing) |
| SmemCopyAtomB | `Copy_Atom<SM75_U16x8_LDSM_T, ElementInput>` (transposing) |
| Accumulator | FP32 |
| Pipeline stages | 2 (double-buffered) |

### SmemLayoutAtom Details

**Operand A** — `Swizzle<2,3,3>` with `Shape<_8,_32>, Stride<_32,_1>`:
- K-contiguous storage (stride-1 in K), 8-row × 32-column atom
- `Swizzle<B=2, M=3, S=3>` XORs bits `[3:5)` with bits `[6:8)`, eliminating bank conflicts
- Preserves 8-element contiguity along K, enabling 128-bit vectorized LDG+STS
- From CUTLASS `DefaultGemm_TensorOpSm80_OperandA<half_t, RowMajor>`

**Operand B** — `Swizzle<3,3,3>` with `Shape<_64,_8>, Stride<_1,_64>`:
- N-contiguous storage (stride-1 in N), 64-row × 8-column atom
- Matches gmem layout (B is row-major, N is contiguous), enabling 128-bit cp.async
- Used with `SM75_U16x8_LDSM_T` which transposes N-contiguous smem data to K-contiguous registers for MMA
- From CUTLASS `DefaultGemm_TensorOpSm80_OperandB<half_t, RowMajor>` (maps to OperandA ColumnMajor)

### MMA Iteration Counts

For a tile of shape (tM, tN, tK) with 2x2 warps and `Tile<_32, _32, _16>`:

| Tile | M steps | N steps | K blocks | Total MMA atoms per thread |
|------|---------|---------|----------|---------------------------|
| 64x64x32 | 2 | 2 | 2 | 8 |
| 128x64x32 | 4 | 2 | 2 | 16 |
| 64x128x32 | 2 | 4 | 2 | 16 |
| 128x128x32 | 4 | 4 | 2 | 32 |

Each MMA step covers 32x32x16 of the output tile. `K_BLOCK_MAX = size<2>(tCrA) = tK / 16 = 2`.

### Why 2x2 Warps for All Tiles

Earlier attempts used larger warp layouts (4x2, 2x4, 4x4) for larger tiles to increase occupancy. This failed because:

- `make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, tiled_mma)` with >2 warps in N gives "TiledCopy uses too few vals for selected CopyAtom" — the B fragment per-warp becomes too small for the 4xU32 LDSM granularity.
- With >2 warps in M, smem copy partition sizes (CPY_K) mismatch between source and retiled destination.

The CUTLASS default config for SM80 FP16 GEMM (`DefaultGemmConfigurationToCutlass3Types`) also uses `Layout<Shape<_2,_2,_1>>` for a 128x128x32 tile, covering the full tile through iteration rather than thread count.

---

## Key Implementation Details

### The retile_D Pattern

The `retile_D()` call is essential for the smem-to-register copy path. Without it, the MMA register fragment layout doesn't match the LDSM copy atom's destination layout:

```cpp
// MMA fragment (partitioned by thr_mma)
Tensor tCrA = thr_mma.partition_fragment_A(sA);  // (MMA, MMA_M, MMA_K)

// LDSM copy partitions
Tensor tCsA           = smem_thr_copy_A.partition_S(sA);       // (CPY, CPY_M, CPY_K) — smem source
Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);       // (CPY, CPY_M, CPY_K) — register dest

// Copy uses retiled view, MMA uses original fragment
copy(smem_tiled_copy_A, tCsA(_, _, k), tCrA_copy_view(_, _, k));  // LDSM
cute::gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), accum);        // MMA
```

`retile_D(tCrA)` creates a **view** of the same register data with a layout that matches `partition_S(sA)` in modes 0 and 1 (the per-thread element count and the M/N division). This is the canonical pattern from `sm80_mma_multistage.hpp` (lines 251-256):

```cpp
// From CUTLASS sm80_mma_multistage.hpp:
Tensor tCsA           = smem_thr_copy_A.partition_S(sA);              // (CPY,CPY_M,CPY_K,PIPE)
Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);              // (CPY,CPY_M,CPY_K)
CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));      // CPY_M
CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));      // CPY_K
```

### Gmem Load Strategies

Both operands use 128-bit `cp.async` for async gmem→smem transfers, bypassing registers.

**Operand A (gathered)** — cp.async with manual gather:

Each thread resolves the gather index (synchronous 4-byte int32 LDG), then issues cp.async for the 128-bit K-contiguous data along the gathered row:

```cpp
int phys_row = gather_map[m_global];  // sync LDG for index
const void *gmem_src = &ptr_A[phys_row * K_dim + k_global];
uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(m_local, k_local));
asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
    :: "r"(smem_addr), "l"(gmem_src), "n"(16));
```

**Operand B (dense)** — cp.async direct:

B has no indirection — the gmem address is computed directly from tile coordinates:

```cpp
const void *gmem_src = &ptr_B[k_global * N + n_global];
uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));
asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
    :: "r"(smem_addr), "l"(gmem_src), "n"(16));
```

Both use zero-fill (`src_size=0`) for out-of-bounds accesses. Both are covered by the same `cp_async_fence()`/`cp_async_wait<0>()` group in the pipelined mainloop.

### Epilogue

The epilogue uses the identity tensor pattern to map accumulator elements to output coordinates:

```cpp
Tensor tCrC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<tM>{}, Int<tN>{})));

for (int i = 0; i < size(accum); ++i) {
    auto coord = tCrC(i);            // Returns (m_local, n_local) coordinate
    int m_global = m_start + get<0>(coord);
    int n_global = n_start + get<1>(coord);

    if (m_global < M && n_global < N) {
        int phys_row = out_map[m_global];
        float result = alpha * accum(i);
        if (beta != 0.0f)
            result += beta * ptr_C[phys_row * N + n_global];
        ptr_D[phys_row * N + n_global] = result;
    }
}
```

`partition_C` of an identity tensor gives each thread a mapping from its accumulator index to the (M, N) tile coordinate. The epilogue is purely element-wise with scatter via `out_map`.

---

## Python API

```python
import warpconvnet._C as _C

# AD Gather-Scatter: D[out_map] = alpha * A[in_map] @ B + beta * C[out_map]
status = _C.gemm.cute_gemm_AD_gather_scatter(
    tensor_a,       # (M_A, K) FP16/BF16
    tensor_b,       # (K, N) FP16/BF16
    tensor_c,       # (M_C, N) FP32
    tensor_d,       # (M_C, N) FP32 output
    indices_a,      # (gather_size,) int32
    indices_d,      # (gather_size,) int32
    mma_tile=3,     # 0=128x128, 1=128x64, 2=64x128, 3=64x64
    alpha=1.0,
    beta=0.0,
)
# Returns 0 on success, negative on error

# TrAB Gather (not yet implemented, returns error)
status = _C.gemm.cute_gemm_trAB_gather(...)
```

Tile enum values:
- `0` = `Tile128x128x32`
- `1` = `Tile128x64x32`
- `2` = `Tile64x128x32`
- `3` = `Tile64x64x32`

---

## Benchmark Results

**Hardware**: NVIDIA RTX 6000 Ada (SM89), CUDA 12.8, PyTorch 2.10.0+cu128

### CuTe 3.x vs CUTLASS 2.x — Current State (Phase 4)

CuTe 3.x / CUTLASS 2.x time ratio (lower is better, <1.0 means CuTe wins):

| Config (M x K x N, gather) | 64x64 | 128x64 | 128x128 |
|----------------------------|-------|--------|---------|
| 100K x 32 x 32, g=50K | 0.93x | 0.93x | 1.11x |
| 100K x 64 x 64, g=50K | 1.13x | 1.15x | 1.32x |
| 100K x 64 x 128, g=50K | 1.26x | 1.16x | 1.42x |
| 200K x 128 x 128, g=100K | **0.97x** | **0.93x** | 1.03x |
| 100K x 128 x 128, g=5K | 1.06x | 1.11x | 1.29x |

**Summary**: CuTe 3.x now **beats CUTLASS 2.x** on the largest compute-heavy config (200K×128×128 at 0.93x with 128×64 tile). Small-K configs (K=32) also show CuTe winning. The remaining gap on medium configs comes from CuTe using 2-stage vs CUTLASS 2.x's 5-stage pipeline, and 128 threads vs 2.x's 256-512 threads for larger tiles.

### Optimization History

CuTe/2.x ratio progression across optimization phases (best tile per config):

| Config | Baseline | Ph1 (vec A) | Ph2 (vec B + LDSM_T) | Ph3 (cp.async B + pipeline) | Ph4 (cp.async A) |
|--------|----------|-------------|----------------------|-----------------------------|-------------------|
| 100K×32×32, g=50K | 1.07x | 1.00x | 1.00x | 0.94x | **0.93x** |
| 100K×64×64, g=50K | 1.44x | 1.32x | 1.16x | 1.16x | **1.13x** |
| 200K×128×128, g=100K | 1.22x | 1.25x | 1.08x | 1.06x | **0.93x** |
| 100K×64×128, g=50K | 1.56x | 1.53x | 1.20x | 1.20x | **1.16x** |

Key optimizations per phase:
- **Phase 1**: 128-bit vectorized LDG.128+STS.128 for gathered A along K → 3-11%
- **Phase 2**: N-contiguous smem B (`Swizzle<3,3,3>`), transposing LDSM_T, vectorized 128-bit B loads → 10-25%
- **Phase 3**: `cp.async` for B (register bypass), 2-stage double-buffered pipelining → 2-7%
- **Phase 4**: `cp.async` for gathered A (register bypass, truly async pipeline) → 3-12%

### BF16 vs FP16 (CuTe 3.x)

| Config | BF16 (ms) | FP16 (ms) |
|--------|-----------|-----------|
| 100K x 64 x 64, g=50K, Tile64x64 | 0.040 | 0.041 |
| 100K x 128 x 128, g=50K, Tile128x128 | 0.077 | 0.078 |

BF16 and FP16 have identical performance (same tensor core throughput on Ada).

---

## Comparison with CUTLASS 2.x Implementation

| Aspect | CUTLASS 2.x | CuTe 3.x (this impl) |
|--------|-------------|----------------------|
| **API** | `GemmUniversal` with `GatherA=true, ScatterD=true` | Manual kernel with raw pointer gather |
| **Gmem load A** | Vectorized 128-bit `cp.async` with built-in gather | 128-bit `cp.async` with manual gather (index LDG + async data transfer) |
| **Gmem load B** | Vectorized 128-bit `cp.async` | Vectorized 128-bit `cp.async` (N-contiguous) |
| **Smem layout B** | K-contiguous (OperandA RowMajor) | N-contiguous (Swizzle<3,3,3>) + LDSM_T |
| **Smem pipeline** | Multi-stage (NumStages=5) | 2-stage double-buffered |
| **Smem-to-reg A** | LDSM_N via CollectiveMma | LDSM_N via `make_tiled_copy_A` + `retile_D` |
| **Smem-to-reg B** | LDSM_N (K-contiguous smem) | LDSM_T (N-contiguous smem, transpose on load) |
| **Thread count** | Varies by tile (128-512) | Fixed 128 for all tiles |
| **Epilogue** | CUTLASS epilogue with scatter | Manual element-wise scatter |
| **Extensibility** | Locked to GemmUniversal API | Full kernel control for custom gather patterns |

### Remaining Performance Gap

CuTe 3.x now beats CUTLASS 2.x on large compute-heavy configs (0.93x at 200K×128×128) but is still 1.1-1.3x slower on medium configs. The remaining gap comes from:

1. **Fewer pipeline stages**: CUTLASS 2.x uses 5 stages vs our 2. More stages better hide gmem latency but require proportionally more smem.

2. **Thread count**: CUTLASS 2.x uses more threads for larger tiles (256-512), distributing load work more evenly.

3. **Gather index overhead**: Each cp.async for A requires a synchronous 4-byte LDG to resolve `gather_map[m_global]` before the async data transfer. This serial dependency limits the async benefit.

---

## Optimization History

### Phase 1: Vectorized A Loads (DONE — commit `a24abaa4`)

128-bit `LDG.128` + `STS.128` for gathered A along K. Works because K is contiguous in both gmem (within each gathered row) and smem (`Swizzle<2,3,3>` preserves 8-element contiguity along K).

**Result**: 3-11% speedup, parity (1.00x) vs CUTLASS 2.x for small-K configs.

### Phase 2: N-Contiguous B + LDSM_T + Vectorized B Loads (DONE — commit `8e06f09f`)

Switched B smem from K-contiguous (`Swizzle<2,3,3>`) to N-contiguous (`Swizzle<3,3,3>`) to match gmem layout. Added `SM75_U16x8_LDSM_T` (transposing LDSM) for smem→register. Vectorized B loads with 128-bit LDG+STS along N.

**Result**: 10-25% speedup, closing the gap from 1.3-1.5x to 1.06-1.20x.

### Phase 3: cp.async for B + 2-Stage Pipelining (DONE — commit `92f6f986`)

Replaced synchronous LDG+STS for B with `cp.async.ca.shared.global.L2::128B` (128-bit async gmem→smem bypassing registers). Added 2-stage double-buffering with 3D smem layouts `(M/N, K, NumStages)` to overlap next K-tile loads with current K-tile MMA compute.

**Result**: 2-7% speedup on compute-heavy configs. 128×128 tile regresses due to doubled smem reducing occupancy.

See `PHASE3_CPASYNC_PIPELINING_PLAN.md` for detailed implementation notes and synchronization correctness proofs.

### Phase 4: cp.async for Gathered A (DONE — commit pending)

Replaced synchronous LDG.128+STS.128 for gathered A with cp.async. Each thread resolves the gather index (synchronous 4-byte LDG for the int32 index), then issues `cp.async` for the K-contiguous data (async 128-bit gmem→smem, bypasses registers). This makes both A and B fully async, covered by the same `cp_async_fence`/`cp_async_wait` group.

**Key insight**: CuTe's `ComposedLayout` via `make_gather_tensor` cannot be used with vectorized cp.async because `partition_S` / `tile2thrfrg` flattens the `ComposedLayout` to a plain `Layout`, breaking `upcast<N>`. Instead, we bypass CuTe's copy infrastructure entirely — compute gathered addresses manually and issue inline PTX `cp.async`.

**Result**: 3-12% speedup. CuTe 3.x now **beats CUTLASS 2.x** on 200K×128×128 (0.93x with 128×64 tile).

---

## Future Work

### TrAB Gather

**Goal**: Implement `D = alpha * A[idx_a].T @ B[idx_b] + beta * C` for weight gradient backward pass.

Both A and B are gathered, and A is transposed (contraction along the gathered M dimension). This is fundamentally harder because the contraction dimension has scattered memory access. Requires split-K parallelism for performance.

### Warp-Specialized Kernel

**Goal**: Overlap gmem loads with MMA compute using producer/consumer warp specialization.

- **Producer warps** (1-2): Handle gmem→smem loads (gathered A + dense B)
- **Consumer warps** (2-3): Handle smem→reg copies and MMA compute

Requires SM80+ warp-level `arrive`/`wait` barriers and careful smem partitioning.

### Autotune Tile Selection

**Goal**: Map `(gather_size, K, N)` to optimal tile config automatically.

Current best tiles: `64x64` for small problems and small K, `128x64` for most configs, `128x128` regresses due to smem pressure with 2-stage pipelining.

---

## Build

The CuTe GEMM kernel is compiled as part of the main package:

```bash
source .venv/bin/activate
uv pip install -e . --no-build-isolation
```

`setup.py` includes `cutlass_cute_gemm_gather_scatter.cu` in the CUDA extension sources and adds `3rdparty/cutlass/examples/common` to the include path (for `gather_tensor.hpp`).

## Tests

```bash
# Correctness tests (25 tests, ~0.3s)
pytest tests/csrc/test_cutlass_cute_gemm.py -v

# Benchmarks
python tests/csrc/benchmark_cute_gemm.py
```

## Error Codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | `kSuccess` | Kernel completed successfully |
| -1 | `kErrorProblemNotSupported` | Problem shape not supported |
| -2 | `kErrorKernelInitialization` | Failed to set smem attributes |
| -3 | `kErrorKernelExecution` | CUDA launch error |
| -4 | `kErrorUnsupportedConfig` | Tile/operation not implemented (e.g., TrAB) |
| -5 | `kErrorInvalidParameters` | Invalid input parameters |
| -6 | `kErrorMixedInputUnsupported` | Mixed precision not supported |
