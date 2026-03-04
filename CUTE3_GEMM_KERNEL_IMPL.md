# CuTe 3.x GEMM Kernel with Gather/Scatter — Implementation Reference

## Overview

This document describes the CuTe 3.x GEMM kernel for sparse convolution in WarpConvNet. The kernel computes:

```
D[out_map[i], j] = alpha * sum_k(A[in_map[i], k] * B[k, j]) + beta * C[out_map[i], j]
```

where `in_map` and `out_map` are integer gather/scatter index arrays that select which rows of A to read and which rows of D to write. This operation is the core of explicit-GEMM sparse convolution, where the kernel map (produced by coordinate hashing) determines how input features are gathered for each convolution kernel offset, and how outputs are scattered back to the output feature tensor.

The implementation uses CUTLASS 3.x CuTe abstractions for tensor core operations (TiledMMA, LDSM copy atoms, swizzled shared memory layouts) while handling the irregular gather/scatter pattern through manual element-wise global memory loads.

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
Global Memory                  Shared Memory              Registers
============                  =============              =========

A[in_map[i], k] ──LDG.128──>  sA(m_local, k_local)  ──LDSM──>  tCrA(MMA,MMA_M,MMA_K)
                  along K      (swizzled layout)       retile_D
                  + STS.128    K-contiguous in
                               gmem AND smem                             │
B[k, n]         ──element──>  sB(n_local, k_local)  ──LDSM──>  tCrB(MMA,MMA_N,MMA_K)
                  wise load    (swizzled layout)       retile_D          │
                                                                    ┌────▼────┐
                                                                    │ TiledMMA │
                                                                    │ (TC ops) │
                                                                    └────┬────┘
                                                                         │
D[out_map[i], j] <──element── accum(i)                                   │
                    wise store  = alpha * accum + beta * C[out_map[i], j] ┘
```

### Why Manual Loads Instead of CollectiveMma

CUTLASS 3.x `CollectiveMma` with `MainloopSm80CpAsyncUnpredicated` uses vectorized 128-bit `cp.async` instructions for gmem-to-smem transfers. These require contiguous, aligned memory access patterns. The gather operation (`A[in_map[i], :]`) introduces indirection that breaks contiguity in the M dimension — each row may come from an arbitrary physical location in A.

Several approaches were attempted to use CuTe's `ComposedLayout` (via `make_gather_tensor`) to express the gather as a layout property:

1. **ComposedLayout with cp.async**: The `upcast<N>` specialization for `ComposedLayout` (needed by vectorized copies) requires a `ScaledBasis` stride in the inner layout. After CuTe's internal partitioning operations (`partition_S`, `tile2thrfrg`, `compose`), the `ComposedLayout` gets flattened to a regular `Layout` with dynamic strides, losing the `ScaledBasis` information. The vectorized copy atom then fails with "src failed to vectorize into registers."

2. **Flat Layout with CustomStride**: Placing `CustomStride<IndexedGather, Stride>` directly in a flat layout causes compilation errors because CuTe's internal operations (`get`, `tile_unzip`, `repeat`, `append`) don't support `CustomStride` as a stride element in tuples.

3. **Manual element-wise loads (current approach)**: Each thread loads individual elements using scalar global memory reads with explicit gather index lookups, then writes them to shared memory using CuTe's swizzled coordinate access. This bypasses all vectorization requirements. The dense B matrix could theoretically use vectorized loads, but its TN-format memory layout (stride N along the K dimension) also prevents contiguous 128-bit loads along K.

### Kernel Structure

```
CuteGemmKernel<TileConfig>::operator()
│
├── Setup: shared memory tensors sA(tM,tK), sB(tN,tK)
├── MMA setup: TiledMma, accum = partition_fragment_C(...)
├── Register fragments: tCrA = partition_fragment_A(sA), tCrB = partition_fragment_B(sB)
├── LDSM copy setup:
│   ├── smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA, tiled_mma)
│   ├── tCsA = partition_S(sA)           // smem source partitions
│   └── tCrA_copy_view = retile_D(tCrA)  // register destination (retiled for LDSM)
│
├── MAINLOOP: for each K tile
│   ├── _load_gathered_tile(A, in_map, sA, ...)   // gmem → smem (element-wise)
│   ├── _load_dense_B_tile(B, sB, ...)             // gmem → smem (element-wise)
│   ├── __syncthreads()
│   ├── for each K block within tile:              // K_BLOCK_MAX = tK / MMA_K
│   │   ├── copy(smem_tiled_copy_A, tCsA(k), tCrA_copy_view(k))  // LDSM
│   │   ├── copy(smem_tiled_copy_B, tCsB(k), tCrB_copy_view(k))  // LDSM
│   │   └── cute::gemm(tiled_mma, tCrA(k), tCrB(k), accum)       // MMA
│   └── __syncthreads()
│
└── _epilogue: D[out_map[i], j] = alpha * accum(i,j) + beta * C[out_map[i], j]
```

---

## Tile Configurations

All configurations use the same warp layout and copy atoms, matching CUTLASS's `DefaultGemmConfigurationToCutlass3Types` for SM80:

| Property | Value |
|----------|-------|
| Warp layout | `Layout<Shape<_2, _2, _1>>` (2x2 = 4 warps, 128 threads) |
| MMA Tile | `Tile<_32, _32, _16>` (32x32 output per step, K=16 per MMA) |
| MMA Atom (FP16) | `SM80_16x8x16_F32F16F16F32_TN` |
| MMA Atom (BF16) | `SM80_16x8x16_F32BF16BF16F32_TN` |
| SmemLayoutAtom | `Swizzle<2,3,3>` composed with `Layout<Shape<_8,_32>, Stride<_32,_1>>` |
| SmemCopyAtom (A,B) | `Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>` |
| Accumulator | FP32 |
| Pipeline stages | 1 (single-buffer, no multi-stage) |

### SmemLayoutAtom Details

The shared memory layout atom `Swizzle<2,3,3>` with base layout `Shape<_8,_32>, Stride<_32,_1>` defines an 8-row x 32-column tile with:

- **K-contiguous storage**: Stride `_1` in the K dimension (columns), stride `_32` in the M/N dimension (rows)
- **Bank-conflict-free swizzle**: `Swizzle<B=2, M=3, S=3>` XORs bits `[3:5)` of the address with bits `[6:8)`, providing a 4-way swizzle that eliminates bank conflicts when 32 threads in a warp access elements along the M/N dimension (each thread reading from a different row)
- **Compatibility**: This is the exact atom from CUTLASS's `DefaultGemm_TensorOpSm80_OperandA<half_t, RowMajor, 8, 32>` (K=32 specialization). When `tile_to_shape`'d to `(tM, tK)` or `(tN, tK)`, it tiles the 8-row atom to fill the full M or N dimension.

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

### Element-Wise Gmem Loads

Both A (gathered) and B (dense) use the same cooperative loading pattern:

```cpp
constexpr int total = tM * tK;  // or tN * tK for B
for (int idx = threadIdx.x; idx < total; idx += MaxThreadsPerBlock) {
    int row_local = idx / tK;
    int k_local   = idx % tK;
    // ... bounds check and load ...
    smem_tile(row_local, k_local) = val;
}
```

The `smem_tile(row, col)` coordinate access automatically applies the swizzled SmemLayout, placing data at the correct bank-conflict-free address. Each of the 128 threads loads `ceil(tM*tK / 128)` elements. For a 64x32 tile, that's `2048/128 = 16` elements per thread.

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

### CuTe 3.x vs CUTLASS 2.x (FP16, min of 50 runs, best tile per config)

| Config (M x K x N, gather) | Best Tile | CuTe 3.x (ms) | CUTLASS 2.x (ms) | CuTe / 2.x |
|----------------------------|-----------|----------------|-------------------|-------------|
| 100K x 32 x 32, g=50K | 128x64x32 | 0.018 | 0.018 | **1.00x** |
| 100K x 64 x 64, g=50K | 128x64x32 | 0.034 | 0.026 | 1.32x |
| 100K x 64 x 128, g=50K | 128x128x32 | 0.062 | 0.040 | 1.56x |
| 50K x 128 x 128, g=25K | 128x64x32 | 0.042 | 0.028 | 1.52x |
| 200K x 128 x 128, g=100K | 128x128x32 | 0.218 | 0.166 | 1.31x |
| 100K x 128 x 128, g=5K | 64x64x32 | 0.035 | 0.027 | 1.31x |

**Summary**: CuTe 3.x is 1.0-1.6x slower than CUTLASS 2.x. The gap is smallest for small-K configs where the vectorized A load (128-bit LDG+STS along K) eliminates most of the load overhead. The remaining gap comes from B's element-wise gmem loads (B has N-contiguous gmem but K-contiguous smem, requiring a transpose that prevents direct cp.async).

### Phase 1 Optimization Impact (Vectorized A Loads)

Comparison of CuTe/2.x ratios before and after vectorized A loads (Tile128x64x32):

| Config | Before (scalar A) | After (vec A) | CuTe Speedup |
|--------|-------------------|---------------|--------------|
| 100K x 32 x 32, g=50K | 1.07x | **1.00x** | 10% |
| 100K x 64 x 64, g=50K | 1.44x | **1.32x** | 11% |
| 100K x 64 x 128, g=50K | 1.56x | **1.53x** | 3% |
| 50K x 128 x 128, g=25K | 1.57x | **1.52x** | 9% |
| 200K x 128 x 128, g=100K | 1.22x | 1.25x | ~0% |
| 100K x 128 x 128, g=5K | 1.43x | **1.38x** | 5% |

The A vectorization gives 3-11% speedup. The improvement is larger when A loading dominates (smaller K). The A load uses 128-bit `LDG.128` from gmem and 128-bit `STS.128` to smem, possible because K is contiguous in both gmem (within each gathered row) and smem (the `Swizzle<2,3,3>` layout preserves 8-element contiguity along K).

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
| **Gmem load A** | Vectorized 128-bit `cp.async` with built-in gather | Vectorized 128-bit LDG+STS along K (gathered rows) |
| **Gmem load B** | Vectorized 128-bit `cp.async` | Element-wise scalar loads (N→K transpose) |
| **Smem pipeline** | Multi-stage (NumStages=5, overlaps gmem load with compute) | Single-stage (no overlap possible with scalar loads) |
| **Smem-to-reg** | LDSM via CollectiveMma framework | LDSM via `make_tiled_copy_A/B` + `retile_D` |
| **MMA compute** | `CollectiveMma` orchestrated pipeline | `cute::gemm(tiled_mma, ...)` per K block |
| **Thread count** | Varies by tile (128-512) | Fixed 128 for all tiles |
| **Epilogue** | CUTLASS epilogue with scatter | Manual element-wise scatter |
| **Tile configs** | Shared with implicit GEMM | Dedicated 8 specializations |
| **Extensibility** | Locked to GemmUniversal API | Full kernel control for custom gather patterns |

### Why Both Exist

CUTLASS 2.x is faster today because its `GemmUniversal` with `GatherA/ScatterD` uses the full CUTLASS pipeline (vectorized loads, multi-stage smem buffering, optimized epilogues). The CuTe 3.x kernel provides:

1. **Foundation for CuTe ComposedLayout integration**: When CuTe's `upcast` specialization for `ComposedLayout` is fixed upstream (or we contribute a fix), the kernel can switch from manual loads to vectorized `CollectiveMma` with gather expressed as a layout property — matching CUTLASS 2.x performance with simpler, more composable code.

2. **Platform for custom gather patterns**: The manual kernel structure makes it straightforward to implement non-standard gather/scatter patterns (e.g., TrAB gather, block-sparse gather, multi-head attention patterns) that don't fit the `GatherA/ScatterD` boolean template.

3. **Explicit control for future optimizations**: Direct access to the load/compute/store pipeline enables targeted optimizations (vectorized dense B loads, software pipelining, warp specialization) without working around framework constraints.

---

## Future Work

### Phase 1: Vectorized A Loads (DONE)

**Status**: Implemented. The gathered A load uses 128-bit `LDG.128` from gmem and 128-bit `STS.128` to smem along the K dimension. This works because K is contiguous in both gmem (within each gathered row) and smem (the `Swizzle<2,3,3>` layout preserves 8-element contiguity along K).

**Result**: 3-11% speedup for most configurations, with parity (1.00x) vs CUTLASS 2.x achieved for small-K configs.

**Key insight**: The `Swizzle<2,3,3>` XOR pattern modifies bits [3:5) via XOR with bits [6:8). Since 8-element K-groups only differ in bits [0:3), and the swizzle only affects bit 3+ while preserving within-group contiguity, vectorized 128-bit stores to swizzled smem are always safe and aligned.

### Phase 2: Vectorized Dense B Loads with Transposing LDSM

**Goal**: Replace element-wise B loads with vectorized cp.async, closing the remaining 1.3-1.5x gap to CUTLASS 2.x.

B is (K, N) row-major (N-contiguous in gmem), but smem stores sB(n, k) with K-contiguous layout. This layout mismatch prevents simple vectorized cp.async (which requires source and destination to be contiguous in the same dimension). A direct vectorized N-load to K-contiguous smem causes severe smem bank conflicts (16-32 way conflicts from threads writing to the same K column in different atom rows).

**Approach**: Change the B smem layout and copy atom to handle the transpose:

1. **SmemLayoutAtomB**: Switch to N-contiguous layout (`Stride<_1, _tN>`) with appropriate swizzle
2. **SmemCopyAtomB**: Switch from `SM75_U32x4_LDSM_N` to `SM75_U16x8_LDSM_T` (transposing LDSM) for smem→register copy
3. **GmemTiledCopyB**: Use `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` with N-vectorized loads
4. **Multi-stage pipeline**: cp.async enables 2-3 stage buffering for B, overlapping B loads with compute

This matches the CUTLASS approach for RowMajor B in TN GEMMs (see `cooperative_gemm.cu::CooperativeGemm4` which uses `SM75_U16x8_LDSM_T` for operand B).

**Blockers**: Requires careful validation of the LDSM_T copy with the new SmemLayout, and the A load (vectorized LDG) must remain compatible with single-stage semantics while B uses multi-stage.

### Phase 3: ComposedLayout for Gathered A

**Goal**: Match CUTLASS 2.x performance by enabling vectorized `cp.async` for gathered A.

The gather can be expressed as a CuTe `ComposedLayout`:
```cpp
auto A_gathered = make_gather_tensor(
    make_gmem_ptr(ptr_A), make_shape(M, K),
    make_stride(K, 1),    // physical (M, K) row-major
    in_map                 // gather indices
);
// A_gathered has ComposedLayout: outer gather * inner identity
```

The current blocker is that CuTe's `partition_S` / `tile2thrfrg` operations flatten `ComposedLayout` to a regular `Layout` with dynamic strides, which breaks the `upcast<N>` specialization needed by vectorized `cp.async` copy atoms.

**Potential fixes**:
1. **Upstream CUTLASS fix**: Modify `tile2thrfrg` to preserve `ComposedLayout` through partitioning
2. **Custom copy atom**: Write a gather-aware copy atom that bypasses `upcast` by computing addresses directly from the gather indices
3. **Pre-permuted smem load**: Use a thread-level gather + `cp.async` pattern where each thread computes its gathered source address, then issues `cp.async` for the K-contiguous portion of that row

### Phase 4: TrAB Gather

**Goal**: Implement `D = alpha * A[idx_a].T @ B[idx_b] + beta * C` for backward pass.

TrAB (Transpose-A, B-gather) is needed for the weight gradient computation in sparse convolution. Both A and B are gathered, and A is transposed (contraction along the M dimension, not K).

This is fundamentally harder than AD gather because:
- The contraction dimension is the gathered row dimension
- Vectorized loads along contraction require reading from scattered rows
- Split-K parallelism is needed for performance (the contraction dimension can be large)

**Approach**: Build on Phase 3's ComposedLayout infrastructure, or use a dedicated reduction kernel where each thread block accumulates partial results along the gathered contraction dimension, followed by a cross-block reduction.

### Phase 5: Warp-Specialized Kernel

**Goal**: Overlap gmem loads with smem-to-reg copies and compute using warp specialization.

Instead of all warps participating in both load and compute:
- **Producer warps** (1-2 warps): Handle gmem-to-smem loads (both gathered A and dense B)
- **Consumer warps** (2-3 warps): Handle smem-to-reg copies and MMA compute

This requires SM80+ warp-level synchronization (`arrive`/`wait` barriers) and careful smem partitioning, but can significantly improve throughput by overlapping the load and compute phases that currently run sequentially.

### Phase 6: Autotune Tile Selection

**Goal**: Automatically select the best tile configuration based on problem dimensions.

Current benchmarks show:
- `128x64x32` is fastest for most configurations
- `64x64x32` is competitive for small problems
- `128x128x32` is often slower due to register pressure with 128 threads

Implement a heuristic or small autotune cache that maps `(gather_size, K, N)` to the optimal tile config, similar to the existing `sparse_conv_cutlass_autotune_cache.py`.

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
