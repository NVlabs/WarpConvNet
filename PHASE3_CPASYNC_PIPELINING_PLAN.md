# Phase 3: cp.async for B + 2-Stage Double-Buffered Pipelining

## Setup

Working directly on the `cute3-vectorized-a-loads` branch (commits: Phase 1 `a24abaa4`, Phase 2 `8e06f09f`).

```bash
source .venv/bin/activate
# Rebuild after edits:
uv pip install -e . --no-build-isolation
# Run tests:
pytest tests/csrc/test_cutlass_cute_gemm.py -v
```

---

## Context

The CuTe 3.x GEMM kernel with gather/scatter currently uses a single-stage mainloop:
**Load A → Load B → `__syncthreads` → Compute (LDSM + MMA) → `__syncthreads`** for each K-tile.

This leaves gmem load latency fully exposed — the tensor cores sit idle while waiting for data.

Phase 3 adds two optimizations:

1. **`cp.async` for dense B loads** — bypasses registers entirely, issuing async gmem→smem transfers via the SM80 copy engine
2. **2-stage double-buffering** — overlaps the next K-tile's loads with the current K-tile's MMA compute

Operand A (gathered via random-access indices) must remain with manual LDG+STS because `cp.async` only supports contiguous memory transfers.

---

## Architecture Overview

### Current Single-Stage Mainloop (Phase 2)

```
for each k_tile:
    ┌─────────────────────────────────────┐
    │  Load A[k_tile] → smem  (LDG+STS)  │  ← gmem latency exposed
    │  Load B[k_tile] → smem  (LDG+STS)  │  ← gmem latency exposed
    │  __syncthreads()                    │
    │  for each k_block:                  │
    │    LDSM smem → registers            │
    │    MMA compute                      │
    │  __syncthreads()                    │
    └─────────────────────────────────────┘
```

### New 2-Stage Pipelined Mainloop (Phase 3)

```
PROLOG:
    Load A[k=0] → smem[stage=0]  (LDG+STS)
    Load B[k=0] → smem[stage=0]  (cp.async)
    cp_async_fence()
    cp_async_wait<0>()
    __syncthreads()

MAINLOOP (k_tile = 1..num_k_tiles-1):
    ┌──────────────────────────────────────────────────────────┐
    │  Load A[k_tile] → smem[next_stage]   (LDG+STS)          │
    │  Load B[k_tile] → smem[next_stage]   (cp.async)         │  ← async, overlaps
    │  cp_async_fence()                                        │    with compute ↓
    │                                                          │
    │  Compute from smem[curr_stage]:                          │  ← MMA runs while
    │    for each k_block:                                     │    B loads complete
    │      LDSM smem[curr_stage] → registers                  │    in background
    │      MMA                                                 │
    │                                                          │
    │  cp_async_wait<0>()                                      │
    │  __syncthreads()                                         │
    └──────────────────────────────────────────────────────────┘

EPILOG:
    Compute last k_tile from smem[last_stage]
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `warpconvnet/csrc/include/cute_gemm_config.h` | `NumStages = 1` → `NumStages = 2` in all 8 specializations |
| `warpconvnet/csrc/include/cute_gemm_kernel.h` | 3D smem layouts, new cp.async B load function, pipelined mainloop |
| `warpconvnet/csrc/include/cute_gemm_launch.h` | No changes needed (smem size auto-adjusts via `cosize_v`) |

---

## Detailed Implementation

### Step 1: `cute_gemm_config.h` — NumStages = 2

Change `static constexpr int NumStages = 1;` to `static constexpr int NumStages = 2;` in all 8 `CuteTileConfig` specializations (4 tile sizes × {half_t, bfloat16_t}).

This is the only change in this file. The 3D smem layout construction uses `NumStages` from the config and lives in the kernel struct.

---

### Step 2: `cute_gemm_kernel.h` — Add cp.async Include

Add after the existing includes (after `#include "cute/algorithm/copy.hpp"`):

```cpp
#include "cute/arch/copy_sm80.hpp"  // cp_async_fence, cp_async_wait
```

This provides:
- `cute::cp_async_fence()` — issues `cp.async.commit_group` (non-blocking)
- `cute::cp_async_wait<N>()` — blocks until at most N commit groups remain pending
- `cute::cast_smem_ptr_to_uint()` — converts generic pointer to 32-bit smem address for PTX

---

### Step 3: `cute_gemm_kernel.h` — 3D SmemLayouts with Stages Dimension

Replace the current 2D SmemLayout declarations:

```cpp
// BEFORE (single-stage):
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(Int<tM>{}, Int<tK>{})));
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(Int<tN>{}, Int<tK>{})));
```

With 3D layouts that include the stages dimension:

```cpp
// AFTER (multi-stage):
static constexpr int NumStages = TileConfig::NumStages;

using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(Int<tM>{}, Int<tK>{}, Int<NumStages>{})));  // (tM, tK, Stages)
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(Int<tN>{}, Int<tK>{}, Int<NumStages>{})));  // (tN, tK, Stages)
```

**How `tile_to_shape` handles the 3D shape:**

The 2D atom (e.g., `SmemLayoutAtomA` with shape `(8, 32)`) is tiled across the first two dimensions to cover `(tM, tK)`. The third dimension (Stages) is appended as a simple strided mode with stride = `cosize(2D_tiled_atom)`. This means:

- `sA(:, :, 0)` occupies bytes `[0, tM * tK * sizeof(half))`
- `sA(:, :, 1)` occupies bytes `[tM * tK * sizeof(half), 2 * tM * tK * sizeof(half))`

Each stage slice has exactly the same swizzled layout as the original single-stage smem.

**SharedStorage remains unchanged** — `cosize_v<SmemLayoutA>` automatically doubles because the layout now includes the stages dimension:

```cpp
struct SharedStorage {
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_a;  // now 2x larger
    cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_b;  // now 2x larger
};
```

---

### Step 4: `cute_gemm_kernel.h` — New `_load_dense_B_tile_cpasync` Function

Add a new private method that replaces `LDG(reg) + STS(smem)` with `cp.async.ca.shared.global`:

```cpp
/// Load a dense B tile into smem using cp.async (128-bit async gmem→smem).
///
/// Same iteration pattern as _load_dense_B_tile: K outer, N-vec inner.
/// Each thread issues cp.async for its assigned 128-bit chunks.
/// Out-of-bounds accesses use cp.async zero-fill (src_size=0).
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

        // Smem destination address (32-bit shared memory pointer for PTX)
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&smem_tile(n_local, k_local));

        bool pred = (k_global < K_dim) && (n_global + kVec <= N);

        if (pred) {
            // Full 128-bit cp.async: gmem → smem, bypasses registers
            const void *gmem_src = &ptr_B[k_global * N + n_global];
            asm volatile(
                "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
                :: "r"(smem_addr), "l"(gmem_src), "n"(16));
        } else {
            // Zero-fill: cp.async with src_size=0 writes zeros to smem
            asm volatile(
                "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
                :: "r"(smem_addr), "l"(ptr_B), "n"(16), "r"(0));
        }
    }
}
```

**Key details:**

| Aspect | Explanation |
|--------|-------------|
| `cp.async.ca.shared.global.L2::128B` | 128-bit async copy, cache at all levels, L2 128B hint |
| Zero-fill variant | 4th operand `src_size=0` → writes 16 bytes of zeros to smem, no gmem read |
| `cast_smem_ptr_to_uint` | Converts generic pointer to 32-bit shared memory address for PTX |
| Register bypass | Unlike LDG+STS, cp.async goes directly from gmem to smem via the copy engine |
| Alignment | `n_local` is always a multiple of `kVec=8`, so smem addr is 16-byte aligned (Swizzle<3,3,3> preserves 8-element contiguity along N) |

The existing `_load_dense_B_tile` (synchronous LDG+STS) can be kept as dead code or removed.

---

### Step 5: `cute_gemm_kernel.h` — Pipelined Mainloop

Replace the entire `operator()` body with the pipelined version. Key structural changes:

#### 5a. Tensor Creation (3D)

```cpp
SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});  // (tM, tK, NumStages)
Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{});  // (tN, tK, NumStages)
```

#### 5b. MMA Fragment Setup

Fragments are partitioned from a 2D slice (stage 0) — shape doesn't depend on stage:

```cpp
Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));  // (MMA,MMA_M,MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));  // (MMA,MMA_N,MMA_K)
```

#### 5c. Smem→Register Copy Partitioning (4D)

Partitioning the 3D smem tensor produces a 4D result:

```cpp
Tensor tCsA = smem_thr_copy_A.partition_S(sA);  // (CPY, CPY_M, CPY_K, NumStages)
Tensor tCsB = smem_thr_copy_B.partition_S(sB);  // (CPY, CPY_N, CPY_K, NumStages)
```

In the compute loop, index with `tCsA(_, _, k_block, stage)` to select the correct K-sub-block within the correct pipeline stage.

#### 5d. Prolog

```cpp
// Load k_tile=0 into stage[0]
_load_gathered_tile(ptr_A, in_map, sA(_, _, 0),
                    m_start, 0, M, K_dim, true);
_load_dense_B_tile_cpasync(ptr_B, sB(_, _, 0),
                           n_start, 0, N, K_dim);
cute::cp_async_fence();    // commit B's async group
cute::cp_async_wait<0>();  // wait for B (A was synchronous)
__syncthreads();           // make smem visible to all threads
```

#### 5e. Mainloop (overlap load-next with compute-curr)

```cpp
CUTLASS_PRAGMA_NO_UNROLL
for (int k_tile = 1; k_tile < num_k_tiles; ++k_tile) {
    int curr_stage = (k_tile - 1) & 1;
    int next_stage = k_tile & 1;
    int k_start = k_tile * tK;

    // Issue loads for NEXT k_tile into next_stage
    _load_gathered_tile(ptr_A, in_map, sA(_, _, next_stage),
                        m_start, k_start, M, K_dim, true);
    _load_dense_B_tile_cpasync(ptr_B, sB(_, _, next_stage),
                               n_start, k_start, N, K_dim);
    cute::cp_async_fence();

    // Compute CURRENT k_tile from curr_stage
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, curr_stage),
             tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, curr_stage),
             tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
    }

    // Wait for next_stage loads to complete
    cute::cp_async_wait<0>();
    __syncthreads();
}
```

#### 5f. Epilog (compute last k_tile)

```cpp
{
    int last_stage = (num_k_tiles - 1) & 1;
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_, _, k_block, last_stage),
             tCrA_copy_view(_, _, k_block));
        copy(smem_tiled_copy_B, tCsB(_, _, k_block, last_stage),
             tCrB_copy_view(_, _, k_block));
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), accum);
    }
}
```

#### 5g. Edge Case: num_k_tiles == 0

Add early return before the prolog:

```cpp
if (num_k_tiles == 0) {
    _epilogue(accum, ptr_C, ptr_D, out_map,
              m_start, n_start, M, N, alpha, beta, tiled_mma);
    return;
}
```

---

### Step 6: No Changes to `cute_gemm_launch.h`

The launcher uses `Kernel::SharedStorageSize` which is `sizeof(SharedStorage)`. Since `cosize_v<SmemLayoutA/B>` now includes the stages dimension, the smem allocation automatically doubles. The existing `cudaFuncSetAttribute` guard handles the case where smem exceeds 48 KB (which it won't — max is 32 KB for 128×128×32 with 2 stages).

---

## Synchronization Pattern — Correctness Proof

### Why no WAR hazard on `smem[curr]`

During the mainloop body, threads:
1. **Write** to `smem[next_stage]` (A via STS, B via cp.async)
2. **Read** from `smem[curr_stage]` (LDSM for MMA)

Since `next_stage != curr_stage`, there is no read-write conflict.

### Why no RAW hazard on `smem[next_stage]`

- B's cp.async: `cp_async_wait<0>()` blocks until the committed async group completes
- A's manual STS: stores are synchronous within the issuing thread
- `__syncthreads()` after the wait ensures all threads see both A and B data before the next iteration, which reads from `next_stage` (now becoming `curr_stage`)

### Why the post-compute `__syncthreads` is necessary

Without it, thread T1 might still be reading `smem[curr_stage]` in MMA while thread T2 (faster) starts the next iteration and overwrites `smem[curr_stage]` (which becomes `next_stage`). The barrier ensures all threads finish their reads before any writes.

---

## Smem Budget

RTX 6000 Ada (SM 8.9) has 100 KB of shared memory per SM.

| Tile | smem_a (2 stages) | smem_b (2 stages) | Total | Headroom |
|------|-------------------|-------------------|-------|----------|
| 64×64×32 | 8 KB | 8 KB | **16 KB** | 84 KB |
| 128×64×32 | 16 KB | 8 KB | **24 KB** | 76 KB |
| 64×128×32 | 8 KB | 16 KB | **24 KB** | 76 KB |
| 128×128×32 | 16 KB | 16 KB | **32 KB** | 68 KB |

All well under the 48 KB threshold that requires `cudaFuncSetAttribute`.

---

## Potential Pitfalls and Edge Cases

### 1. `num_k_tiles == 1`

Only one K-tile. The prolog loads stage[0], the mainloop loop body executes zero times, and the epilog computes from stage[0]. `last_stage = (1-1) & 1 = 0`. Correct.

### 2. `num_k_tiles == 0`

K_dim = 0. Early-return guard handles this: output = `beta * C` (accumulator is zero).

### 3. cp.async vs regular STS ordering

`cp_async_fence()` / `cp_async_wait<N>()` only track cp.async operations. A's regular STS stores are unaffected — they complete in program order within each thread and become visible to other threads after `__syncthreads()`. No interference between the two mechanisms.

### 4. Smem pointer alignment for cp.async

`cp.async` requires 16-byte aligned smem destination for 128-bit copies. `SmemLayoutAtomB` uses `Swizzle<3,3,3>` with N-contiguous stride-1. When `n_local` is a multiple of `kVec=8` (always true in our loop), the smem address is `base + n_local * 2 bytes = base + 16 * nv`, which is 16-byte aligned.

### 5. Gmem pointer alignment for cp.async

B is row-major `(K, N)`, accessed at `ptr_B[k_global * N + n_global]`. For the non-zero-fill path (`pred=true`), `n_global` is `n_start + nv * kVec` where `n_start = n_tile * tN` (always a multiple of `kVec`). So `n_global` is a multiple of 8, giving a 16-byte aligned gmem address (8 elements × 2 bytes = 16). PyTorch CUDA tensors are 256-byte aligned at the base.

### 6. Partial N tiles

When `N` is not a multiple of `tN`, the rightmost tile has partial columns. The predicate `n_global + kVec <= N` correctly routes these to the zero-fill path. No element-wise fallback needed (unlike the LDG+STS version).

---

## CUTLASS Reference Patterns

### cp.async primitives (`cute/arch/copy_sm80.hpp`)

```cpp
// Non-blocking commit — groups outstanding cp.async ops
cute::cp_async_fence();  // asm: cp.async.commit_group

// Block until at most N groups remain pending
cute::cp_async_wait<0>();  // asm: cp.async.wait_all
cute::cp_async_wait<1>();  // asm: cp.async.wait_group 1

// Smem pointer conversion for inline PTX
uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
```

### CUTLASS multistage mainloop (`sm80_mma_multistage.hpp`)

```cpp
// 3D smem layout with stages
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<Stages>{})));

// 4D partitioned smem tensor
Tensor tCsA = smem_thr_copy_A.partition_S(sA);  // (CPY, CPY_M, CPY_K, Stages)

// Select stage in compute loop
copy(smem_tiled_copy_A, tCsA(_,_,k_block,smem_pipe_read), tCrA_copy_view(_,_,k_block));
```

### cp.async zero-fill (`SM80_CP_ASYNC_CACHEALWAYS_ZFILL`)

```cpp
// PTX: cp.async.ca.shared.global.L2::128B [smem], [gmem], cp_size, src_size
// src_size = 0: writes zeros to smem (no gmem read)
// src_size = 16: copies 16 bytes from gmem to smem
asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n"
    :: "r"(smem_addr), "l"(gmem_ptr), "n"(16), "r"(src_size));
```

---

## Verification

```bash
source .venv/bin/activate

# Correctness: all 25 tests must pass
pytest tests/csrc/test_cutlass_cute_gemm.py -v

# Performance: compare CuTe 3.x vs CUTLASS 2.x
python tests/csrc/benchmark_cute_gemm.py
```

### Benchmark Results (RTX 6000 Ada)

CuTe 3.x / CUTLASS 2.x time ratio (lower is better, <1.0 means CuTe wins):

| Config (MxKxN, gather) | 64×64 | 128×64 | 128×128 |
|------------------------|-------|--------|---------|
| 100K×32×32, g=50K | 0.94x | 0.94x | 1.16x |
| 100K×64×64, g=50K | 1.16x | 1.21x | 1.35x |
| 100K×64×128, g=50K | 1.24x | 1.20x | 1.47x |
| 200K×128×128, g=100K | 1.07x | 1.06x | 1.23x |
| 100K×128×128, g=5K | 1.12x | 1.19x | 1.41x |

**Phase 2 → Phase 3 improvements:**
- 200K×128×128, 64×64: 1.15x → 1.07x (7% improvement)
- 200K×128×128, 128×64: 1.08x → 1.06x (2% improvement)
- 100K×64×128, 128×64: 1.29x → 1.20x (7% improvement)
- Small-K configs (K=32) with only 1 K-tile show minimal change since pipelining can't overlap when there's nothing to pipeline

**128×128 tile regression:** Doubled smem (32 KB with 2 stages) reduces occupancy from ~6 to ~3 blocks/SM, causing 128×128 to regress. This is an inherent occupancy vs. latency-hiding trade-off.

**Remaining gap to CUTLASS 2.x:** The CuTe kernel's gathered A loads still use synchronous LDG+STS (cannot use cp.async for random-access gather). This is the dominant remaining bottleneck.
