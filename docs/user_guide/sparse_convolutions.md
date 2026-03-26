# Sparse Convolutions

WarpConvNet implements spatially sparse convolutions on voxel grids using multiple CUDA backends with automatic algorithm selection.

## Overview

WarpConvNet provides two types of sparse convolutions:

- **Regular Sparse Convolution**: General-purpose convolution for feature learning
- **Depthwise Sparse Convolution**: Channel-wise convolution for efficient feature processing

Both include a **unified auto-tuning system** that benchmarks algorithm candidates at runtime and caches the best configuration per problem shape.

## Two GEMM Operations

A sparse convolution backward pass decomposes into two mathematically distinct GEMM operations:

| Operation             | Math                          | Used By        | Cache Namespace     |
| --------------------- | ----------------------------- | -------------- | ------------------- |
| **AB gather-scatter** | `D[scatter] = A[gather] @ B`  | Forward, dgrad | `AB_gather_scatter` |
| **AtB gather-gather** | `D = A[gather]^T @ B[gather]` | Wgrad          | `AtB_gather_gather` |

Forward and dgrad share the same kernel (gather input, dense weight, scatter to output). Wgrad uses a reduction kernel (gather both operands, dense output per offset). Each operation is auto-tuned independently.

## Convolution Kernel Backends

### Per-Offset Backends

These backends process each kernel offset as a separate GEMM call:

| Backend                 | Implementation                                                              | Strengths                                                                      |
| ----------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `explicit_gemm`         | Gather features into a dense buffer, call `torch.mm`, scatter-add results   | Simple, reliable fallback. No CUDA alignment requirements.                     |
| `implicit_gemm`         | Custom CUDA kernel that fuses gather, GEMM, and scatter-add into one launch | Best at small channels (C \<= 64) where launch overhead matters less.          |
| `cutlass_implicit_gemm` | CUTLASS fused gather-GEMM-scatter kernel                                    | High throughput at large channels. Auto-pads unaligned channels internally.    |
| `cute_implicit_gemm`    | CuTe 3.x fused gather-GEMM-scatter kernel                                   | Vectorized A-operand loads (`cp.async`). Competitive at small-medium channels. |

### Fused Multi-Offset Backends

These backends process multiple (or all) kernel offsets in a single launch:

| Backend                  | Implementation                                                                         | Strengths                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `cute_grouped`           | CuTe 3.x grouped GEMM — all offsets in one launch via binary-search dispatch           | Dominant wgrad winner (64%). Amortizes launch overhead at medium-large channels. |
| `cutlass_grouped_hybrid` | CUTLASS for large offsets + `torch.bmm` for grouped small offsets                      | Strong at large N with medium-large channels.                                    |
| `mask_implicit_gemm`     | Mask-based fused kernel — iterates all K offsets per output row using bitmask skipping | **Dominant AB winner (56% fwd, 74% dgrad)**. No atomicAdd. CuTe tensor core MMA. |

### How `mask_implicit_gemm` Works

Unlike per-offset and grouped backends that launch separate work per offset, the mask kernel processes **all K offsets in a single launch**. For each output row:

1. Look up which offsets are active via a bitmask (`pair_mask`)
2. For each active offset, gather from input and accumulate with the offset's weight
3. Write output directly — no atomicAdd needed since each output row is exclusive

For dgrad, a reverse pair_table is constructed so the same forward kernel can be reused with swapped dimensions, avoiding atomicAdd entirely (~2x speedup over the old atomicAdd dgrad).

## Auto-Tuning System

### How It Works

On the first forward (or backward) pass for a new problem shape, WarpConvNet:

1. Selects a set of **candidate algorithms** based on the convolution dimensions (N, C_in, C_out, K)
2. Runs each candidate with warmup + timed iterations
3. Picks the fastest and caches the result keyed by `(log10(N_in), log10(N_out), C_in, C_out, K, dtype, SM)`
4. Subsequent calls with the same shape hit the cache instantly

Results are persisted to `~/.cache/warpconvnet/benchmark_cache_generic.msgpack` and survive across Python sessions.

### Adaptive Candidate Selection

The candidate set adapts to the problem dimensions. Based on benchmark analysis of 148 configs (SM 8.9, cuBLAS 12.9.1.4):

**AB gather-scatter (forward + dgrad)** — 7-11 candidates:

| N range           | ch \<= 256                   | ch > 256                       |
| ----------------- | ---------------------------- | ------------------------------ |
| Small (N \<= 10K) | mask (92-100%)               | cute_grouped (58%), mask (25%) |
| Medium (10K-100K) | mask (69%), cutlass (27%)    | cutlass_grouped (67%)          |
| Large (N > 100K)  | mask/cutlass_grouped/cutlass | cutlass (100%)                 |

**AtB gather-gather (wgrad)** — 5-8 candidates:

| N range           | ch \<= 64                                     | ch > 64             |
| ----------------- | --------------------------------------------- | ------------------- |
| Small (N \<= 10K) | cute_grouped (57%), implicit_gemm (36%)       | cute_grouped (100%) |
| Medium (10K-100K) | cute_grouped (57%), explicit_grouped (43%)    | cute_grouped (77%)  |
| Large (N > 100K)  | cutlass_grouped (57%), explicit_grouped (36%) | cute_grouped (100%) |

### Algorithm Modes

| Mode             |   AB Candidates | AtB Candidates | Use Case                                      |
| ---------------- | --------------: | -------------: | --------------------------------------------- |
| `auto` (default) | 7-11 (adaptive) | 5-8 (adaptive) | Normal usage. Covers all winning algorithms.  |
| `trimmed`        |              11 |             27 | Broader search, excludes known dead-weight.   |
| `all`            |              23 |             35 | Exhaustive. For benchmarking or new hardware. |

```bash
# Default: adaptive reduced set (recommended)
export WARPCONVNET_FWD_ALGO_MODE=auto

# Exhaustive: benchmark every algorithm variant
export WARPCONVNET_FWD_ALGO_MODE=all

# Specific algorithm (no benchmarking, just use it)
export WARPCONVNET_FWD_ALGO_MODE=mask_implicit_gemm

# Algorithm list (benchmark only these)
export WARPCONVNET_FWD_ALGO_MODE="[mask_implicit_gemm,cutlass_implicit_gemm]"
```

The same options apply to `WARPCONVNET_BWD_ALGO_MODE` (controls wgrad AtB algorithm).

## Usage

### Basic Usage

```python
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv

# Auto mode (default) -- auto-tunes on first call, cached thereafter
conv = SpatiallySparseConv(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
)
output = conv(input_voxels)
```

### Functional API

```python
from warpconvnet.nn.functional import spatially_sparse_conv

output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
)
```

### Specifying Algorithms

Forward, dgrad, and wgrad can each be controlled independently:

```python
# Different algorithms for each operation
output = spatially_sparse_conv(
    input_voxels, weight, kernel_size=3,
    fwd_algo="mask_implicit_gemm",       # AB gather-scatter for forward
    dgrad_algo="mask_implicit_gemm",     # AB gather-scatter for dgrad
    wgrad_algo="cute_grouped",           # AtB gather-gather for wgrad
)

# Algorithm list -- benchmarks only these
output = spatially_sparse_conv(
    input_voxels, weight, kernel_size=3,
    fwd_algo=["mask_implicit_gemm", "cutlass_implicit_gemm"],
    dgrad_algo=["mask_implicit_gemm", "cute_grouped"],
    wgrad_algo=["cute_grouped", "cutlass_grouped_hybrid"],
)
```

### Depthwise Convolution

```python
from warpconvnet.nn.functional import spatially_sparse_depthwise_conv

output = spatially_sparse_depthwise_conv(
    input_features,
    depthwise_weight,
    kernel_map,
    num_out_coords,
)
```

Depthwise convolution has its own algorithm modes (`explicit_gemm`, `implicit_gemm`, `auto`) controlled by:

```bash
export WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE=auto
export WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE=auto
```

## Environment Variables

### Algorithm Selection

```bash
# AB gather-scatter algorithm for forward and dgrad (default: auto)
export WARPCONVNET_FWD_ALGO_MODE=auto

# AtB gather-gather algorithm for wgrad (default: auto)
export WARPCONVNET_BWD_ALGO_MODE=auto
```

Accepted values: `auto`, `all`, `trimmed`, any single algorithm name, or a bracket list like `[algo1,algo2]`.

Valid algorithm names: `explicit_gemm`, `implicit_gemm`, `cutlass_implicit_gemm`, `cute_implicit_gemm`, `explicit_gemm_grouped`, `implicit_gemm_grouped`, `cutlass_grouped_hybrid`, `cute_grouped`, `mask_implicit_gemm`.

### Cache and Logging

```bash
# Cache directory (default: ~/.cache/warpconvnet)
export WARPCONVNET_BENCHMARK_CACHE_DIR=~/.cache/warpconvnet

# Suppress auto-tuning logs (default: true)
export WARPCONVNET_AUTOTUNE_LOG=false
```

### Inspecting the Cache

Use `scripts/inspect_benchmark_cache.py` to view cached results:

```bash
python scripts/inspect_benchmark_cache.py
python scripts/inspect_benchmark_cache.py namespace=AB_gather_scatter --best-only
```

Use `scripts/analyze_autotune_cache.py` to generate statistical analysis of algorithm win rates:

```bash
python scripts/analyze_autotune_cache.py --markdown --output analysis.md
```

See [Inspecting the Benchmark Cache](./inspect_benchmark_cache.md) for details.

## Performance Characteristics

### When Each Backend Wins

Based on empirical analysis on RTX 6000 Ada with cuBLAS 12.9.1.4:

| Condition                  | Best AB Backend         | Best AtB Backend                      |
| -------------------------- | ----------------------- | ------------------------------------- |
| ch \<= 256, any N          | `mask_implicit_gemm`    | `cute_grouped`                        |
| ch > 256, small N          | `cute_grouped`          | `cute_grouped`                        |
| ch > 256, large N          | `cutlass_implicit_gemm` | `cute_grouped`                        |
| ch \<= 64, small N (wgrad) | —                       | `implicit_gemm` or `explicit_grouped` |

## Troubleshooting

**Slow first run**: Normal — auto-tuning benchmarks candidates. Subsequent runs use the cache. Use `auto` mode (not `all`) to minimize tuning time. To skip auto-tuning entirely, [pre-populate the cache](./populate_benchmark_cache.md) before your first run.

**Clear cache when switching GPUs**:

```bash
rm -rf ~/.cache/warpconvnet/
```

**CUTLASS not available**: Some backends require specific GPU compute capability. Fall back to:

```bash
export WARPCONVNET_FWD_ALGO_MODE="[explicit_gemm,implicit_gemm,mask_implicit_gemm]"
```

## Source Files

| File                                                              | Contents                                             |
| ----------------------------------------------------------------- | ---------------------------------------------------- |
| `warpconvnet/nn/functional/sparse_conv/detail/unified.py`         | Auto-tuning dispatch, config construction            |
| `warpconvnet/nn/functional/sparse_conv/detail/algo_params.py`     | Adaptive candidate selection, algorithm enums        |
| `warpconvnet/nn/functional/sparse_conv/detail/autotune.py`        | Benchmark runners, cache init/merge                  |
| `warpconvnet/nn/functional/sparse_conv/detail/dispatch.py`        | Algorithm execution dispatch                         |
| `warpconvnet/nn/functional/sparse_conv/detail/mask_gemm.py`       | Mask-based fused kernel dispatch, reverse pair_table |
| `warpconvnet/nn/functional/sparse_conv/detail/cute_grouped.py`    | CuTe grouped GEMM (AB + TrAB)                        |
| `warpconvnet/nn/functional/sparse_conv/detail/cutlass.py`         | CUTLASS per-offset gather-scatter                    |
| `warpconvnet/nn/functional/sparse_conv/detail/explicit.py`        | Explicit GEMM via cuBLAS                             |
| `warpconvnet/nn/functional/sparse_conv/detail/implicit_direct.py` | SIMT implicit GEMM                                   |
| `warpconvnet/utils/benchmark_cache.py`                            | Generic benchmark cache with persistence             |
| `warpconvnet/constants.py`                                        | Environment variable parsing                         |
