# Sparse Convolutions

WarpConvNet implements spatially sparse convolutions on voxel grids using multiple CUDA backends with automatic algorithm selection.

## Overview

WarpConvNet provides two types of sparse convolutions:

- **Regular Sparse Convolution**: General-purpose convolution for feature learning
- **Depthwise Sparse Convolution**: Channel-wise convolution for efficient feature processing

Both include a **unified auto-tuning system** that benchmarks algorithm candidates at runtime and caches the best configuration per problem shape.

## Convolution Kernel Backends

A sparse convolution decomposes into per-offset GEMMs: for a 3x3x3 kernel (27 offsets), each offset produces a gather-GEMM-scatter operation. WarpConvNet provides multiple backends that implement this pattern with different trade-offs.

### Per-Offset Backends

These backends process each kernel offset as a separate GEMM call:

| Backend                 | Implementation                                                              | Strengths                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `explicit_gemm`         | Gather features into a dense buffer, call `torch.mm`, scatter-add results   | Simple, reliable fallback. No CUDA alignment requirements.                                                                                        |
| `implicit_gemm`         | Custom CUDA kernel that fuses gather, GEMM, and scatter-add into one launch | Avoids intermediate buffers. Best at small channels (C \<= 64) where launch overhead matters less than memory traffic.                            |
| `cutlass_implicit_gemm` | CUTLASS fused gather-GEMM-scatter kernel                                    | High throughput at large channels. Auto-tunes MMA tile shape and split-K internally. Requires channels aligned to 8.                              |
| `cute_implicit_gemm`    | CuTe 3.x fused gather-GEMM-scatter kernel                                   | Similar to CUTLASS but uses the CuTe 3.x programming model with vectorized A-operand loads (`cp.async`). Competitive at small-to-medium channels. |

### Grouped Backends

These backends batch multiple small offsets together to reduce launch overhead. See [Adaptive GEMM Grouping](./adaptive_gemm_grouping.md) for details on the bucketing algorithm.

| Backend                  | Implementation                                                                                    | Strengths                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `explicit_gemm_grouped`  | Gather into a padded `[B, M_max, C_in]` buffer, call `torch.bmm`                                  | Simple batched path. Rarely optimal in practice.                                                                                       |
| `implicit_gemm_grouped`  | Custom CUDA kernel processing all pairs from a bucket in one launch with per-row weight selection | Zero padding waste. Competitive at small channels with large kernel volumes (e.g., 5x5x5).                                             |
| `cutlass_grouped_hybrid` | CUTLASS for large offsets + `torch.bmm` for grouped small offsets                                 | Amortizes CUTLASS per-offset setup cost. Strong at medium-to-large channels.                                                           |
| `cute_grouped`           | CuTe 3.x grouped GEMM with batched offset execution                                               | **Most frequently optimal backend overall** (44% win rate across all cached configs). Dominates at C >= 96. Auto-tunes MMA tile shape. |

## Auto-Tuning System

### How It Works

On the first forward (or backward) pass for a new problem shape, WarpConvNet:

1. Selects a set of **candidate algorithms** based on the convolution dimensions
2. Runs each candidate with warmup + timed iterations
3. Picks the fastest and caches the result keyed by `(log2(N_in), log2(N_out), C_in, C_out, kernel_volume, dtype)`
4. Subsequent calls with the same shape hit the cache instantly

Results are persisted to `~/.cache/warpconvnet/benchmark_cache_generic.msgpack` and survive across Python sessions.

### Adaptive Candidate Selection

Not all algorithms are competitive at all problem sizes. To reduce auto-tuning time, the candidate set adapts to the convolution dimensions:

**Small channels** (max(C_in, C_out) \<= 64) -- 10 candidates:

- `cutlass_implicit_gemm`, `cutlass_grouped_hybrid`
- `cute_grouped` (3 MMA tile variants)
- `implicit_gemm` (2 block sizes) -- wins ~30% of small-channel configs
- `cute_implicit_gemm` -- wins at small N with small channels
- `implicit_gemm_grouped` (2 saturation values) -- wins at 3->32, 7->13

**Large channels** (max(C_in, C_out) > 64) -- 5 candidates:

- `cutlass_implicit_gemm`, `cutlass_grouped_hybrid`
- `cute_grouped` (3 MMA tile variants)

At large channels, `implicit_gemm` and its variants never win, so they are excluded. This cuts auto-tuning time by 50% for large-channel layers without any performance loss.

### `auto` vs `all` Mode

The `WARPCONVNET_FWD_ALGO_MODE` and `WARPCONVNET_BWD_ALGO_MODE` environment variables control which candidate set is used:

| Mode             | Forward Candidates | Backward Candidates | Use Case                                                                      |
| ---------------- | -----------------: | ------------------: | ----------------------------------------------------------------------------- |
| `auto` (default) |    5-10 (adaptive) |                  12 | Normal usage. Covers >96% of winning algorithms.                              |
| `all`            |                 19 |                 32+ | Exhaustive search. Use when validating on new hardware or after code changes. |

```bash
# Default: adaptive reduced set (recommended)
export WARPCONVNET_FWD_ALGO_MODE=auto

# Exhaustive: benchmark every algorithm variant
export WARPCONVNET_FWD_ALGO_MODE=all

# Specific algorithm (no benchmarking, just use it)
export WARPCONVNET_FWD_ALGO_MODE=cute_grouped

# Algorithm list (benchmark only these)
export WARPCONVNET_FWD_ALGO_MODE="[cute_grouped,cutlass_implicit_gemm]"
```

The same options apply to `WARPCONVNET_BWD_ALGO_MODE`.

### Backward Pass Candidate Set

The backward pass has its own reduced candidate set (12 candidates in `auto` mode):

| Backend                  | Backward Win Rate |
| ------------------------ | ----------------: |
| `cutlass_implicit_gemm`  |             58.9% |
| `cutlass_grouped_hybrid` |             21.9% |
| `explicit_gemm_grouped`  |              7.3% |
| `cute_grouped`           |              6.0% |
| `explicit_gemm`          |              4.6% |
| `implicit_gemm`          |              1.3% |

Unlike forward where `cute_grouped` dominates, backward is dominated by `cutlass_implicit_gemm` across nearly all channel configurations.

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

```python
# Single algorithm (string or enum)
output = spatially_sparse_conv(
    input_voxels, weight, kernel_size=3,
    fwd_algo="cute_grouped",
    bwd_algo="cutlass_implicit_gemm",
)

# Algorithm list -- benchmarks only these
output = spatially_sparse_conv(
    input_voxels, weight, kernel_size=3,
    fwd_algo=["cute_grouped", "cutlass_implicit_gemm"],
    bwd_algo=["cutlass_implicit_gemm", "cutlass_grouped_hybrid"],
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
# Forward algorithm (default: auto)
export WARPCONVNET_FWD_ALGO_MODE=auto

# Backward algorithm (default: auto)
export WARPCONVNET_BWD_ALGO_MODE=auto
```

Accepted values: `auto`, `all`, any single algorithm name, or a bracket list like `[algo1,algo2]`.

Valid algorithm names: `explicit_gemm`, `implicit_gemm`, `cutlass_implicit_gemm`, `cute_implicit_gemm`, `explicit_gemm_grouped`, `implicit_gemm_grouped`, `cutlass_grouped_hybrid`, `cute_grouped`.

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
python scripts/inspect_benchmark_cache.py namespace=sparse_conv_forward --best-only
```

Use `scripts/analyze_autotune_cache.py` to generate statistical analysis of algorithm win rates:

```bash
python scripts/analyze_autotune_cache.py --markdown --output analysis.md
```

See [Inspecting the Benchmark Cache](./inspect_benchmark_cache.md) for details.

## Performance Characteristics

### When Each Backend Wins

Based on empirical analysis across 292 forward configurations on RTX 6000 Ada:

| Condition                       | Best Backend            | Why                                                               |
| ------------------------------- | ----------------------- | ----------------------------------------------------------------- |
| C_in, C_out >= 96               | `cute_grouped`          | Batched CuTe GEMM amortizes launch overhead; high FLOP/byte ratio |
| C_in, C_out \<= 64              | `implicit_gemm`         | Fused kernel avoids intermediate buffers; low per-launch cost     |
| C_in=3, C_out=32 (initial conv) | `implicit_gemm_grouped` | Small C_in means gather is cheap; grouping helps at kv=125        |
| C=96 with large N (>64K)        | `cutlass_implicit_gemm` | CUTLASS saturates GPU at large problem sizes                      |
| Backward pass (all sizes)       | `cutlass_implicit_gemm` | Dominates 59% of backward configs                                 |

### Margins Are Tight

Auto-tuning is important because algorithm performance differences are small:

- **Median margin** between #1 and #2: 1.9% (forward), 3.4% (backward)
- **71% of configs** have \<5% margin between winner and runner-up
- No single algorithm can be hardcoded without measurable regression

## Troubleshooting

**Slow first run**: Normal -- auto-tuning benchmarks candidates. Subsequent runs use the cache. Use `auto` mode (not `all`) to minimize tuning time.

**Clear cache when switching GPUs**:

```bash
rm -rf ~/.cache/warpconvnet/
```

**CUTLASS not available**: Some backends require specific GPU compute capability and channel alignment (multiple of 8). Fall back to:

```bash
export WARPCONVNET_FWD_ALGO_MODE="[explicit_gemm,implicit_gemm]"
export WARPCONVNET_BWD_ALGO_MODE="[explicit_gemm,implicit_gemm]"
```

**Algorithm availability warnings**: Check that your CUDA toolkit and GPU support the requested backend. CuTe 3.x backends require the CuTe extension to be compiled.

## Source Files

| File                                                              | Contents                                                   |
| ----------------------------------------------------------------- | ---------------------------------------------------------- |
| `warpconvnet/nn/functional/sparse_conv/detail/unified.py`         | Auto-tuning system, adaptive candidate selection, dispatch |
| `warpconvnet/nn/functional/sparse_conv/detail/explicit.py`        | `explicit_gemm` forward/backward                           |
| `warpconvnet/nn/functional/sparse_conv/detail/implicit_direct.py` | `implicit_gemm` forward/backward                           |
| `warpconvnet/nn/functional/sparse_conv/detail/cutlass.py`         | `cutlass_implicit_gemm` forward/backward                   |
| `warpconvnet/nn/functional/sparse_conv/detail/cute_grouped.py`    | `cute_grouped` forward/backward                            |
| `warpconvnet/nn/functional/sparse_conv/detail/grouping.py`        | Adaptive offset grouping / bucketing                       |
| `warpconvnet/utils/benchmark_cache.py`                            | Generic benchmark cache with persistence                   |
| `warpconvnet/constants.py`                                        | Environment variable parsing                               |
