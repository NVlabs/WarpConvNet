# Auto-Tuning

**Created**: 2026-04-18 17:02:44
**Edited**: 2026-04-18 17:02:44

WarpConvNet's spatially sparse convolution has many backend algorithms
(see [Spatially Sparse Convolutions — Algorithms](./sparse_convolutions.md#algorithms-overview)).
None of them wins across all problem shapes: the optimal pick depends on
coordinate count, input/output channels, kernel volume, dtype, and the
GPU. This page describes how WarpConvNet chooses.

## Why auto-tune

A single sparse-conv layer runs **three math kernels** per training step
(forward = AB, dgrad = ABt, wgrad = AtB — see
[Three math kernels per layer](./sparse_convolutions.md#three-math-kernels-per-layer)),
each with its own optimal algorithm:

- Relative winners shift dramatically with channel count (e.g. 64 vs 256).
- Small-$N$ shapes favor mask-based fused kernels; large-$N$ shapes
  favor CUTLASS.
- Wgrad (AtB gather-gather) has different arithmetic intensity than
  AB/ABt and picks differently from forward at the same shape.
- Dgrad (ABt) picks differently from fwd (AB) because the
  $C_\text{in} \leftrightarrow C_\text{out}$ swap changes the optimal
  tile shape and split-K factor.

Picking by hand is infeasible. WarpConvNet benchmarks the candidate set
at the runtime shape on first use and caches the winner, per op, in
three independent cache namespaces (`AB_gather_scatter`,
`ABt_gather_scatter`, `AtB_gather_gather`).

## How it works

On the first forward (or backward) pass for a new problem shape,
WarpConvNet:

1. Selects a set of **candidate algorithms** based on the convolution
   dimensions $(N, C_{\text{in}}, C_{\text{out}}, K)$ and dtype.
2. Runs each candidate with `warmup=2`, `iters=5` and records median time
   via CUDA events.
3. Picks the fastest and caches the result keyed by
   $(\lceil\log_{10} N_{\text{in}}\rceil, \lceil\log_{10} N_{\text{out}}\rceil, C_{\text{in}}, C_{\text{out}}, K, G, \text{use\_fp16\_accum}, \text{dtype}, \text{SM})$.
4. Subsequent calls with the same shape hit the cache instantly.

Results are persisted to
`~/.cache/warpconvnet/benchmark_cache_generic.msgpack` and survive across
Python sessions. The cache merges back-in results from other ranks so
that rank 0's autotune pass populates every rank.

## Adaptive candidate selection

The candidate set adapts to the problem dimensions. Based on benchmark
analysis of 148 configs on SM 8.9 with cuBLAS 12.9.1.4:

**AB gather-scatter (forward)** and **ABt gather-scatter (dgrad)** share
the same candidate pool — 7-11 candidates per op; each op is tuned
independently against its own cache namespace:

| $N$ range                             | $ch \le 256$                                      | $ch > 256$                               |
| ------------------------------------- | ------------------------------------------------- | ---------------------------------------- |
| Small ($N \le 10{,}000$)              | `production` (92-100%)                            | `cute_grouped` (58%), `production` (25%) |
| Medium ($10{,}000 < N \le 100{,}000$) | `production` (69%), `cutlass_implicit_gemm` (27%) | `cutlass_grouped_hybrid` (67%)           |
| Large ($N > 100{,}000$)               | `production` / `cutlass_grouped` / `cutlass`      | `cutlass_implicit_gemm` (100%)           |

**AtB gather-gather (wgrad)** — 5-8 candidates:

| $N$ range | $ch \le 64$                                                   | $ch > 64$             |
| --------- | ------------------------------------------------------------- | --------------------- |
| Small     | `cute_grouped` (57%), `implicit_gemm` (36%)                   | `cute_grouped` (100%) |
| Medium    | `cute_grouped` (57%), `explicit_gemm_grouped` (43%)           | `cute_grouped` (77%)  |
| Large     | `cutlass_grouped_hybrid` (57%), `explicit_gemm_grouped` (36%) | `cute_grouped` (100%) |

## Modes

Three global modes for the AB and AtB candidate sets:

| Mode             |   AB candidates | AtB candidates | When to use                                                                       |
| ---------------- | --------------: | -------------: | --------------------------------------------------------------------------------- |
| `auto` (default) | 7-11 (adaptive) | 5-8 (adaptive) | Normal training / inference. Covers every winning algorithm at its optimal shape. |
| `trimmed`        |              11 |             27 | Broader search. Includes slower-converging alternatives but excludes dead-weight. |
| `all`            |              23 |             35 | Exhaustive. For benchmarking new hardware or new backends; slowest first run.     |

```bash
# Default: adaptive reduced set (recommended)
export WARPCONVNET_FWD_ALGO_MODE=auto

# Exhaustive: benchmark every algorithm variant
export WARPCONVNET_FWD_ALGO_MODE=all

# Specific algorithm (no benchmarking, just use it)
export WARPCONVNET_FWD_ALGO_MODE=production

# Algorithm list (benchmark only these)
export WARPCONVNET_FWD_ALGO_MODE="[production,cutlass_implicit_gemm]"
```

The same options apply to `WARPCONVNET_DGRAD_ALGO_MODE` (dgrad ABt
algorithm) and `WARPCONVNET_WGRAD_ALGO_MODE` (wgrad AtB algorithm).

## Specifying algorithms

Forward, dgrad, and wgrad can be controlled independently:

```python
from warpconvnet.nn.functional import spatially_sparse_conv

# Different algorithms for each op
output = spatially_sparse_conv(
    input_voxels, weight, kernel_size=3,
    fwd_algo="production",       # AB for forward
    dgrad_algo="production",     # AB for dgrad
    wgrad_algo="cute_grouped",   # AtB for wgrad
)

# Algorithm list -- benchmark only these
output = spatially_sparse_conv(
    input_voxels, weight, kernel_size=3,
    fwd_algo=["production", "cutlass_implicit_gemm"],
    dgrad_algo=["production", "cute_grouped"],
    wgrad_algo=["cute_grouped", "cutlass_grouped_hybrid"],
)
```

### Strict name filter

Named algorithms are resolved strictly. A typo raises `ValueError` rather
than silently falling back to autotune:

```python
spatially_sparse_conv(..., fwd_algo="explicit_gem")  # typo
# ValueError: Unknown algorithm(s) in filter: ['explicit_gem'].
# Not present in adaptive pool or exhaustive _ALL_AB_PARAMS.
# Fix the algo name or extend the pool.
```

Parameterless algorithms (`explicit_gemm`, `cutlass_implicit_gemm`,
`cute_implicit_gemm`) are synthesised as `(name, {})` when they are not
in the current adaptive pool, so those names always work regardless of
mode.

## Environment variables

| Variable                                   | Default                | Description                                                                                               |
| ------------------------------------------ | ---------------------- | --------------------------------------------------------------------------------------------------------- |
| `WARPCONVNET_FWD_ALGO_MODE`                | `auto`                 | AB gather-scatter algorithm for forward. Shared candidate pool with dgrad.                                |
| `WARPCONVNET_DGRAD_ALGO_MODE`              | `auto`                 | ABt gather-scatter algorithm for dgrad. Shared candidate pool with forward; tuned + cached independently. |
| `WARPCONVNET_WGRAD_ALGO_MODE`              | `auto`                 | AtB gather-gather algorithm for wgrad.                                                                    |
| `WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE` | `auto`                 | Depthwise forward algorithm.                                                                              |
| `WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE` | `auto`                 | Depthwise backward algorithm.                                                                             |
| `WARPCONVNET_USE_FP16_ACCUM`               | `false`                | Global default for the fp16 accumulator flag. See [Accumulator Precision](./accumulator_precision.md).    |
| `WARPCONVNET_BENCHMARK_CACHE_DIR`          | `~/.cache/warpconvnet` | Cache directory.                                                                                          |
| `WARPCONVNET_AUTOTUNE_LOG`                 | `true`                 | Set to `false` to suppress auto-tuning log messages.                                                      |

Accepted values for the mode variables: `auto`, `all`, `trimmed`, a
single algorithm name, or a bracket list like `[algo1,algo2]`.

Valid algorithm names: `explicit_gemm`, `implicit_gemm`,
`cutlass_implicit_gemm`, `cute_implicit_gemm`, `explicit_gemm_grouped`,
`implicit_gemm_grouped`, `cutlass_grouped_hybrid`, `cute_grouped`,
`production`. Unknown names raise `ValueError`.

## Cache

Results are keyed per problem shape and persisted to
`~/.cache/warpconvnet/benchmark_cache_generic.msgpack`.

```bash
# Clear cache (e.g. after switching GPUs)
rm -rf ~/.cache/warpconvnet/

# Inspect cached entries
python scripts/inspect_benchmark_cache.py
python scripts/inspect_benchmark_cache.py namespace=AB_gather_scatter --best-only   # forward
python scripts/inspect_benchmark_cache.py namespace=ABt_gather_scatter --best-only  # dgrad
python scripts/inspect_benchmark_cache.py namespace=AtB_gather_gather --best-only   # wgrad

# Analyze win rates and margins across all configs
python scripts/analyze_autotune_cache.py --markdown --output analysis.md
```

See [Inspect Benchmark Cache](./inspect_benchmark_cache.md) for the full
inspector CLI and
[Pre-Populate Benchmark Cache](./populate_benchmark_cache.md) for
filling the cache ahead of production.

## Performance characteristics

Based on empirical analysis on RTX 6000 Ada with cuBLAS 12.9.1.4:

| Condition                      | Best AB backend         | Best AtB backend                           |
| ------------------------------ | ----------------------- | ------------------------------------------ |
| $ch \le 256$, any $N$          | `production`            | `cute_grouped`                             |
| $ch > 256$, small $N$          | `cute_grouped`          | `cute_grouped`                             |
| $ch > 256$, large $N$          | `cutlass_implicit_gemm` | `cute_grouped`                             |
| $ch \le 64$, small $N$ (wgrad) | —                       | `implicit_gemm` or `explicit_gemm_grouped` |

The cost of the first autotune pass on a previously-unseen shape is
roughly `(warmup + iters) * n_candidates * kernel_time`. For `auto`
mode this is typically under one second on a warm GPU; for `all` mode
it can take tens of seconds.

## Troubleshooting

**Slow first run**: normal — autotune benchmarks candidates. Use `auto`
(not `all`) to minimize tuning time. To skip autotune entirely,
[pre-populate the cache](./populate_benchmark_cache.md) before your
first run.

**Cache mismatch across GPUs**: the SM capability is embedded in cache
keys, so entries from one GPU will not be picked up on another. Clear
the cache with `rm -rf ~/.cache/warpconvnet/` when switching hardware.

**CUTLASS not available**: some backends require specific compute
capability. Fall back with an explicit list:

```bash
export WARPCONVNET_FWD_ALGO_MODE="[explicit_gemm,implicit_gemm,production]"
```

**`ValueError: Unknown algorithm(s) in filter`**: you passed a name
that is not in the adaptive or exhaustive pool. Check the valid names
list above.

## Source files

| File                                                          | Contents                                                    |
| ------------------------------------------------------------- | ----------------------------------------------------------- |
| `warpconvnet/nn/functional/sparse_conv/detail/unified.py`     | Top-level auto-tune dispatch.                               |
| `warpconvnet/nn/functional/sparse_conv/detail/algo_params.py` | Adaptive candidate selection, mode handling, strict filter. |
| `warpconvnet/nn/functional/sparse_conv/detail/autotune.py`    | Benchmark runners, cache init/merge, callback registration. |
| `warpconvnet/utils/benchmark_cache.py`                        | Generic benchmark cache with msgpack persistence.           |
| `warpconvnet/constants.py`                                    | Environment variable parsing.                               |
