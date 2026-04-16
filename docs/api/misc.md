# Miscellaneous

## Environment variables

WarpConvNet reads the following environment variables at import time.
All are optional — defaults are chosen for typical use.

Defined in `warpconvnet/constants.py`.

### Algorithm selection

| Variable                                   | Default | Description                                                                                                                                                                                                        |
| ------------------------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `WARPCONVNET_FWD_ALGO_MODE`                | `auto`  | Forward convolution algorithm. `auto` benchmarks a reduced candidate set. `all` benchmarks every algorithm. Can also be a single name (e.g., `implicit_gemm`) or a list (`[implicit_gemm,cutlass_implicit_gemm]`). |
| `WARPCONVNET_BWD_ALGO_MODE`                | `auto`  | Backward convolution algorithm. Same format as forward.                                                                                                                                                            |
| `WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE` | `auto`  | Depthwise forward algorithm (`explicit_gemm`, `implicit_gemm`, or `auto`).                                                                                                                                         |
| `WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE` | `auto`  | Depthwise backward algorithm.                                                                                                                                                                                      |

Valid algorithm names: `explicit_gemm`, `implicit_gemm`,
`cutlass_implicit_gemm`, `cute_implicit_gemm`, `explicit_gemm_grouped`,
`implicit_gemm_grouped`, `cutlass_grouped_hybrid`, `cute_grouped`,
`production`, `production`, `auto`, `all`, `trimmed`.

### Benchmark cache

| Variable                                   | Default                | Description                                                |
| ------------------------------------------ | ---------------------- | ---------------------------------------------------------- |
| `WARPCONVNET_BENCHMARK_CACHE_DIR`          | `~/.cache/warpconvnet` | Directory for the auto-tuning benchmark cache              |
| `WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE` | (unset)                | If set, overrides the default cache directory              |
| `WARPCONVNET_AUTOTUNE_LOG`                 | `true`                 | Set to `false` or `0` to suppress auto-tuning log messages |

### Other

| Variable                                | Default | Description                                                          |
| --------------------------------------- | ------- | -------------------------------------------------------------------- |
| `WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP` | `false` | Skip symmetric kernel map optimization                               |
| `WARPCONVNET_SKIP_EXTENSION`            | `0`     | Set to `1` to skip loading the C++ extension (for docs builds, etc.) |
