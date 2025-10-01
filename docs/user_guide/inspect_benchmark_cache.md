## Inspecting the Benchmark Cache

WarpConvNet benchmarks sparse convolution algorithms at runtime and caches the results for fast reuse across sessions. This page documents the `scripts/inspect_benchmark_cache.py` helper script, which pretty-prints the cached results so you can understand what was tried and which configurations performed best.

### What it shows

- **Namespaces**: Logical groups of cached results (e.g., `sparse_conv_forward`, `sparse_conv_backward`, low-level kernels).
- **Per-configuration results**: For each input configuration (number of coords, channels, kernel volume, dtype), the script lists algorithms that were benchmarked and the measured time.
- **Ordering**: Results are shown best-first by default within each configuration.

## Quick start

Run without arguments to see the namespace tree:

```bash
python scripts/inspect_benchmark_cache.py
```

Show details for a specific namespace (e.g., forward sparse conv):

```bash
python scripts/inspect_benchmark_cache.py namespace=sparse_conv_forward
```

Only show the best algorithm per configuration:

```bash
python scripts/inspect_benchmark_cache.py namespace=sparse_conv_forward --best-only
```

Show the top K results per configuration:

```bash
python scripts/inspect_benchmark_cache.py namespace=sparse_conv_forward --top-k 3
```

Search namespaces or keys when passing extra arguments:

```bash
# List namespaces then search for entries containing "wmma"
python scripts/inspect_benchmark_cache.py wmma

# Search inside a specific namespace
python scripts/inspect_benchmark_cache.py namespace=sparse_conv_forward wmma
```

## Sample output

Below is an excerpt from a real run inspecting the `sparse_conv_forward` namespace. Times are in milliseconds; lower is better.

```text
Loading benchmark cache...
Cache file location: /home/<user>/.cache/warpconvnet/benchmark_cache_generic.pkl
Cache file size: 44,320 bytes
Last modified: 2025-09-08 13:33:35

============================================================
NAMESPACE TREE
============================================================
Total namespaces: 6

- implicit_gemm_AD_gather_scatter: 37 entry(ies)
- implicit_gemm_trAB_gather: 24 entry(ies)
- sparse_conv_backward: 6 entry(ies)
- sparse_conv_forward: 11 entry(ies)
- wmma_implicit_gemm_sm80: 23 entry(ies)
- wmma_split_k_implicit_gemm_sm80: 13 entry(ies)

============================================================
NAMESPACE: SPARSE_CONV_FORWARD
============================================================
Total configurations: 11

----------------------------------------
Configuration 1:
----------------------------------------
Config Parameters:
  log_num_in_coords: 21
  log_num_out_coords: 21
  in_channels: 3
  out_channels: 32
  kernel_volume: 27
  in_dtype: torch.float16

Results:
  [
    [
      "implicit_gemm"
      {
        fwd_block_size: 16
      }
      4.149
    ]
    [
      "implicit_gemm"
      {
        fwd_block_size: 32
      }
      7.833
    ]
    ["wmma_implicit_gemm", {}, 10.814]
    ["explicit_gemm", {}, 13.789]
    [
      "implicit_gemm"
      {
        fwd_block_size: 4
      }
      15.120
    ]
  ]

----------------------------------------
Configuration 2:
----------------------------------------
Config Parameters:
  log_num_in_coords: 21
  log_num_out_coords: 21
  in_channels: 32
  out_channels: 32
  kernel_volume: 27
  in_dtype: torch.float16

Results:
  [
    ["cutlass_implicit_gemm", {}, 4.613]
    [
      "implicit_gemm"
      {
        fwd_block_size: 16
      }
      8.107
    ]
    ["wmma_implicit_gemm", {}, 14.126]
    ["explicit_gemm", {}, 19.792]
  ]
```

## Interpreting results

- **Configuration**: A unique combination of problem shape and dtype: `num_in_coords`, `num_out_coords` (logged as powers of 2 for brevity), `in_channels`, `out_channels`, `kernel_volume`, and `in_dtype`.
- **Algorithms**: Each entry is `[algo_name, params, time_ms]`.
  - `implicit_gemm` may include parameters like `fwd_block_size`, `gemm_block_size`, `split_k_threads_per_block`, `split_k_factor` depending on forward/backward.
  - `cutlass_implicit_gemm` and `wmma_implicit_gemm` typically list `{}` because they auto-tune internally.
  - `explicit_gemm` uses a dense path and lists `{}`.
- **Best-first**: The first result per configuration is the fastest among those benchmarked.

## Relationship to environment variables

The cache reflects runs filtered by your environment variable settings in `warpconvnet/constants.py`:

- `WARPCONVNET_FWD_ALGO_MODE`
- `WARPCONVNET_BWD_ALGO_MODE`

These accept a single algorithm (e.g., `implicit_gemm`, `cutlass_implicit_gemm`, `wmma_implicit_gemm`, `explicit_gemm`) or a list like `[implicit_gemm,wmma_implicit_gemm,cutlass_implicit_gemm]`. The inspector shows what was actually benchmarked inside that filtered search space.

See the sparse convolutions guide for details on recommended settings and AUTO mode.

## Benchmark cache management

The benchmark cache is automatically managed:

- **Persistent Storage**: Results are saved to `~/.cache/warpconvnet/`
- **Configuration-Specific**: Different cache entries exist for different input sizes, channels, kernel volumes, and dtypes
- **Background Saving**: Cache updates can happen in background threads
- **Manual Reset**: Clear cache with `rm -rf ~/.cache/warpconvnet/` if needed

## Tips and troubleshooting

- **Clear cache** when switching GPUs or after significant software changes:
  ```bash
  rm -rf ~/.cache/warpconvnet/
  ```
- **Algorithm availability** depends on your GPU and toolchain:
  - CUTLASS requires compatible compute capability.
  - WMMA requires Tensor Cores and compatible compute capability.
- **First run is slower**: Benchmarking is performed once per unique configuration; subsequent runs reuse the cached best.
- **Focus the search**: Use env var lists to limit benchmarking to known-good algorithms during development.

## Script location

The inspector script lives at:

- `scripts/inspect_benchmark_cache.py`

You can open it for more flags and formatting logic, or invoke it directly as shown above.
