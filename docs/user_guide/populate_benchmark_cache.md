## Pre-Populating the Benchmark Cache

WarpConvNet auto-tunes sparse convolution algorithms on the first forward (and backward) pass for each new problem shape. While the results are cached for future runs, the initial tuning adds latency to the first few iterations.

To eliminate this cold-start cost entirely, you can **pre-populate the cache** using the provided script. This is especially useful for:

- **Production deployments** where first-iteration latency matters
- **Shared clusters** where a single cache file can be distributed to all users
- **CI/CD pipelines** that need deterministic timing from the first iteration

### Quick start

```bash
# Pre-populate with default configs (MinkUNet/MaxViT-UNet channel progressions,
# 7 voxel counts, ks=3 and ks=5, fp16 and bf16 — 728 configs total)
python scripts/populate_benchmark_cache.py

# Quick smoke test (6 configs, ~1 minute)
python scripts/populate_benchmark_cache.py --preset quick

# Preview what will be benchmarked without running anything
python scripts/populate_benchmark_cache.py --dry-run
```

### What it benchmarks

The default configuration grid covers common 3D deep learning architectures:

| Dimension | Values | Source |
| --------- | ------ | ------ |
| **Voxel counts** | 30K, 65K, 130K, 260K, 500K, 1M, 2M | Indoor (ScanNet) to outdoor (nuScenes/Waymo) |
| **Channel pairs** | 3→32, 32→64, 64→128, 128→256, 256→256, ... (26 pairs) | MinkUNet18/34, MaxViT-UNet, SparseConvUNet |
| **Kernel sizes** | 3, 5 | Standard 3×3×3 and 5×5×5 |
| **Dtypes** | float16, bfloat16 | Mixed-precision training |

After log₂-deduplication (voxel counts that map to the same cache bucket are merged), this produces **728 unique configurations**.

### Customizing the grid

```bash
# Only benchmark specific channel pairs
python scripts/populate_benchmark_cache.py --channels 32,64 128,256

# Only specific voxel counts
python scripts/populate_benchmark_cache.py --num-voxels 100000 500000

# Only forward pass
python scripts/populate_benchmark_cache.py --forward-only

# Exhaustive algorithm search (slower but tests all candidates)
python scripts/populate_benchmark_cache.py --algo-mode all

# Combine options
python scripts/populate_benchmark_cache.py \
    --channels 64,128 128,256 256,256 \
    --num-voxels 200000 1000000 \
    --kernel-sizes 3 \
    --dtypes float16 \
    --forward-only
```

### Resuming interrupted runs

The `--resume` flag skips configurations that already have a cache entry. This is useful for long runs that may be interrupted:

```bash
# First run (interrupted after 200 configs)
python scripts/populate_benchmark_cache.py

# Resume — picks up where it left off
python scripts/populate_benchmark_cache.py --resume
```

### Distributing cache files

The cache is stored at `~/.cache/warpconvnet/benchmark_cache_generic.msgpack`. It is **architecture-specific** — the GPU's SM capability (e.g., SM 8.0 for A100, SM 8.9 for RTX 6000 Ada) is embedded in cache keys. To distribute:

1. Run the script on each target GPU architecture
2. Copy the resulting `.msgpack` file to the target machine's `~/.cache/warpconvnet/`

```bash
# On the source machine (e.g., A100)
python scripts/populate_benchmark_cache.py
ls -lh ~/.cache/warpconvnet/benchmark_cache_generic.msgpack

# Copy to target machines
scp ~/.cache/warpconvnet/benchmark_cache_generic.msgpack \
    user@target:~/.cache/warpconvnet/
```

!!! warning "Do not mix cache files from different GPU architectures"
    Cache entries from an A100 (SM 8.0) will not match lookups on an RTX 4090 (SM 8.9). Each GPU architecture needs its own cache. If you accidentally mix them, clear the cache with `rm -rf ~/.cache/warpconvnet/` and re-run the script.

### Full CLI reference

```
usage: populate_benchmark_cache.py [-h]
    [--preset {default,quick}]
    [--num-voxels N [N ...]]
    [--channels C_in,C_out [C_in,C_out ...]]
    [--kernel-sizes K [K ...]]
    [--dtypes {float16,bfloat16,float32} ...]
    [--algo-mode MODE]
    [--forward-only] [--backward-only]
    [--batch-size B]
    [--dry-run] [--resume]
    [--device DEVICE]
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--preset` | `default` | `default` (728 configs) or `quick` (6 configs) |
| `--num-voxels` | preset | Override voxel counts |
| `--channels` | preset | Override channel pairs as `C_in,C_out` |
| `--kernel-sizes` | preset | Override kernel sizes |
| `--dtypes` | preset | Override dtypes |
| `--algo-mode` | `auto` | Algorithm selection: `auto`, `all`, or specific name |
| `--forward-only` | off | Skip backward pass benchmarking |
| `--backward-only` | off | Skip forward pass benchmarking |
| `--batch-size` | 1 | Batch size for voxel generation |
| `--dry-run` | off | List configs without running |
| `--resume` | off | Skip configs already in cache |
| `--device` | `cuda:0` | CUDA device to benchmark on |

### Relationship to environment variables

The script respects `WARPCONVNET_BENCHMARK_CACHE_DIR` if set:

```bash
export WARPCONVNET_BENCHMARK_CACHE_DIR=/shared/warpconvnet_cache
python scripts/populate_benchmark_cache.py
```

The `--algo-mode` flag sets `WARPCONVNET_FWD_ALGO_MODE` and `WARPCONVNET_BWD_ALGO_MODE` internally. See [Sparse Convolutions](./sparse_convolutions.md) for details on algorithm modes.
