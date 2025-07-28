# Sparse Convolutions

WarpConvNet implements efficient spatially sparse convolutions on voxel grids using CUDA and Warp kernels.

## Overview

WarpConvNet provides two types of sparse convolutions:

- **Regular Sparse Convolution**: General-purpose convolution for feature learning
- **Depthwise Sparse Convolution**: Channel-wise convolution for efficient feature processing

Both implementations feature a **unified benchmarking system** that automatically finds optimal algorithm configurations for your specific hardware and data characteristics.

## Algorithm Selection

### Available Algorithms

#### Regular Sparse Convolution

- `EXPLICIT_GEMM`: Traditional matrix multiplication approach
- `IMPLICIT_GEMM`: Custom CUDA kernels with implicit GEMM operations
- `CUTLASS_IMPLICIT_GEMM`: NVIDIA CUTLASS-based high-performance kernels
- `AUTO`: Automatically benchmark and select the best algorithm

#### Depthwise Sparse Convolution

- `EXPLICIT`: Element-wise multiplication approach
- `IMPLICIT`: Custom CUDA kernels for depthwise operations
- `AUTO`: Automatically benchmark and select the best algorithm

### Unified Benchmarking System

The unified benchmarking system ensures **consistent parameter optimization** across all algorithm inputs:

- **Single Algorithm**: `fwd_algo=IMPLICIT_GEMM` → Benchmarks all parameter combinations for IMPLICIT_GEMM
- **Algorithm List**: `fwd_algo=[IMPLICIT_GEMM, CUTLASS_IMPLICIT_GEMM]` → Benchmarks both algorithms, selects best
- **AUTO Mode**: `fwd_algo=AUTO` → Benchmarks all available algorithms, selects best

**Key Insight**: Algorithm input acts as a **search space filter** - benchmarking always occurs to find optimal parameters within the specified space.

## Usage Examples

### Basic Usage

```python
import torch
from warpconvnet.nn.functional import spatially_sparse_conv
from warpconvnet.nn.functional.sparse_conv import SPARSE_CONV_FWD_ALGO_MODE

# Single algorithm (triggers parameter optimization)
output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    fwd_algo=SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM,
    bwd_algo=SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM,
)

# Algorithm list (limits search space, finds best within list)
output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    fwd_algo=[
        SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM,
        SPARSE_CONV_FWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM
    ],
    bwd_algo=[
        SPARSE_CONV_BWD_ALGO_MODE.IMPLICIT_GEMM,
        SPARSE_CONV_BWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM
    ],
)

# AUTO mode (benchmarks all available algorithms)
output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    fwd_algo=SPARSE_CONV_FWD_ALGO_MODE.AUTO,
    bwd_algo=SPARSE_CONV_BWD_ALGO_MODE.AUTO,
)
```

### Module API

```python
from warpconvnet.nn.modules import SpatiallySparseConv

# Single algorithm with module
conv = SpatiallySparseConv(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    fwd_algo=SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM,
    bwd_algo=SPARSE_CONV_BWD_ALGO_MODE.IMPLICIT_GEMM,
)

# Algorithm list with module
conv = SpatiallySparseConv(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    fwd_algo=[
        SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM,
        SPARSE_CONV_FWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM
    ],
    bwd_algo=[
        SPARSE_CONV_BWD_ALGO_MODE.IMPLICIT_GEMM,
        SPARSE_CONV_BWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM
    ],
)
```

### Depthwise Convolution

```python
from warpconvnet.nn.functional import spatially_sparse_depthwise_conv
from warpconvnet.nn.functional.sparse_conv_depth import SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE

# Depthwise convolution with algorithm list
output = spatially_sparse_depthwise_conv(
    input_features,
    depthwise_weight,
    kernel_map,
    num_out_coords,
    fwd_algo=[
        SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT,
        SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT
    ],
    bwd_algo=[
        SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
        SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT
    ],
)
```

## Environment Variable Configuration

You can set global defaults using environment variables that support both single algorithms and algorithm lists:

### Regular Sparse Convolution

```bash
# Single algorithm
export WARPCONVNET_FWD_ALGO_MODE=implicit_gemm
export WARPCONVNET_BWD_ALGO_MODE=implicit_gemm

# Algorithm list (limits search space)
export WARPCONVNET_FWD_ALGO_MODE="[implicit_gemm,cutlass_implicit_gemm]"
export WARPCONVNET_BWD_ALGO_MODE="[implicit_gemm,cutlass_implicit_gemm]"

# AUTO mode (benchmark all algorithms)
export WARPCONVNET_FWD_ALGO_MODE=auto
export WARPCONVNET_BWD_ALGO_MODE=auto
```

### Depthwise Sparse Convolution

```bash
# Single algorithm
export WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE=explicit
export WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE=explicit

# Algorithm list
export WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE="[explicit,implicit]"
export WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE="[explicit,implicit]"

# AUTO mode
export WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE=auto
export WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE=auto
```

### Using Environment Variables

```python
# When no algorithm specified, uses environment variables
output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    # fwd_algo and bwd_algo default to environment variables
)

# Explicit None also uses environment variables
output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    fwd_algo=None,  # Uses WARPCONVNET_FWD_ALGO_MODE
    bwd_algo=None,  # Uses WARPCONVNET_BWD_ALGO_MODE
)
```

## Benchmarking and Performance Optimization

### How Benchmarking Works

1. **Algorithm Filtering**: The system determines which algorithms to benchmark based on your input
2. **Parameter Generation**: For each algorithm, generates all possible parameter combinations
3. **Performance Testing**: Runs each combination multiple times and measures execution time
4. **Optimal Selection**: Chooses the fastest algorithm and parameter configuration
5. **Caching**: Stores results for future use with similar configurations

### Parameter Examples

The system automatically optimizes parameters like:

**IMPLICIT_GEMM**:

- `fwd_block_size`: 4, 16, 32
- `gemm_block_size`: 4, 16, 32
- `split_k_factor`: 2, 4, 8

**CUTLASS_IMPLICIT_GEMM**:

- `mma_tile`: 0, 1, 2, 3
- `split_k_slices`: 1, 2, 4, 8
- `accumulator_type`: float32

### Performance Benefits

The unified benchmarking system provides:

- **Consistent Optimization**: All execution paths lead to parameter optimization
- **Hardware Adaptation**: Automatically finds best configuration for your GPU
- **Future-Proof**: New algorithm parameters are automatically optimized
- **Search Space Control**: Algorithm lists limit benchmarking scope for faster startup

## String and Mixed Input Support

The system supports flexible input formats:

```python
# String inputs
output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    fwd_algo="implicit_gemm",  # String format
    bwd_algo="implicit_gemm",
)

# String lists
output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    fwd_algo=["implicit_gemm", "cutlass_implicit_gemm"],  # String list
    bwd_algo=["implicit_gemm", "cutlass_implicit_gemm"],
)

# Mixed enum and string lists
output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    fwd_algo=[SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM, "cutlass_implicit_gemm"],
    bwd_algo=[SPARSE_CONV_BWD_ALGO_MODE.IMPLICIT_GEMM, "cutlass_implicit_gemm"],
)
```

## Cache Management

The benchmark cache is automatically managed:

- **Persistent Storage**: Results saved to `~/.cache/warpconvnet/`
- **Configuration-Specific**: Different cache entries for different input sizes/types
- **Background Saving**: Cache updates happen in background threads
- **Manual Reset**: Clear cache with `rm -rf ~/.cache/warpconvnet/` if needed

## Best Practices

### For Development

- Use `AUTO` mode to explore all available algorithms
- Use algorithm lists to limit search space during hyperparameter tuning
- Monitor first-run performance (benchmarking overhead) vs. cached runs

### For Production

- Use specific algorithms once you know what works best
- Set environment variables for consistent behavior across runs
- Consider using algorithm lists if you want to restrict to tested algorithms

### For New Hardware

- Clear cache when switching GPUs: `rm -rf ~/.cache/warpconvnet/`
- Use `AUTO` mode to discover optimal algorithms for new hardware
- Algorithm lists help compare specific algorithms on new hardware

## Troubleshooting

### Common Issues

**Slow First Run**:

- Normal behavior - benchmarking finds optimal parameters
- Subsequent runs use cached results and are fast
- Use algorithm lists to reduce initial benchmarking time

**Cache Issues**:

- Clear cache: `rm -rf ~/.cache/warpconvnet/`
- Check permissions on cache directory
- Cache is configuration-specific (input size, types, etc.)

**Algorithm Availability**:

- Some algorithms require specific CUDA versions
- CUTLASS algorithms need compatible GPU compute capability
- Check logs for algorithm availability warnings
- CUTLASS may not work on all GPUs
  - Use `export WARPCONVNET_FWD_ALGO_MODE="[explicit_gemm,implicit_gemm]"; export WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE="[explicit_gemm,implicit_gemm]"` to force use explicit and implicit gemm.

### Performance Tips

- Use algorithm lists to focus on algorithms known to work well
- Environment variables provide global defaults
- Cache results are persistent across Python sessions
- Benchmarking overhead is paid once per configuration
