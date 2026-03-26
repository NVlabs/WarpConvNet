# Benchmark: WarpConvNet vs SpConv MinkUNet18

## Environment

| Parameter         | Value                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| GPU               | NVIDIA RTX 6000 Ada (SM 8.9, 48 GB)                                                                 |
| Driver            | 570.x                                                                                               |
| CUDA (nvcc)       | 12.9                                                                                                |
| PyTorch           | 2.10.0+cu128                                                                                        |
| Model             | MinkUNet18 (21.7M parameters)                                                                       |
| Batch size        | 4                                                                                                   |
| Voxel size        | 0.02                                                                                                |
| Voxels            | ~420,168 (identical for both -- CPU voxelization)                                                   |
| Data              | Synthetic ScanNet-like point clouds                                                                 |
| Warmup            | 5 iterations                                                                                        |
| Benchmark         | 10 iterations (median)                                                                              |
| `LD_LIBRARY_PATH` | `""` (avoids cuBLAS version mismatch, see [cublas_version_mismatch.md](cublas_version_mismatch.md)) |

## Results

### fp16 mixed precision (`16-mixed`) -- recommended

|             | SpConv (ms) | WarpConvNet (ms) | Speedup      |
| ----------- | ----------- | ---------------- | ------------ |
| Forward     | 202         | 109              | **1.9x**     |
| Backward    | 465         | 180              | **2.6x**     |
| **Fwd+Bwd** | **667**     | **289**          | **2.3x**     |
| Peak Memory | 12,786 MB   | 5,607 MB         | **56% less** |

### fp32 full precision (`32-true`)

|             | SpConv (ms) | WarpConvNet (ms) | Speedup      |
| ----------- | ----------- | ---------------- | ------------ |
| Forward     | 197         | 148              | **1.3x**     |
| Backward    | 468         | 414              | **1.1x**     |
| **Fwd+Bwd** | **665**     | **562**          | **1.18x**    |
| Peak Memory | 12,786 MB   | 10,356 MB        | **19% less** |

## Key Kernels

WarpConvNet's performance advantage comes from two fused CuTe GEMM kernels that replace the traditional per-offset kernel launch pattern used by SpConv.

### 1. Fused Grouped AD Gather-Scatter (forward + grad_input)

**Kernel**: `CuteGemmGroupedKernel` in `cute_gemm_grouped_kernel.h`

Standard sparse convolution with kernel size 3x3x3 has 27 offsets. SpConv and traditional implementations launch a **separate GEMM kernel per offset** (27 launches). The fused grouped AD kernel processes all offsets in a **single kernel launch**:

- Each threadblock determines its group (offset) via binary search on a prefix-sum array of M-tiles
- All groups share input features (A) and output (D), each has its own weight pointer (B)
- Output accumulation uses `atomicAdd` since multiple groups may scatter to overlapping rows
- Grid: `(total_m_tiles_across_all_groups, N_tiles)`

This kernel is used for both **forward** and **grad_input backward**. It dominates forward performance for medium-to-large channel counts (64+), selected by the auto-tuner for 18 of 27 forward configs.

**Impact**: 2x faster than SpConv's implicit GEMM for high-channel decoder blocks (block4, block5, block6 with 128-384 channels).

### 2. Fused Grouped TrAB Gather (grad_weight backward)

**Kernel**: `CuteGemmGroupedTrABKernel` in `cute_gemm_grouped_kernel.h`

The weight gradient computation requires `D_k = A[in_map_k]^T @ B[out_map_k]` for each offset k. Previously this was a **loop of 27 separate TrAB kernel launches**, which was the primary backward bottleneck (103ms for the grad_weight loop alone at N=330K, C=192->128).

The fused grouped TrAB kernel processes all offsets in a single launch:

- Grid: `(K_dim_tiles, N_tiles, num_groups)` -- groups in the z-dimension
- Each threadblock picks its group from `blockIdx.z`, loads A and B with group-specific gather indices
- Each group writes to its own output matrix (no atomicAdd needed)
- Shares the same pipelined mainloop as the non-grouped TrAB kernel

**Impact**: 11-17x faster for weight gradient at realistic scales:

| Config (N, C_in, C_out) | Per-offset TrAB (ms) | Fused Grouped TrAB (ms) | Speedup |
| ----------------------- | -------------------- | ----------------------- | ------- |
| 69K, 128, 128           | 42.5                 | 2.6                     | 16.7x   |
| 69K, 192, 128           | 42.4                 | 2.6                     | 16.7x   |
| 41K, 384, 256           | 16.0                 | 1.4                     | 11.4x   |

This kernel flipped the fp32 backward from **13% slower** than SpConv to **12% faster**.

## Per-Layer Breakdown (fp32)

| Layer      | SpConv (ms) | WarpConvNet (ms) | Ratio     | Notes                             |
| ---------- | ----------- | ---------------- | --------- | --------------------------------- |
| conv0      | 2.68        | 2.78             | 1.04x     | K=5, C: 3->32                     |
| conv1      | 1.24        | 1.96             | 1.58x     | Stride-2 downsample               |
| block1     | 3.39        | 4.32             | 1.27x     | C: 32->32, 2 BasicBlocks          |
| conv2      | 0.97        | 1.63             | 1.68x     | Stride-2 downsample               |
| block2     | 11.44       | 13.55            | 1.18x     | C: 32->64, 2 BasicBlocks          |
| conv3      | 1.42        | 1.64             | 1.15x     | Stride-2 downsample               |
| block3     | 11.17       | 10.56            | 0.95x     | C: 64->128, 2 BasicBlocks         |
| conv4      | 1.87        | 1.16             | **0.62x** | Stride-2 downsample               |
| block4     | 8.05        | 4.63             | **0.58x** | C: 128->256, 2 BasicBlocks        |
| convtr4    | 1.46        | 1.29             | 0.88x     | Transposed conv                   |
| **block5** | **63.07**   | **31.51**        | **0.50x** | C: 384->256, 2 BasicBlocks (skip) |
| convtr5    | 2.71        | 3.00             | 1.11x     | Transposed conv                   |
| **block6** | **55.19**   | **36.28**        | **0.66x** | C: 192->128, 2 BasicBlocks (skip) |
| convtr6    | 2.11        | 3.25             | 1.54x     | Transposed conv                   |
| block7     | 19.11       | 13.76            | 0.72x     | C: 128->96, 2 BasicBlocks (skip)  |
| convtr7    | 1.98        | 3.19             | 1.61x     | Transposed conv                   |
| block8     | 11.82       | 10.78            | 0.91x     | C: 128->96, 2 BasicBlocks (skip)  |
| final      | 0.35        | 0.35             | 1.00x     | 1x1 conv                          |

WarpConvNet wins at high channel counts (block4-block8) and loses slightly at stride-2/transposed convolutions (small operations, \<3ms absolute impact).

## Algorithm Selection

The auto-tuner selects the best algorithm per convolution config:

- **AB gather-scatter (forward + dgrad)**: `mask_implicit_gemm` wins 56% forward, 74% dgrad. `cutlass_implicit_gemm` wins at large N + ch>256.
- **AtB gather-gather (wgrad)**: `cute_grouped` wins 64%. `cutlass_grouped_hybrid` wins at large N.

## Reproducing

```bash
# From the warpconvnet repo root, with both warpconvnet and spconv installed:
LD_LIBRARY_PATH="" python benchmark_per_layer.py --lib both --precision 16-mixed
LD_LIBRARY_PATH="" python benchmark_per_layer.py --lib both --precision 32-true
```

To force re-benchmarking, clear the cache:

```bash
rm ~/.cache/warpconvnet/benchmark_cache_generic.msgpack
```
