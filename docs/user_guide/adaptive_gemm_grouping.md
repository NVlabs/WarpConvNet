# Adaptive GEMM Grouping

Sparse convolution processes each kernel offset independently: for a 3x3x3 kernel, 26 non-identity offsets each produce a separate GEMM call (or CUDA kernel launch). When individual offsets have small pair counts, these launches become inefficient due to low GPU utilization and per-launch overhead.

Adaptive GEMM grouping addresses this by batching similar-sized offsets together, executing them as a single operation instead of many small ones.

## How It Works

### Offset Classification

Each kernel offset k has M_k coordinate pairs. Offsets are classified as:

- **Large** (M_k >= `saturation_m`): Already GPU-saturating. Processed individually using the standard per-offset path.
- **Small** (M_k < `saturation_m`): Grouped into buckets for batched execution.

### Bucketing Algorithm

Small offsets are sorted by pair count and greedily merged into buckets. A new offset is added to the current bucket only if the padding waste stays below a threshold:

$$
\text{redundancy} = \frac{M_{\max} \times B - \sum M_k}{\sum M_k} \leq \text{threshold}
$$

where B is the bucket size and M_max is the largest pair count in the bucket. The default threshold is 10%, meaning at most 10% of the padded computation is wasted.

### Example Distribution (500K points, 3x3x3 kernel)

| Offset Type | Count | Pair Count Range |
| ----------- | ----- | ---------------- |
| Identity    | 1     | ~500,000         |
| Face        | 6     | 5,000 - 50,000   |
| Edge        | 12    | 500 - 5,000      |
| Corner      | 8     | 50 - 2,000       |

With `saturation_m=5000`, the 6 face offsets are processed individually while the 20 edge+corner offsets are grouped into a few buckets.

## Grouped Backends

Four grouped variants are available, each applying the grouping strategy to its respective backend.

### Explicit GEMM Grouped (`EXPLICIT_GEMM_GROUPED`)

Gathers input features into a padded `[B, M_max, C_in]` buffer using vectorized indexing, executes `torch.bmm`, then scatter-adds results back to the output. No Python for-loops; the gather and scatter use pre-computed flat indices.

**Forward:**

```
features[cat_in_map] → padded buffer → torch.bmm(buffer, stacked_weights) → scatter_add to output
```

### Implicit GEMM Grouped (`IMPLICIT_GEMM_GROUPED`)

Uses a custom CUDA kernel (`implicit_gemm_grouped`) that processes all pairs from a bucket in a single launch. Each pair selects its weight matrix via a `weight_idx` array, achieving **zero padding waste**.

The kernel uses a dual-path optimization:

- **Uniform blocks** (~98% of blocks): All rows in the thread block share the same weight index. Both input features (A) and weights (B) are tiled in shared memory, matching the performance of the ungrouped kernel.
- **Boundary blocks** (~2%): Rows at weight-group transitions use different weights. A is tiled in shared memory; B is loaded per-thread from L2 cache.

```
CUDA kernel: A[in_map[i]] @ B[weight_idx[i]] → atomicAdd to C[out_map[i]]
```

### CUTLASS Hybrid Grouped (`CUTLASS_GROUPED_HYBRID`)

Combines CUTLASS fused gather-GEMM-scatter for large offsets with `torch.bmm` for grouped small offsets. Amortizes CUTLASS per-offset setup cost.

### CuTe Grouped (`CUTE_GROUPED`)

Uses the CuTe 3.x programming model for grouped GEMM with batched offset execution. Auto-tunes the MMA tile shape (`mma_tile` parameter). This is the **most frequently optimal forward backend overall** (44% win rate), dominating at channels >= 96. It processes all offsets in a single grouped GEMM call, avoiding per-offset launch overhead entirely.

```
CuTe 3.x grouped GEMM: A[in_map] @ B[weight_idx] → C[out_map]
```

## Algorithm Selection

The grouped variants are included in the unified autotuning system. When using `AUTO` mode, the benchmarking system automatically evaluates grouped variants alongside ungrouped ones and selects the fastest for your data characteristics.

You can also select grouped algorithms explicitly:

```python
from warpconvnet.nn.functional.sparse_conv import (
    SPARSE_CONV_AB_ALGO_MODE,
    SPARSE_CONV_ATB_ALGO_MODE,
)

output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    fwd_algo=SPARSE_CONV_AB_ALGO_MODE.CUTLASS_GROUPED_HYBRID,
    wgrad_algo=SPARSE_CONV_ATB_ALGO_MODE.EXPLICIT_GEMM_GROUPED,
)
```

Or via environment variables:

```bash
export WARPCONVNET_FWD_ALGO_MODE=cutlass_grouped_hybrid
export WARPCONVNET_DGRAD_ALGO_MODE=cutlass_grouped_hybrid
export WARPCONVNET_WGRAD_ALGO_MODE=explicit_gemm_grouped
```

### Grouping Parameters

The grouping behavior is controlled by `saturation_m` (passed via the autotuning parameter grid):

| `saturation_m` | Effect                                                               |
| -------------- | -------------------------------------------------------------------- |
| 1000           | More offsets grouped; beneficial when most offsets are small         |
| 5000 (default) | Balanced; groups edge/corner offsets, leaves face offsets individual |
| 20000          | Fewer offsets grouped; only smallest offsets are batched             |

The `grouping_threshold` (default 0.1) controls maximum padding waste per bucket.

## Performance Characteristics

Grouping helps most when:

- **Many small offsets** with low pair counts (high per-launch overhead relative to compute)
- **Large channel dimensions** (C >= 64), where batched GEMM is more efficient
- **CUTLASS backend**, which has higher per-offset setup cost

Grouping helps least when:

- **Dense kernel maps** where most offsets are GPU-saturating
- **Small channel dimensions** (C \<= 16), where launch overhead is already low
- **Very few offsets** (e.g., 3x1x1 kernel with only 2 non-identity offsets)

### Benchmark (RTX 6000 Ada, 3x3x3 kernel)

| Backend                 | 10K coords | 100K coords | 500K coords |
| ----------------------- | ---------- | ----------- | ----------- |
| Explicit Grouped (C=64) | 1.32x      | 1.03x       | 1.01x       |
| Implicit Grouped (C=32) | 0.65x      | 1.22x       | 0.44x       |
| CUTLASS Hybrid (C=64)   | 0.91x      | 0.67x       | **6.04x**   |

The CUTLASS hybrid achieves 6x speedup at 500K coordinates with C=64, where per-offset CUTLASS launch overhead dominates the ungrouped path.

## Source Files

| File                                                              | Contents                                                                                |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `warpconvnet/nn/functional/sparse_conv/detail/grouping.py`        | Bucketing algorithm, `GroupedKernelMap`, `prepare_grouped_kernel_map()`                 |
| `warpconvnet/nn/functional/sparse_conv/detail/explicit.py`        | `_explicit_gemm_forward_grouped()`, `_explicit_gemm_backward_grouped()`                 |
| `warpconvnet/nn/functional/sparse_conv/detail/implicit_direct.py` | `_implicit_gemm_forward_grouped()`, `_implicit_gemm_backward_grouped()`                 |
| `warpconvnet/nn/functional/sparse_conv/detail/cutlass.py`         | `_cutlass_implicit_gemm_forward_grouped()`, `_cutlass_implicit_gemm_backward_grouped()` |
| `warpconvnet/nn/functional/sparse_conv/detail/cute_grouped.py`    | `_cute_grouped_forward()`, `_cute_grouped_backward()` (CuTe 3.x)                        |
| `warpconvnet/csrc/implicit_gemm.cu`                               | `implicit_gemm_grouped` CUDA kernel                                                     |
| `warpconvnet/csrc/bindings/gemm_bindings.cpp`                     | pybind11 binding for `_C.gemm.implicit_gemm_grouped`                                    |
| `warpconvnet/nn/functional/sparse_conv/detail/unified.py`         | Autotuning integration                                                                  |
