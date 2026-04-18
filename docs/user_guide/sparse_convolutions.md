# Spatially Sparse Convolutions

**Created**: 2026-04-18 17:02:44
**Edited**: 2026-04-18 17:02:44

WarpConvNet implements **spatially sparse convolutions** on voxel grids. This
page defines what "spatially sparse" means, gives the mathematical
formulation, describes convolution variants (standard, group, depthwise),
and enumerates the three math kernels that make up a training step. The
per-shape algorithm-selection system is described on its own page
([Auto-Tuning](./autotune.md)).

## Spatial, feature, and weight sparsity

"Sparsity" in neural networks is an overloaded term. Three distinct flavors
show up in 3D work and they have completely different implications:

| Kind                 | What is sparse                                                             | Typical source                                                   | How WarpConvNet handles it                                                         |
| -------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Spatial sparsity** | Most grid coordinates are empty; only occupied coordinates carry features. | Native to 3D point clouds, LiDAR, voxel grids, occupancy fields. | Primary target. All convolutions on `Voxels` operate only on occupied coordinates. |
| **Feature sparsity** | Individual feature-channel values are zero (e.g. post-ReLU).               | Activation sparsity, gated MoE, quantization.                    | Orthogonal; not exploited by the conv kernels here.                                |
| **Weight sparsity**  | Pruned kernel weights are structurally zero.                               | Pruning for model compression.                                   | Orthogonal; compatible but not exploited.                                          |

The distinction between *spatial* and *weight* sparsity (and the motivation
to study them jointly) is the subject of [1]. This page is about **spatial
sparsity** — only the first row of that table.

## Mathematical formulation

Let $\mathcal{C}^{\text{in}}, \mathcal{C}^{\text{out}} \subset \mathbb{Z}^D$
be the input and output coordinate sets on a $D$-dimensional integer grid
(typically $D = 2$ or $D = 3$). Let $\mathcal{K} \subset \mathbb{Z}^D$ be
the set of kernel offsets (e.g. the 27 offsets of a $3 \times 3 \times 3$
kernel in $D = 3$). The **generalized sparse convolution** [2] is:

$$
\mathbf{y}_{\mathbf{u}} = \sum_{\mathbf{i} \in \mathcal{N}(\mathbf{u}, \mathcal{K}, \mathcal{C}^{\text{in}})} \mathbf{W}_{\mathbf{i}} \, \mathbf{x}_{\mathbf{u} + \mathbf{i}}
\qquad \text{for } \mathbf{u} \in \mathcal{C}^{\text{out}}
$$

with the active-offset set

$$
\mathcal{N}(\mathbf{u}, \mathcal{K}, \mathcal{C}^{\text{in}}) = \{\, \mathbf{i} \in \mathcal{K} : \mathbf{u} + \mathbf{i} \in \mathcal{C}^{\text{in}} \,\}.
$$

Here $\mathbf{x}_{\mathbf{u}} \in \mathbb{R}^{C_{\text{in}}}$ and
$\mathbf{y}_{\mathbf{u}} \in \mathbb{R}^{C_{\text{out}}}$ are per-coordinate
feature vectors, and $\mathbf{W}_{\mathbf{i}} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$
is the weight matrix associated with kernel offset $\mathbf{i}$.

The cost is proportional to $\sum_{\mathbf{u} \in \mathcal{C}^{\text{out}}} |\mathcal{N}(\mathbf{u}, \mathcal{K}, \mathcal{C}^{\text{in}})|$
rather than $|\mathcal{C}^{\text{out}}| \cdot |\mathcal{K}|$, which is the
entire point of spatial sparsity: work scales with occupied neighbor pairs,
not with the dense grid volume.

### How $\mathcal{C}^{\text{out}}$ is chosen

The output coordinate set is controlled by three flags on
`SparseConv3d` / `spatially_sparse_conv`: `stride`, `transposed`, and
`generative`.

| Regime                    | Flags                         | $\mathcal{C}^{\text{out}}$                                                                                                                        | Notes                                                                           |
| ------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **stride=1**              | `stride=1`                    | $\mathcal{C}^{\text{out}} = \mathcal{C}^{\text{in}}$                                                                                              | Preserves coordinate set. Pioneered as *submanifold sparse convolution* in [3]. |
| **Downsampling**          | `stride>1`                    | Downsampled coordinates (one per stride cell).                                                                                                    | Standard pooling/strided convolution.                                           |
| **Transposed**            | `transposed=True`, `stride>1` | Upsampled coordinates by factor `stride` over $\mathcal{C}^{\text{in}}$.                                                                          | Deconvolution / learned upsampling.                                             |
| **Generative (stride=1)** | `generative=True`, `stride=1` | $\mathcal{C}^{\text{in}}$ **expanded by the kernel support** — every grid point reachable from any input coordinate via a kernel offset is added. | Densification: the coordinate set *grows*.                                      |
| **Generative (stride>1)** | `generative=True`, `stride>1` | Stride (or upsample, if `transposed=True`) the input coordinates first, then expand by the kernel support.                                        | Used in generative decoders / diffusion models that produce new occupied cells. |

The stride-1 case — convolving over exactly the occupied coordinates
without introducing new output sites — was introduced by Graham and
van der Maaten [3] as *submanifold sparse convolution*. WarpConvNet uses
the stride-based terminology internally (see `SparseConv3d(stride=1)`) but
the idea is the one from [3]. The generalized form with arbitrary strides
and coordinate sets follows Choy, Gwak, Savarese [2].

**Generative convolution** (`generative=True`) is the only regime that
**adds** new coordinates rather than reducing or preserving them. For
each input coordinate $\mathbf{u}$, every grid point $\mathbf{u} + \mathbf{i}$
for $\mathbf{i} \in \mathcal{K}$ is added to $\mathcal{C}^{\text{out}}$
(deduplicated). This is the standard tool for occupancy densification
and generative sparse decoders — outputs can only appear within the
kernel reach of existing inputs, so successive generative layers
progressively fill the occupied region.

```python
# Generative, stride=1: expand the occupied set by the kernel support.
# Output has more coordinates than input.
conv_gen = SparseConv3d(64, 128, kernel_size=3, stride=1, generative=True)

# Generative, transposed, stride=2: upsample input by 2x, then expand.
# Used in the decoder side of a sparse U-Net for generative tasks.
up_gen = SparseConv3d(64, 128, kernel_size=3, stride=2,
                      transposed=True, generative=True)
```

## Convolution variants

### Standard convolution

The default. `SparseConv3d(C_in, C_out, kernel_size=K, stride=s)` with
weight tensor $\mathbf{W} \in \mathbb{R}^{K \times C_{\text{in}} \times C_{\text{out}}}$
implements the equation above exactly.

```python
from warpconvnet.nn.modules.sparse_conv import SparseConv3d

# stride=1, preserves C_out = C_in coordinate set (the stride-1 regime above)
conv = SparseConv3d(64, 128, kernel_size=3, stride=1)

# stride=2, downsampling convolution
down = SparseConv3d(64, 128, kernel_size=3, stride=2)
```

### Group convolution

Partitions input and output channels into $G$ groups, each processed
independently. The weight tensor is reshaped to $\mathbf{W} \in \mathbb{R}^{K \times G \times C_{\text{in}}/G \times C_{\text{out}}/G}$
and the sum in the equation above is restricted per-group: group $g$ of
the output attends only to group $g$ of the input. Compute and parameter
count both drop by $G$; spatial connectivity (the coordinate set and
kernel support) is unchanged.

```python
# G=4: weight is [27, 4, 16, 32]
conv_g4 = SparseConv3d(64, 128, kernel_size=3, groups=4)
```

**Constraints:**

- `C_in` and `C_out` must be divisible by `groups`.
- Forward / dgrad pad per-group channel counts up to a multiple of
  `kVec=8` internally. Per-group C >= 8 for vectorized fp16 loads.
- Wgrad requires per-group channel counts to be **exactly** divisible by
  `kVec=8` (no padding fallback). Pick per-group C as a multiple of 8.
- Kernel-map state (pair_table, pair_mask) is spatial and is built once
  per layer call, then reused across groups.

**Tested matrix** (RTX 6000 Ada, SM 8.9, AMP fp16, WideGroupUNet smoke):
forward + backward correctness validated end-to-end for

- `kernel_size` ∈ {3, 5, 7}
- `groups` ∈ {1, 2, 4}

spanning all 9 combinations. `kernel_size=7` exercises the $K=343$ /
`mask_words=12` dispatch path end-to-end.

### Depthwise convolution

Group convolution with $G = C_{\text{in}} = C_{\text{out}}$. Weight tensor
is $\mathbf{W} \in \mathbb{R}^{K \times C \times 1 \times 1}$; each channel
is convolved independently.

```python
# Depthwise: each of 64 channels convolved independently
conv_dw = SparseConv3d(64, 64, kernel_size=3, groups=64)
```

A dedicated depthwise path lives under
`warpconvnet.nn.functional.spatially_sparse_depthwise_conv` with its own
algorithm modes (`WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE` /
`WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE`).

## Three math kernels per layer

A complete training step through one spatially sparse convolution layer
requires **three mathematically distinct GEMM operations**:

| Pass                                            | Math                                           | WarpConvNet GEMM class | Cache namespace     |
| ----------------------------------------------- | ---------------------------------------------- | ---------------------- | ------------------- |
| Forward                                         | $\mathbf{Y} = \mathbf{A},\mathbf{W}$           | **AB** gather-scatter  | `AB_gather_scatter` |
| Backward $\partial/\partial \mathbf{X}$ (dgrad) | $\mathbf{dX} = \mathbf{dY},\mathbf{W}^{!\top}$ | **AB** gather-scatter  | `AB_gather_scatter` |
| Backward $\partial/\partial \mathbf{W}$ (wgrad) | $\mathbf{dW} = \mathbf{A}^{!\top},\mathbf{dY}$ | **AtB** gather-gather  | `AtB_gather_gather` |

- **AB gather-scatter**: gather rows of $\mathbf{A}$ by the input-side
  kernel map, multiply by the dense per-offset weight matrix $\mathbf{W}$,
  scatter-add into the output buffer. Forward and dgrad share the same
  kernel class with the roles of source/target swapped (warpconvnet builds
  a reverse pair_table so the forward-path mask kernel can run dgrad with
  no atomics). Each output row is written by a single thread block → no
  atomicAdd needed.
- **AtB gather-gather**: gather rows of both operands by the input and
  output kernel maps, compute a reduction of outer products into the
  per-offset weight-gradient buffer. No scatter step; one dense output
  tile per kernel offset. Shape and reuse pattern are different enough
  from AB that the optimal algorithm differs per shape.

This split matters because **the three ops need three independent
autotune searches**: there is no single algorithm that wins for all of
forward, dgrad, and wgrad simultaneously, and the AtB gather-gather
pattern has different arithmetic intensity from AB gather-scatter even at
the same $(N, C_{\text{in}}, C_{\text{out}}, K)$ shape.

## Algorithms (overview)

Each math kernel class above has multiple CUDA backend implementations.
A short taxonomy:

### AB backends (forward + dgrad)

| Backend                  | Implementation                                                                          | Strengths                                                               |
| ------------------------ | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `explicit_gemm`          | Gather to dense buffer, `torch.mm`, scatter-add.                                        | Reliable fallback. No alignment requirements.                           |
| `implicit_gemm`          | Fused CUDA kernel; SIMT gather + GEMM + scatter-add in one launch.                      | Small channels (C ≤ 64).                                                |
| `cutlass_implicit_gemm`  | CUTLASS fused gather-GEMM-scatter.                                                      | Tensor-core throughput at large channels. Auto-pads unaligned channels. |
| `cute_implicit_gemm`     | CuTe 3.x fused kernel with `cp.async` vectorized A-loads.                               | Competitive at small-medium channels.                                   |
| `cute_grouped`           | CuTe 3.x grouped GEMM — all offsets in one launch via binary-search dispatch.           | Small-N medium-channel; winner for C > 256 at small N.                  |
| `cutlass_grouped_hybrid` | CUTLASS for large offsets + `torch.bmm` for grouped small offsets.                      | Strong at large N + medium-large channels.                              |
| `production`             | Mask-based fused kernel — iterates all $K$ offsets per output row via bitmask skipping. | Dominant AB winner on most shapes. No atomicAdd. CuTe tensor core MMA.  |

### AtB backends (wgrad)

Same name list as above, but implementing the
$\mathbf{A}^{!\top},\mathbf{dY}$ gather-gather pattern. `cute_grouped`
wins the majority of wgrad shapes empirically.

### How `production` works

Unlike per-offset and grouped backends that launch separate work per
offset, the `production` kernel processes **all K offsets in a single
launch**. For each output row:

1. Look up which offsets are active via a bitmask (`pair_mask`).
2. For each active offset, gather from input and accumulate with the
   offset's weight.
3. Write output directly — no atomicAdd needed since each output row is
   exclusive.

For dgrad, a reverse pair_table is constructed so the same forward
kernel can be reused with swapped dimensions, avoiding atomicAdd entirely
(~2x speedup over the old atomicAdd dgrad).

### Picking an algorithm

Empirically, no single backend wins across all shapes. Picking by hand is
infeasible. **WarpConvNet auto-tunes per problem shape** — see the
[Auto-Tuning](./autotune.md) page for candidate selection, modes,
environment variables, and how to specify algorithms explicitly.

## Usage

### Basic

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

### Functional

```python
from warpconvnet.nn.functional import spatially_sparse_conv

output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
)
```

### Depthwise

```python
from warpconvnet.nn.functional import spatially_sparse_depthwise_conv

output = spatially_sparse_depthwise_conv(
    input_features,
    depthwise_weight,
    kernel_map,
    num_out_coords,
)
```

For algorithm control (`fwd_algo`, `dgrad_algo`, `wgrad_algo`), env
variables, and the strict algorithm-name filter, see
[Auto-Tuning](./autotune.md).

## See also

- [Auto-Tuning](./autotune.md) — per-shape algorithm selection, caching,
  env variables.
- [Accumulator Precision](./accumulator_precision.md) — fp32 vs fp16
  accumulator in the mask GEMM.
- [Adaptive GEMM Grouping](./adaptive_gemm_grouping.md) — how small
  kernel offsets are batched into grouped GEMMs.
- [Inspect Benchmark Cache](./inspect_benchmark_cache.md) — dump cached
  autotune results.
- [Pre-Populate Benchmark Cache](./populate_benchmark_cache.md) — fill
  the cache ahead of production workloads.

## Source files

| File                                                              | Contents                                                           |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| `warpconvnet/nn/functional/sparse_conv/detail/unified.py`         | Top-level dispatch, config construction.                           |
| `warpconvnet/nn/functional/sparse_conv/detail/algo_params.py`     | Adaptive candidate selection, algorithm enums, F16Acc pool gating. |
| `warpconvnet/nn/functional/sparse_conv/detail/autotune.py`        | Benchmark runners, cache init/merge.                               |
| `warpconvnet/nn/functional/sparse_conv/detail/dispatch.py`        | Algorithm execution dispatch.                                      |
| `warpconvnet/nn/functional/sparse_conv/detail/mask_gemm.py`       | Mask-based fused kernel dispatch, reverse pair_table.              |
| `warpconvnet/nn/functional/sparse_conv/detail/cute_grouped.py`    | CuTe grouped GEMM (AB + AtB).                                      |
| `warpconvnet/nn/functional/sparse_conv/detail/cutlass.py`         | CUTLASS per-offset gather-scatter.                                 |
| `warpconvnet/nn/functional/sparse_conv/detail/explicit.py`        | Explicit GEMM via cuBLAS.                                          |
| `warpconvnet/nn/functional/sparse_conv/detail/implicit_direct.py` | SIMT implicit GEMM.                                                |
| `warpconvnet/utils/benchmark_cache.py`                            | Generic benchmark cache with persistence.                          |
| `warpconvnet/constants.py`                                        | Environment variable parsing.                                      |

## References

1. Lee, J., Choy, C., Park, J. *Putting 3D Spatially Sparse Networks on a Diet.* arXiv:2112.01316, 2021. [[arxiv]](https://arxiv.org/abs/2112.01316)
2. Choy, C., Gwak, J., Savarese, S. *4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks.* CVPR 2019. arXiv:1904.08755. [[arxiv]](https://arxiv.org/abs/1904.08755)
3. Graham, B., van der Maaten, L. *Submanifold Sparse Convolutional Networks.* arXiv:1706.01307, 2017. [[arxiv]](https://arxiv.org/abs/1706.01307)
   See also Graham, Engelcke, van der Maaten. *3D Semantic Segmentation with Submanifold Sparse Convolutional Networks.* CVPR 2018. arXiv:1711.10275. [[arxiv]](https://arxiv.org/abs/1711.10275)
