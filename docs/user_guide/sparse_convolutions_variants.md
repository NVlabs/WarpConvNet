# Sparse Convolution Variants & API

**Created**: 2026-05-03 13:00:00 PST
**Edited**: 2026-05-03 13:00:00 PST

User-facing layer constructors and functional entry points for spatially
sparse convolutions. For the math definitions and visual intuition, start
with [Concepts](sparse_convolutions.md). For the GEMM-level
implementation, see [Internals](sparse_convolutions_internals.md).

All variants apply the same generalized sparse convolution

$$
\mathbf{y}_{\mathbf{u}} \;=\; \sum_{\mathbf{i} \in \mathcal{N}(\mathbf{u},\,\mathcal{K},\,\mathcal{C}^{\text{in}})} \mathbf{W}_{\mathbf{i}}\, \mathbf{x}_{\mathbf{u} + \mathbf{i}}
\qquad \mathbf{u} \in \mathcal{C}^{\text{out}}
$$

over a chosen $(\mathcal{C}^{\text{in}}, \mathcal{C}^{\text{out}}, \mathcal{K})$
triple. The variants differ in how the **per-offset weight tensor**
$\{\mathbf{W}_{\mathbf{i}}\}_{\mathbf{i} \in \mathcal{K}}$ is shaped /
restricted, not in how $\mathcal{C}^{\text{out}}$ is chosen.

## Standard convolution

Default. Per-offset weights are full $C_{\text{out}} \times C_{\text{in}}$ matrices:

$$
\mathbf{W} \in \mathbb{R}^{|\mathcal{K}| \,\times\, C_{\text{in}} \,\times\, C_{\text{out}}}.
$$

```python
from warpconvnet.nn.modules.sparse_conv import SparseConv3d

# stride=1, preserves C_out = C_in coordinate set (the stride-1 regime)
conv = SparseConv3d(64, 128, kernel_size=3, stride=1)

# stride=2, downsampling convolution
down = SparseConv3d(64, 128, kernel_size=3, stride=2)
```

## Group convolution

Partition input and output channels into $G$ groups, each processed
independently. The weight tensor restructures as

$$
\mathbf{W} \in \mathbb{R}^{|\mathcal{K}| \,\times\, G \,\times\, (C_{\text{in}}/G) \,\times\, (C_{\text{out}}/G)},
$$

and the convolution sum splits along the group axis: group $g$ of the
output attends only to group $g$ of the input. Compute and parameter count
both drop by $G$; spatial connectivity ($\mathcal{C}^{\text{out}}$,
$\mathcal{K}$) is unchanged.

```python
# G=4: weight is [27, 4, 16, 32]
conv_g4 = SparseConv3d(64, 128, kernel_size=3, groups=4)
```

**Constraints:**

- $C_{\text{in}}$ and $C_{\text{out}}$ must be divisible by `groups`.
- Forward / dgrad pad per-group channel counts up to a multiple of
  `kVec=8` internally. Per-group $C \ge 8$ for vectorized fp16 loads.
- Wgrad requires per-group channel counts to be **exactly** divisible by
  `kVec=8` (no padding fallback). Pick per-group $C$ as a multiple of 8.
- Kernel-map state (`pair_table`, `pair_mask`) is spatial and is built
  once per layer call, then reused across groups.

**Tested matrix** (RTX 6000 Ada, SM 8.9, AMP fp16, WideGroupUNet smoke):
forward + backward correctness validated end-to-end for

- `kernel_size` ∈ {3, 5, 7}
- `groups` ∈ {1, 2, 4}

spanning all 9 combinations. `kernel_size=7` exercises the $K=343$ /
`mask_words=12` dispatch path end-to-end.

## Depthwise convolution

Group convolution at the limit $G = C_{\text{in}} = C_{\text{out}} = C$.
Each channel is convolved independently:

$$
\mathbf{W} \in \mathbb{R}^{|\mathcal{K}| \,\times\, C},
\qquad
y_{\mathbf{u}, c} \;=\; \sum_{\mathbf{i} \in \mathcal{N}(\mathbf{u}, \mathcal{K}, \mathcal{C}^{\text{in}})} W_{\mathbf{i}, c}\, x_{\mathbf{u}+\mathbf{i},\, c}.
$$

```python
# Depthwise: each of 64 channels convolved independently
conv_dw = SparseConv3d(64, 64, kernel_size=3, groups=64)
```

A dedicated depthwise path lives under
`warpconvnet.nn.functional.spatially_sparse_depthwise_conv` with its own
algorithm modes (`WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE` /
`WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE`).

## Generative convolution

`generative=True` makes $\mathcal{C}^{\text{out}}$ *grow* beyond
$\mathcal{C}^{\text{in}}$ by the kernel support:

$$
\mathcal{C}^{\text{out}} \;=\; \bigl\{\, \mathbf{u} + \mathbf{i} : \mathbf{u} \in \mathcal{C}^{\text{in}},\ \mathbf{i} \in \mathcal{K} \,\bigr\}.
$$

(With `stride>1` and / or `transposed=True`, $\mathcal{C}^{\text{in}}$ is
strided / upsampled first, then expanded by $\mathcal{K}$.)

This is the standard tool for occupancy densification and sparse
generative decoders — outputs can only appear within the kernel reach of
existing inputs, so successive generative layers progressively fill the
occupied region.

```python
# Generative, stride=1: expand the occupied set by the kernel support.
# Output has more coordinates than input.
conv_gen = SparseConv3d(64, 128, kernel_size=3, stride=1, generative=True)

# Generative, transposed, stride=2: upsample input by 2x, then expand.
# Used in the decoder side of a sparse U-Net for generative tasks.
up_gen = SparseConv3d(64, 128, kernel_size=3, stride=2,
                      transposed=True, generative=True)
```

## Functional entry points

The module wrappers above call into a small set of functional ops:

```python
from warpconvnet.nn.functional import spatially_sparse_conv

output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
)
```

```python
from warpconvnet.nn.functional import spatially_sparse_depthwise_conv

output = spatially_sparse_depthwise_conv(
    input_features,
    depthwise_weight,
    kernel_map,
    num_out_coords,
)
```

For algorithm control (`fwd_algo`, `dgrad_algo`, `wgrad_algo`),
environment variables, and the strict algorithm-name filter, see
[Auto-Tuning](autotune.md) and [Internals](sparse_convolutions_internals.md).

## See also

- [Concepts](sparse_convolutions.md) — math definitions, visual regimes.
- [Internals](sparse_convolutions_internals.md) — three GEMM ops per layer,
  algorithm backends.
- [Auto-Tuning](autotune.md) — per-shape algorithm selection.
