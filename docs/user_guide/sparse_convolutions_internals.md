# Sparse Convolution Internals

**Created**: 2026-05-03 13:00:00 PST
**Edited**: 2026-05-03 13:00:00 PST

How WarpConvNet executes the sparse convolution
$\mathbf{y}_{\mathbf{u}} = \sum_{\mathbf{i}} \mathbf{W}_{\mathbf{i}}\, \mathbf{x}_{\mathbf{u}+\mathbf{i}}$
on the GPU. For the math definition, see
[Concepts](sparse_convolutions.md). For user-facing layer constructors,
see [Variants & API](sparse_convolutions_variants.md).

## Kernel map

The sparse convolution is data-driven: at runtime, for each input/output
coordinate pair we must know *which* offset $\mathbf{i} \in \mathcal{K}$
connects them. WarpConvNet materializes this as the **kernel map** —
for every $\mathbf{i} \in \mathcal{K}$, a list of (input row, output row)
pairs:

$$
\mathrm{KMap}[\mathbf{i}] \;=\; \bigl\{\, (r^{\text{in}}, r^{\text{out}}) : r^{\text{in}} \in \mathrm{Idx}(\mathcal{C}^{\text{in}}),\ r^{\text{out}} \in \mathrm{Idx}(\mathcal{C}^{\text{out}}),\ \mathrm{coord}(r^{\text{out}}) + \mathbf{i} = \mathrm{coord}(r^{\text{in}}) \,\bigr\}.
$$

Two encodings:

- **`pair_table`**: dense per-offset list of (input_row, output_row).
- **`pair_mask`**: per-output-row bitmask over $\mathcal{K}$ flagging
  which offsets are active for that row. Used by the `production`
  backend to avoid per-offset launches.

Built once per `(C^in, C^out, K)` shape via the
[`PackedHashTable`](packed_hash_table.md) coordinate index, then reused
across forward, dgrad, and wgrad.

## Three math kernels per layer

A complete training step through one sparse convolution layer requires
**three mathematically distinct GEMM operations**:

| Pass    | Math                                             | GEMM class             | Cache namespace      |
| ------- | ------------------------------------------------ | ---------------------- | -------------------- |
| Forward | $\mathbf{Y} = \mathbf{A}\,\mathbf{W}$            | **AB** gather-scatter  | `AB_gather_scatter`  |
| Dgrad   | $\mathbf{dX} = \mathbf{dY}\,\mathbf{W}^{\!\top}$ | **ABt** gather-scatter | `ABt_gather_scatter` |
| Wgrad   | $\mathbf{dW} = \mathbf{A}^{\!\top}\,\mathbf{dY}$ | **AtB** gather-gather  | `AtB_gather_gather`  |

Here $\mathbf{A}$ is the input feature matrix
($N_{\text{in}} \times C_{\text{in}}$), $\mathbf{Y}$ is the output
feature matrix ($N_{\text{out}} \times C_{\text{out}}$), and
$\mathbf{W}_{\mathbf{i}} \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$
is the per-offset weight (one slice per $\mathbf{i} \in \mathcal{K}$).

### AB gather-scatter (forward)

For each offset $\mathbf{i}$ in turn:

$$
\forall (r^{\text{in}}, r^{\text{out}}) \in \mathrm{KMap}[\mathbf{i}]: \quad
\mathbf{Y}[r^{\text{out}}, :]\ \mathrel{+}=\ \mathbf{A}[r^{\text{in}}, :]\,\mathbf{W}_{\mathbf{i}}.
$$

Gather rows of $\mathbf{A}$ by the input-side kernel map, multiply by
the dense per-offset weight matrix $\mathbf{W}_{\mathbf{i}}$, scatter-add
into the output buffer. Each output row is written by a single thread
block → no `atomicAdd` needed.

### ABt gather-scatter (dgrad)

Same gather-scatter shape as forward, but with
$\mathbf{B} = \mathbf{W}_{\mathbf{i}}^{\!\top}$:

$$
\forall (r^{\text{in}}, r^{\text{out}}) \in \mathrm{KMap}[\mathbf{i}]: \quad
\mathbf{dX}[r^{\text{in}}, :]\ \mathrel{+}=\ \mathbf{dY}[r^{\text{out}}, :]\,\mathbf{W}_{\mathbf{i}}^{\!\top}.
$$

WarpConvNet builds a **reverse pair_table** so the forward-path mask
kernel can run dgrad with no atomics and no explicit transpose tensor at
runtime — the transpose is folded into the iteration order. Distinct
from AB because the optimal tile shape and split-K depend on the
$C_{\text{in}} \leftrightarrow C_{\text{out}}$ swap.

### AtB gather-gather (wgrad)

For each offset $\mathbf{i}$:

$$
\mathbf{dW}_{\mathbf{i}} \;=\; \sum_{(r^{\text{in}}, r^{\text{out}}) \in \mathrm{KMap}[\mathbf{i}]} \mathbf{A}[r^{\text{in}}, :]^{\!\top}\,\mathbf{dY}[r^{\text{out}}, :].
$$

Gather rows of *both* operands by the input and output kernel maps,
reduce outer products into the per-offset weight-gradient buffer. No
scatter step; one dense output tile per kernel offset. Shape and reuse
pattern are different enough from AB/ABt that the optimal algorithm
differs per shape.

### Why the split matters

Each of the three ops is **auto-tuned independently** in its own cache
namespace (`AB_gather_scatter`, `ABt_gather_scatter`,
`AtB_gather_gather`). There is no single algorithm that wins for all of
forward, dgrad, and wgrad at the same
$(N, C_{\text{in}}, C_{\text{out}}, |\mathcal{K}|)$ shape, and the
gather-gather pattern has different arithmetic intensity from
gather-scatter even holding the shape fixed.

## Algorithm taxonomy

Each math kernel class above has multiple CUDA backend implementations.

### AB / ABt backends (forward + dgrad)

The forward (AB) and dgrad (ABt) passes share this backend list — they
have the same gather-scatter structure, only $\mathbf{B}$ differs
($\mathbf{W}$ for fwd, $\mathbf{W}^{\!\top}$ for dgrad). Each op picks
its own winner per shape.

| Backend                  | Implementation                                                                        | Strengths                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `explicit_gemm`          | Gather to dense buffer, `torch.mm`, scatter-add.                                      | Reliable fallback. No alignment requirements.                              |
| `implicit_gemm`          | Fused CUDA kernel; SIMT gather + GEMM + scatter-add in one launch.                    | Small channels ($C \le 64$).                                               |
| `cutlass_implicit_gemm`  | CUTLASS fused gather-GEMM-scatter.                                                    | Tensor-core throughput at large channels. Auto-pads unaligned channels.    |
| `cute_implicit_gemm`     | CuTe 3.x fused kernel with `cp.async` vectorized A-loads.                             | Competitive at small-medium channels.                                      |
| `cute_grouped`           | CuTe 3.x grouped GEMM — all offsets in one launch via binary-search dispatch.         | Small-N medium-channel; winner for $C > 256$ at small $N$.                 |
| `cutlass_grouped_hybrid` | CUTLASS for large offsets + `torch.bmm` for grouped small offsets.                    | Strong at large $N$ + medium-large channels.                               |
| `production`             | Mask-based fused kernel — iterates all $K^D$ offsets per output row via bitmask skip. | Dominant AB/ABt winner on most shapes. No atomicAdd. CuTe tensor-core MMA. |

### AtB backends (wgrad)

Same name list as above, but implementing the
$\mathbf{A}^{\!\top}\,\mathbf{dY}$ gather-gather pattern. `cute_grouped`
wins the majority of wgrad shapes empirically.

### How `production` works

Unlike per-offset and grouped backends that launch separate work per
offset, the `production` kernel processes **all $|\mathcal{K}|$ offsets
in a single launch**. For each output row $r^{\text{out}}$:

1. Load `pair_mask[r_out]` — a bitmask of length $|\mathcal{K}|$ flagging
   which offsets contribute to this output row.
2. For each set bit $\mathbf{i}$, gather $\mathbf{A}[r^{\text{in}}_{\mathbf{i}}, :]$
   from the input and accumulate
   $\mathbf{A}[r^{\text{in}}_{\mathbf{i}}, :]\,\mathbf{W}_{\mathbf{i}}$
   into the output tile.
3. Write output tile directly — no `atomicAdd` needed since each output
   row is exclusive to one thread block.

For dgrad, a reverse `pair_table` is constructed so the same forward
kernel runs with swapped dimensions, avoiding atomics entirely (~2× over
the old `atomicAdd` dgrad).

### Picking an algorithm

Empirically, no single backend wins across all shapes. Picking by hand is
infeasible. **WarpConvNet auto-tunes per problem shape** — see the
[Auto-Tuning](autotune.md) page for candidate selection, modes, environment
variables, and how to specify algorithms explicitly.

## See also

- [Concepts](sparse_convolutions.md) — math definitions, visual regimes.
- [Variants & API](sparse_convolutions_variants.md) — `SparseConv3d`,
  group / depthwise variants.
- [Auto-Tuning](autotune.md) — per-shape algorithm selection, caching,
  env variables.
- [Accumulator Precision](accumulator_precision.md) — fp32 vs fp16
  accumulator in the mask GEMM.
- [Adaptive GEMM Grouping](adaptive_gemm_grouping.md) — how small kernel
  offsets are batched into grouped GEMMs.
- [Inspect Benchmark Cache](inspect_benchmark_cache.md) — dump cached
  autotune results.
- [Pre-Populate Benchmark Cache](populate_benchmark_cache.md) — fill the
  cache ahead of production workloads.
- [Packed Hash Table](packed_hash_table.md) — coordinate index that
  backs the kernel-map build.

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
