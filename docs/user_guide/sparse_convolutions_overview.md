# Spatially Sparse Convolutions — Overview

**Created**: 2026-05-03 12:45:00 PST
**Edited**: 2026-05-03 12:45:00 PST

Spatially sparse convolution material is split across three pages.
This page is a map — pick the one that matches what you want to do.

## Which page should I read?

| If you want to…                                                                                                                 | Read                                              |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Understand what "spatially sparse" means; see the math; watch animations of dense / sparse / stride=1 / generalized convolution | [Concepts](sparse_convolutions.md)                |
| Pick a layer (`SparseConv3d`, group conv, depthwise, generative) and write code                                                 | [Variants & API](sparse_convolutions_variants.md) |
| Understand the GEMM-level implementation, kernel maps, the AB / ABt / AtB ops, or pick a backend                                | [Internals](sparse_convolutions_internals.md)     |

## At a glance

```text
                            ┌──────────────────────────┐
                            │  Concepts                │   math definition
new reader  ───────────────▶│  sparse_convolutions.md  │   + 4 animations
                            └──────────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
   ┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
   │ Variants & API                      │   │ Internals                           │
   │ sparse_convolutions_variants.md     │   │ sparse_convolutions_internals.md    │
   │                                     │   │                                     │
   │ standard / group / depthwise /      │   │ kernel map, AB / ABt / AtB GEMMs,   │
   │ generative; usage examples          │   │ algorithm taxonomy, source files    │
   └─────────────────────────────────────┘   └─────────────────────────────────────┘
       (API user)                                 (contributor / perf engineer)
```

## What WarpConvNet implements

WarpConvNet implements **spatially sparse convolutions** on integer-grid
voxel coordinates: convolutions where only the *occupied* coordinates of
a sparse 3D tensor carry features, and work scales with occupied
neighbor pairs instead of dense grid volume.

The generalized form (one equation, four pages of consequences):

$$
\mathbf{y}_{\mathbf{u}} \;=\; \sum_{\mathbf{i} \in \mathcal{N}(\mathbf{u},\,\mathcal{K},\,\mathcal{C}^{\text{in}})} \mathbf{W}_{\mathbf{i}}\, \mathbf{x}_{\mathbf{u} + \mathbf{i}}
\qquad \mathbf{u} \in \mathcal{C}^{\text{out}}.
$$

For full definitions of $\mathcal{C}^{\text{in}}$, $\mathcal{C}^{\text{out}}$,
$\mathcal{K}$, $\mathcal{N}$ and the visual intuition for each regime,
start with [Concepts](sparse_convolutions.md).

## Related pages

- [Auto-Tuning](autotune.md) — how WarpConvNet picks a backend per shape.
- [Packed Hash Table](packed_hash_table.md) — coordinate index that
  backs the kernel-map build.
- [Bilateral & Permutohedral Filters](bilateral_permutohedral_filters.md)
  — sparse convolution lifted to high-dimensional lattices.
- [Accumulator Precision](accumulator_precision.md) — fp32 vs fp16 in the
  mask GEMM.
