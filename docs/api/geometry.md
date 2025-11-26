# Geometry

::: warpconvnet.geometry

## Geometry containers

Every geometry type in WarpConvNet wraps a `Coords` instance and a `Features`
instance that share the same ragged-batch metadata.

- `warpconvnet.geometry.base.coords.Coords` stores concatenated coordinates plus an
  `offsets` vector marking where each example begins.
- `warpconvnet.geometry.base.features.Features` (and the `CatFeatures` and `PadFeatures`
  specializations) stores feature tensors that obey the same offsets so coordinates and
  features always stay aligned.
- `warpconvnet.geometry.base.geometry.Geometry` wires the pair together, validates their
  shapes, and exposes device/dtype utilities with AMP-aware accessors.

This shared contract lets subclasses freely switch between point clouds, voxels, or grids
without duplicating batching logic. See [Batched coordinate layout](./geometry_batched.md)
for a deeper explanation of how concatenated tensors and offsets interact.

## Types

WarpConvNet ships several geometry containers that unify coordinate systems
with their associated features. Use these types as the canonical interfaces for
points, voxels, dense grids, and FIGConvNet factor grids.

### Points

Flexible point-cloud geometry supporting ragged batches, feature paddings, and
neighbor search utilities for sparse convolution modules.

::: warpconvnet.geometry.types.points.Points

### Voxels

Sparse voxel geometry that accepts integer coordinates with tensor strides and
offers helpers to move between dense tensors and CSR-style batched features.

::: warpconvnet.geometry.types.voxels.Voxels

### Grid

Regular dense grid representation that keeps `GridCoords` and `GridFeatures`
in sync, providing utilities for shape validation, format conversions, and
batch-aware initialization.

::: warpconvnet.geometry.types.grid.Grid

### Factor grid

Container that bundles multiple `Grid` instances with distinct factorized
memory formats so that FIGConvNet layers can operate on complementary spatial
perspectives.

::: warpconvnet.geometry.types.factor_grid.FactorGrid
