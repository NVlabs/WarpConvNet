# Geometry Tutorial

See also: the concise type catalog in [Geometry Types](geometry_types.md).

This tutorial demonstrates how to create basic geometry types used by WarpConvNet.

## Creating `Points`

```python
import torch
from warpconvnet.geometry.types.points import Points

# coordinates and features for two batches.
N1, N2 = 1000, 500  # batch size 2, each batch has N1, N2 points
coords = [torch.rand(N1, 3), torch.rand(N2, 3)]
features = [torch.rand(N1, 7), torch.rand(N2, 7)]

points = Points(coords, features)
print(points.batch_size)
```

You can also build `Points` from concatenated tensors and `offsets`:

```python
# same N1, N2 as above
coords_cat = torch.cat([coords[0], coords[1]], dim=0)            # (N1+N2, 3)
feats_cat = torch.cat([features[0], features[1]], dim=0)         # (N1+N2, 7)
offsets = torch.tensor([0, N1, N1 + N2], dtype=torch.int32)

points_cat = Points(coords_cat, feats_cat, offsets)
```

## Creating `Voxels`

```python
from warpconvnet.geometry.types.voxels import Voxels

voxel_size = 0.01
N1, N2, C = 1000, 500, 32  # batch size 2, each batch has N1, N2 voxels, C channels
voxel_coords = [
    (torch.rand(N1, 3) / voxel_size).int(),
    (torch.rand(N2, 3) / voxel_size).int(),
]
voxel_feats = [torch.rand(N1, C), torch.rand(N2, C)]

voxels = Voxels(voxel_coords, voxel_feats)
print(voxels.batch_size)
```

Or, using concatenation plus `offsets` (integer coordinates expected):

```python
coords_cat = torch.cat(voxel_coords, dim=0)                      # (N1+N2, 3) int
feats_cat = torch.cat(voxel_feats, dim=0)                        # (N1+N2, C)
offsets = torch.tensor([0, N1, N1 + N2], dtype=torch.int32)

voxels_cat = Voxels(coords_cat, feats_cat, offsets)
```

## Conversions

Conversions between geometry types live in
`warpconvnet.geometry.types.conversion` and follow a `<src>_to_<dst>` naming
convention, each returning a geometry type.

Downsample `Points` to `Voxels` (also available as the `points.to_voxels(...)`
method):

```python
from warpconvnet.geometry.types.conversion.to_voxels import points_to_voxels
voxels = points_to_voxels(points, voxel_size=0.02, reduction="mean")
```

Rasterize `Points` or `Voxels` onto a dense `Grid`. `grid_shape` is `(H, W, D)`;
point features are aggregated into cells by a neighbor search (`radius`, `knn`,
or `voxel`):

```python
from warpconvnet.geometry.types.conversion.to_grid import (
    points_to_grid,
    voxels_to_grid,
)

grid = points_to_grid(points, grid_shape=(64, 64, 64), search_type="radius")
grid = voxels_to_grid(voxels, grid_shape=(64, 64, 64))
```

Build a factorized multi-plane `FactorGrid` (one grid per memory format):

```python
from warpconvnet.geometry.types.conversion.to_factor_grid import points_to_factor_grid

factor_grid = points_to_factor_grid(
    points,
    grid_shapes=[(64, 64, 1), (64, 1, 64), (1, 64, 64)],
    memory_formats=["b_zc_x_y", "b_xc_y_z", "b_yc_x_z"],
)
```

> Padded-batch helpers used by attention (`GeometryToPaddedBatch`,
> `PaddedBatchToGeometry`) are **not** type conversions — they produce dense
> padded tensors, not a geometry. See the attention modules in
> `warpconvnet.nn.modules.attention`.
