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

Downsample `Points` to `Voxels` and convert grids to/from dense:

```python
from warpconvnet.geometry.types.conversion.to_voxels import points_to_voxels
voxels = points_to_voxels(points, voxel_size=0.02, reduction="mean")
```
