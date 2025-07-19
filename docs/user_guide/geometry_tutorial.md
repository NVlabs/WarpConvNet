# Geometry Tutorial

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

`Points` can be downsampled into `Voxels` and voxel grids can be converted back to dense tensors:

```python
downsampled = points.voxel_downsample(voxel_size)
dense = voxels.to_dense(channel_dim=1)
restored = Voxels.from_dense(dense, dense_tensor_channel_dim=1)
```
