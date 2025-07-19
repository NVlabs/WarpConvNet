# Geometry Types

WarpConvNet defines a set of geometry containers that combine batched coordinates and features. They provide utility functions for neighbor search, conversions and indexing. The tests in `tests/types` showcase typical usage while the source code under `warpconvnet/geometry/types` contains the full definitions.

## Points

`Points` represent unordered sets of positions with per-point features.

```python
import torch
from warpconvnet.geometry.types.points import Points

coords = [torch.rand(1000, 3), torch.rand(500, 3)]
features = [torch.rand(1000, 7), torch.rand(500, 7)]
points = Points(coords, features)
print(points.batch_size)  # 2
```

Downsampling and neighbor search utilities are provided. See `tests/types/test_points.py` for additional examples.

## Voxels

`Voxels` store integer grid coordinates and features. They can be constructed directly or converted from dense tensors.

```python
from warpconvnet.geometry.types.voxels import Voxels

voxel_size = 0.01
voxel_coords = [(torch.rand(1000, 3) / voxel_size).int(),
                (torch.rand(500, 3) / voxel_size).int()]
voxel_feats = [torch.rand(1000, 7), torch.rand(500, 7)]
voxels = Voxels(voxel_coords, voxel_feats)
```

Conversion to and from dense representations is demonstrated in `tests/types/test_voxels.py`.

## Grid

`Grid` is a dense voxel grid with a fixed shape and memory format.

```python
from warpconvnet.geometry.types.grid import Grid, GridMemoryFormat

grid = Grid.from_shape(
    grid_shape=(4, 6, 8),
    num_channels=16,
    memory_format=GridMemoryFormat.b_x_y_z_c,
    batch_size=2,
)
```

Conversion helpers like `points_to_grid` are illustrated in `tests/types/test_grid.py` and `tests/types/test_to_grid.py`.

## FactorGrid

`FactorGrid` groups multiple `Grid` objects with different factorized memory formats.

```python
from warpconvnet.geometry.types.factor_grid import FactorGrid
from warpconvnet.geometry.features.grid import GridMemoryFormat

factor = FactorGrid.create_from_grid_shape(
    [(2, 32, 64), (16, 2, 64), (16, 32, 2)],
    num_channels=7,
    memory_formats=[
        GridMemoryFormat.b_zc_x_y,
        GridMemoryFormat.b_xc_y_z,
        GridMemoryFormat.b_yc_x_z,
    ],
    batch_size=2,
)
```

Detailed usage can be found in `tests/types/test_factor_grid.py`.

For complete API documentation consult the modules under `warpconvnet/geometry/types/`.
