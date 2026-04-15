# Basic Concepts

WarpConvNet represents 3D data using **geometry containers** that pair coordinate
information with per-point or per-voxel features. All containers derive from a
common `Geometry` base class and support variable-length batching, device
transfer, and dtype conversion out of the box.

## Geometry containers

| Type           | Coordinates           | Use case                                           |
| -------------- | --------------------- | -------------------------------------------------- |
| **Points**     | Real-valued $(x,y,z)$ | Unstructured point clouds, continuous convolutions |
| **Voxels**     | Integer grid indices  | Sparse convolutions on quantized grids             |
| **Grid**       | Dense regular grid    | Dense 3D convolutions                              |
| **FactorGrid** | Factorized grid views | Factorized 3D convolutions (memory-efficient)      |

See [Geometry Types](geometry_types.md) for details on each container and
[Geometry Tutorial](geometry_tutorial.md) for code examples.

## Batching

Variable-length batches are stored in **concatenated** (ragged) format by
default: all points across a batch are packed into a single tensor, and an
`offsets` tensor of shape `(B+1,)` marks where each sample begins and ends.
This avoids padding waste for point clouds of different sizes.

When an operation requires uniform tensor shapes (e.g., `torch.bmm`),
call `geometry.to_pad()` to switch to padded format. Use `geometry.to_cat()`
to convert back.

## Modules

Neural network layers in `warpconvnet.nn` accept and return `Geometry` objects,
so they compose naturally with `torch.nn` layers via
`warpconvnet.nn.Sequential`:

```python
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.sequential import Sequential
import torch.nn as nn

net = Sequential(
    SparseConv3d(32, 64, kernel_size=3, stride=2),
    nn.BatchNorm1d(64),
    nn.ReLU(),
)
```

Standard `torch.nn` layers (e.g., `BatchNorm1d`, `ReLU`, `Linear`) are applied
to the feature tensor automatically.

## Conversions

Containers can be converted between types:

- `Points.to_voxels(voxel_size, reduction)` — quantize to sparse voxels
- `Voxels.to_dense(...)` — materialize a dense grid tensor
- `Voxels.to_grid(...)` — wrap as a `Grid` geometry

These conversions are differentiable where possible.
