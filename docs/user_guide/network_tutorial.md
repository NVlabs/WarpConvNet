# Network Creation Tutorial

This tutorial walks through building neural networks with WarpConvNet modules. The code snippets are simplified versions of the example scripts in the `examples` folder.

## MNIST Example

The `examples/mnist.py` script constructs a small network for classifying 2‑D images using sparse convolutions:

```python
import torch
import torch.nn as nn
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_pool import sparse_max_pool
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            SparseConv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            SparseConv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(14 * 14 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = Voxels.from_dense(x)
        x = self.layers(x)
        x = sparse_max_pool(x, kernel_size=(2, 2), stride=(2, 2))
        x = x.to_dense(channel_dim=1, spatial_shape=(14, 14))
        x = torch.flatten(x, 1)
        return self.head(x)
```

## ModelNet Example

For 3‑D data a combination of point and sparse convolutions can be used. The following snippet is adapted from `examples/modelnet.py`:

```python
import torch
import torch.nn as nn
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv3d

class UseAllConvNet(nn.Module):
    def __init__(self, voxel_size: float = 0.05):
        super().__init__()
        self.point_conv = Sequential(
            PointConv(24, 64, neighbor_search_args=RealSearchConfig("knn", knn_k=16)),
            nn.LayerNorm(64),
            nn.ReLU(),
            PointConv(64, 64, neighbor_search_args=RealSearchConfig("radius", radius=voxel_size)),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.voxel_size = voxel_size
        self.sparse_conv = Sequential(
            SparseConv3d(64, 64, kernel_size=3, stride=1),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        pc = Points.from_list_of_coordinates(coords, encoding_channels=8, encoding_range=1)
        pc = self.point_conv(pc)
        st = pc.to_voxels(reduction="max", voxel_size=self.voxel_size)
        st = self.sparse_conv(st)
        return st.to_dense(channel_dim=1)
```

These examples demonstrate how geometry‑aware modules such as `PointConv` and `SparseConv3d` can be combined using `Sequential` to form end‑to‑end networks. Refer to the full example scripts for the training loops and data handling.
