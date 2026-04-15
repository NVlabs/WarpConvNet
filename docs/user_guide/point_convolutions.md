# Point Convolutions

Point convolutions operate directly on unstructured point clouds (`Points`
geometry) without voxelization. They aggregate features from neighboring points
using a configurable neighbor search and learned transformations.

## PointConv

`warpconvnet.nn.modules.point_conv.PointConv` is the main module.

```python
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
import torch.nn as nn

net = Sequential(
    PointConv(
        in_channels=3,
        out_channels=64,
        neighbor_search_args=RealSearchConfig("knn", knn_k=16),
    ),
    nn.LayerNorm(64),
    nn.ReLU(),
)
```

### Key parameters

| Parameter                      | Description                                                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `in_channels` / `out_channels` | Input and output feature dimensions                                                                                  |
| `neighbor_search_args`         | A `RealSearchConfig` specifying the search method (`"knn"` or `"radius"`) and its parameters                         |
| `reductions`                   | Aggregation functions over neighbors (default: `("mean",)`)                                                          |
| `use_rel_pos`                  | Include relative coordinates as edge features                                                                        |
| `use_rel_pos_encode`           | Include sinusoidal positional encoding of relative coordinates                                                       |
| `out_point_type`               | `"same"` (keep coordinates), `"downsample"` (voxel-based downsampling), or `"provided"` (user-supplied query points) |

### Neighbor search

Configure via `RealSearchConfig`:

```python
# K-nearest neighbors
knn_config = RealSearchConfig("knn", knn_k=16)

# Radius search
radius_config = RealSearchConfig("radius", radius=0.1, max_neighbors=32)
```

### Downsampling

To produce a lower-resolution point cloud, set `out_point_type="downsample"`:

```python
from warpconvnet.ops.reductions import REDUCTIONS

down_conv = PointConv(
    64, 128,
    neighbor_search_args=RealSearchConfig("knn", knn_k=16),
    out_point_type="downsample",
    pooling_voxel_size=0.1,
    pooling_reduction=REDUCTIONS.MEAN,
)
```

See the [API reference](../api/nn.md) for the full signature.
