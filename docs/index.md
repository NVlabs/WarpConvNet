# WarpConvNet

Welcome to the WarpConvNet documentation.

WarpConvNet is a high-performance 3D deep learning library built on NVIDIA's Warp framework. It provides efficient implementations of point cloud processing, sparse voxel convolutions, attention mechanisms for 3D data, and geometric operations.

## 🚀 Quick Start

```python
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
from warpconvnet.ops.reductions import REDUCTIONS

point_conv = Sequential(
    PointConv(24, 64, neighbor_search_args=RealSearchConfig("knn", knn_k=16)),
    nn.LayerNorm(64),
    nn.ReLU(),
)
sparse_conv = Sequential(
    SparseConv3d(64, 128, kernel_size=3, stride=2),
    nn.ReLU(),
)

coords: Float[Tensor, "B N 3"]  # noqa: F821
pc: Points = Points.from_list_of_coordinates(coords, encoding_channels=8, encoding_range=1)
pc = point_conv(pc)
vox: Voxels = pc.to_voxels(reduction=REDUCTIONS.MEAN, voxel_size=0.05)
vox = sparse_conv(vox)
dense: Tensor = vox.to_dense(channel_dim=1, min_coords=(-5, -5, -5), max_coords=(2, 2, 2))
# feed `dense` to your 3D CNN head for classification
```

See `examples/modelnet.py` for a full training script.

## 📚 Documentation Sections

- **Getting Started**: Installation and quick start guides
- **User Guide**: Comprehensive tutorials and concepts
- **API Reference**: Complete API documentation
- **Examples**: Real-world usage examples
- **Diagrams**: Architecture and data flow visualizations

## 🔗 Links

- [GitHub Repository](https://github.com/nvidia/warpconvnet)
- [Installation Guide](getting_started/installation.md)
- [Quick Start Tutorial](getting_started/quickstart.md)
