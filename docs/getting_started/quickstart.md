# Quick Start

After [installation](installation.md), verify everything works with a small
example.

## Minimal example

```python
import torch
import torch.nn as nn
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.ops.reductions import REDUCTIONS

# Create random point clouds (batch of 2, ~1000 points each)
coords = [torch.rand(1000, 3) for _ in range(2)]
features = [torch.rand(1000, 16) for _ in range(2)]
pc = Points.from_list_of_coordinates(coords, features=features)

# Voxelize and run a sparse convolution
vox = pc.to_voxels(reduction=REDUCTIONS.MEAN, voxel_size=0.05)
net = Sequential(SparseConv3d(16, 32, kernel_size=3), nn.ReLU())
out = net(vox)
print(out)  # Voxels geometry with 32 feature channels
```

## Run the ModelNet example

```bash
python examples/modelnet.py
```

This trains a 3D classification model on ModelNet40. The dataset is downloaded
automatically on first run.

## Run the ScanNet example

```bash
pip install ".[models]"
python examples/scannet.py train.batch_size=12 model=mink_unet
```

This trains a semantic segmentation model on ScanNet. The script uses
[Hydra](https://hydra.cc/) for configuration — pass `--help` to see all options.
