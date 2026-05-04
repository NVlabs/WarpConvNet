# MinkUNet

`warpconvnet.models.MinkUNet*` implements the
[Minkowski U-Net](https://arxiv.org/abs/1904.08755) family from
Choy, Gwak, Savarese, *4D Spatio-Temporal ConvNets*, CVPR 2019.

A symmetric encoder–decoder over voxelized inputs:

- Encoder: 4 strided `SparseConv3d` stages with `BasicBlock` / `BottleneckBlock`
  residual blocks
- Decoder: matching transposed-conv stages with skip connections from the
  encoder
- Output: per-voxel logits

## Variants

| Class         | Block             | Layers per stage          | Params |
| ------------- | ----------------- | ------------------------- | ------ |
| `MinkUNet18`  | `BasicBlock`      | (2, 2, 2, 2, 2, 2, 2, 2)  | ~22 M  |
| `MinkUNet34`  | `BasicBlock`      | (2, 3, 4, 6, 2, 2, 2, 2)  | ~38 M  |
| `MinkUNet50`  | `BottleneckBlock` | (2, 3, 4, 6, 2, 2, 2, 2)  | ~52 M  |
| `MinkUNet101` | `BottleneckBlock` | (2, 3, 4, 23, 2, 2, 2, 2) | ~83 M  |

`PointMinkUNet18` / `PointMinkUNet34` wrap the voxel network with
`PointToVoxel` so the input/output types are `Points`.

## Signature

```python
class MinkUNetBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        planes: Tuple[int, ...] = (32, 64, 128, 256, 256, 128, 96, 96),
        layers: Tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2, 2),
        init_dim: int = 32,
        block_type: type = BasicBlock,
    ): ...
```

## Usage

### Voxels in / voxels out

```python
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models import MinkUNet34

model = MinkUNet34(in_channels=3, out_channels=20).cuda()
voxels: Voxels = ...                 # already voxelized
logits = model(voxels)               # Voxels with (N, 20) features
```

### Points in / points out

```python
from warpconvnet.geometry.types.points import Points
from warpconvnet.models import PointMinkUNet18

model = PointMinkUNet18(in_channels=3, out_channels=20).cuda()
pc: Points = ...
out = model(pc)                       # Points with (N, 20) features
```

### Hydra

```yaml
model:
  _target_: warpconvnet.models.MinkUNet34
  in_channels: 3
  out_channels: 20
```

## Reference

- Choy, Gwak, Savarese. *4D Spatio-Temporal ConvNets: Minkowski Convolutional
  Neural Networks.* CVPR 2019.
