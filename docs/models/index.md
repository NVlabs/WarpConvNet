# Models

`warpconvnet.models` is a small zoo of reference 3D point-cloud and sparse-voxel
networks. Every model is an importable `nn.Module` that consumes
[`Points`](../user_guide/geometry_types.md) (or `Voxels`) and returns a
`Geometry` or tensor.

```python
from warpconvnet.models import (
    DGCNN, DGCNNEncoder,
    FIGConvNet, FIGConvNetDrivAer,
    MaskFormer, MaskTransformer,
    MinkUNet18, MinkUNet34, MinkUNet50, MinkUNet101,
    PointMinkUNet18, PointMinkUNet34,
    PointNet,
    PointTransformerV3,
    SpaCeFormer,
)
```

## Model index

| Model                                           | Input                    | Task                            | Paper                          |
| ----------------------------------------------- | ------------------------ | ------------------------------- | ------------------------------ |
| [`PointNet`](pointnet.md)                       | Points                   | Classification / Segmentation   | Qi et al. 2017                 |
| [`DGCNN`](dgcnn.md)                             | Points                   | Classification / Segmentation   | Wang et al. 2019               |
| [`PointTransformerV3`](point_transformer_v3.md) | Points (serialized)      | Segmentation                    | Wu et al. 2024                 |
| [`MinkUNet18/34/50/101`](mink_unet.md)          | Voxels                   | Semantic segmentation           | Choy et al. 2019               |
| [`PointMinkUNet18/34`](mink_unet.md)            | Points → Voxels → Points | Per-point segmentation          | —                              |
| [`FIGConvNet`](figconv.md)                      | Points                   | Regression / Dense prediction   | Choy et al. 2024               |
| [`MaskFormer`](maskformer.md)                   | Points                   | Instance / mask prediction      | Cheng et al. 2021 (3D variant) |
| [`SpaCeFormer`](space_former.md)                | Voxels                   | Hierarchical sparse-voxel U-Net | Choy et al. 2026               |

## Importing in Hydra

Hydra `_target_` strings resolve to the public path, so any of the above
classes work as a drop-in target:

```yaml
model:
  _target_: warpconvnet.models.MinkUNet34
  in_channels: 3
  out_channels: 20
```

```bash
python examples/train/scannet.py model._target_=warpconvnet.models.MinkUNet34
```
