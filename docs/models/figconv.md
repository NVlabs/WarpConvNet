# FIGConvNet

`warpconvnet.models.FIGConvNet` implements *Factorized Implicit Global
Convolution* networks from
[Choy et al., *Factorized Implicit Global Convolution for Automotive
Computational Fluid Dynamics Prediction*, CVPR 2024](https://arxiv.org/abs/2401.01122).

Points are projected onto three 2D
[`FactorGrid`](../user_guide/geometry_types.md) memory formats
(`b_xc_y_z`, `b_yc_x_z`, `b_zc_x_y`). A multi-resolution encoder–decoder
applies large-kernel global convolutions per axis, then features are
re-aggregated to points via interpolation or graph-conv.

## Variants

- **`FIGConvNet`** — base architecture; outputs both per-point predictions
  and an MLP-pooled global value (e.g. drag coefficient).
- **`FIGConvNetDrivAer`** — DrivAer-specific subclass with the dataset's
  default resolutions / pooling layout.

## Signature

```python
class FIGConvNet(BaseSpatialModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],            # length == num_levels + 1
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        mlp_channels: List[int] = [2048, 2048],
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[
            Tuple[GridMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        ...
    ): ...
```

`hidden_channels` must have **`num_levels + 1`** entries (one per
encoder/decoder level), e.g. `[16, 32, 64, 128]` for the default
`num_levels=3`.

## Usage

```python
import torch
from warpconvnet.geometry.types.points import Points
from warpconvnet.models import FIGConvNet

pc = Points(
    [torch.rand(20000, 3) for _ in range(2)],
    [torch.rand(20000, 3) for _ in range(2)],
).cuda()

model = FIGConvNet(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    hidden_channels=[16, 32, 64, 128],   # num_levels + 1 entries
).cuda()

per_point, global_pred = model(pc)        # per-point and pooled outputs
```

## Reference

- Choy, Lee, Hamdi, Catalano, Wu, Quéraud, Lin, Spence, Spence, Romero,
  Maggio. *Factorized Implicit Global Convolution for Automotive CFD
  Prediction.* CVPR 2024.
