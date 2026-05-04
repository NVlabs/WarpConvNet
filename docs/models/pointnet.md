# PointNet

`warpconvnet.models.PointNet` is the classic permutation-invariant point set
classifier from
[Qi et al., *PointNet*, CVPR 2017](https://arxiv.org/abs/1612.00593).

## Architecture

1. Per-point MLP (`Linear → BN → ReLU` blocks) in `[64, 128, 1024]` channels
2. Optional STN (spatial transformer) on the input + an internal feature STN
3. Global max-pool over points
4. Classifier head `[512, 256, num_classes]`

## Signature

```python
class PointNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_dims: List[int] = [64, 128, 1024],
        classifier_dims: List[int] = [512, 256],
        use_stn: bool = True,
    ): ...
```

## Usage

```python
import torch
from warpconvnet.geometry.types.points import Points
from warpconvnet.models import PointNet

device = "cuda"
B, N, C = 4, 2048, 3
coords = [torch.rand(N, 3) for _ in range(B)]
features = [torch.rand(N, C) for _ in range(B)]
pc = Points(coords, features).to(device)

model = PointNet(in_channels=C, out_channels=40).to(device)
logits = model(pc)            # (B, 40)
```

## Reference

- Qi, Su, Mo, Guibas. *PointNet: Deep Learning on Point Sets for 3D
  Classification and Segmentation.* CVPR 2017.
