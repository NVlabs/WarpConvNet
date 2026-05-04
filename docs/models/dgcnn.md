# DGCNN

`warpconvnet.models.DGCNN` implements the dynamic-graph CNN of
[Wang et al., *Dynamic Graph CNN for Learning on Point Clouds*, TOG 2019](https://arxiv.org/abs/1801.07829).

Each layer rebuilds a kNN graph in feature space and applies a `PointConv`
edge function. Concatenated multi-scale features feed a global pool +
classifier head.

## Components

- **`DGCNNEncoder`** — stack of `PointConvBlock` stages with skip
  concatenation; outputs a feature `Geometry`.
- **`DGCNN`** — encoder + global pool + linear classifier head.

## Signature

```python
class DGCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_channels: Union[int, List[int]] = (64, 64, 128, 256),
        classifier_channels: List[int] = [512, 256],
        knn_k: int = 20,
    ): ...
```

## Usage

```python
import torch
from warpconvnet.geometry.types.points import Points
from warpconvnet.models import DGCNN, DGCNNEncoder

device = "cuda"
pc = Points(
    [torch.rand(2048, 3) for _ in range(4)],
    [torch.rand(2048, 3) for _ in range(4)],
).to(device)

cls = DGCNN(in_channels=3, out_channels=40).to(device)
logits = cls(pc)              # (B, 40)

# Or use just the encoder for per-point features:
encoder = DGCNNEncoder(in_channels=3).to(device)
feat = encoder(pc)            # Points with concatenated multi-scale features
```

## Reference

- Wang, Sun, Liu, Sarma, Bronstein, Solomon. *Dynamic Graph CNN for
  Learning on Point Clouds.* ACM TOG, 2019.
