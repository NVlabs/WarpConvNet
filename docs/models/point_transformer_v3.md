# PointTransformerV3

`warpconvnet.models.PointTransformerV3` is a serialized-attention
encoder–decoder from
[Wu et al., *Point Transformer V3: Simpler, Faster, Stronger*, CVPR 2024](https://arxiv.org/abs/2312.10035).

Points are serialized along space-filling curves (Z-order, Hilbert), then
processed with patch-based local attention. A symmetric U-shape encoder–decoder
with learnable serialization order shuffling delivers competitive accuracy
at very high throughput.

## Architecture

- **Patch attention** — `PatchAttention` over fixed-size token windows on
  the serialization curve
- **Encoder** — 5 levels (default depths `(3, 3, 3, 6, 3)`), increasing
  channel widths `(48, 96, 192, 384, 512)`, sparse downsampling between
  levels
- **Decoder** — 4 levels mirroring the encoder, with skip connections via
  `SerializedUnpooling`
- **Final head** — per-point linear

## Signature

```python
class PointTransformerV3(BaseSpatialModel):
    def __init__(
        self,
        in_channels: int,
        enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2),
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32),
        enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Tuple[int, ...] = (2, 2, 2, 2),
        dec_channels: Tuple[int, ...] = (64, 96, 128, 256),
        dec_num_head: Tuple[int, ...] = (4, 6, 8, 16),
        dec_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024),
        shuffle_orders: bool = True,
        order: Tuple[POINT_ORDERING, ...] = ("z", "hilbert"),
    ): ...
```

## Usage

```python
import torch
from warpconvnet.geometry.types.points import Points
from warpconvnet.models import PointTransformerV3
from warpconvnet.nn.modules.sparse_pool import PointToVoxel

pc = Points(
    [torch.rand(N, 3) for N in (5000, 8000, 6000)],
    [torch.rand(N, 7) for N in (5000, 8000, 6000)],
).cuda()

model = PointToVoxel(
    PointTransformerV3(in_channels=7, shuffle_orders=True),
    voxel_size=0.02,
    reduction="mean",
).cuda()

out = model(pc)                     # Points with 48-D per-point features
```

## Reference

- Wu, Jiang, Zhang, Zhao, Hu, Lu, Liu, Bei, Zhao. *Point Transformer V3:
  Simpler, Faster, Stronger.* CVPR 2024.
