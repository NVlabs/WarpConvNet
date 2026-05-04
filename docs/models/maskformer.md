# MaskFormer

`warpconvnet.models.MaskFormer` is a 3D adaptation of the
[MaskFormer / Mask2Former](https://arxiv.org/abs/2107.06278) family for
**point-cloud instance / mask prediction**. A backbone produces dense
per-point features, a transformer decoder cross-attends learnable queries
against those features, and the final masks are computed by inner-product
between queries and per-point feature embeddings.

## Components

- **Backbone** — any `BaseSpatialModel` that returns a per-point `Geometry`
  (e.g. `MinkUNet18`, `PointTransformerV3`, wrapped in
  `PointToVoxel` if voxelization is desired).
- **`MaskTransformer`** — `num_decoders` layers of self-attention over
  queries followed by cross-attention against scene features.
- **`MaskFormer`** — couples the backbone, mask transformer, a class head,
  and an inner-product mask head.

## Signature

```python
class MaskFormer(BaseSpatialModel):
    def __init__(
        self,
        backbone: BaseSpatialModel,
        hidden_dim: int,
        num_queries: int,
        num_heads: int,
        num_decoders: int,
        dim_feedforward: int,
        dropout: float,
        num_classes: int,
    ): ...
    def forward(self, x: Points) -> Tuple[
        Float[Tensor, "B Q num_classes+1"],
        List[Float[Tensor, "Q N_b"]],
    ]: ...
```

`logits.shape == (B, num_queries, num_classes + 1)` (the extra slot is the
"no object" / background class). `masks` is a list of length `B`; each
entry is `(num_queries, N_b)` raw logits over the points of scene `b`.

## Usage

```python
import torch
from warpconvnet.dataset.scannet import ScanNetInstanceDataset
from warpconvnet.geometry.types.points import Points
from warpconvnet.models import MaskFormer, MinkUNet18
from warpconvnet.nn.modules.sparse_pool import PointToVoxel

device = "cuda"
ds = ScanNetInstanceDataset(
    root="/path/to/scannet_preprocessed",
    split="val",
    label_set="scannet20",      # or "scannet200" for 198 fine-grained classes
    voxel_size=0.04,
)
samples = [ds[i] for i in range(2)]
pc = Points.from_list_of_coordinates(
    [torch.from_numpy(s["coords"]).float() for s in samples],
    features=[torch.from_numpy(s["colors"]).float() / 255.0 for s in samples],
).to(device)

backbone = PointToVoxel(
    inner_module=MinkUNet18(in_channels=3, out_channels=96),
    voxel_size=0.04,
    concat_unpooled_pc=False,
)

model = MaskFormer(
    backbone=backbone,
    hidden_dim=96,
    num_queries=100,
    num_heads=8,
    num_decoders=6,
    dim_feedforward=256,
    dropout=0.1,
    num_classes=20,            # set to 200 for the ScanNet200 label set
).to(device)

logits, masks = model(pc)
# logits: (2, 100, 21)        # (B, num_queries, num_classes + 1)
# masks:  [tensor(100, N0), tensor(100, N1)]
```

## Training

A reference Hydra-driven training loop with Hungarian matching and the
standard MaskFormer loss (CE + BCE + Dice) is at
[`examples/train/maskformer.py`](https://github.com/NVlabs/WarpConvNet/blob/main/examples/train/maskformer.py).

```bash
python examples/train/maskformer.py \
    paths.data_dir=/path/to/scannet_preprocessed \
    train.batch_size=2 \
    train.lr=1e-4
```

The script defaults to `data.label_set=scannet20`; pass
`data.label_set=scannet200` to switch to the 198-class label set.

## Dataset

Use [`ScanNetInstanceDataset`](../api/dataset.md) for ScanNet/ScanNet200 with
the Mask3D-preprocessed layout. Raw ScanNet meshes are
[ToS-gated](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf); see the
dataset docstring for preprocessing pointers.

## Reference

- Cheng, Schwing, Kirillov. *Per-Pixel Classification is Not All You Need
  for Semantic Segmentation.* NeurIPS 2021.
- Cheng, Misra, Schwing, Kirillov, Girdhar. *Masked-attention Mask
  Transformer for Universal Image Segmentation.* CVPR 2022.
- Schult et al. *Mask3D: Mask Transformer for 3D Instance Segmentation.*
  ICRA 2023 (3D adaptation reference).
