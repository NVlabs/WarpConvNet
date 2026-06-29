# SpaceFormer

Mixed **space/curve** sparse attention for 3D point clouds: each level picks between
window-grouped 3D attention (`space`) and serialized space-filling-curve attention
(`curve`). This package provides two models that share the same backbone.

**Project page:** https://nvlabs.github.io/SpaCeFormer/

| Model                | File                  | Use for                                                           |
| -------------------- | --------------------- | ----------------------------------------------------------------- |
| `SpaCeFormer`        | `space_former.py`     | the backbone → **semantic segmentation** (per-point class logits) |
| `SpaCeFormerInstSeg` | `space_former_seg.py` | proposal-free **instance segmentation** (open-vocabulary)         |

```python
from warpconvnet.models.spaceformer import SpaCeFormer, SpaCeFormerInstSeg, build_spaceformer
```

## Semantic segmentation — `SpaCeFormer` (backbone)

Set `out_channels=num_classes`; the backbone appends a final linear head and returns a
`Points` whose features are per-point class logits.

```python
import torch
from warpconvnet.geometry.types.points import Points
from warpconvnet.models.spaceformer import SpaCeFormer

model = SpaCeFormer(in_channels=3, out_channels=20).cuda()   # e.g. ScanNet-20
pc = Points(batched_coordinates=coord, batched_features=feat, offsets=offset).cuda()
out = model(pc)                 # Points
logits = out.features           # [N, num_classes] per-point class logits
```

Tune the architecture with the `enc_*` / `dec_*` / `*_attn_types` arguments (compact
codes, e.g. `enc_attn_types="ssccc"`). See `space_former.py` for the full signature.

## Instance segmentation — `SpaCeFormerInstSeg`

A Mask2Former-style query decoder (learned queries + 3D RoPE) on the backbone. `forward`
returns **raw** predictions `{logit, mask, clip_feat}`; turning `clip_feat` into
open-vocabulary labels (SigLIP2 text + prompt ensembling) and post-processing the masks
(NMS / min-points) is downstream and ships with the release/demo, not in WarpConvNet.

```python
import torch
from warpconvnet.models.spaceformer import build_spaceformer, load_spaceformer_checkpoint
from huggingface_hub import hf_hub_download

net = build_spaceformer(device=torch.device("cuda"))     # released architecture
ckpt = hf_hub_download("chrischoy/SpaCeFormer", "spaceformer_512_siglip2_ssccc.ckpt")
load_spaceformer_checkpoint(net, ckpt)                   # weights-only, strict=False

# coord [N,3] float meters; feat [N,3] RGB in [-1,1]; offset [0, N]
out = net({"coord": coord, "feat": feat, "offset": offset})
# {"logit": [B,Q,2], "mask": List[[N,Q]], "clip_feat": [B,Q,1152]}
```

For the end-to-end open-vocab demo (labeling + post-processing + Gradio UI), see the
release/demo repo and the HuggingFace Space.

## Naming

`SpaCeFormer` is the **backbone**; `SpaCeFormerInstSeg` is the **instance-seg model** that
wraps it — the capitalization follows the paper's "space-curve" etymology.

## License

Apache-2.0.
