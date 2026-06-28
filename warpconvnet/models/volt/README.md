# Volt — Volume Transformer

A WarpConvNet-native implementation of **Volt**, a ViT-style transformer for
sparse-voxel semantic segmentation.

The scene is partitioned into non-overlapping `K³` voxel patches (`K=5`), each linearly
embedded into a token. A pre-norm transformer with **global per-scene attention**
(variable-length flash-attention) and **anisotropic axial RoPE** processes the tokens,
which are then un-embedded back to voxel resolution. The model is built on WarpConvNet
primitives — `Voxels` for geometry I/O, `flash_attn_varlen_qkvpacked` for attention,
and `SparseConv3d` for the convolutional variants.

## Variants and results

Every variant is a single keyword change to `Volt`, exposed through `build_volt` /
`VOLT_VARIANTS`. Results are ScanNet v2 validation mIoU with test-time augmentation.

| Variant                | Configuration                               | Params |      mIoU |
| ---------------------- | ------------------------------------------- | -----: | --------: |
| `volt-s`               | base (384-d, 6 heads)                       |  23.7M |     76.06 |
| `volt-convattn`        | `conv_before_attn=True`                     |  71.5M |     76.41 |
| `volt-b`               | `embed_dim=768, num_heads=12`               |  87.8M |     76.53 |
| `volt-convblock`       | `tokenizer_type="convblock"`                |  26.9M |     77.01 |
| `volt-all3`            | `volt-b` + `convblock` + `conv_before_attn` | 284.9M |     77.93 |
| `volt-blockattn`       | `convblock` + `conv_before_attn`            |  74.7M |     78.00 |
| **`volt-b-convblock`** | `volt-b` + `convblock`                      |  93.8M | **78.23** |

`volt-b-convblock` is the strongest configuration at **78.23 mIoU**, above the published
Volt-S result of 77.3.

### Configuration knobs

- `tokenizer_type` — `"linear"` (default) uses the per-slot patch embed; `"convblock"`
  prepends a ResNet-style non-strided `SparseConv3d` stem that adds local context before
  the same per-slot embed.
- `conv_before_attn=True` — adds a per-block stride-1 `SparseConv3d` residual on the
  token grid before attention, giving each layer both local and global mixing.
- `embed_dim` / `num_heads` — model scale (e.g. 384/6 → 768/12).

## Usage

```python
from warpconvnet.models.volt import build_volt

model = build_volt("volt-b-convblock")          # best configuration

# WarpConvNet-native: Voxels -> Voxels (features replaced with per-voxel outputs)
voxels_out = model(voxels_in)

# Raw-tensor entry point for external harnesses:
feats = model.forward_tensors(feat, grid_coord, batch)  # [N, up_mlp_dim]
```

`build_volt` returns the backbone, which outputs `up_mlp_dim` features; add a linear
head for per-class logits.

## Training

Trained on ScanNet v2 (voxel size 0.02 m, 6-channel color + normal input) for 100 epochs.
