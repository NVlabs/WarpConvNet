# Normalizations

WarpConvNet provides normalization layers that operate on `Geometry` objects.
Each wraps a standard PyTorch normalization and applies it to the underlying
feature tensor, so you can drop them into any `Sequential` pipeline.

## Available layers

| Class                | Underlying module         | Notes                                                                         |
| -------------------- | ------------------------- | ----------------------------------------------------------------------------- |
| `BatchNorm`          | `torch.nn.BatchNorm1d`    | Standard batch normalization                                                  |
| `LayerNorm`          | `torch.nn.LayerNorm`      | Per-sample normalization                                                      |
| `InstanceNorm`       | `torch.nn.InstanceNorm1d` | Per-instance normalization                                                    |
| `GroupNorm`          | `torch.nn.GroupNorm`      | Channel-group normalization                                                   |
| `RMSNorm`            | `torch.nn.RMSNorm`        | Root-mean-square normalization                                                |
| `SegmentedLayerNorm` | Custom CUDA kernel        | Layer norm that respects variable-length segments within a concatenated batch |

All classes live in `warpconvnet.nn.modules.normalizations`.

## Usage

```python
from warpconvnet.nn.modules.normalizations import BatchNorm, SegmentedLayerNorm
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
import torch.nn as nn

net = Sequential(
    SparseConv3d(32, 64, kernel_size=3),
    BatchNorm(64),
    nn.ReLU(),
)
```

You can also use `torch.nn.BatchNorm1d(64)` directly — `Sequential` will apply
it to the feature tensor automatically. The WarpConvNet wrappers are provided
for cases where you need the layer to accept a `Geometry` object directly
(e.g., when used outside of `Sequential`).

## Segmented layer norm

`SegmentedLayerNorm` computes per-sample statistics over the concatenated
batch, respecting the `offsets` boundaries. This is useful when samples have
very different lengths and you want each sample normalized independently
without converting to padded format.

```python
norm = SegmentedLayerNorm(channels=64)
output = norm(geometry)  # computes mean/var per sample
```

## Functional API

Lower-level functions are available in `warpconvnet.nn.functional.normalizations`:

- `segmented_layer_norm(x, offsets, gamma, beta, eps)` — segmented layer norm
- `segmented_range_norm(features, row_offsets, eps)` — normalize each segment to [0, 1]
- `segmented_norm(x, offsets, eps)` — basic segmented normalization
