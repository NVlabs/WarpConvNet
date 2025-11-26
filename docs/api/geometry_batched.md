# Batched coordinate layout

::: warpconvnet.geometry.base.geometry.Geometry

Geometry containers in WarpConvNet treat batched data as a pair of concatenated tensors
plus an `offsets` vector. This “batched” layout stores every sample back-to-back inside a
single tensor while recording where each sample begins so ragged batches stay addressable.

## Concatenated coordinates

`warpconvnet.geometry.base.coords.Coords` keeps two tensors:

- `batched_tensor`: shape `[N, D]` with all coordinates concatenated in batch order.
- `offsets`: shape `[B + 1]` with `offsets[b]` marking the starting row for batch `b`.

The difference `offsets[b + 1] - offsets[b]` gives the number of coordinates in sample
`b`. Because the data lives in one tensor, it is cheap to move, sort, or feed to CUDA
kernels that expect CSR-style layouts.

## Features that share offsets

`warpconvnet.geometry.base.features.Features` mirrors the coordinate batching strategy.
Both `CatFeatures` (concatenated) and `PadFeatures` (padded) expose the same offsets so
`Geometry` subclasses can hop between dense and ragged views without recomputing metadata.

- Concatenated features: `[N, C]` storage with the same CSR-style offsets as the
  coordinates.
- Padded features: `[B, L_max, C]` storage when kernels need dense memory, but each row
  still references the shared offsets to keep track of valid entries.

## Putting it together

The base `Geometry` class validates that coordinates and features have identical offsets,
then exposes helpers such as `geometry.batch_indexed_coordinates` and AMP-aware feature
accessors. Subclasses (e.g., `Points`, `Voxels`, `Grid`) inherit these guarantees and add
type-specific methods on top.

```python
import torch
from warpconvnet.geometry.types.points import Points

coords = torch.tensor(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [5, 5, 5],
        [6, 5, 5],
    ],
    dtype=torch.int32,
)
offsets = torch.tensor([0, 3, 5], dtype=torch.int32)
features = torch.randn(5, 32)

points = Points(coords, features, offsets=offsets)
# points.batched_coordinates.offsets == points.batched_features.offsets
```

In this example two ragged samples (lengths 3 and 2) share one coordinate tensor and one
feature tensor. The shared offsets allow the geometry module to slice, move devices, or
convert between concatenated and padded features without copying metadata.
