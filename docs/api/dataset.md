# Dataset

Dataset loaders for common 3D benchmarks.

Defined in `warpconvnet/dataset/`.

## ModelNet40Dataset

```python
from warpconvnet.dataset.modelnet import ModelNet40Dataset
```

```python
ModelNet40Dataset(
    root_dir: str = "./data/modelnet40",
    split: str = "train",  # "train" or "test"
)
```

PyTorch `Dataset` for [ModelNet40](https://modelnet.cs.princeton.edu/) point
cloud classification (40 object categories, 2048 points per shape).

Downloads and extracts the HDF5 dataset automatically on first use.

Each sample is a dict:

| Key        | Shape       | Dtype   | Description        |
| ---------- | ----------- | ------- | ------------------ |
| `"coords"` | `(2048, 3)` | float32 | Point coordinates  |
| `"labels"` | scalar      | int64   | Class label (0-39) |

## ScanNetDataset

```python
from warpconvnet.dataset.scannet import ScanNetDataset
```

```python
ScanNetDataset(
    root: str = "./data/scannet",
    split: str = "train",          # "train" or "val"
    voxel_size: float | None = None,  # optional voxel downsampling
    out_type: str = "voxel",       # "point" or "voxel"
    min_coord: tuple | None = None,  # optional coordinate offset
)
```

PyTorch `Dataset` for [ScanNet](http://www.scan-net.org/) semantic
segmentation using the pre-processed data from the
[OpenScene](https://pengsongyou.github.io/openscene) project.

Downloads the dataset automatically on first use (~1.3 GB).

Each sample is a dict:

| Key        | Shape    | Dtype   | Description                                |
| ---------- | -------- | ------- | ------------------------------------------ |
| `"coords"` | `(N, 3)` | float32 | Point coordinates                          |
| `"colors"` | `(N, 3)` | float32 | RGB colors                                 |
| `"labels"` | `(N,)`   | int64   | Semantic labels (20 classes, 255 = ignore) |

If `voxel_size` is set, points are voxel-downsampled before returning.
