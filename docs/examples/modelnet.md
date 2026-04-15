# ModelNet Example

This example trains a 3D shape classification model on
[ModelNet40](https://modelnet.cs.princeton.edu/) (40 object categories).
It demonstrates combining `PointConv`, `SparseConv3d`, and dense `Conv3d`
in a single network.

## Dataset

**ModelNet40** contains 12,311 meshed CAD models across 40 categories
(airplane, chair, table, etc.). The script uses the HDF5 point-cloud variant
with 2,048 points per shape. The dataset is downloaded automatically on
first run to `./data/modelnet40/`.

- Training set: 9,843 shapes
- Test set: 2,468 shapes
- Each sample: `(2048, 3)` float32 coordinates + integer class label

## Network architecture (`UseAllConvNet`)

The model chains three processing stages:

1. **PointConv** (continuous point cloud) — two `PointConv` layers (24 → 64 → 64 channels) with KNN (k=16) and radius search
2. **SparseConv3d** (sparse voxels) — points are voxelized (`voxel_size=0.05`), then processed by five sparse convolution layers (64 → 512 channels) with two stride-2 downsampling layers
3. **Dense Conv3d** (regular grid) — sparse voxels are materialized to a dense tensor, then two `Conv3d` layers followed by a linear classifier output 40 class logits

## Run

```bash
python examples/modelnet.py
```

The script uses [Fire](https://github.com/google/python-fire) for CLI
arguments. All parameters of the `main()` function can be overridden:

```bash
python examples/modelnet.py --batch_size=16 --epochs=50 --lr=5e-4
```

### Arguments

| Argument                | Default             | Description                        |
| ----------------------- | ------------------- | ---------------------------------- |
| `--root_dir`            | `./data/modelnet40` | Dataset download / cache directory |
| `--batch_size`          | `32`                | Training batch size                |
| `--test_batch_size`     | `100`               | Test batch size                    |
| `--epochs`              | `100`               | Number of training epochs          |
| `--lr`                  | `1e-3`              | AdamW learning rate                |
| `--scheduler_step_size` | `10`                | StepLR decay period (epochs)       |
| `--gamma`               | `0.7`               | StepLR decay factor                |
| `--device`              | `cuda`              | Device (`cuda` or `cpu`)           |

## Expected output

Each epoch prints a progress bar with the training loss, followed by test-set
evaluation:

```
Loss: 0.032145, LR: [0.001]: 100%|██████████| 308/308
Test set: Average loss: 0.3842, Accuracy: 2315/2468 (93.80%)
```

After 100 epochs, expect **~90-93% test accuracy** depending on random seed
and hardware.
