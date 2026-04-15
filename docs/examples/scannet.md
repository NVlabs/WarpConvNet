# ScanNet Example

This example trains a semantic segmentation model on
[ScanNet](http://www.scan-net.org/) indoor scenes using a MinkUNet-style
encoder-decoder built with sparse convolutions.

## Dataset

The script uses the pre-processed ScanNet 3D point clouds from the
[OpenScene](https://pengsongyou.github.io/openscene) project. Each scene is
stored as `(coords, colors, labels)`:

- **coords**: `(N, 3)` float32 — 3D point positions
- **colors**: `(N, 3)` float32 — RGB color features
- **labels**: `(N,)` int — semantic class labels (20 classes, 255 = ignore)

The 20 semantic classes include: wall, floor, cabinet, bed, chair, sofa,
table, door, window, bookshelf, picture, counter, desk, curtain, refrigerator,
shower curtain, toilet, sink, bathtub, and other furniture.

The dataset is downloaded automatically on first run (~1.3 GB) to
`./data/scannet_3d/`.

!!! warning "No data augmentation"
This example does **not** apply augmentations (random rotation, scaling,
color jitter, etc.). For high-quality training results, implement your own
augmentation pipeline.

## Network architecture

The default model is **MinkUNet18**, a U-Net with sparse convolution
encoder and decoder blocks connected by skip connections. Available models:

| Model                   | Description                 |
| ----------------------- | --------------------------- |
| `mink_unet.MinkUNet18`  | Lightweight U-Net (default) |
| `mink_unet.MinkUNet34`  | Deeper encoder              |
| `mink_unet.MinkUNet50`  | ResNet-50 style blocks      |
| `mink_unet.MinkUNet101` | ResNet-101 style blocks     |

Input points are voxelized at `voxel_size=0.02` and wrapped via
`PointToSparseWrapper`, which handles the point-to-voxel conversion and
maps output features back to the original point resolution.

The model outputs per-point logits with shape `(N, 20)`.

## Setup

Install the optional model and training dependencies:

```bash
pip install "warpconvnet[models]"
```

Additional requirements: `hydra-core`, `omegaconf`, `torchmetrics`.

## Run

```bash
python examples/scannet.py
```

The script uses [Hydra](https://hydra.cc/) for configuration. Override any
parameter on the command line:

```bash
# Smaller batch size for limited GPU memory
python examples/scannet.py train.batch_size=4

# Use a deeper model
python examples/scannet.py model._target_=mink_unet.MinkUNet34

# Change voxel size and learning rate
python examples/scannet.py data.voxel_size=0.05 train.lr=0.01
```

### Configuration reference

**Paths:**

| Key                | Default             | Description                    |
| ------------------ | ------------------- | ------------------------------ |
| `paths.data_dir`   | `./data/scannet_3d` | Dataset directory              |
| `paths.output_dir` | `./results/`        | Output directory               |
| `paths.ckpt_path`  | `null`              | Checkpoint path to resume from |

**Training:**

| Key                 | Default | Description                  |
| ------------------- | ------- | ---------------------------- |
| `train.batch_size`  | `12`    | Training batch size          |
| `train.lr`          | `0.001` | AdamW learning rate          |
| `train.epochs`      | `100`   | Number of training epochs    |
| `train.step_size`   | `20`    | StepLR decay period (epochs) |
| `train.gamma`       | `0.7`   | StepLR decay factor          |
| `train.num_workers` | `8`     | DataLoader workers           |

**Test:**

| Key                | Default | Description        |
| ------------------ | ------- | ------------------ |
| `test.batch_size`  | `12`    | Test batch size    |
| `test.num_workers` | `4`     | DataLoader workers |

**Data:**

| Key                 | Default | Description                           |
| ------------------- | ------- | ------------------------------------- |
| `data.num_classes`  | `20`    | Number of semantic classes            |
| `data.voxel_size`   | `0.02`  | Voxelization resolution (meters)      |
| `data.ignore_index` | `255`   | Label index to ignore in loss/metrics |

**Model:**

| Key                  | Default                | Description                                                  |
| -------------------- | ---------------------- | ------------------------------------------------------------ |
| `model._target_`     | `mink_unet.MinkUNet18` | Model class to instantiate                                   |
| `model.in_channels`  | `3`                    | Input feature channels (RGB)                                 |
| `model.out_channels` | `20`                   | Output channels (num classes)                                |
| `model.in_type`      | `voxel`                | Input type (`voxel` wraps model with `PointToSparseWrapper`) |

**General:**

| Key         | Default | Description                     |
| ----------- | ------- | ------------------------------- |
| `device`    | `cuda`  | Device                          |
| `use_wandb` | `false` | Enable Weights & Biases logging |
| `seed`      | `42`    | Random seed                     |

## Expected output

Each epoch prints a progress bar followed by test-set evaluation with
accuracy and mean IoU:

```
Train Epoch: 1 Loss:  2.143: 100%|██████████| 104/104
Test set: Average loss:  1.8234, Accuracy:  42.15%, mIoU:  18.73%
```

After 100 epochs with default settings, expect roughly:

- **Overall accuracy**: ~75-80%
- **mIoU**: ~55-65%

Results will vary with augmentation, model choice, and voxel size. This
example is intended as a starting point, not a benchmark-tuned recipe.
