# WarpConvNet

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="#installation"><img alt="pip install" src="https://img.shields.io/badge/pip%20install-warpconvnet-blue?logo=pypi&logoColor=white"></a>
  <a href="https://nvlabs.github.io/WarpConvNet/"><img alt="Docs" src="https://img.shields.io/badge/Docs-Website-blue?logo=mkdocs"></a>
  <a href="https://github.com/NVlabs/WarpConvNet/actions/workflows/docs.yml"><img alt="Docs Build" src="https://github.com/NVlabs/WarpConvNet/actions/workflows/docs.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-green"></a>
  <a href="https://developer.nvidia.com/cuda-zone"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-Enabled-76B900?logo=nvidia&logoColor=white"></a>
</p>

## Overview

WarpConvNet is a high-performance library for 3D deep learning, built on NVIDIA's CUDA framework. It provides efficient implementations of:

- Point cloud processing
- Sparse voxel convolutions
- Attention mechanisms for 3D data
- Geometric operations and transformations

### Minimal example (ModelNet-style)

```python
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
from warpconvnet.ops.reductions import REDUCTIONS

point_conv = Sequential(
    PointConv(24, 64, neighbor_search_args=RealSearchConfig("knn", knn_k=16)),
    nn.LayerNorm(64),
    nn.ReLU(),
)
sparse_conv = Sequential(
    SparseConv3d(64, 128, kernel_size=3, stride=2),
    nn.ReLU(),
)

coords: Float[Tensor, "B N 3"]  # noqa: F821
pc: Points = Points.from_list_of_coordinates(coords, encoding_channels=8, encoding_range=1)
pc = point_conv(pc)
vox: Voxels = pc.to_voxels(reduction=REDUCTIONS.MEAN, voxel_size=0.05)
vox = sparse_conv(vox)
dense: Tensor = vox.to_dense(channel_dim=1, min_coords=(-5, -5, -5), max_coords=(2, 2, 2))
# feed `dense` to your 3D CNN head for classification
```

See `examples/modelnet.py` for a full training script.

## Installation

Recommend using [`uv`](https://docs.astral.sh/uv/) to install the dependencies. When using `uv`, prepend with `uv pip install ...`.

```bash
# Install PyTorch first (specify your CUDA version)
export CUDA=cu128  # For CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA}

# Install core dependencies
pip install build ninja
pip install cupy-cuda12x  # use cupy-cuda11x for CUDA 11.x
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install flash-attn --no-build-isolation

# Install warpconvnet from source
git clone https://github.com/NVlabs/WarpConvNet.git
cd WarpConvNet
git submodule update --init 3rdparty/cutlass
pip install .

# If this fails, please create an issue on https://github.com/NVlabs/WarpConvNet/issues and try running the following commands:
cd WarpConvNet
# Option 1
python setup.py build_ext --inplace
# Option 2
pip install -e . --no-deps
```

Available optional dependency groups:

- `warpconvnet[dev]`: Development tools (pytest, coverage, pre-commit)
- `warpconvnet[docs]`: Documentation building tools
- `warpconvnet[models]`: Additional dependencies for model training (wandb, hydra, etc.)

## Directory Structure

```
./
├── 3rdparty/            # Third-party dependencies
│   └── cutlass/         # CUDA kernels
├── docker/              # Docker build files
├── docs/                # Documentation sources
├── examples/            # Example applications
├── scripts/             # Development utilities
├── tests/               # Test suite
│   ├── base/            # Core functionality tests
│   ├── coords/          # Coordinate operation tests
│   ├── features/        # Feature processing tests
│   ├── nn/              # Neural network tests
│   ├── csrc/            # C++/CUDA test utilities
│   └── types/           # Geometry type tests
└── warpconvnet/         # Main package
    ├── csrc/            # C++/CUDA extensions
    ├── dataset/         # Dataset utilities
    ├── geometry/        # Geometric operations
    │   ├── base/        # Core definitions
    │   ├── coords/      # Coordinate operations
    │   ├── features/    # Feature operations
    │   └── types/       # Geometry types
    ├── nn/              # Neural networks
    │   ├── functional/  # Neural network functions
    │   └── modules/     # Neural network modules
    ├── ops/             # Basic operations
    └── utils/           # Utility functions
```

For complete directory structure, run `bash scripts/dir_struct.sh`.

## Quick Start

### ModelNet Classification

```bash
python examples/modelnet.py
```

### ScanNet Semantic Segmentation

```bash
pip install warpconvnet[models]
cd warpconvnet/models
python examples/scannet.py train.batch_size=12 model=mink_unet
```

## Docker Usage

Build and run with GPU support:

```bash
# Build container
cd docker
docker build -t warpconvnet .

# Run container
docker run --gpus all \
    --shm-size=32g \
    -it \
    -v "/home/${USER}:/root" \
    -v "$(pwd):/workspace" \
    warpconvnet:latest
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/nn/
pytest tests/coords/

# Run with benchmarks
pytest tests/ --benchmark-only
```

### Building Documentation

```bash
# Install documentation dependencies
uv pip install -r docs/requirements.txt

# Build docs
mkdocs build -f mkdocs-readthedocs.yml

# Serve locally
mkdocs serve -f mkdocs-readthedocs.yml
```

📖 **Documentation**: [https://nvlabs.github.io/WarpConvNet/](https://nvlabs.github.io/WarpConvNet/)

The documentation is automatically built and deployed to GitHub Pages on every push to the main branch.

## License

Apache 2.0

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{warpconvnet2025,
  author = {Chris Choy and NVIDIA Research},
  title = {WarpConvNet: High-Performance 3D Deep Learning Library},
  year = {2025},
  publisher = {NVIDIA Corporation},
  howpublished = {\url{https://github.com/NVlabs/warpconvnet}}
}
```
