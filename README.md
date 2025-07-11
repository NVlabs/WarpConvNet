# WarpConvNet

## Overview

WarpConvNet is a high-performance library for 3D deep learning, built on NVIDIA's Warp framework. It provides efficient implementations of:

- Point cloud processing
- Sparse voxel convolutions
- Attention mechanisms for 3D data
- Geometric operations and transformations

## Installation

Recommend using [`uv`](https://docs.astral.sh/uv/) to install the dependencies.

```bash
# Install PyTorch first (specify your CUDA version)
export CUDA=cu128  # For CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA}

# Install WarpConvNet with CUDA-specific dependencies (choose one):
# For CUDA 12.8:
pip install warpconvnet[torch270-cuda128]

# Install Flash Attention (requires special installation)
pip install flash-attn --no-build-isolation

# Alternative: Install dependencies separately
pip install cupy-cuda12x  # use cupy-cuda11x for CUDA 11.x
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install flash-attn --no-build-isolation
pip install warpconvnet

# Or install from source
git clone https://github.com/NVlabs/WarpConvNet.git
cd WarpConvNet
git submodule update --init 3rdparty/cutlass
pip install .
```

Available optional dependency groups:

- `warpconvnet[dev]`: Development tools (pytest, coverage, pre-commit)
- `warpconvnet[docs]`: Documentation building tools
- `warpconvnet[models]`: Additional dependencies for model training (wandb, hydra, etc.)
- `warpconvnet[torch270-cuda128]`: torch-scatter and cupy for torch 2.7.0 and CUDA 12.8

## Directory Structure

```
./
├── 3rdparty/            # Third-party dependencies
│   └── cutlass/         # CUDA kernels
├── docker/              # Docker build files
│   ├── build.sh
│   └── Dockerfile
├── docs/                # Documentation sources
├── examples/            # Example applications
├── scripts/             # Development utilities
│   ├── build_docs.py
│   ├── dir_struct.sh
│   └── serve_docs.py
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
    ├── models/          # Sample models
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
# Build docs
python scripts/build_docs.py

# Serve locally
python scripts/serve_docs.py
```

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
