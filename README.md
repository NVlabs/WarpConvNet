# WarpConvNet

## Overview

WarpConvNet is a high-performance library for 3D deep learning, built on NVIDIA's Warp framework. It provides efficient implementations of:

- Point cloud processing
- Sparse voxel convolutions
- Attention mechanisms for 3D data
- Geometric operations and transformations

## Installation

```bash
# Specify the CUDA version
export CUDA=cu121  # For CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+${CUDA}.html
pip install warpconvnet
```

Available optional dependency groups:
- `warpconvnet[dev]`: Development tools (pytest, coverage, pre-commit)
- `warpconvnet[docs]`: Documentation building tools
- `warpconvnet[models]`: Additional dependencies for model training (wandb, hydra, etc.)

## Directory Structure

```
./
├── docker/             # Docker build files
│   ├── build.sh
│   └── Dockerfile
├── examples/           # Example applications
│   ├── mnist.py
│   └── modelnet.py
├── scripts/            # Development utilities
│   ├── build_docs.py
│   ├── dir_struct.sh
│   └── serve_docs.py
├── tests/              # Test suite
│   ├── base/           # Core functionality tests
│   ├── coords/         # Coordinate operation tests
│   ├── features/       # Feature processing tests
│   ├── nn/             # Neural network tests
│   └── types/          # Geometry type tests
└── warpconvnet/        # Main package
    ├── geometry/       # Geometric operations
    │   ├── base/       # Core definitions
    │   ├── coords/     # Coordinate operations
    │   ├── features/   # Feature operations
    │   └── types/      # Geometry types
    ├── nn/             # Neural networks
    │   ├── functional/ # Neural network functions
    │   └── modules/    # Neural network modules
    ├── ops/            # Basic operations
    └── utils/          # Utility functions
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
cd warpconvnet/docker
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
