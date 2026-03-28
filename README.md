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

## Sparse Convolution Auto-Tuning

WarpConvNet automatically benchmarks CUDA kernel algorithms on the first forward/backward pass and caches results to `~/.cache/warpconvnet/`. Subsequent runs reuse cached results with no overhead.

Sparse convolution decomposes into two distinct GEMM operations:

- **AB gather-scatter** (forward and dgrad): `D[scatter] = A[gather] @ B` — fused gather-GEMM-scatter
- **AtB gather-gather** (wgrad): `D = A[gather]^T @ B[gather]` — reduction GEMM

Each operation is auto-tuned independently with its own candidate set and cache namespace.

**The first iteration will be slower** while auto-tuning runs. You will see log messages like:

```
Auto-tuning sparse convolution algorithms. The first few iterations will be slow...
Auto-tune forward complete: mask_implicit_gemm (mma_tile=3) — 0.21ms
```

### Algorithm Selection Modes

The auto-tuner supports three candidate selection modes, controlled by environment variables:

| Mode                 | AB candidates | AtB candidates | Description                                                                                     |
| -------------------- | ------------: | -------------: | ----------------------------------------------------------------------------------------------- |
| **`auto`** (default) |           4-5 |            3-4 | Fastest auto-tune. Only the dominant winner per channel/N region. Recommended for training.     |
| **`trimmed`**        |          5-11 |            6-9 | Moderate search. Includes runner-ups for edge cases. Default for `populate_benchmark_cache.py`. |
| **`all`**            |            23 |             35 | Exhaustive. Benchmarks every algorithm. Use for new hardware validation or after code changes.  |

### Environment Variables

| Variable                          | Default                | Description                                 |
| --------------------------------- | ---------------------- | ------------------------------------------- |
| `WARPCONVNET_FWD_ALGO_MODE`       | `auto`                 | AB gather-scatter algorithm (forward/dgrad) |
| `WARPCONVNET_BWD_ALGO_MODE`       | `auto`                 | AtB gather-gather algorithm (wgrad)         |
| `WARPCONVNET_AUTOTUNE_LOG`        | `true`                 | Set to `false` to suppress auto-tuning logs |
| `WARPCONVNET_BENCHMARK_CACHE_DIR` | `~/.cache/warpconvnet` | Cache directory                             |

```bash
# Suppress auto-tuning logs
export WARPCONVNET_AUTOTUNE_LOG=false

# Pin a specific algorithm (skip auto-tuning entirely)
export WARPCONVNET_FWD_ALGO_MODE=mask_implicit_gemm

# Exhaustive search (slow, for benchmarking)
export WARPCONVNET_FWD_ALGO_MODE=all

# Benchmark only specific algorithms
export WARPCONVNET_FWD_ALGO_MODE="[mask_implicit_gemm,cutlass_implicit_gemm]"
```

Available algorithms: `explicit_gemm`, `implicit_gemm`, `cutlass_implicit_gemm`, `cute_implicit_gemm`, `cute_grouped`, `explicit_gemm_grouped`, `cutlass_grouped_hybrid`, `mask_implicit_gemm`.

### Pre-Populating the Cache

To skip auto-tuning entirely, pre-populate the cache with the `populate_benchmark_cache.py` script. Multi-GPU support is available via `--gpus`:

```bash
# Single GPU (uses trimmed mode by default)
python scripts/populate_benchmark_cache.py

# All GPUs in parallel
python scripts/populate_benchmark_cache.py --gpus all

# Exhaustive search on specific GPUs
python scripts/populate_benchmark_cache.py --gpus 0,1 --algo-mode all
```

See [Pre-Populate Benchmark Cache](https://nvlabs.github.io/WarpConvNet/user_guide/populate_benchmark_cache/) for details.

For algorithm backends and cache inspection, see the [Sparse Convolutions](https://nvlabs.github.io/WarpConvNet/user_guide/sparse_convolutions/) and [Inspecting the Benchmark Cache](https://nvlabs.github.io/WarpConvNet/user_guide/inspect_benchmark_cache/) documentation.

## Installation

Recommend using [`uv`](https://docs.astral.sh/uv/) to install the dependencies. When using `uv`, prepend with `uv pip install ...`.

### Pre-built wheels (recommended)

Pre-built wheels are available for common PyTorch + CUDA combinations on Linux x86_64. This is the fastest way to install — no compiler or CUDA toolkit needed.

```bash
# Install PyTorch first (specify your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install warpconvnet (pre-built wheel, matched to your torch + CUDA)
pip install warpconvnet --extra-index-url https://nvlabs.github.io/WarpConvNet/whl/
```

Available wheels:

| PyTorch | CUDA                | Python           |
| ------- | ------------------- | ---------------- |
| 2.10.x  | cu130, cu128, cu126 | 3.10, 3.11, 3.12 |
| 2.5.x   | cu124, cu121        | 3.10, 3.11, 3.12 |

### Install from PyPI (builds from source)

If no pre-built wheel matches your environment, you can install from PyPI. This builds the CUDA extensions from source (~10 minutes).

```bash
# Install PyTorch first (specify your CUDA version)
export CUDA=cu128  # For CUDA 12.8
export TORCH_CUDA_ARCH_LIST="8.9 8.0"  # A100 is 80, RTX 6000 Ada is 89

pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA}

# Install build dependencies
pip install build ninja

# Install warpconvnet (builds from source)
pip install warpconvnet
```

### Install from source (development)

```bash
# Install PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
pip install build ninja
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install flash-attn --no-build-isolation

# Clone and install
git clone https://github.com/NVlabs/WarpConvNet.git
cd WarpConvNet
git submodule update --init 3rdparty/cutlass
pip install -e . --no-build-isolation

# If this fails, please create an issue on https://github.com/NVlabs/WarpConvNet/issues and try:
# Option 1
python setup.py build_ext --inplace
# Option 2
pip install -e . --no-deps --no-build-isolation --force-reinstall
```

### Optional: Pre-Populate the Benchmark Cache

To eliminate first-run auto-tuning latency, you can pre-populate the cache for common configurations:

```bash
# Quick smoke test (~1 minute)
python scripts/populate_benchmark_cache.py --preset quick

# Full grid for production (364 configs — takes longer)
python scripts/populate_benchmark_cache.py
```

The cache file (`~/.cache/warpconvnet/benchmark_cache_generic.msgpack`) is GPU-architecture-specific and can be distributed to other machines with the same GPU type. See the [Pre-Populate Benchmark Cache](https://nvlabs.github.io/WarpConvNet/user_guide/populate_benchmark_cache/) guide for details.

### Optional dependency groups

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
mkdocs build

# Serve locally
mkdocs serve
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
