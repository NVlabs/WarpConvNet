# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.4 with CUDA support
- NVIDIA GPU (Ampere or newer recommended)
- CUDA toolkit (nvcc) matching your PyTorch CUDA version
- `ninja` (for C++/CUDA JIT compilation)

## Install from source

```bash
# 1. Install PyTorch for your CUDA version (example: CUDA 12.8)
export CUDA=cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA}

# 2. Install build dependencies
pip install build ninja

# 3. Install WarpConvNet and its dependencies
pip install cupy-cuda12x
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install flash-attn --no-build-isolation
pip install .
```

!!! note "CUDA version string"
The `cu128` version string follows PyTorch's convention: `cu` + major + minor
digits without a dot. For example, CUDA 12.1 → `cu121`, CUDA 12.8 → `cu128`.
Check [pytorch.org](https://pytorch.org/get-started/locally/) for available versions.

## Verify the installation

```bash
python -c "import warpconvnet; print('WarpConvNet installed successfully')"
```

## Optional: model extras

To run the ScanNet example and other model-training scripts, install with the
`models` extra:

```bash
pip install ".[models]"
```

## Troubleshooting

### cuBLAS fp16 correctness issue

Some versions of `nvidia-cublas-cu12` shipped with PyTorch produce incorrect
results for fp16/bf16 matrix multiplications. WarpConvNet prints a warning at
import time if this affects your environment. See the
[Troubleshooting](../user_guide/troubleshooting.md) guide for
details and fixes.
