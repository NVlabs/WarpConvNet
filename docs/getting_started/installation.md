# Installation

WarpConvNet requires Python >= 3.9 and a working CUDA toolchain. The library can be installed from source.

```bash
# Install PyTorch for your CUDA version
export CUDA=cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA}

# Install dependencies and WarpConvNet
pip install build ninja
pip install cupy-cuda12x
pip install git+https://github.com/rusty1s/pytorch_scatter.git
pip install flash-attn --no-build-isolation
pip install .
```
