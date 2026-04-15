# Compilation Guide

WarpConvNet ships CUDA C++ extensions that must be compiled against your
specific PyTorch and CUDA versions. This page covers all compilation methods
and common issues.

## Prerequisites

- **NVIDIA GPU** with compute capability >= 7.0 (Volta or newer)
- **CUDA toolkit** with `nvcc` matching your PyTorch CUDA version
- **PyTorch** with CUDA support (CPU-only builds are not supported)
- **ninja** build system (for parallel compilation)
- **C++17** compatible compiler (GCC >= 7, Clang >= 5)

## Method 1: Pre-built wheels (no compilation)

The fastest option. Pre-built wheels are available for common
PyTorch + CUDA combinations:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install "warpconvnet==1.5.0+torch2.10cu128" \
    --find-links https://github.com/NVlabs/WarpConvNet/releases/latest/download/
```

Replace the version string to match your PyTorch + CUDA combo.
See [available wheels](https://github.com/NVlabs/WarpConvNet/releases).

## Method 2: pip install from source

Builds the CUDA extension automatically. Takes ~10 minutes.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install build ninja

# From PyPI
pip install warpconvnet

# Or from a local clone
git clone https://github.com/NVlabs/WarpConvNet.git
cd WarpConvNet
git submodule update --init 3rdparty/cutlass
pip install -e . --no-build-isolation
```

### Targeting specific GPU architectures

By default, WarpConvNet auto-detects your current GPU and compiles only for
that architecture. To target specific architectures (e.g., for deployment
across different GPUs):

```bash
export TORCH_CUDA_ARCH_LIST="8.0 8.9"  # A100 + RTX 6000 Ada
pip install -e . --no-build-isolation
```

Common architecture codes:

| GPU family                            | Compute capability |
| ------------------------------------- | ------------------ |
| Volta (V100)                          | 7.0                |
| Turing (RTX 20xx)                     | 7.5                |
| Ampere (A100, RTX 30xx)               | 8.0, 8.6           |
| Ada Lovelace (RTX 40xx, RTX 6000 Ada) | 8.9                |
| Hopper (H100)                         | 9.0                |
| Blackwell (B200)                      | 10.0               |

## Method 3: build_ext (in-place compilation only)

Use `setup.py build_ext --inplace` when you want to compile the C++ extension
without a full pip install. This is useful for development iteration — it
builds `warpconvnet/_C.*.so` directly into the source tree.

```bash
git clone https://github.com/NVlabs/WarpConvNet.git
cd WarpConvNet
git submodule update --init 3rdparty/cutlass
pip install build ninja  # build dependencies

python setup.py build_ext --inplace
```

### setuptools-scm version detection

WarpConvNet uses **setuptools-scm** to derive the package version from git
tags (e.g., tag `v1.5.0` produces version `1.5.0`). This can fail in several
situations:

- **Detached HEAD** (e.g., during `git rebase`)
- **Shallow clones** (`git clone --depth 1`)
- **No git tags** in the clone
- **Worktrees** without tag visibility

When version detection fails, you'll see an error like:

```
LookupError: setuptools-scm was unable to detect version
```

**Fix:** Set the `SETUPTOOLS_SCM_PRETEND_VERSION` environment variable to
bypass git-based version detection:

```bash
# build_ext only (development)
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python setup.py build_ext --inplace

# Full editable install
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -e . --no-build-isolation

# Clean rebuild (removes stale build artifacts)
rm -rf build/
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python setup.py build_ext --inplace
```

The version string is only used for metadata — it doesn't affect
functionality. Use `0.0.0` for local development or any valid version string.

## Verify the build

After compilation, verify the extension loads correctly:

```bash
python -c "import warpconvnet; print('OK')"
```

If the C++ extension fails to load, you'll see an `ImportError` with details
about the missing symbol or ABI mismatch.

## Troubleshooting

### `ninja: build stopped: subcommand failed`

The actual error is usually above the ninja message. Scroll up to find the
`nvcc` or `g++` error. Common causes:

- **CUDA version mismatch**: `nvcc --version` must match the CUDA version
  your PyTorch was built with. Check with
  `python -c "import torch; print(torch.version.cuda)"`.
- **Missing CUTLASS submodule**: Run `git submodule update --init 3rdparty/cutlass`.

### `undefined symbol` at import time

The compiled extension was built against a different PyTorch version than
the one currently installed. Rebuild:

```bash
rm -rf build/
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python setup.py build_ext --inplace
```

### Slow compilation

Compilation builds 30+ CUDA source files. To speed it up:

- Install `ninja` (`pip install ninja`) — enables parallel compilation.
- Limit target architectures:
  `export TORCH_CUDA_ARCH_LIST="8.9"` (your GPU only).
- Use `ccache` if available.

### Multiple Python versions

Each Python version needs its own compiled `.so` file
(e.g., `_C.cpython-311-x86_64-linux-gnu.so`). If you switch Python versions,
rebuild the extension.
