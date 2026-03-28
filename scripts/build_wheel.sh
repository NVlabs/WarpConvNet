#!/bin/bash
# Build a single warpconvnet wheel for the current torch+CUDA environment.
#
# Usage:
#   ./scripts/build_wheel.sh            # auto-detect torch+CUDA version
#   ./scripts/build_wheel.sh torch2.10cu128   # explicit local version tag
#
# The wheel is placed in dist/repaired/ with the local version tag embedded.
# Version is derived from git tags via setuptools-scm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Auto-detect torch and CUDA versions if no argument given
if [ $# -ge 1 ]; then
    LOCAL_VERSION="$1"
else
    TORCH_VERSION=$(python -c "import torch; v=torch.__version__.split('+')[0]; parts=v.split('.'); print(f'{parts[0]}.{parts[1]}')")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "unknown")
    LOCAL_VERSION="torch${TORCH_VERSION}cu${CUDA_VERSION}"
    echo "Auto-detected: ${LOCAL_VERSION}"
fi

# Get base version from git tags via setuptools-scm (strip local segment if present)
BASE_VERSION=$(python -m setuptools_scm 2>/dev/null | cut -d'+' -f1 || echo "0.0.0")
export SETUPTOOLS_SCM_PRETEND_VERSION="${BASE_VERSION}+${LOCAL_VERSION}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0a}"

echo "Building wheel version: ${SETUPTOOLS_SCM_PRETEND_VERSION}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

# Clean previous builds
rm -rf build/ dist/ ./*.egg-info/

# Build
python setup.py bdist_wheel

echo ""
echo "Built wheel:"
ls -lh dist/*.whl

# Try auditwheel repair if available
if command -v auditwheel &>/dev/null; then
    echo ""
    echo "Running auditwheel repair..."
    mkdir -p dist/repaired
    auditwheel repair dist/*.whl \
        --plat manylinux_2_28_x86_64 \
        -w dist/repaired/ \
        --exclude libtorch.so \
        --exclude libtorch_cpu.so \
        --exclude libtorch_cuda.so \
        --exclude libtorch_python.so \
        --exclude libc10.so \
        --exclude libc10_cuda.so \
        --exclude libcudart.so \
        --exclude libcublas.so \
        --exclude libcublasLt.so \
        --exclude libcuda.so || {
        echo "auditwheel repair failed, copying original wheel"
        cp dist/*.whl dist/repaired/
    }
    echo ""
    echo "Repaired wheel:"
    ls -lh dist/repaired/*.whl
else
    echo ""
    echo "auditwheel not found, skipping repair. Install with: pip install auditwheel patchelf"
    mkdir -p dist/repaired
    cp dist/*.whl dist/repaired/
fi

echo ""
echo "Done. Wheel ready at: dist/repaired/"
