# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from setuptools import setup

import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Get CUDA toolkit path
def get_cuda_path():
    try:
        # Try to get CUDA path from nvcc
        result = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc_path = result.stdout.strip()
            return os.path.dirname(os.path.dirname(nvcc_path))
    except Exception as e:
        print(f"Error getting CUDA path: {e}")
        pass

    # Fallback to common CUDA installation paths
    for path in ["/usr/local/cuda", "/opt/cuda", "/usr/local/cuda-12", "/usr/local/cuda-11"]:
        if os.path.exists(path):
            return path

    return "/usr/local/cuda"


cuda_home = get_cuda_path()
print(f"Using CUDA path: {cuda_home}")

# Get the absolute path of the workspace
workspace_dir = os.path.dirname(os.path.abspath(__file__))

# Define include directories
include_dirs = [
    torch.utils.cpp_extension.include_paths()[0],  # PyTorch includes
    torch.utils.cpp_extension.include_paths()[1],  # PyTorch CUDA includes
    os.path.join(workspace_dir, "3rdparty/cutlass/include"),  # CUTLASS includes
    os.path.join(workspace_dir, "3rdparty/cutlass/tools/util/include"),  # CUTLASS util includes
    os.path.join(workspace_dir, "warpconvnet/csrc/include"),  # Project includes
    f"{cuda_home}/include",  # CUDA includes
]

# Define library directories
library_dirs = [
    f"{cuda_home}/lib64",
    torch.utils.cpp_extension.library_paths()[0],
]

# Define libraries
libraries = [
    "cudart",
    "cublas",
]

# Define compile arguments
cxx_args = [
    "-std=c++17",
    "-O3",
    "-DWITH_CUDA",
    "-Wno-changes-meaning",
    "-fpermissive",
]

nvcc_args = [
    "-std=c++17",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DWITH_CUDA",
    "-arch=sm_80",  # Adjust based on your GPU architecture
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "--allow-unsupported-compiler",
    "--compiler-options=-fpermissive,-w",
]

# Check DISABLE_BFLOAT16
if os.environ.get("DISABLE_BFLOAT16", "0") == "1":
    print("Disabling BFLOAT16 support")
    cxx_args.append("-DDISABLE_BFLOAT16")
    nvcc_args.append("-DDISABLE_BFLOAT16")

# Check DEBUG flag
if os.environ.get("DEBUG", "0") == "1":
    print("Enabling DEBUG mode")
    cxx_args.append("-DDEBUG")
    nvcc_args.append("-DDEBUG")

# Define the extension
ext_modules = [
    CUDAExtension(
        name="warpconvnet._C",
        sources=[
            "warpconvnet/csrc/warpconvnet_pybind.cpp",
            "warpconvnet/csrc/cutlass_gemm_gather_scatter.cu",
            "warpconvnet/csrc/cutlass_gemm_gather_scatter_sm80_fp32.cu",
            "warpconvnet/csrc/cub_sort.cu",
            "warpconvnet/csrc/voxel_mapping_kernels.cu",
            "warpconvnet/csrc/implicit_fma_kernel.cu",
            "warpconvnet/csrc/implicit_reduction.cu",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        },
        language="c++",
    )
]

setup(
    name="warpconvnet",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension,
    },
    zip_safe=False,
    python_requires=">=3.8",
)
