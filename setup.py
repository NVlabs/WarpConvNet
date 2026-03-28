# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from setuptools import setup

# Allow sdist generation without torch installed.
# When torch is not available, setup() runs with no ext_modules (source-only).
try:
    import torch
    import torch.utils.cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    version_str = getattr(torch, "__version__", "")
    if isinstance(version_str, str) and "cpu" in version_str.lower():
        print(
            f"ERROR: warpconvnet requires a CUDA-enabled PyTorch build; detected CPU-only PyTorch ({version_str}). "
            "Please install a CUDA build of PyTorch.",
            file=sys.stderr,
        )
        raise SystemExit(1)

workspace_dir = os.path.dirname(os.path.abspath(__file__))

# Defaults for sdist-only mode (no torch)
ext_modules = []
cmdclass = {}

if _HAS_TORCH:
    # ---------------------------------------------------------------------------
    # CUDA extension build (requires torch + CUDA toolkit)
    # ---------------------------------------------------------------------------

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

    # Define include directories
    include_dirs = [
        torch.utils.cpp_extension.include_paths()[0],  # PyTorch includes
        torch.utils.cpp_extension.include_paths()[1],  # PyTorch CUDA includes
        os.path.join(workspace_dir, "3rdparty/cutlass/include"),  # CUTLASS includes
        os.path.join(
            workspace_dir, "3rdparty/cutlass/tools/util/include"
        ),  # CUTLASS util includes
        os.path.join(workspace_dir, "warpconvnet/csrc/include"),  # Project includes
        os.path.join(
            workspace_dir, "3rdparty/cutlass/examples/common"
        ),  # CUTLASS examples (gather_tensor.hpp)
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
        "cuda",  # CUDA driver API (cuTensorMapEncodeTiled for SM90 TMA)
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
        # Intentionally omit -gencode/-arch flags. PyTorch will inject these
        # based on TORCH_CUDA_ARCH_LIST or its internal defaults.
        "--allow-unsupported-compiler",
        "--compiler-options=-fpermissive,-w",
    ]

    # Informative log about TORCH_CUDA_ARCH_LIST usage
    cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if cuda_arch_list:
        print(f"TORCH_CUDA_ARCH_LIST detected: {cuda_arch_list}")
    else:
        print("TORCH_CUDA_ARCH_LIST not set; using PyTorch default arch list")

    # Explicitly generate -gencode flags from TORCH_CUDA_ARCH_LIST.
    # We must do this ourselves because adding any explicit -gencode (e.g. sm_90a)
    # prevents PyTorch's BuildExtension from injecting its own gencode flags.
    _has_sm100_target = False
    _has_sm90_target = False
    _has_sm80_target = False
    _arch_values = []
    _MIN_ARCH = 7.0  # Minimum supported architecture (Volta)

    if cuda_arch_list:
        for arch in cuda_arch_list.replace(",", " ").replace(";", " ").split():
            arch = arch.strip().rstrip("+")
            try:
                _arch_values.append(float(arch))
            except ValueError:
                pass
    else:
        # When no explicit arch list, detect from current GPU
        try:
            cap = torch.cuda.get_device_capability()
            _arch_values.append(float(f"{cap[0]}.{cap[1]}"))
            if cap[0] >= 10:
                _has_sm100_target = True
            if cap[0] >= 9:
                _has_sm90_target = True
            if cap[0] >= 8:
                _has_sm80_target = True
        except Exception:
            pass

    # Filter out architectures below minimum
    _skipped = [v for v in _arch_values if v < _MIN_ARCH]
    _arch_values = [v for v in _arch_values if v >= _MIN_ARCH]
    if _skipped:
        print(
            f"WARNING: Skipping unsupported architectures < sm_{int(_MIN_ARCH * 10)}: "
            f"{', '.join(f'sm_{int(v * 10)}' for v in _skipped)}"
        )
    if not _arch_values and not any(v >= 9.0 for v in _skipped):
        # No valid arch remaining and no sm_90 — default to sm_70
        _arch_values = [7.0]
        print("No supported architecture found; defaulting to sm_70")

    # Determine SM80+ presence first (needed to decide whether SM7X gencode is safe)
    for arch_val in _arch_values:
        if arch_val >= 10.0:
            _has_sm100_target = True
            _has_sm90_target = True
            _has_sm80_target = True
        elif arch_val >= 9.0:
            _has_sm90_target = True
            _has_sm80_target = True
        elif arch_val >= 8.0:
            _has_sm80_target = True

    # Generate gencode flags for all requested architectures.
    # When SM80+ targets are present, skip SM7X gencode: CUTLASS .cu files are compiled
    # with all gencode flags, and CUTLASS cp.async / tensor-core MMA fail on compute_7X.
    _skipped_sm7x = []
    _has_ptx_target = False
    for arch_val in _arch_values:
        arch_int = int(arch_val * 10)  # e.g. 7.0 -> 70, 8.0 -> 80, 9.0 -> 90, 10.0 -> 100
        if arch_val >= 10.0 and arch_val < 10.0 + 0.01:
            pass  # sm_100a added separately below
        elif arch_val >= 10.0:
            # Future arch (e.g. 12.0): emit PTX for forward compatibility
            nvcc_args.append(f"-gencode=arch=compute_{arch_int},code=compute_{arch_int}")
            _has_ptx_target = True
            print(f"Adding PTX gencode for SM{arch_int} (forward compatibility)")
        elif arch_val >= 9.0:
            pass  # sm_90a added separately below
        elif arch_val >= 8.0:
            nvcc_args.append(f"-gencode=arch=compute_{arch_int},code=sm_{arch_int}")
            print(f"Adding gencode for SM{arch_int}")
        elif _has_sm80_target:
            # Skip SM7X gencode when SM80+ is also targeted (CUTLASS incompatible)
            _skipped_sm7x.append(arch_val)
        else:
            nvcc_args.append(f"-gencode=arch=compute_{arch_int},code=sm_{arch_int}")
            print(f"Adding gencode for SM{arch_int}")

    if _skipped_sm7x:
        print(
            f"WARNING: Skipping SM7X gencode ({', '.join(f'sm_{int(v * 10)}' for v in _skipped_sm7x)}) "
            f"because SM80+ targets are present (CUTLASS requires sm_80+ for all gencode targets)"
        )

    # SM80+ CUTLASS support (cp.async, tensor core MMA)
    if _has_sm80_target:
        cxx_args.append("-DWARPCONVNET_SM80_ENABLED=1")
        nvcc_args.append("-DWARPCONVNET_SM80_ENABLED=1")
        print("Adding WARPCONVNET_SM80_ENABLED (CUTLASS cp.async, tensor core MMA)")

    # For SM90 (Hopper) WGMMA support, use sm_90a (not just sm_90).
    # sm_90a enables __CUDA_ARCH_FEAT_SM90_ALL needed for WGMMA instructions.
    if _has_sm90_target:
        nvcc_args.append("-gencode=arch=compute_90a,code=sm_90a")
        cxx_args.append("-DWARPCONVNET_SM90_ENABLED=1")
        nvcc_args.append("-DWARPCONVNET_SM90_ENABLED=1")
        print("Adding SM90a (Hopper WGMMA) gencode flag and WARPCONVNET_SM90_ENABLED")

    # For SM100 (Blackwell) support, use sm_100a (like sm_90a for Hopper).
    # sm_100a enables __CUDA_ARCH_FEAT_SM100_ALL needed for Blackwell-specific features.
    if _has_sm100_target:
        nvcc_args.append("-gencode=arch=compute_100a,code=sm_100a")
        cxx_args.append("-DWARPCONVNET_SM100_ENABLED=1")
        nvcc_args.append("-DWARPCONVNET_SM100_ENABLED=1")
        print("Adding SM100a (Blackwell) gencode flag and WARPCONVNET_SM100_ENABLED")

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
                "warpconvnet/csrc/bindings/gemm_bindings.cpp",
                "warpconvnet/csrc/bindings/fma_bindings.cpp",
                "warpconvnet/csrc/bindings/utils_bindings.cpp",
                "warpconvnet/csrc/cutlass_gemm_gather_scatter.cu",
                "warpconvnet/csrc/cutlass_cute_gemm_gather_scatter.cu",
                "warpconvnet/csrc/cutlass_cute_gemm_mask.cu",
                "warpconvnet/csrc/cutlass_cute_gemm_staged.cu",
                "warpconvnet/csrc/cutlass_cute_gemm_sm90.cu",  # SM90 (Hopper) WGMMA GEMM
                "warpconvnet/csrc/cutlass_gemm_gather_scatter_sm80_fp32.cu",
                "warpconvnet/csrc/cub_sort.cu",
                "warpconvnet/csrc/voxel_mapping_kernels.cu",
                "warpconvnet/csrc/implicit_fma_kernel.cu",
                "warpconvnet/csrc/implicit_reduction.cu",
                "warpconvnet/csrc/segmented_arithmetic.cu",
                "warpconvnet/csrc/implicit_gemm.cu",
                "warpconvnet/csrc/mask_implicit_gemm.cu",
                "warpconvnet/csrc/mask_data_kernels.cu",
                "warpconvnet/csrc/implicit_gemm_split_k.cu",
                "warpconvnet/csrc/bindings/sampling_bindings.cpp",
                "warpconvnet/csrc/farthest_point_sampling.cu",
                "warpconvnet/csrc/bindings/coords_bindings.cpp",
                "warpconvnet/csrc/coords_launch.cu",
                "warpconvnet/csrc/hashmap_kernels.cu",
                "warpconvnet/csrc/discrete_kernels.cu",
                "warpconvnet/csrc/morton_code.cu",
                "warpconvnet/csrc/find_first_gt_bsearch.cu",
                "warpconvnet/csrc/radius_search_kernels.cu",
                "warpconvnet/csrc/fused_kernel_map.cu",
                "warpconvnet/csrc/cutlass_cute_gemm_grouped_sm90.cu",
                "warpconvnet/csrc/window_grouping_kernels.cu",
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

    cmdclass = {"build_ext": BuildExtension}
else:
    print("PyTorch not found — building source distribution only (no CUDA extensions).")

# Local version suffix for pre-built wheels (e.g. "+torch2.10cu128").
# pyproject.toml reads version from VERSION.md (dynamic), so we patch the file
# in-place and restore it after setup() to inject the local tag.
_local_version = os.environ.get("WARPCONVNET_LOCAL_VERSION", "")
_version_file = os.path.join(workspace_dir, "VERSION.md")
_original_version = None

if _local_version:
    with open(_version_file) as f:
        _original_version = f.read()
    _tagged_version = f"{_original_version.strip()}+{_local_version}"
    with open(_version_file, "w") as f:
        f.write(_tagged_version + "\n")
    print(f"Using local version: {_tagged_version}")

try:
    setup(
        name="warpconvnet",
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        zip_safe=False,
        python_requires=">=3.8",
    )
finally:
    # Restore VERSION.md so the working tree stays clean
    if _original_version is not None:
        with open(_version_file, "w") as f:
            f.write(_original_version)
