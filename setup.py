# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
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


def _git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace_dir, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


BUILD_COMMIT = _git_commit()
print(f"warpconvnet build commit: {BUILD_COMMIT}")


def _generate_warpgemm_codegen():
    """Optionally regenerate warpgemm-sourced artifacts (WARPGEMM_REGEN=1).

    Wcn ships standalone — the canonical safe files (gemm_mma_tiles.h,
    cute_gemm_config.h, mask_gemm_dispatch_table.inc) are committed under
    csrc/include/ and csrc/mask_gemm/. Selectively-committed kernel headers
    live in csrc/mask_gemm/include/. None of these require warpgemm to be
    importable for the build to succeed.

    When WARPGEMM_REGEN=1 is set AND warpgemm is importable, this function
    refreshes those committed snapshots in place by calling
    warpgemm.codegen.write_mask_to() into a temp dir and copying the safe
    files into their existing committed locations. Kernel header templates
    and warpgemm_*.cuh helper fragments contain warpgemm-internal IP and
    are NOT committed; the warpgemm-equipped path can write them directly
    into csrc/mask_gemm/include/ as a curated set tracked by name.
    """

    if os.environ.get("WARPGEMM_REGEN", "0") != "1":
        return

    csrc_dir = os.path.join(workspace_dir, "warpconvnet", "csrc")
    include_dir = os.path.join(csrc_dir, "include")
    mask_gemm_dir = os.path.join(csrc_dir, "mask_gemm")
    mask_gemm_include_dir = os.path.join(mask_gemm_dir, "include")
    offset_gemm_dir = os.path.join(csrc_dir, "offset_gemm")
    os.makedirs(mask_gemm_include_dir, exist_ok=True)
    os.makedirs(offset_gemm_dir, exist_ok=True)
    try:
        from warpgemm.codegen.offset_gemm import write_to as _write_offset_gemm
        from warpgemm.autotune import write_tile_metadata_to as _write_tile_metadata
        from warpgemm.codegen import write_mask_to as _write_mask
    except ImportError as exc:
        print(f"WARPGEMM_REGEN=1 set but warpgemm not importable: {exc}; skipping codegen")
        return
    offset_gemm_written = _write_offset_gemm(offset_gemm_dir)
    print(
        f"warpgemm offset_gemm codegen: wrote {len(offset_gemm_written)} files to "
        f"{offset_gemm_dir}"
    )
    tm_path = _write_tile_metadata(mask_gemm_dir)
    print(f"warpgemm tile_metadata codegen: wrote {tm_path}")

    # Refresh selectively-committed kernel headers (curated set already
    # present in csrc/mask_gemm/include/) and the safe canonical artifacts
    # (gemm_mma_tiles.h + cute_gemm_config.h → csrc/include/;
    # mask_gemm_dispatch_table.inc → csrc/mask_gemm/) by writing into a
    # temp dir and copying only the files that should be tracked.
    tracked_mask_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(mask_gemm_include_dir)
        if f.startswith("MaskGemm_") and f.endswith(".h")
    )
    if tracked_mask_names:
        # canonical=False: emit ONLY the requested MaskGemm_*.h (+ always-emitted
        # helper .cuh fragments). Without it, write_mask_to also dumps the shared
        # canonical SoT artifacts (gemm_mma_tiles.h, cute_gemm_config.h, cute_gemm
        # bodies, tile_enums, dispatch.inc) into mask_gemm/include/ as UNTRACKED
        # strays, which a later clean build resolves from the wrong -I dir and
        # fails on. Those canonical files are routed to their real homes by the
        # dedicated copy block below.
        mask_written = _write_mask(
            mask_gemm_include_dir, names=tracked_mask_names, canonical=False
        )
        print(
            f"warpgemm mask_gemm codegen: wrote {len(mask_written)} files to "
            f"{mask_gemm_include_dir}"
        )
    else:
        print(
            f"warpgemm mask_gemm codegen: no tracked MaskGemm_*.h under "
            f"{mask_gemm_include_dir}; skipping (initial seed must be done by hand)"
        )

    # Safe canonical files: gemm_mma_tiles.h + cute_gemm_config.h →
    # csrc/include/; mask_gemm_dispatch_table.inc → csrc/mask_gemm/.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as canonical_tmp:
        _write_mask(canonical_tmp)
        for fname, dst_dir in [
            ("gemm_mma_tiles.h", include_dir),
            ("cute_gemm_config.h", include_dir),
            ("cute_gemm_kernel.h", include_dir),
            ("cute_gemm_grouped_kernel.h", include_dir),
            ("mask_gemm_tile_enums.h", include_dir),
            ("mask_gemm_dispatch_table.inc", mask_gemm_dir),
        ]:
            src = os.path.join(canonical_tmp, fname)
            dst = os.path.join(dst_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"warpgemm canonical refresh: {fname} → {dst}")


def _patch_cutlass_cuda_host_adapter():
    """Make `#include <cuda.h>` unconditional in CUTLASS's cuda_host_adapter.hpp.

    CUTLASS 4.1's cuda_host_adapter.hpp conditionally includes <cuda.h>
    based on macros it sets earlier in the same file. Under CUDA 12.9 +
    nvcc 12.9 some compilation paths fail to define `cuuint32_t` /
    `cuuint64_t` before <cudaTypedefs.h> consumes them in forward
    declarations of `cuTensorMapEncodeIm2colWide_v12080`, leading to:

        cudaTypedefs.h(917): error: identifier "cuuint32_t" is undefined

    which then cascades into CUTLASS parse failures (Status undefined,
    cute::cute:: double-resolution, etc.).

    Fix: idempotent in-place edit of cuda_host_adapter.hpp to make the
    `#include <cuda.h>` unconditional, ensuring cuuint32_t is always
    defined before <cudaTypedefs.h> is reached.

    Apply via a marker comment so the patch is detected on subsequent
    builds and not re-applied.
    """
    path = os.path.join(
        workspace_dir, "3rdparty", "cutlass", "include", "cutlass", "cuda_host_adapter.hpp"
    )
    if not os.path.exists(path):
        return
    marker = "/* WCN: cuda.h pre-include patched */"
    try:
        with open(path) as f:
            content = f.read()
        if marker in content:
            return  # Already patched
        old_block = (
            "// Include <cuda.h> for CUDA Driver API calls if any of these capabilities are enabled.\n"
            "#if defined(CUDA_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED) ||        \\\n"
            "    defined(CUDA_HOST_ADAPTER_TENSORMAP_ENABLED)\n"
            "\n"
            "#include <cuda.h>\n"
            "\n"
            "#endif // defined(CUDA_HOST_ADAPTER_LAUNCH_ATTRIBUTES_ENABLED) ||\n"
            "       // defined(CUDA_HOST_ADAPTER_TENSORMAP_ENABLED)"
        )
        new_block = (
            f"// {marker[3:-3]}\n"
            "// Unconditional <cuda.h> include: guarantees cuuint32_t/cuuint64_t\n"
            "// typedefs exist before <cudaTypedefs.h> forward-declares CUDA 12.8+\n"
            "// tensor-map APIs that consume them. CUTLASS upstream guard misses\n"
            "// some compilation paths under CUDA 12.9. Safe no-op when cuda.h\n"
            "// would have been included anyway by the original conditional.\n"
            "#include <cuda.h>"
        )
        if old_block not in content:
            return  # Upstream changed shape; skip rather than corrupt.
        new_content = content.replace(old_block, new_block)
        with open(path, "w") as f:
            f.write(new_content)
        print(f"CUTLASS cuda_host_adapter.hpp patched for CUDA 12.9 compat: {path}")
    except OSError as e:
        print(f"Warning: failed to patch CUTLASS cuda_host_adapter.hpp: {e}")


# Defaults for sdist-only mode (no torch)
ext_modules = []
cmdclass = {}

if _HAS_TORCH:
    _patch_cutlass_cuda_host_adapter()
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
    _commit_define = f'-DWARPCONVNET_BUILD_COMMIT="{BUILD_COMMIT}"'

    cxx_args = [
        "-std=c++17",
        "-O3",
        "-DWITH_CUDA",
        _commit_define,
        "-Wno-changes-meaning",
        "-fpermissive",
    ]

    nvcc_args = [
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-DWITH_CUDA",
        _commit_define,
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
            # Strip PyTorch-style suffixes (e.g. "9.0a" arch-specific, "8.0+PTX")
            # before parsing as a float.
            if arch.upper().endswith("PTX"):
                arch = arch[: -len("PTX")].rstrip("+")
            arch = arch.rstrip("aA")
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

    _generate_warpgemm_codegen()

    # Define the extension
    # Hand-written cute + implicit instantiations are now generated by
    # warpgemm.codegen.offset_gemm into warpconvnet/csrc/offset_gemm/. Pick those up
    # via glob; the legacy hand-written .cu files remain on disk for one cycle
    # but are not compiled (deletion follows in F2).
    generated_sources = sorted(glob.glob("warpconvnet/csrc/offset_gemm/*.cu"))
    ext_modules = [
        CUDAExtension(
            name="warpconvnet._C",
            sources=[
                "warpconvnet/csrc/warpconvnet_pybind.cpp",
                "warpconvnet/csrc/bindings/gemm_bindings.cpp",
                "warpconvnet/csrc/bindings/fma_bindings.cpp",
                "warpconvnet/csrc/bindings/utils_bindings.cpp",
                "warpconvnet/csrc/cutlass_gemm_gather_scatter.cu",
                "warpconvnet/csrc/cutlass_cute_gemm_staged.cu",  # SM80 staged — not in generated tier
                "warpconvnet/csrc/cutlass_gemm_gather_scatter_sm80_fp32.cu",
                "warpconvnet/csrc/cub_sort.cu",
                "warpconvnet/csrc/voxel_mapping_kernels.cu",
                "warpconvnet/csrc/implicit_fma_kernel.cu",
                "warpconvnet/csrc/implicit_reduction.cu",
                "warpconvnet/csrc/segmented_arithmetic.cu",
                "warpconvnet/csrc/mask_data_kernels.cu",
                "warpconvnet/csrc/bindings/sampling_bindings.cpp",
                "warpconvnet/csrc/farthest_point_sampling.cu",
                "warpconvnet/csrc/bindings/coords_bindings.cpp",
                "warpconvnet/csrc/bindings/mask_gemm_bindings.cu",
                "warpconvnet/csrc/coords_launch.cu",
                "warpconvnet/csrc/morton_code.cu",
                "warpconvnet/csrc/find_first_gt_bsearch.cu",
                "warpconvnet/csrc/radius_search_kernels.cu",
                "warpconvnet/csrc/mask_gemm_kernels_fwd.cu",
                "warpconvnet/csrc/mask_gemm_kernels_dgrad.cu",
                "warpconvnet/csrc/mask_gemm_kernels_wgrad.cu",
                "warpconvnet/csrc/window_grouping_kernels.cu",
                "warpconvnet/csrc/bindings/cuhash_bindings.cpp",
                "warpconvnet/csrc/cuhash_hash_table.cu",
                "warpconvnet/csrc/cuhash_kernel_map.cu",
                "warpconvnet/csrc/cuhash_packed128.cu",
                "warpconvnet/csrc/bindings/fused_rope_bindings.cpp",
                "warpconvnet/csrc/fused_rope_kernel.cu",
                *generated_sources,
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


# Bake git commit hash into warpconvnet/_build_info.py at build time so
# `import warpconvnet` can report which binary was loaded.
# For pre-built wheels, set SETUPTOOLS_SCM_PRETEND_VERSION externally
# (e.g. "1.4.2+torch2.10cu128") to inject a local version tag.
# setuptools-scm reads version from git tags by default.
# All metadata (name, version, dependencies) comes from pyproject.toml.
# setup.py only provides ext_modules and cmdclass for the CUDA build.
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
