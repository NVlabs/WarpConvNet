# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

# Import constants to set the default values
from warpconvnet.constants import (
    WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP,
    WARPCONVNET_FWD_ALGO_MODE,
    WARPCONVNET_DGRAD_ALGO_MODE,
    WARPCONVNET_WGRAD_ALGO_MODE,
    get_fp16_accum,
    set_fp16_accum,
)

_SKIP_EXTENSION = os.environ.get("WARPCONVNET_SKIP_EXTENSION", "0") == "1"

if not _SKIP_EXTENSION:
    try:
        from . import _C  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Failed to import the compiled WarpConvNet extension. Build it via "
            "`python setup.py build_ext --inplace` or install the pre-built wheel."
        ) from exc
else:
    _C = None  # type: ignore[assignment]

# Commit hash is baked into _C via -DWARPCONVNET_BUILD_COMMIT at compile time.
_BUILD_COMMIT = getattr(_C, "__build_commit__", "unknown") if _C is not None else "unknown"

try:
    from ._version import version as __version__  # written by setuptools-scm at build time
except ImportError:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("warpconvnet")
    except Exception:
        __version__ = "unknown"

print(f"warpconvnet {__version__} (commit {_BUILD_COMMIT[:12]})")

# Register pytree nodes, allow_in_graph markers, and compiler.disable
# wrappers so that torch.compile(model) works out of the box.
from . import _compile  # noqa: F401
