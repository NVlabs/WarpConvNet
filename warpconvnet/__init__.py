# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

# Import constants to set the default values
from warpconvnet.constants import (
    WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP,
    WARPCONVNET_FWD_ALGO_MODE,
    WARPCONVNET_BWD_ALGO_MODE,
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

# Register pytree nodes, allow_in_graph markers, and compiler.disable
# wrappers so that torch.compile(model) works out of the box.
from . import _compile  # noqa: F401
