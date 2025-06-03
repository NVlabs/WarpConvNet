# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from warpconvnet.utils.logger import get_logger

logger = get_logger(__name__)

# VALID bools
VALID_BOOLS = ["true", "false", "1", "0"]
WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP = False

# For bool, true, false, 0, 1 are all valid values
env_skip_sym_kernel_map = os.environ.get("WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP")
if env_skip_sym_kernel_map is not None:
    env_skip_sym_kernel_map = env_skip_sym_kernel_map.lower()
    if env_skip_sym_kernel_map not in VALID_BOOLS:
        raise ValueError(
            f"WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP must be one of {VALID_BOOLS}, got {env_skip_sym_kernel_map}"
        )
    WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP = env_skip_sym_kernel_map in ["true", "1"]
    logger.info(
        f"WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP is set to {WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP} by environment variable"
    )

# --- Types ---

# --- Functions ---
