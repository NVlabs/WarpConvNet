# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional, Union
from warpconvnet.utils.logger import get_logger

logger = get_logger(__name__)


def _get_env_bool(env_var_name: str, default_value: bool) -> bool:
    """Helper function to read and validate boolean environment variables."""
    valid_bools = ["true", "false", "1", "0"]
    env_value = os.environ.get(env_var_name)

    if env_value is None:
        return default_value

    env_value = env_value.lower()
    if env_value not in valid_bools:
        raise ValueError(f"{env_var_name} must be one of {valid_bools}, got {env_value}")

    result = env_value in ["true", "1"]
    logger.info(f"{env_var_name} is set to {result} by environment variable")
    return result


def _get_env_string(env_var_name: str, default_value: str, valid_values: List[str]) -> str:
    """Helper function to read and validate string environment variables."""
    env_value = os.environ.get(env_var_name)

    if env_value is None:
        return default_value

    env_value = env_value.lower()
    if env_value not in valid_values:
        raise ValueError(f"{env_var_name} must be one of {valid_values}, got {env_value}")

    logger.info(f"{env_var_name} is set to {env_value} by environment variable")
    return env_value


# Boolean constants
WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP = _get_env_bool(
    "WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP", False
)

# String constants with validation
VALID_ALGOS = ["explicit_gemm", "auto"]
WARPCONVNET_FWD_ALGO_MODE = _get_env_string("WARPCONVNET_FWD_ALGO_MODE", "auto", VALID_ALGOS)
WARPCONVNET_BWD_ALGO_MODE = _get_env_string("WARPCONVNET_BWD_ALGO_MODE", "auto", VALID_ALGOS)

VALID_DEPTHWISE_ALGOS = ["explicit", "implicit", "auto"]
WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE = _get_env_string(
    "WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE", "auto", VALID_DEPTHWISE_ALGOS
)
WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE = _get_env_string(
    "WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE", "auto", VALID_DEPTHWISE_ALGOS
)

# --- Types ---

# --- Functions ---
