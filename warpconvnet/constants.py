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


def _get_env_string(
    env_var_name: str, default_value: str, valid_values: Optional[List[str]] = None
) -> str:
    """Helper function to read and validate string environment variables."""
    env_value = os.environ.get(env_var_name)

    if env_value is None:
        return default_value

    env_value = env_value.lower()
    if valid_values is not None and env_value not in valid_values:
        raise ValueError(f"{env_var_name} must be one of {valid_values}, got {env_value}")

    logger.info(f"{env_var_name} is set to {env_value} by environment variable")
    return env_value


def _get_env_string_list(
    env_var_name: str,
    default_value: Union[str, List[str]],
    valid_values: Optional[List[str]] = None,
) -> Union[str, List[str]]:
    """Helper function to read and validate string or list environment variables.

    Supports formats:
    - Single value: "auto" or "implicit_gemm"
    - List format: "[implicit_gemm,cutlass_implicit_gemm]"
    """
    env_value = os.environ.get(env_var_name)

    if env_value is None:
        return default_value

    env_value = env_value.strip()

    # Check if it's a list format [item1,item2,...]
    if env_value.startswith("[") and env_value.endswith("]"):
        # Parse list format
        list_content = env_value[1:-1].strip()
        if not list_content:
            # Empty list, return default
            return default_value

        # Split by comma and clean each item
        items = [item.strip().lower() for item in list_content.split(",")]

        # Validate each item if valid_values provided
        if valid_values is not None:
            for item in items:
                if item not in valid_values:
                    raise ValueError(
                        f"{env_var_name} contains invalid algorithm '{item}'. Valid values: {valid_values}"
                    )

        logger.info(f"{env_var_name} is set to {items} by environment variable")
        return items
    else:
        # Single value format
        env_value = env_value.lower()
        if valid_values is not None and env_value not in valid_values:
            raise ValueError(f"{env_var_name} must be one of {valid_values}, got {env_value}")

        logger.info(f"{env_var_name} is set to {env_value} by environment variable")
        return env_value


# Boolean constants
WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP = _get_env_bool(
    "WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP", False
)

# String constants with validation
VALID_ALGOS = ["explicit_gemm", "implicit_gemm", "cutlass_implicit_gemm", "auto"]

# Algorithm selection constants
# These environment variables support both single algorithm and list of algorithms:
#
# Single algorithm examples:
#   export WARPCONVNET_FWD_ALGO_MODE=implicit_gemm
#   export WARPCONVNET_BWD_ALGO_MODE=cutlass_implicit_gemm
#   export WARPCONVNET_FWD_ALGO_MODE=auto  # (default) benchmark all algorithms
#
# Multiple algorithm examples (will benchmark only the specified algorithms):
#   export WARPCONVNET_FWD_ALGO_MODE="[implicit_gemm,cutlass_implicit_gemm]"
#   export WARPCONVNET_BWD_ALGO_MODE="[explicit_gemm,implicit_gemm]"
#
# When multiple algorithms are specified, the system will benchmark only those
# algorithms instead of all available algorithms and select the best one.
WARPCONVNET_FWD_ALGO_MODE = _get_env_string_list("WARPCONVNET_FWD_ALGO_MODE", "auto", VALID_ALGOS)
WARPCONVNET_BWD_ALGO_MODE = _get_env_string_list("WARPCONVNET_BWD_ALGO_MODE", "auto", VALID_ALGOS)

VALID_DEPTHWISE_ALGOS = ["explicit_gemm", "implicit_gemm", "auto"]

# Depthwise convolution algorithm selection constants
# Similar to regular convolution, these support both single and multiple algorithm specification:
#
# Examples:
#   export WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE=implicit_gemm
#   export WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE="[explicit_gemm,implicit_gemm]"
WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE = _get_env_string_list(
    "WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE", "auto", VALID_DEPTHWISE_ALGOS
)
WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE = _get_env_string_list(
    "WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE", "auto", VALID_DEPTHWISE_ALGOS
)

# Sparse conv benchmark cache
WARPCONVNET_BENCHMARK_CACHE_DIR = _get_env_string(
    "WARPCONVNET_BENCHMARK_CACHE_DIR", "~/.cache/warpconvnet"
)

WARPCONVNET_BENCHMARK_CACHE_VERSION = 3.0

# --- Types ---

# --- Functions ---
