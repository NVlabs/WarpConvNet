# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Algorithm enums, benchmark parameter lists, and parameter filtering for
# sparse convolution forward/backward algorithm selection.

from typing import Any, Dict, List, Tuple, Union

from enum import Enum

from warpconvnet.constants import (
    WARPCONVNET_FWD_ALGO_MODE,
    WARPCONVNET_BWD_ALGO_MODE,
)
from warpconvnet.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Backend availability flags (set once at import time)
# ---------------------------------------------------------------------------

try:
    from .cute import (  # noqa: F401
        _cute_implicit_gemm_forward_logic,
        _cute_implicit_gemm_backward_logic,
    )

    _HAS_CUTE_BACKEND = True
except Exception:
    _HAS_CUTE_BACKEND = False

try:
    from .cute_grouped import (  # noqa: F401
        _cute_grouped_forward_logic,
        _cute_grouped_backward_logic,
    )

    _HAS_CUTE_GROUPED = _HAS_CUTE_BACKEND
except Exception:
    _HAS_CUTE_GROUPED = False

# ---------------------------------------------------------------------------
# Enums for granular algorithm control
# ---------------------------------------------------------------------------


class SPARSE_CONV_FWD_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    CUTLASS_IMPLICIT_GEMM = "cutlass_implicit_gemm"
    CUTE_IMPLICIT_GEMM = "cute_implicit_gemm"
    EXPLICIT_GEMM_GROUPED = "explicit_gemm_grouped"
    IMPLICIT_GEMM_GROUPED = "implicit_gemm_grouped"
    CUTLASS_GROUPED_HYBRID = "cutlass_grouped_hybrid"
    CUTE_GROUPED = "cute_grouped"
    AUTO = "auto"  # Benchmark and select the best algorithm
    ALL = "all"  # Benchmark ALL candidates (slow, exhaustive)


class SPARSE_CONV_BWD_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    CUTLASS_IMPLICIT_GEMM = "cutlass_implicit_gemm"
    CUTE_IMPLICIT_GEMM = "cute_implicit_gemm"
    EXPLICIT_GEMM_GROUPED = "explicit_gemm_grouped"
    IMPLICIT_GEMM_GROUPED = "implicit_gemm_grouped"
    CUTLASS_GROUPED_HYBRID = "cutlass_grouped_hybrid"
    CUTE_GROUPED = "cute_grouped"
    AUTO = "auto"  # Benchmark and select the best algorithm
    ALL = "all"  # Benchmark ALL candidates (slow, exhaustive)


# ---------------------------------------------------------------------------
# Forward benchmark candidate lists
# ---------------------------------------------------------------------------

# Base forward benchmark candidates shared across all channel sizes.
_BENCHMARK_FORWARD_PARAMS_BASE = [
    ("cutlass_implicit_gemm", {}),
    ("cutlass_grouped_hybrid", {"saturation_m": 2000}),
    *(
        []
        if not _HAS_CUTE_GROUPED
        else [
            ("cute_grouped", {"mma_tile": 3}),
            ("cute_grouped", {"mma_tile": 0}),
            ("cute_grouped", {"mma_tile": 1}),
        ]
    ),
]

# Additional candidates for small channels (max(C_in, C_out) <= 64).
# implicit_gemm wins 14/38 at 32->32, 11/28 at 64->64 but never wins at C>=96.
# cute_implicit_gemm wins 9 times total, all at small channels / small N.
_BENCHMARK_FORWARD_PARAMS_SMALL_CH = [
    *[("implicit_gemm", {"fwd_block_size": block_size}) for block_size in [16, 32]],
    *([] if not _HAS_CUTE_BACKEND else [("cute_implicit_gemm", {})]),
]

# Additional candidates for small channels (max <= 64).
# implicit_gemm_grouped wins 10/292 overall (3.4%), all at small channels:
# 5 wins at 3->32 kv=125, 5 wins at 32->64/32->32/7->13 kv=8/27.
# Margins vs implicit_gemm are tiny (0.1%-4.9%) so these are only needed
# when implicit_gemm is already a candidate (i.e., small channels).
_BENCHMARK_FORWARD_PARAMS_GROUPED = [
    ("implicit_gemm_grouped", {"fwd_block_size": 16, "saturation_m": 2000}),
    ("implicit_gemm_grouped", {"fwd_block_size": 16, "saturation_m": 5000}),
]


def _get_adaptive_forward_params(
    in_channels: int, out_channels: int, kernel_volume: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get forward benchmark candidates adapted to the convolution dimensions.

    Based on cache analysis of 292 configs:
    - Small channels (max(C_in,C_out) <= 64): implicit_gemm wins ~30%,
      implicit_gemm_grouped wins ~3.4% (all at small channels)
    - Large channels (max >= 96): implicit_gemm/grouped never win forward
    """
    max_ch = max(in_channels, out_channels)
    params = list(_BENCHMARK_FORWARD_PARAMS_BASE)
    if max_ch <= 64:
        params.extend(_BENCHMARK_FORWARD_PARAMS_SMALL_CH)
        params.extend(_BENCHMARK_FORWARD_PARAMS_GROUPED)
    return params


# Full forward benchmark candidates ("all"): exhaustive search including rarely-winning variants.
_ALL_BENCHMARK_FORWARD_PARAMS = [
    ("cutlass_implicit_gemm", {}),
    *([] if not _HAS_CUTE_BACKEND else [("cute_implicit_gemm", {})]),
    *[("implicit_gemm", {"fwd_block_size": block_size}) for block_size in [4, 16, 32]],
    ("explicit_gemm", {}),
    # Grouped variants: adaptive offset grouping with batched execution
    *[("explicit_gemm_grouped", {"saturation_m": m}) for m in [2000, 5000, 10000]],
    *[
        ("implicit_gemm_grouped", {"fwd_block_size": bs, "saturation_m": m})
        for bs in [16, 32]
        for m in [2000, 5000]
    ],
    *[("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]],
    *(
        []
        if not _HAS_CUTE_GROUPED
        else [
            ("cute_grouped", {"mma_tile": 3}),
            ("cute_grouped", {"mma_tile": 0}),
            ("cute_grouped", {"mma_tile": 1}),
        ]
    ),
]

# ---------------------------------------------------------------------------
# Backward benchmark candidate lists
# ---------------------------------------------------------------------------

# Reduced backward benchmark candidates (default "auto"): only algorithms that
# win or appear in top-3 frequently. 9 candidates vs 32 in the full set.
_BENCHMARK_BACKWARD_PARAMS = [
    ("cutlass_implicit_gemm", {}),
    ("explicit_gemm", {}),
    ("explicit_gemm_grouped", {"saturation_m": 2000}),
    (
        "implicit_gemm",
        {"gemm_block_size": 16, "split_k_threads_per_block": 256, "split_k_factor": 2},
    ),
    *[("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]],
    *(
        []
        if not _HAS_CUTE_GROUPED
        else [
            ("cute_grouped", {"mma_tile": 3}),
            ("cute_grouped", {"mma_tile": 0}),
            ("cute_grouped", {"mma_tile": 1}),
        ]
    ),
]

# Full backward benchmark candidates ("all"): exhaustive search.
_ALL_BENCHMARK_BACKWARD_PARAMS = [
    ("cutlass_implicit_gemm", {}),
    *([] if not _HAS_CUTE_BACKEND else [("cute_implicit_gemm", {})]),
    *[
        (
            "implicit_gemm",
            {
                "gemm_block_size": gemm_block_size,
                "split_k_threads_per_block": split_k_threads_per_block,
                "split_k_factor": split_k_factor,
            },
        )
        for gemm_block_size in [4, 16, 32]
        for split_k_threads_per_block in [256]
        for split_k_factor in [2, 4, 8, 16]
    ],
    ("explicit_gemm", {}),
    # Grouped backward variants
    *[("explicit_gemm_grouped", {"saturation_m": m}) for m in [2000, 5000, 10000]],
    *[
        (
            "implicit_gemm_grouped",
            {
                "gemm_block_size": bs,
                "split_k_threads_per_block": 256,
                "split_k_factor": sf,
                "saturation_m": m,
            },
        )
        for bs in [16, 32]
        for sf in [4, 8]
        for m in [2000, 5000]
    ],
    *[("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]],
    *(
        []
        if not _HAS_CUTE_GROUPED
        else [
            ("cute_grouped", {"mma_tile": 3}),
            ("cute_grouped", {"mma_tile": 0}),
            ("cute_grouped", {"mma_tile": 1}),
        ]
    ),
]

# ---------------------------------------------------------------------------
# Parameter filtering
# ---------------------------------------------------------------------------


def _get_filtered_forward_params() -> List[Tuple[str, Dict[str, Any]]]:
    """Get forward benchmark parameters filtered by environment variable.

    For "auto", returns the adaptive (reduced) set for a generic large-channel config.
    For "all", returns the full exhaustive set.
    """
    return _filter_benchmark_params_by_env_config(
        _BENCHMARK_FORWARD_PARAMS_BASE, WARPCONVNET_FWD_ALGO_MODE, is_forward=True
    )


def _get_filtered_backward_params() -> List[Tuple[str, Dict[str, Any]]]:
    """Get backward benchmark parameters filtered by environment variable."""
    return _filter_benchmark_params_by_env_config(
        _BENCHMARK_BACKWARD_PARAMS, WARPCONVNET_BWD_ALGO_MODE, is_forward=False
    )


def _filter_benchmark_params_by_env_config(
    all_params: List[Tuple[Union[str, Any], Dict[str, Any]]],
    env_config: Union[str, List[Union[str, Any]]],
    is_forward: bool = True,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Filter benchmark parameters based on environment variable configuration.

    Args:
        all_params: All available benchmark parameters (the reduced "auto" set)
        env_config: Environment variable value (string or list of algorithm names)
        is_forward: Whether this is for forward pass (affects enum type)

    Returns:
        Filtered list of benchmark parameters
    """
    if env_config == "all":
        # When "all", use the full exhaustive candidate set
        full_params = (
            _ALL_BENCHMARK_FORWARD_PARAMS if is_forward else _ALL_BENCHMARK_BACKWARD_PARAMS
        )
        return [(str(algo), params) for algo, params in full_params]

    if env_config == "auto":
        # When "auto", use reduced candidate set (default)
        return [(str(algo), params) for algo, params in all_params]

    # Convert environment config to list of algorithm names
    if isinstance(env_config, str):
        target_algos = [env_config]
    else:
        target_algos = [str(a) for a in env_config]

    if not target_algos:
        logger.warning("No valid algorithms found, using all algorithms")
        return all_params

    # Filter parameters to only include target algorithms
    filtered_params: List[Tuple[str, Dict[str, Any]]] = []
    for algo_tag, params in all_params:
        algo_str = str(algo_tag)
        if algo_str in target_algos:
            filtered_params.append((algo_str, params))

    if not filtered_params:
        logger.warning(
            f"No benchmark parameters found for algorithms {target_algos}, using all algorithms"
        )
        return all_params

    return filtered_params
