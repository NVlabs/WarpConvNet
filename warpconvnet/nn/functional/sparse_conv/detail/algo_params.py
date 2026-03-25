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

# Hardware capability detection
_HAS_SM80_HARDWARE = False
_HAS_SM90_HARDWARE = False
try:
    import torch

    _sm_cap = torch.cuda.get_device_capability()
    _HAS_SM80_HARDWARE = _sm_cap[0] >= 8
    _HAS_SM90_HARDWARE = _sm_cap[0] >= 9
except Exception:
    pass

# CUTLASS/CuTe backends — require SM80+ hardware AND compiled support
_HAS_CUTLASS_BACKEND = False
try:
    import warpconvnet._C as _C

    _HAS_CUTLASS_BACKEND = _HAS_SM80_HARDWARE and hasattr(
        _C.gemm, "cutlass_gemm_AD_gather_scatter"
    )
except Exception:
    pass

# Override CuTe backend availability: require SM80+ hardware
if not _HAS_SM80_HARDWARE:
    _HAS_CUTE_BACKEND = False
    _HAS_CUTE_GROUPED = False

# SM90 WGMMA backends — available when compiled with SM90 support and running on SM90+ hardware
try:
    _HAS_CUTE_SM90 = _HAS_SM90_HARDWARE and hasattr(
        _C.gemm, "cute_gemm_sm90_AD_gather_scatter"
    )
    _HAS_CUTE_GROUPED_SM90 = _HAS_SM90_HARDWARE and hasattr(
        _C.gemm, "cute_gemm_sm90_grouped_AD_gather_scatter"
    )
except Exception:
    _HAS_CUTE_SM90 = False
    _HAS_CUTE_GROUPED_SM90 = False

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
    CUTE_IMPLICIT_GEMM_SM90 = "cute_implicit_gemm_sm90"
    CUTE_GROUPED_SM90 = "cute_grouped_sm90"
    MASK_IMPLICIT_GEMM = "mask_implicit_gemm"
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
    CUTE_IMPLICIT_GEMM_SM90 = "cute_implicit_gemm_sm90"
    CUTE_GROUPED_SM90 = "cute_grouped_sm90"
    MASK_IMPLICIT_GEMM = "mask_implicit_gemm"
    AUTO = "auto"  # Benchmark and select the best algorithm
    ALL = "all"  # Benchmark ALL candidates (slow, exhaustive)


# ---------------------------------------------------------------------------
# Forward benchmark candidate lists
# ---------------------------------------------------------------------------
#
# Time-weighted cache analysis of 465 forward configs (fp16/bf16/fp32, SM 8.9):
# Total optimal time: 896.6ms across all configs.
#
# Removal impact (ms lost if algo removed from full set):
#   cutlass_implicit_gemm : 27.3ms  — dominates large N, large channels, kv=27
#   implicit_gemm_grouped : 15.0ms  — critical for small channels (<=32), all N
#   cute_grouped          :  9.9ms  — best at small-medium N, medium channels
#   implicit_gemm         :  6.1ms  — wins small channels and small N
#   cutlass_grouped_hybrid:  2.0ms  — competitive everywhere but rarely unique
#   cute_implicit_gemm    :  0.5ms  — marginal, only small N + small channels
#   explicit_gemm         :  0.0ms  — never uniquely fastest (always tied)
#   explicit_gemm_grouped :  1.25ms — wins at xlarge N, asymmetric channels
#
# Greedy set cover order:
#   1. cutlass_implicit_gemm (base regret 4.4%)
#   2. cute_grouped          (+35.3ms saved, regret 0.5%)
#   3. cutlass_grouped_hybrid(+2.1ms, regret 0.2%)
#   4. implicit_gemm         (+1.7ms, regret 1.8%)
#   5. implicit_gemm_grouped (+15.0ms, regret 0.1%)
#   6. explicit_gemm         (+0.5ms, regret 0.1%)
#   7. cute_implicit_gemm    (+0.5ms, regret 0.0%)
#
# Cross-axis findings (N bucket × channel bucket):
#   - N<=4K: cute_grouped + implicit_gemm dominate; cutlass barely participates
#   - 4K<N<=64K, ch<=64: implicit_gemm_grouped most valuable (3.5ms removal)
#   - 4K<N<=64K, ch>64: cute_grouped most valuable (3.7ms removal)
#   - N>64K, ch>64: cutlass_implicit_gemm dominates; cute_grouped drops off
#   - N>64K, ch<=64: implicit_gemm_grouped critical (10.7ms removal at 64K-512K)
#   - kv=125: only implicit_gemm and implicit_gemm_grouped win

# Algo building blocks (assembled conditionally by _get_adaptive_forward_params)
_FWD_CUTLASS_IMPLICIT = [("cutlass_implicit_gemm", {})] if _HAS_CUTLASS_BACKEND else []
_FWD_CUTLASS_GROUPED = (
    [("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]]
    if _HAS_CUTLASS_BACKEND
    else []
)
# SM90 WGMMA forward building blocks (only on Hopper+ hardware with SM90 compiled support)
_FWD_CUTE_SM90 = (
    []
    if not _HAS_CUTE_SM90
    else [
        ("cute_implicit_gemm_sm90", {"mma_tile": 100}),
        ("cute_implicit_gemm_sm90", {"mma_tile": 103}),
        ("cute_implicit_gemm_sm90", {"mma_tile": 104}),
    ]
)
_FWD_CUTE_GROUPED_SM90 = (
    []
    if not _HAS_CUTE_GROUPED_SM90
    else [
        ("cute_grouped_sm90", {"mma_tile": 103, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 100, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 101, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 104, "use_cp_async": True}),
    ]
)
# implicit_gemm: 0% forward wins in time-weighted analysis (208 configs).
# Kept for backward (3.4% dgrad, 4.8% wgrad) but removed from forward.
_FWD_IMPLICIT_GEMM_16 = []  # was [("implicit_gemm", {"fwd_block_size": 16})]
_FWD_IMPLICIT_GEMM_32 = []  # was [("implicit_gemm", {"fwd_block_size": 32})]
_FWD_CUTE_GROUPED = (
    []
    if not _HAS_CUTE_GROUPED
    else [
        ("cute_grouped", {"mma_tile": 3}),
        ("cute_grouped", {"mma_tile": 0}),
        ("cute_grouped", {"mma_tile": 1}),
    ]
)
# implicit_gemm_grouped: 0% wins in all directions (forward, dgrad, wgrad).
_FWD_IMPLICIT_GEMM_GROUPED = []  # was 2 configs
_FWD_MASK_IMPLICIT_GEMM = [
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 0}),  # Tile128x128x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 1}),  # Tile128x64x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 2}),  # Tile64x128x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 3}),  # Tile64x64x32
]
# cute_implicit_gemm: 0% wins in all directions. Dead weight.
_FWD_CUTE_IMPLICIT = []  # was [("cute_implicit_gemm", {})]
# explicit_gemm: 0% forward wins. Kept as fallback only, not in auto-tune candidates.
_FWD_EXPLICIT = []  # was [("explicit_gemm", {})]
# explicit_gemm_grouped: 0% forward wins, 2.4% dgrad, 1.4% wgrad.
_FWD_EXPLICIT_GROUPED = []  # was 2 configs


import math as _math


def _get_adaptive_forward_params(
    in_channels: int,
    out_channels: int,
    kernel_volume: int,
    num_in_coords: int = 0,
    voxel_size: Union[Tuple[int, ...], None] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get forward benchmark candidates adapted to the convolution dimensions.

    Uses N (num_in_coords), max(C_in,C_out), and kernel_volume to select only
    algorithms that have non-trivial removal impact in that region, based on
    time-weighted analysis of 465 configs.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_volume: Product of kernel dimensions (e.g. 27 for 3x3x3).
        num_in_coords: Number of input voxels (0 = unknown).
        voxel_size: Tensor stride / voxel size tuple (None = unknown).
    """
    max_ch = max(in_channels, out_channels)
    log_n = _math.ceil(_math.log2(num_in_coords)) if num_in_coords > 1 else 0

    # kv=125 (5x5x5): only implicit_gemm variants win
    if kernel_volume >= 64:
        params: List[Tuple[str, Dict[str, Any]]] = []
        params.extend(_FWD_CUTE_GROUPED)  # May win for large kv via fused launch
        params.extend(_FWD_IMPLICIT_GEMM_16)
        params.extend(_FWD_IMPLICIT_GEMM_32)
        params.extend(_FWD_IMPLICIT_GEMM_GROUPED)
        params.extend(_FWD_MASK_IMPLICIT_GEMM)
        return params

    # --- kv <= 27 (3x3x3 or 2x2x2) below ---

    # Small N (N <= 4K, logN <= 12): cute_grouped + implicit_gemm dominate
    # cutlass_implicit_gemm has zero removal impact at these sizes
    if 0 < log_n <= 12:
        params = []
        params.extend(_FWD_CUTE_GROUPED)
        params.extend(_FWD_IMPLICIT_GEMM_16)
        params.extend(_FWD_MASK_IMPLICIT_GEMM)  # Wins at small-medium C
        if max_ch <= 64:
            params.extend(_FWD_IMPLICIT_GEMM_GROUPED)
        params.extend(_FWD_CUTE_SM90)
        params.extend(_FWD_CUTE_GROUPED_SM90)
        return params

    # Medium N (4K < N <= 64K, logN 13-16)
    if 0 < log_n <= 16:
        params = []
        params.extend(_FWD_CUTE_GROUPED)  # highest removal impact in this range
        params.extend(_FWD_CUTLASS_IMPLICIT)
        params.extend(_FWD_CUTLASS_GROUPED)
        params.extend(_FWD_IMPLICIT_GEMM_16)
        params.extend(_FWD_MASK_IMPLICIT_GEMM)  # Competitive at C<=96
        if max_ch <= 64:
            params.extend(_FWD_IMPLICIT_GEMM_GROUPED)
            params.extend(_FWD_CUTE_IMPLICIT)
        params.extend(_FWD_CUTE_SM90)
        params.extend(_FWD_CUTE_GROUPED_SM90)
        return params

    # Large N (N > 64K, logN > 16) or unknown N (num_in_coords=0)
    params = []
    params.extend(_FWD_CUTLASS_IMPLICIT)  # dominates this range
    params.extend(_FWD_CUTLASS_GROUPED)
    params.extend(_FWD_IMPLICIT_GEMM_16)
    params.extend(_FWD_MASK_IMPLICIT_GEMM)  # Competitive at C<=64
    if max_ch <= 64:
        # implicit_gemm_grouped: 10.7ms removal impact at 64K-512K, ch<=64
        params.extend(_FWD_IMPLICIT_GEMM_GROUPED)
        params.extend(_FWD_IMPLICIT_GEMM_32)
        params.extend(_FWD_CUTE_IMPLICIT)
    if log_n > 19:
        # explicit_gemm_grouped: 1.25ms removal impact at xlarge N, asymmetric channels
        params.extend(_FWD_EXPLICIT_GROUPED)
    if log_n <= 19 or log_n == 0:
        # cute_grouped still contributes up to ~512K
        params.extend(_FWD_CUTE_GROUPED)
    params.extend(_FWD_CUTE_SM90)
    params.extend(_FWD_CUTE_GROUPED_SM90)
    return params


# Full forward benchmark candidates ("all"): exhaustive search.
# Trimmed based on time-weighted analysis of 208 configs:
# Removed: cute_implicit_gemm (0%), implicit_gemm (0%), implicit_gemm_grouped (0%),
#          explicit_gemm (0%), explicit_gemm_grouped (0%)
_ALL_BENCHMARK_FORWARD_PARAMS = [
    *([] if not _HAS_CUTLASS_BACKEND else [("cutlass_implicit_gemm", {})]),
    # cute_implicit_gemm: removed (0% wins across all directions)
    # implicit_gemm: removed from forward (0% wins; kept for backward)
    # explicit_gemm: removed from forward (0% wins; kept as fallback)
    # explicit_gemm_grouped: removed from forward (0% wins)
    # implicit_gemm_grouped: removed (0% wins across all directions)
    *(
        [("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]]
        if _HAS_CUTLASS_BACKEND
        else []
    ),
    *(
        []
        if not _HAS_CUTE_GROUPED
        else [
            ("cute_grouped", {"mma_tile": 3}),
            ("cute_grouped", {"mma_tile": 0}),
            ("cute_grouped", {"mma_tile": 1}),
        ]
    ),
    # SM90 WGMMA candidates for exhaustive search
    *(
        []
        if not _HAS_CUTE_SM90
        else [
            ("cute_implicit_gemm_sm90", {"mma_tile": tile})
            for tile in [100, 101, 102, 103, 104]
        ]
    ),
    *(
        []
        if not _HAS_CUTE_GROUPED_SM90
        else [
            ("cute_grouped_sm90", {"mma_tile": tile, "use_cp_async": cp})
            for tile in [100, 101, 102, 103, 104]
            for cp in [True, False]
        ]
    ),
    # Mask-based fused implicit GEMM (all tile configs)
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 0}),  # Tile128x128x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 1}),  # Tile128x64x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 2}),  # Tile64x128x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 3}),  # Tile64x64x32
]

# ---------------------------------------------------------------------------
# Backward benchmark candidate lists
# ---------------------------------------------------------------------------
#
# Time-weighted cache analysis of 458 backward configs (fp16/bf16, SM 8.9):
# Total optimal time: 1826.6ms across all configs.
#
# Removal impact (ms lost if algo removed):
#   cutlass_grouped_hybrid: 119.2ms — dominant at large N, all channel sizes
#   cute_grouped          :  46.7ms — critical at small-medium N
#   implicit_gemm         :  19.8ms — critical for small channels (<=32)
#   cutlass_implicit_gemm :  13.9ms — important at xlarge N, large channels
#   explicit_gemm         :   2.8ms — minor, mostly at large N + small channels
#   explicit_gemm_grouped :   0.2ms — negligible unique value
#   cute_implicit_gemm    :   0.0ms — never wins backward
#   implicit_gemm_grouped :   0.0ms — never wins backward
#
# Cross-axis findings:
#   - N<=4K: cute_grouped wins 100%
#   - 4K<N<=64K: cute_grouped (26.2ms removal), implicit_gemm (7.9ms)
#   - 64K<N<=512K, ch>64: cutlass_grouped_hybrid (56.2ms removal)
#   - 64K<N<=512K, ch<=64: implicit_gemm (11.9ms), cute_grouped (4.1ms)
#   - N>512K, ch>64: cutlass_grouped_hybrid (51.7ms), cutlass_implicit_gemm (10.4ms)
#   - N>512K, ch<=64: cutlass_grouped_hybrid (8.8ms), explicit_gemm (0.2ms)
#   - kv=125: explicit_gemm (59%) + explicit_gemm_grouped (32%) + implicit_gemm (9%)

# Algo building blocks for backward
_BWD_CUTLASS_IMPLICIT = [("cutlass_implicit_gemm", {})] if _HAS_CUTLASS_BACKEND else []
_BWD_CUTLASS_GROUPED = (
    [("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]]
    if _HAS_CUTLASS_BACKEND
    else []
)
_BWD_CUTE_GROUPED = (
    []
    if not _HAS_CUTE_GROUPED
    else [
        ("cute_grouped", {"mma_tile": 3}),
        ("cute_grouped", {"mma_tile": 0}),
        ("cute_grouped", {"mma_tile": 1}),
    ]
)
_BWD_IMPLICIT_GEMM = [
    (
        "implicit_gemm",
        {"gemm_block_size": 16, "split_k_threads_per_block": 256, "split_k_factor": 2},
    ),
]
_BWD_MASK_IMPLICIT_GEMM = [("mask_implicit_gemm", {"block_size": 16})]
# SM90 WGMMA backward building blocks
_BWD_CUTE_SM90 = (
    []
    if not _HAS_CUTE_SM90
    else [
        ("cute_implicit_gemm_sm90", {"mma_tile": 100}),
        ("cute_implicit_gemm_sm90", {"mma_tile": 103}),
        ("cute_implicit_gemm_sm90", {"mma_tile": 104}),
    ]
)
_BWD_CUTE_GROUPED_SM90 = (
    []
    if not _HAS_CUTE_GROUPED_SM90
    else [
        ("cute_grouped_sm90", {"mma_tile": 103, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 100, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 101, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 104, "use_cp_async": True}),
    ]
)
_BWD_EXPLICIT = [("explicit_gemm", {})]
_BWD_EXPLICIT_GROUPED = [("explicit_gemm_grouped", {"saturation_m": 2000})]

# Static backward params (used by _get_filtered_backward_params for env var filtering)
_BENCHMARK_BACKWARD_PARAMS = [
    *_BWD_CUTLASS_IMPLICIT,
    *_BWD_EXPLICIT,
    *_BWD_EXPLICIT_GROUPED,
    *_BWD_IMPLICIT_GEMM,
    *_BWD_CUTLASS_GROUPED,
    *_BWD_CUTE_GROUPED,
    *_BWD_CUTE_SM90,
    *_BWD_CUTE_GROUPED_SM90,
]


def _get_adaptive_backward_params(
    in_channels: int,
    out_channels: int,
    kernel_volume: int,
    num_in_coords: int = 0,
    voxel_size: Union[Tuple[int, ...], None] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get backward benchmark candidates adapted to the convolution dimensions.

    Uses N (num_in_coords), max(C_in,C_out), and kernel_volume to select only
    algorithms with non-trivial removal impact in that region, based on
    time-weighted analysis of 458 configs.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_volume: Product of kernel dimensions (e.g. 27 for 3x3x3).
        num_in_coords: Number of input voxels (0 = unknown).
        voxel_size: Tensor stride / voxel size tuple (None = unknown).
    """
    max_ch = max(in_channels, out_channels)
    log_n = _math.ceil(_math.log2(num_in_coords)) if num_in_coords > 1 else 0

    # kv=125 (5x5x5): include cute_grouped (2.9x faster than implicit_gemm)
    if kernel_volume >= 64:
        params: List[Tuple[str, Dict[str, Any]]] = []
        params.extend(_BWD_CUTE_GROUPED)  # 0.72ms vs implicit_gemm 2.09ms at kv=125
        params.extend(_BWD_EXPLICIT)
        params.extend(_BWD_EXPLICIT_GROUPED)
        params.extend(_BWD_IMPLICIT_GEMM)
        # Note: mask_implicit_gemm backward has SIMT-only kernel with OOB bug at large N
        # Disabled until CuTe backward is implemented
        # params.extend(_BWD_MASK_IMPLICIT_GEMM)
        return params

    # --- kv <= 27 below ---

    # Small N (N <= 4K, logN <= 12): cute_grouped dominates (100%)
    if 0 < log_n <= 12:
        params = []
        params.extend(_BWD_CUTE_GROUPED)
        params.extend(_BWD_CUTLASS_IMPLICIT)  # minor but covers edge cases
        params.extend(_BWD_CUTE_SM90)
        params.extend(_BWD_CUTE_GROUPED_SM90)
        return params

    # Medium N (4K < N <= 64K, logN 13-16)
    if 0 < log_n <= 16:
        params = []
        params.extend(_BWD_CUTE_GROUPED)  # 26.2ms removal impact
        params.extend(_BWD_CUTLASS_IMPLICIT)
        params.extend(_BWD_CUTLASS_GROUPED)
        if max_ch <= 64:
            params.extend(_BWD_IMPLICIT_GEMM)  # 7.9ms removal at ch<=64
        params.extend(_BWD_CUTE_SM90)
        params.extend(_BWD_CUTE_GROUPED_SM90)
        return params

    # Large N (N > 64K, logN > 16) or unknown N
    params = []
    params.extend(_BWD_CUTLASS_GROUPED)  # dominant: 119.2ms removal total
    params.extend(_BWD_CUTLASS_IMPLICIT)
    if max_ch <= 64:
        params.extend(_BWD_IMPLICIT_GEMM)  # 11.9ms removal at 64K-512K ch<=64
        params.extend(_BWD_EXPLICIT)  # minor but non-zero at ch<=32
        params.extend(_BWD_EXPLICIT_GROUPED)
    else:
        params.extend(_BWD_EXPLICIT)  # 1.7ms removal at 65-128 channels
    if log_n <= 19 or log_n == 0:
        # cute_grouped still valuable up to ~512K
        params.extend(_BWD_CUTE_GROUPED)
    params.extend(_BWD_CUTE_SM90)
    params.extend(_BWD_CUTE_GROUPED_SM90)
    return params


# Full backward benchmark candidates ("all"): exhaustive search.
# Trimmed: cute_implicit_gemm (0% all directions), implicit_gemm_grouped (0% all directions)
_ALL_BENCHMARK_BACKWARD_PARAMS = [
    *([] if not _HAS_CUTLASS_BACKEND else [("cutlass_implicit_gemm", {})]),
    # cute_implicit_gemm: removed (0% wins across all directions)
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
    # implicit_gemm_grouped: removed (0% wins across all directions)
    *(
        [("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]]
        if _HAS_CUTLASS_BACKEND
        else []
    ),
    *(
        []
        if not _HAS_CUTE_GROUPED
        else [
            ("cute_grouped", {"mma_tile": 3}),
            ("cute_grouped", {"mma_tile": 0}),
            ("cute_grouped", {"mma_tile": 1}),
        ]
    ),
    # SM90 WGMMA backward exhaustive candidates
    *(
        []
        if not _HAS_CUTE_SM90
        else [
            ("cute_implicit_gemm_sm90", {"mma_tile": tile})
            for tile in [100, 101, 102, 103, 104]
        ]
    ),
    *(
        []
        if not _HAS_CUTE_GROUPED_SM90
        else [
            ("cute_grouped_sm90", {"mma_tile": tile, "use_cp_async": cp})
            for tile in [100, 101, 102, 103, 104]
            for cp in [True, False]
        ]
    ),
    # Mask-based fused implicit GEMM backward disabled (SIMT kernel has OOB at large N)
    # TODO: Re-enable once CuTe backward mask kernel is implemented
]

# ---------------------------------------------------------------------------
# Parameter filtering
# ---------------------------------------------------------------------------


# Static superset of all forward "auto" candidates (union of all adaptive branches).
# Used by _get_filtered_forward_params for env var filtering.
_BENCHMARK_FORWARD_PARAMS_ALL_AUTO = [
    *_FWD_CUTLASS_IMPLICIT,
    *_FWD_CUTLASS_GROUPED,
    *_FWD_IMPLICIT_GEMM_16,
    *_FWD_IMPLICIT_GEMM_32,
    *_FWD_CUTE_GROUPED,
    *_FWD_IMPLICIT_GEMM_GROUPED,
    *_FWD_CUTE_IMPLICIT,
    *_FWD_EXPLICIT,
    *_FWD_CUTE_SM90,
    *_FWD_CUTE_GROUPED_SM90,
    *_FWD_EXPLICIT_GROUPED,
]


def _get_filtered_forward_params() -> List[Tuple[str, Dict[str, Any]]]:
    """Get forward benchmark parameters filtered by environment variable.

    For "auto", returns the static superset of all adaptive candidates.
    For "all", returns the full exhaustive set.
    """
    return _filter_benchmark_params_by_env_config(
        _BENCHMARK_FORWARD_PARAMS_ALL_AUTO, WARPCONVNET_FWD_ALGO_MODE, is_forward=True
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
            _ALL_BENCHMARK_FORWARD_PARAMS
            if is_forward
            else _ALL_BENCHMARK_BACKWARD_PARAMS
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
