# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Algorithm enums, benchmark parameter lists, and parameter filtering for
# sparse convolution AB (gather-scatter) and AtB (gather-gather) algorithm selection.

from typing import Any, Dict, List, Tuple, Union

from enum import Enum

from warpconvnet.constants import (
    WARPCONVNET_FWD_ALGO_MODE as WARPCONVNET_AB_ALGO_MODE,
    WARPCONVNET_BWD_ALGO_MODE as WARPCONVNET_ATB_ALGO_MODE,
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
    _HAS_CUTE_SM90 = _HAS_SM90_HARDWARE and hasattr(_C.gemm, "cute_gemm_sm90_AD_gather_scatter")
    _HAS_CUTE_GROUPED_SM90 = _HAS_SM90_HARDWARE and hasattr(
        _C.gemm, "cute_gemm_sm90_grouped_AD_gather_scatter"
    )
except Exception:
    _HAS_CUTE_SM90 = False
    _HAS_CUTE_GROUPED_SM90 = False

# ---------------------------------------------------------------------------
# Enums for granular algorithm control
# ---------------------------------------------------------------------------


class SPARSE_CONV_AB_ALGO_MODE(Enum):
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
    TRIMMED = "trimmed"  # Benchmark reduced set (excludes dead-weight)


class SPARSE_CONV_ATB_ALGO_MODE(Enum):
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
    TRIMMED = "trimmed"  # Benchmark reduced set (excludes dead-weight)


# ---------------------------------------------------------------------------
# AB (gather-scatter) benchmark candidate building blocks
# ---------------------------------------------------------------------------
#
# Used by both forward and dgrad (both are A @ B with gather-scatter).
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

# Algo building blocks (assembled conditionally by _get_adaptive_AB_params)
_AB_CUTLASS_IMPLICIT = [("cutlass_implicit_gemm", {})] if _HAS_CUTLASS_BACKEND else []
_AB_CUTLASS_GROUPED = (
    [("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]]
    if _HAS_CUTLASS_BACKEND
    else []
)
# SM90 WGMMA building blocks (only on Hopper+ hardware with SM90 compiled support)
_AB_CUTE_SM90 = (
    []
    if not _HAS_CUTE_SM90
    else [
        ("cute_implicit_gemm_sm90", {"mma_tile": 100}),
        ("cute_implicit_gemm_sm90", {"mma_tile": 103}),
        ("cute_implicit_gemm_sm90", {"mma_tile": 104}),
    ]
)
_AB_CUTE_GROUPED_SM90 = (
    []
    if not _HAS_CUTE_GROUPED_SM90
    else [
        ("cute_grouped_sm90", {"mma_tile": 103, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 100, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 101, "use_cp_async": True}),
        ("cute_grouped_sm90", {"mma_tile": 104, "use_cp_async": True}),
    ]
)
_AB_CUTE_GROUPED = (
    []
    if not _HAS_CUTE_GROUPED
    else [
        ("cute_grouped", {"mma_tile": 3}),
        ("cute_grouped", {"mma_tile": 0}),
        ("cute_grouped", {"mma_tile": 1}),
    ]
)
_AB_MASK_IMPLICIT_GEMM = [
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 0}),  # Tile128x128x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 1}),  # Tile128x64x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 2}),  # Tile64x128x32
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 3}),  # Tile64x64x32
]

# ---------------------------------------------------------------------------
# AtB (gather-gather / wgrad) specific building blocks
# ---------------------------------------------------------------------------

_ATB_IMPLICIT_GEMM = [
    (
        "implicit_gemm",
        {"gemm_block_size": 16, "split_k_threads_per_block": 256, "split_k_factor": 2},
    ),
]
_ATB_EXPLICIT = [("explicit_gemm", {})]
_ATB_EXPLICIT_GROUPED = [("explicit_gemm_grouped", {"saturation_m": 2000})]


import math as _math


def _get_adaptive_AB_params(
    in_channels: int,
    out_channels: int,
    kernel_volume: int,
    num_in_coords: int = 0,
    voxel_size: Union[Tuple[int, ...], None] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get AB (gather-scatter) benchmark candidates adapted to the convolution dimensions.

    Used for both forward and dgrad passes (both are A @ B with gather-scatter).

    Based on time-weighted analysis of 148 configs (SM 8.9, cuBLAS 12.9.1.4):
      mask_implicit_gemm: 56% wins — dominates at ch<=256
      cutlass_grouped_hybrid: 20% — wins at large N + large channels
      cutlass_implicit_gemm: 17% — wins at large N + ch>256
      cute_grouped: 6% — wins at small N + ch>256
    """
    max_ch = max(in_channels, out_channels)
    log_n = _math.ceil(_math.log2(num_in_coords)) if num_in_coords > 1 else 0

    # kv >= 64 (5x5x5): no benchmark data, include broad set
    if kernel_volume >= 64:
        params: List[Tuple[str, Dict[str, Any]]] = []
        params.extend(_AB_CUTE_GROUPED)
        params.extend(_AB_MASK_IMPLICIT_GEMM)
        params.extend(_AB_CUTLASS_IMPLICIT)
        params.extend(_AB_CUTLASS_GROUPED)
        params.extend(_AB_CUTE_SM90)
        params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # --- kv <= 27 below ---

    # Small N (N <= 10K, log10 <= 4): mask dominates (92-100%), cute_grouped for ch>256
    if 0 < log_n <= 14:
        params = []
        params.extend(_AB_MASK_IMPLICIT_GEMM)
        params.extend(_AB_CUTE_GROUPED)
        if max_ch > 256:
            params.extend(_AB_CUTLASS_IMPLICIT)
        params.extend(_AB_CUTE_SM90)
        params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # Medium N (10K < N <= 100K, log10 = 5): mask (69%), cutlass (27%)
    if 0 < log_n <= 17:
        params = []
        params.extend(_AB_MASK_IMPLICIT_GEMM)
        params.extend(_AB_CUTLASS_IMPLICIT)
        if max_ch > 256:
            params.extend(_AB_CUTLASS_GROUPED)
        params.extend(_AB_CUTE_GROUPED)
        params.extend(_AB_CUTE_SM90)
        params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # Large N (N > 100K) or unknown N: cutlass dominates ch>256,
    # mask wins ch<=64, cutlass_grouped wins ch 65-256
    params = []
    params.extend(_AB_MASK_IMPLICIT_GEMM)
    params.extend(_AB_CUTLASS_IMPLICIT)
    params.extend(_AB_CUTLASS_GROUPED)
    params.extend(_AB_CUTE_GROUPED)
    params.extend(_AB_CUTE_SM90)
    params.extend(_AB_CUTE_GROUPED_SM90)
    return params


# Trimmed AB benchmark candidates ("trimmed"): excludes algorithms
# that had 0% wins in earlier time-weighted analysis. Used by "all" mode
# prior to the cuBLAS fix; may exclude algorithms that are now competitive.
_TRIMMED_AB_PARAMS = [
    *([] if not _HAS_CUTLASS_BACKEND else [("cutlass_implicit_gemm", {})]),
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
    *(
        []
        if not _HAS_CUTE_SM90
        else [
            ("cute_implicit_gemm_sm90", {"mma_tile": tile}) for tile in [100, 101, 102, 103, 104]
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
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 0}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 1}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 2}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 3}),
]

# Exhaustive AB benchmark candidates ("all"): every algorithm and
# parameter combination. Nothing excluded.
_ALL_AB_PARAMS = [
    # Explicit GEMM (per-offset matmul via cuBLAS)
    ("explicit_gemm", {}),
    *[("explicit_gemm_grouped", {"saturation_m": m}) for m in [2000, 5000, 10000]],
    # Implicit GEMM (SIMT gather-scatter)
    *[("implicit_gemm", {"fwd_block_size": bs}) for bs in [4, 16, 32]],
    *[
        ("implicit_gemm_grouped", {"fwd_block_size": bs, "saturation_m": m})
        for bs in [16]
        for m in [2000, 5000, 10000]
    ],
    # CUTLASS per-offset gather-scatter
    *([] if not _HAS_CUTLASS_BACKEND else [("cutlass_implicit_gemm", {})]),
    *(
        [("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]]
        if _HAS_CUTLASS_BACKEND
        else []
    ),
    # CuTe per-offset gather-scatter
    *([] if not _HAS_CUTE_BACKEND else [("cute_implicit_gemm", {})]),
    # CuTe grouped fused multi-offset
    *([] if not _HAS_CUTE_GROUPED else [("cute_grouped", {"mma_tile": t}) for t in [0, 1, 2, 3]]),
    # SM90 WGMMA
    *(
        []
        if not _HAS_CUTE_SM90
        else [
            ("cute_implicit_gemm_sm90", {"mma_tile": tile}) for tile in [100, 101, 102, 103, 104]
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
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 0}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 1}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 2}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 3}),
]

# ---------------------------------------------------------------------------
# AtB (gather-gather / wgrad) benchmark candidate lists
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

# Static AtB params (used by _get_filtered_AtB_params for env var filtering)
_ATB_PARAMS_AUTO = [
    *_AB_MASK_IMPLICIT_GEMM,
    *_AB_CUTE_GROUPED,
    *_AB_CUTLASS_IMPLICIT,
    *_AB_CUTLASS_GROUPED,
    *_ATB_EXPLICIT,
    *_ATB_EXPLICIT_GROUPED,
    *_ATB_IMPLICIT_GEMM,
    *_AB_CUTE_SM90,
    *_AB_CUTE_GROUPED_SM90,
]


def _get_adaptive_AtB_params(
    in_channels: int,
    out_channels: int,
    kernel_volume: int,
    num_in_coords: int = 0,
    voxel_size: Union[Tuple[int, ...], None] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get AtB (wgrad) benchmark candidates (A^T @ B reduction, no scatter).

    Based on 146-config analysis (SM 8.9, cuBLAS 12.9.1.4):
      cute_grouped 64%, cutlass_grouped_hybrid 12%, explicit_gemm_grouped 10%,
      implicit_gemm 5%, explicit_gemm 5%, cutlass_implicit_gemm 4%
    """
    max_ch = max(in_channels, out_channels)
    log_n = _math.ceil(_math.log2(num_in_coords)) if num_in_coords > 1 else 0

    if kernel_volume >= 64:
        params: List[Tuple[str, Dict[str, Any]]] = []
        params.extend(_AB_CUTE_GROUPED)
        params.extend(_AB_CUTLASS_GROUPED)
        params.extend(_ATB_EXPLICIT)
        params.extend(_ATB_EXPLICIT_GROUPED)
        params.extend(_ATB_IMPLICIT_GEMM)
        params.extend(_AB_CUTE_SM90)
        params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # Small N (N <= 10K): cute_grouped 57-100%, explicit_grouped 15%, implicit_gemm 8%
    if 0 < log_n <= 14:
        params = []
        params.extend(_AB_CUTE_GROUPED)
        params.extend(_ATB_EXPLICIT_GROUPED)
        if max_ch <= 64:
            params.extend(_ATB_IMPLICIT_GEMM)
        params.extend(_AB_CUTE_SM90)
        params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # Medium N (10K-100K): cute_grouped 77%, explicit 23%
    if 0 < log_n <= 17:
        params = []
        params.extend(_AB_CUTE_GROUPED)
        params.extend(_ATB_EXPLICIT)
        params.extend(_ATB_EXPLICIT_GROUPED)
        params.extend(_AB_CUTE_SM90)
        params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # Large N (N > 100K) or unknown: cutlass_grouped 40%, cute_grouped 30%, cutlass 30%
    params = []
    params.extend(_AB_CUTE_GROUPED)
    params.extend(_AB_CUTLASS_GROUPED)
    params.extend(_AB_CUTLASS_IMPLICIT)
    params.extend(_ATB_EXPLICIT_GROUPED)
    if max_ch <= 64:
        params.extend(_ATB_EXPLICIT)
    params.extend(_AB_CUTE_SM90)
    params.extend(_AB_CUTE_GROUPED_SM90)
    return params


# Trimmed AtB benchmark candidates ("trimmed"): excludes algorithms
# that had 0% wins in earlier analysis (pre-cuBLAS fix).
_TRIMMED_ATB_PARAMS = [
    *([] if not _HAS_CUTLASS_BACKEND else [("cutlass_implicit_gemm", {})]),
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
    *[("explicit_gemm_grouped", {"saturation_m": m}) for m in [2000, 5000, 10000]],
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
    *(
        []
        if not _HAS_CUTE_SM90
        else [
            ("cute_implicit_gemm_sm90", {"mma_tile": tile}) for tile in [100, 101, 102, 103, 104]
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
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 0}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 1}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 2}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 3}),
]

# Exhaustive AtB benchmark candidates ("all"): every algorithm and
# parameter combination. Nothing excluded.
_ALL_ATB_PARAMS = [
    # Explicit GEMM
    ("explicit_gemm", {}),
    *[("explicit_gemm_grouped", {"saturation_m": m}) for m in [2000, 5000, 10000]],
    # Implicit GEMM (SIMT)
    *[
        (
            "implicit_gemm",
            {
                "gemm_block_size": gemm_block_size,
                "split_k_threads_per_block": 256,
                "split_k_factor": split_k_factor,
            },
        )
        for gemm_block_size in [4, 16, 32]
        for split_k_factor in [2, 4, 8, 16]
    ],
    *[
        (
            "implicit_gemm_grouped",
            {
                "gemm_block_size": 16,
                "split_k_threads_per_block": 256,
                "split_k_factor": split_k_factor,
                "saturation_m": m,
            },
        )
        for split_k_factor in [2, 4]
        for m in [2000, 5000, 10000]
    ],
    # CUTLASS per-offset
    *([] if not _HAS_CUTLASS_BACKEND else [("cutlass_implicit_gemm", {})]),
    *(
        [("cutlass_grouped_hybrid", {"saturation_m": m}) for m in [2000, 5000, 10000]]
        if _HAS_CUTLASS_BACKEND
        else []
    ),
    # CuTe per-offset
    *([] if not _HAS_CUTE_BACKEND else [("cute_implicit_gemm", {})]),
    # CuTe grouped
    *([] if not _HAS_CUTE_GROUPED else [("cute_grouped", {"mma_tile": t}) for t in [0, 1, 2, 3]]),
    # SM90 WGMMA
    *(
        []
        if not _HAS_CUTE_SM90
        else [
            ("cute_implicit_gemm_sm90", {"mma_tile": tile}) for tile in [100, 101, 102, 103, 104]
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
    # Mask-based fused implicit GEMM
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 0}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 1}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 2}),
    ("mask_implicit_gemm", {"block_size": 16, "mma_tile": 3}),
]

# ---------------------------------------------------------------------------
# Parameter filtering
# ---------------------------------------------------------------------------


# Static superset of all AB "auto" candidates (union of all adaptive branches).
# Used by _get_filtered_AB_params for env var filtering.
_AB_PARAMS_AUTO = [
    *_AB_MASK_IMPLICIT_GEMM,
    *_AB_CUTLASS_IMPLICIT,
    *_AB_CUTLASS_GROUPED,
    *_AB_CUTE_GROUPED,
    *_AB_CUTE_SM90,
    *_AB_CUTE_GROUPED_SM90,
]


def _get_filtered_AB_params() -> List[Tuple[str, Dict[str, Any]]]:
    """Get AB benchmark parameters filtered by environment variable.

    For "auto", returns the static superset of all adaptive candidates.
    For "all", returns the full exhaustive set.
    """
    return _filter_benchmark_params_by_env_config(
        _AB_PARAMS_AUTO, WARPCONVNET_AB_ALGO_MODE, is_forward=True
    )


def _get_filtered_AtB_params() -> List[Tuple[str, Dict[str, Any]]]:
    """Get AtB benchmark parameters filtered by environment variable."""
    return _filter_benchmark_params_by_env_config(
        _ATB_PARAMS_AUTO, WARPCONVNET_ATB_ALGO_MODE, is_forward=False
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
        is_forward: Whether this is for AB pass (affects which exhaustive set to use)

    Returns:
        Filtered list of benchmark parameters
    """
    if env_config == "all":
        # When "all", use the full exhaustive candidate set (nothing excluded)
        full_params = _ALL_AB_PARAMS if is_forward else _ALL_ATB_PARAMS
        return [(str(algo), params) for algo, params in full_params]

    if env_config == "trimmed":
        # When "trimmed", use the reduced set that excludes 0%-win algorithms
        # from earlier time-weighted analysis (pre-cuBLAS fix)
        trimmed = _TRIMMED_AB_PARAMS if is_forward else _TRIMMED_ATB_PARAMS
        return [(str(algo), params) for algo, params in trimmed]

    if env_config == "auto":
        # When "auto", use adaptive candidate set based on dimensions
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
