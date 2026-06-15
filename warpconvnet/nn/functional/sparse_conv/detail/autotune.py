# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Auto-tuning benchmark runners and cache management for sparse convolution
# algorithm selection.

from typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

from enum import Enum

import torch
from torch import Tensor

from warpconvnet._offset_gemm_constants import BACKEND_CUTE_GROUPED_SM90, BACKEND_CUTE_SM90
from warpconvnet.utils.benchmark_cache import (
    SpatiallySparseConvConfig,
    generic_benchmark_get_namespace,
    generic_benchmark_update_entry,
)
from warpconvnet.utils.timer import CUDATimer
from warpconvnet.utils.logger import get_logger

from .algo_params import (
    _get_filtered_AB_params,
    _get_filtered_AtB_params,
)
from .backends import (
    BwdCtx,
    FwdCtx,
    benchmark_backward,
    benchmark_forward,
)

logger = get_logger(__name__)

# Benchmark iterations for auto-tuning. More iterations = more reliable
# winner selection but slower first-iteration auto-tune.
#
# Phase 1 (screening): every candidate gets _BENCHMARK_NUM_ITERS samples,
# median taken. Used to pick top-k for re-timing.
# Phase 2 (tie-break): top _BENCHMARK_TIE_BREAK_TOP_K candidates re-timed
# with _BENCHMARK_TIE_BREAK_NUM_ITERS samples, median wins.
#
# Without phase 2, top candidates within 3% of each other get ranked by
# noise — caused bimodal e2e bench at RES=32 C=1024 (best 1.24x, worst
# 2.0x same shape across runs).
_BENCHMARK_NUM_WARMUP = 3
_BENCHMARK_NUM_ITERS = 7

# Tie-break re-timing: re-time top-K candidates with more samples to
# stabilize ranking when phase-1 medians are within tie-break threshold.
_BENCHMARK_TIE_BREAK_TOP_K = 3
_BENCHMARK_TIE_BREAK_NUM_ITERS = 21
# If phase-1 best vs k-th-best ratio < this, run tie-break. Otherwise
# the gap is larger than expected noise so phase-1 ranking is trusted.
_BENCHMARK_TIE_BREAK_THRESHOLD = 1.10

# Track whether auto-tune banner has been shown (once per process)
_AUTOTUNE_BANNER_SHOWN = False

# ---------------------------------------------------------------------------
# In-memory benchmark result caches (config -> sorted list of results)
# ---------------------------------------------------------------------------

_BENCHMARK_AB_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}  # AB gather-scatter (forward): Y = A @ B
_BENCHMARK_ABT_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}  # ABt gather-scatter (dgrad): dX = dY @ W^T
_BENCHMARK_ATB_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}  # AtB gather-gather (wgrad): dW = A^T @ dY

# ---------------------------------------------------------------------------
# Serialization helpers for cache
# ---------------------------------------------------------------------------


def _serialize_algo_value(algo: Any) -> str:
    if isinstance(algo, Enum):
        return str(algo.value)
    return str(algo)


def _serialize_benchmark_results(
    results: List[Tuple[Union[str, Any], Dict[str, Any], float]],
) -> List[Tuple[str, Dict[str, Any], float]]:
    return [
        (_serialize_algo_value(algo), params, float(metric)) for algo, params, metric in results
    ]


def _normalize_cached_params(algo: str, params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    normalized = dict(params)
    if (
        algo == "cute_implicit_gemm_sm90"
        and "tile_id" not in normalized
        and "mma_tile" in normalized
    ):
        normalized["backend"] = normalized.get("backend", BACKEND_CUTE_SM90)
        normalized["tile_id"] = normalized.pop("mma_tile")
    elif algo == "cute_grouped_sm90" and "tile_id" not in normalized and "mma_tile" in normalized:
        normalized["backend"] = normalized.get("backend", BACKEND_CUTE_GROUPED_SM90)
        normalized["tile_id"] = normalized.pop("mma_tile")
    return normalized


def _normalize_cached_algo(algo: str) -> str:
    # Cache namespaces changed with the registry migration. The algorithm names
    # stay stable; only params are rewritten by _normalize_cached_params.
    return algo


def _normalize_benchmark_results(
    results: Any,
    is_forward: bool,
) -> List[Tuple[str, Dict[str, Any], float]]:
    if results is None:
        return []
    out: List[Tuple[str, Dict[str, Any], float]] = []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        algo_raw, params, metric = item
        algo_str = _normalize_cached_algo(_serialize_algo_value(algo_raw))
        out.append((algo_str, _normalize_cached_params(algo_str, params), float(metric)))
    return out


# ---------------------------------------------------------------------------
# Load cached benchmark results at module initialization
# ---------------------------------------------------------------------------


def _initialize_benchmark_cache():
    """Load cached benchmark results and populate global dictionaries."""
    ab_ns = generic_benchmark_get_namespace("AB_gather_scatter")
    abt_ns = generic_benchmark_get_namespace("ABt_gather_scatter")
    atb_ns = generic_benchmark_get_namespace("AtB_gather_gather")

    if isinstance(ab_ns, dict):
        for k, v in ab_ns.items():
            _BENCHMARK_AB_RESULTS[k] = _normalize_benchmark_results(v, is_forward=True)
    if isinstance(abt_ns, dict):
        for k, v in abt_ns.items():
            _BENCHMARK_ABT_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)
    if isinstance(atb_ns, dict):
        for k, v in atb_ns.items():
            _BENCHMARK_ATB_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)

    n_ab = len(ab_ns) if ab_ns else 0
    n_abt = len(abt_ns) if abt_ns else 0
    n_atb = len(atb_ns) if atb_ns else 0
    if n_ab or n_abt or n_atb:
        logger.info(
            f"Loaded {n_ab} AB_gather_scatter (fwd), {n_abt} ABt_gather_scatter (dgrad), "
            f"{n_atb} AtB_gather_gather (wgrad) benchmark configurations from cache"
        )


def _on_cache_merge(namespace: str, merged_dict: dict) -> None:
    """Callback from GenericBenchmarkCache when disk data is merged.

    Refreshes the in-memory auto-tune results with entries from other ranks.
    """
    if namespace == "AB_gather_scatter":
        for k, v in merged_dict.items():
            if k not in _BENCHMARK_AB_RESULTS:
                _BENCHMARK_AB_RESULTS[k] = _normalize_benchmark_results(v, is_forward=True)
    elif namespace == "ABt_gather_scatter":
        for k, v in merged_dict.items():
            if k not in _BENCHMARK_ABT_RESULTS:
                _BENCHMARK_ABT_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)
    elif namespace == "AtB_gather_gather":
        for k, v in merged_dict.items():
            if k not in _BENCHMARK_ATB_RESULTS:
                _BENCHMARK_ATB_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)


# Initialize cache on module load
_initialize_benchmark_cache()

# Register callback so other ranks' results refresh our in-memory cache
from warpconvnet.utils.benchmark_cache import get_generic_benchmark_cache as _get_cache

_get_cache().register_on_merge_callback(_on_cache_merge)


def _tie_break_top_k(
    sorted_results: List[Tuple[str, Dict[str, Any], float]],
    run_one,
    timer,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Re-time top candidates with more samples to stabilize ranking.

    Phase-1 timed every candidate with `_BENCHMARK_NUM_ITERS` samples.
    When top candidates are close (within `_BENCHMARK_TIE_BREAK_THRESHOLD`),
    the median ranking is dominated by noise. This helper re-times the top
    `_BENCHMARK_TIE_BREAK_TOP_K` candidates with `_BENCHMARK_TIE_BREAK_NUM_ITERS`
    samples each, replaces their phase-1 medians with the new ones, and
    re-sorts.

    Args:
        sorted_results: List of (algo, params, median_ms) sorted ascending
            by median_ms.
        run_one: Callable run_one(algo, params) executing one pass; result
            ignored.
        timer: CUDA event timer with `with timer:` and `.elapsed_time`.

    Returns:
        Re-sorted result list (full length, not just top-k).
    """
    if len(sorted_results) <= 1:
        return sorted_results
    best_time = sorted_results[0][2]
    if best_time <= 0:
        return sorted_results
    # How many candidates fall within the tie-break threshold?
    cutoff = best_time * _BENCHMARK_TIE_BREAK_THRESHOLD
    in_band = [(i, r) for i, r in enumerate(sorted_results) if r[2] <= cutoff]
    in_band = in_band[:_BENCHMARK_TIE_BREAK_TOP_K]
    if len(in_band) <= 1:
        return sorted_results

    rebuilt: List[Tuple[str, Dict[str, Any], float]] = list(sorted_results)
    for i, (algo, params, _) in in_band:
        try:
            iter_times = []
            for _ in range(_BENCHMARK_TIE_BREAK_NUM_ITERS):
                with timer:
                    run_one(algo, params)
                iter_times.append(timer.elapsed_time)
            torch.cuda.synchronize()
            median = sorted(iter_times)[len(iter_times) // 2]
            rebuilt[i] = (algo, params, median)
        except (RuntimeError, Exception):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            continue
    rebuilt.sort(key=lambda x: x[2])
    return rebuilt


# ---------------------------------------------------------------------------
# Forward benchmark runner
# ---------------------------------------------------------------------------


def _run_forward_benchmarks(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    warmup_iters: int = _BENCHMARK_NUM_WARMUP,
    benchmark_iters: int = _BENCHMARK_NUM_ITERS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    groups: int = 1,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Benchmark different forward algorithms and return sorted results (best first)."""
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[Tuple[str, Dict[str, Any], float]] = []
    timer = CUDATimer()

    def _execute_single_fwd_pass(algo_mode: str, params_config: Dict[str, Any]) -> Optional[int]:
        # Route through the shared registry so the benchmarked kernel is exactly
        # the one dispatch.py executes. Unavailable/unknown backends raise here
        # and are skipped by the surrounding try/except (same as returning a
        # non-zero status).
        ctx = FwdCtx(
            in_features=in_features,
            weight=weight,
            kernel_map=kernel_map,
            num_out_coords=num_out_coords,
            compute_dtype=compute_dtype,
            params=params_config,
            fwd_block_size=None,
            groups=groups,
        )
        return benchmark_forward(algo_mode, ctx)

    params_to_use = custom_params if custom_params is not None else _get_filtered_AB_params()
    # Filter out IMPLICIT_GEMM when dtype is float64 (unsupported by kernels)
    dtype_to_check = compute_dtype if compute_dtype is not None else in_features.dtype
    if dtype_to_check == torch.float64:
        params_to_use = [(algo, cfg) for (algo, cfg) in params_to_use if algo != "implicit_gemm"]

    global _AUTOTUNE_BANNER_SHOWN
    num_candidates = len(params_to_use)
    N_in = in_features.shape[0]
    C_in_val = in_features.shape[1]
    C_out_val = weight.shape[2]
    if not _AUTOTUNE_BANNER_SHOWN:
        logger.warning(
            "WarpConvNet: Auto-tuning sparse convolution algorithms. "
            "The first few iterations will be slow while optimal kernels are selected. "
            "Results are cached to ~/.cache/warpconvnet/ for future runs."
        )
        _AUTOTUNE_BANNER_SHOWN = True
    logger.info(
        f"Auto-tuning forward (N={N_in}, C_in={C_in_val}, C_out={C_out_val}, "
        f"{num_candidates} candidates)..."
    )

    for idx, (algo_mode, params_config) in enumerate(params_to_use, 1):
        # Warmup runs
        status = None
        try:
            for _ in range(warmup_iters):
                status = _execute_single_fwd_pass(algo_mode, params_config)
                if isinstance(status, int) and status != 0:
                    break
            # Sync to catch async CUDA errors from this candidate
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (error: {e})")
            # Clear CUDA error state to prevent corruption of subsequent candidates.
            # cudaGetLastError() resets the error flag; synchronize() then succeeds.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # Clear error state by consuming the sync exception  # Clear again after sync failure
            continue

        if isinstance(status, int) and status != 0:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (unsupported)")
            continue

        # Benchmark runs — collect all times and take median for robustness
        iter_times = []

        try:
            for _ in range(benchmark_iters):
                with timer:
                    _execute_single_fwd_pass(algo_mode, params_config)
                iter_times.append(timer.elapsed_time)
            # Sync to catch async errors
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode} — failed during benchmark (error: {e})"
            )
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # Clear error state by consuming the sync exception
            continue

        if iter_times:
            median_time_ms = sorted(iter_times)[len(iter_times) // 2]
            all_benchmark_results.append((algo_mode, params_config, median_time_ms))
            _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode}"
                + (f" ({_param_str})" if _param_str else "")
                + f" — {median_time_ms:.2f}ms"
            )

    if not all_benchmark_results:
        logger.warning("No forward benchmark succeeded. Falling back to explicit_gemm.")
        with timer:
            _execute_single_fwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    # Tie-break re-timing: when top candidates are within threshold (3% by
    # default), re-time them with more samples to stabilize ranking.
    all_benchmark_results = _tie_break_top_k(
        all_benchmark_results,
        run_one=lambda algo, cfg: _execute_single_fwd_pass(algo, cfg),
        timer=timer,
    )

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    _best_param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    logger.info(
        f"Auto-tune forward complete: {best_algo}"
        + (f" ({_best_param_str})" if _best_param_str else "")
        + f" — {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results


# ---------------------------------------------------------------------------
# Backward benchmark runner
# ---------------------------------------------------------------------------


def _run_backward_benchmarks(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    warmup_iters: int = _BENCHMARK_NUM_WARMUP,
    benchmark_iters: int = _BENCHMARK_NUM_ITERS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    needs_input_grad: Tuple[bool, bool] = (True, True),
    groups: int = 1,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Benchmark different backward algorithms and return sorted results (best first).

    Args:
        needs_input_grad: Tuple (need_dgrad, need_wgrad). When benchmarking
            dgrad and wgrad separately, set one to False to measure only the
            other direction.
    """
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[Tuple[str, Dict[str, Any], float]] = []
    timer = CUDATimer()

    def _execute_single_bwd_pass(algo_mode: str, params_config: Dict[str, Any]) -> Optional[int]:
        # Route through the shared registry so the benchmarked kernel is exactly
        # the one dispatch.py executes. weight_T is None here (autotune does not
        # pre-transpose); the mask/cute_grouped backends recompute it as needed.
        ctx = BwdCtx(
            grad_output=grad_output,
            in_features=in_features,
            weight=weight,
            kernel_map=kernel_map,
            num_out_coords=num_out_coords,
            compute_dtype=compute_dtype,
            device=device,
            needs_input_grad=needs_input_grad,
            params=params_config,
            groups=groups,
        )
        return benchmark_backward(algo_mode, ctx)

    params_to_use = custom_params if custom_params is not None else _get_filtered_AtB_params()
    # Filter out IMPLICIT_GEMM when dtype is float64 (unsupported by kernels)
    dtype_to_check = compute_dtype if compute_dtype is not None else grad_output.dtype
    if dtype_to_check == torch.float64:
        params_to_use = [(algo, cfg) for (algo, cfg) in params_to_use if algo != "implicit_gemm"]

    global _AUTOTUNE_BANNER_SHOWN
    num_candidates = len(params_to_use)
    N_in = in_features.shape[0]
    C_in_val = in_features.shape[1]
    C_out_val = weight.shape[2]
    if not _AUTOTUNE_BANNER_SHOWN:
        logger.warning(
            "WarpConvNet: Auto-tuning sparse convolution algorithms. "
            "The first few iterations will be slow while optimal kernels are selected. "
            "Results are cached to ~/.cache/warpconvnet/ for future runs."
        )
        _AUTOTUNE_BANNER_SHOWN = True
    logger.info(
        f"Auto-tuning backward (N={N_in}, C_in={C_in_val}, C_out={C_out_val}, "
        f"{num_candidates} candidates)..."
    )

    for idx, (algo_mode, params_config) in enumerate(params_to_use, 1):
        status = None
        try:
            for _ in range(warmup_iters):
                status = _execute_single_bwd_pass(algo_mode, params_config)
                if isinstance(status, int) and status != 0:
                    break
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (error: {e})")
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # Clear error state by consuming the sync exception
            continue

        if isinstance(status, int) and status != 0:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (unsupported)")
            continue

        # Benchmark runs — collect all times and take median for robustness
        iter_times = []

        if benchmark_iters == 0:
            if warmup_iters == 0:
                continue
        else:
            try:
                for _ in range(benchmark_iters):
                    with timer:
                        _execute_single_bwd_pass(algo_mode, params_config)
                    iter_times.append(timer.elapsed_time)
                torch.cuda.synchronize()
            except (RuntimeError, Exception) as e:
                logger.debug(
                    f"  [{idx}/{num_candidates}] {algo_mode} — failed during benchmark (error: {e})"
                )
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass  # Clear error state by consuming the sync exception
                continue

        if iter_times:
            median_time_ms = sorted(iter_times)[len(iter_times) // 2]
            all_benchmark_results.append((algo_mode, params_config, median_time_ms))
            _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode}"
                + (f" ({_param_str})" if _param_str else "")
                + f" — {median_time_ms:.2f}ms"
            )

    if not all_benchmark_results:
        logger.warning("No backward benchmark succeeded. Falling back to explicit_gemm.")
        with timer:
            _execute_single_bwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    # Tie-break re-timing for stable ranking when top candidates are
    # within threshold. See _tie_break_top_k docstring.
    all_benchmark_results = _tie_break_top_k(
        all_benchmark_results,
        run_one=lambda algo, cfg: _execute_single_bwd_pass(algo, cfg),
        timer=timer,
    )

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    _best_param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    logger.info(
        f"Auto-tune backward complete: {best_algo}"
        + (f" ({_best_param_str})" if _best_param_str else "")
        + f" — {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results
