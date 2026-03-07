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

from warpconvnet.utils.benchmark_cache import (
    SpatiallySparseConvConfig,
    generic_benchmark_get_namespace,
    generic_benchmark_update_entry,
)
from warpconvnet.utils.timer import CUDATimer
from warpconvnet.utils.logger import get_logger

from .explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
    _explicit_gemm_forward_grouped,
    _explicit_gemm_backward_grouped,
)
from .implicit_direct import (
    _implicit_gemm_forward_logic,
    _implicit_gemm_backward_logic,
    _implicit_gemm_forward_grouped,
    _implicit_gemm_backward_grouped,
)
from .cutlass import (
    _cutlass_implicit_gemm_forward_logic,
    _cutlass_implicit_gemm_backward_logic,
    _cutlass_implicit_gemm_forward_grouped,
    _cutlass_implicit_gemm_backward_grouped,
)
from .algo_params import (
    _HAS_CUTE_BACKEND,
    _HAS_CUTE_GROUPED,
    _get_filtered_forward_params,
    _get_filtered_backward_params,
)

if _HAS_CUTE_BACKEND:
    from .cute import (
        _cute_implicit_gemm_forward_logic,
        _cute_implicit_gemm_backward_logic,
    )

if _HAS_CUTE_GROUPED:
    from .cute_grouped import (
        _cute_grouped_forward_logic,
        _cute_grouped_backward_logic,
    )

logger = get_logger(__name__)

# Separate benchmark parameters for independent operations
_BENCHMARK_NUM_RUNS = 2

# Track whether auto-tune banner has been shown (once per process)
_AUTOTUNE_BANNER_SHOWN = False

# ---------------------------------------------------------------------------
# In-memory benchmark result caches (config -> sorted list of results)
# ---------------------------------------------------------------------------

_BENCHMARK_FORWARD_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}
_BENCHMARK_BACKWARD_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}

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
        algo_str = _serialize_algo_value(algo_raw)
        out.append((algo_str, params, float(metric)))
    return out


# ---------------------------------------------------------------------------
# Load cached benchmark results at module initialization
# ---------------------------------------------------------------------------


def _initialize_benchmark_cache():
    """Load cached benchmark results and populate global dictionaries."""
    forward_ns = generic_benchmark_get_namespace("sparse_conv_forward")
    backward_ns = generic_benchmark_get_namespace("sparse_conv_backward")

    # Normalize any stored values to strings
    if isinstance(forward_ns, dict):
        for k, v in forward_ns.items():
            _BENCHMARK_FORWARD_RESULTS[k] = _normalize_benchmark_results(v, is_forward=True)
    if isinstance(backward_ns, dict):
        for k, v in backward_ns.items():
            _BENCHMARK_BACKWARD_RESULTS[k] = _normalize_benchmark_results(v, is_forward=False)
    if forward_ns or backward_ns:
        logger.info(
            f"Loaded {len(forward_ns)} forward and {len(backward_ns)} "
            f"backward benchmark configurations from cache"
        )


# Initialize cache on module load
_initialize_benchmark_cache()


# ---------------------------------------------------------------------------
# Forward benchmark runner
# ---------------------------------------------------------------------------


def _run_forward_benchmarks(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    warmup_iters: int = _BENCHMARK_NUM_RUNS // 2,
    benchmark_iters: int = _BENCHMARK_NUM_RUNS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Benchmark different forward algorithms and return sorted results (best first)."""
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[Tuple[str, Dict[str, Any], float]] = []
    timer = CUDATimer()

    def _execute_single_fwd_pass(algo_mode: str, params_config: Dict[str, Any]) -> Optional[int]:
        if algo_mode == "explicit_gemm":
            _ = _explicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
            )
        elif algo_mode == "implicit_gemm":
            _ = _implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                **params_config,
            )
        elif algo_mode == "cutlass_implicit_gemm":
            status = _cutlass_implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                accumulator_type=params_config.get("accumulator_type", torch.float32),
            )
            if isinstance(status, int) and status != 0:
                return status
        elif algo_mode == "explicit_gemm_grouped":
            _ = _explicit_gemm_forward_grouped(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                saturation_m=params_config.get("saturation_m", 5000),
            )
        elif algo_mode == "implicit_gemm_grouped":
            _ = _implicit_gemm_forward_grouped(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                fwd_block_size=params_config.get("fwd_block_size", 16),
                saturation_m=params_config.get("saturation_m", 5000),
            )
        elif algo_mode == "cute_implicit_gemm":
            if not _HAS_CUTE_BACKEND:
                return -1
            status = _cute_implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
            )
            if isinstance(status, int) and status != 0:
                return status
        elif algo_mode == "cutlass_grouped_hybrid":
            status = _cutlass_implicit_gemm_forward_grouped(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                accumulator_type=params_config.get("accumulator_type", torch.float32),
                saturation_m=params_config.get("saturation_m", 5000),
            )
            if isinstance(status, int) and status != 0:
                return status
        elif algo_mode == "cute_grouped":
            if not _HAS_CUTE_GROUPED:
                return -1
            status = _cute_grouped_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                mma_tile=params_config.get("mma_tile", 3),
            )
            if isinstance(status, int) and status != 0:
                return status
        else:
            raise ValueError(f"Unsupported algo_mode in _execute_single_fwd_pass: {algo_mode}")

    params_to_use = custom_params if custom_params is not None else _get_filtered_forward_params()
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
        except (RuntimeError, Exception) as e:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (error: {e})")
            continue

        if isinstance(status, int) and status != 0:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (unsupported)")
            continue

        # Benchmark runs
        current_algo_min_time_ms = float("inf")

        try:
            for _ in range(benchmark_iters):
                with timer:
                    _execute_single_fwd_pass(algo_mode, params_config)
                current_algo_min_time_ms = min(current_algo_min_time_ms, timer.elapsed_time)
        except (RuntimeError, Exception) as e:
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode} — failed during benchmark (error: {e})"
            )
            continue

        if current_algo_min_time_ms != float("inf"):
            all_benchmark_results.append((algo_mode, params_config, current_algo_min_time_ms))
            _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode}"
                + (f" ({_param_str})" if _param_str else "")
                + f" — {current_algo_min_time_ms:.2f}ms"
            )

    if not all_benchmark_results:
        logger.warning("No forward benchmark succeeded. Falling back to explicit_gemm.")
        with timer:
            _execute_single_fwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

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
    warmup_iters: int = _BENCHMARK_NUM_RUNS // 2,
    benchmark_iters: int = _BENCHMARK_NUM_RUNS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Benchmark different backward algorithms and return sorted results (best first)."""
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[Tuple[str, Dict[str, Any], float]] = []
    timer = CUDATimer()

    def _execute_single_bwd_pass(algo_mode: str, params_config: Dict[str, Any]) -> Optional[int]:
        status = None

        if algo_mode == "explicit_gemm":
            _, _ = _explicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                compute_dtype,
                device,
            )
        elif algo_mode == "implicit_gemm":
            _, _ = _implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                params_config.get("gemm_block_size", 16),
                params_config.get("split_k_threads_per_block", 128),
                params_config.get("split_k_factor", 4),
                compute_dtype,
            )
        elif algo_mode == "cutlass_implicit_gemm":
            status, _ = _cutlass_implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                accumulator_type=params_config.get("accumulator_type", torch.float32),
                device=device,
            )
            if isinstance(status, int) and status != 0:
                return status
        elif algo_mode == "cute_implicit_gemm":
            if not _HAS_CUTE_BACKEND:
                return -1
            result = _cute_implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                device=device,
            )
            if isinstance(result[0], int) and result[0] != 0:
                return result[0]
        elif algo_mode == "explicit_gemm_grouped":
            _, _ = _explicit_gemm_backward_grouped(
                grad_output,
                in_features,
                weight,
                kernel_map,
                compute_dtype,
                device,
                saturation_m=params_config.get("saturation_m", 5000),
            )
        elif algo_mode == "implicit_gemm_grouped":
            _, _ = _implicit_gemm_backward_grouped(
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                gemm_block_size=params_config.get("gemm_block_size", 16),
                split_k_threads_per_block=params_config.get("split_k_threads_per_block", 256),
                split_k_factor=params_config.get("split_k_factor", 4),
                compute_dtype=compute_dtype,
                saturation_m=params_config.get("saturation_m", 5000),
            )
        elif algo_mode == "cutlass_grouped_hybrid":
            result = _cutlass_implicit_gemm_backward_grouped(
                grad_output,
                in_features,
                weight,
                kernel_map,
                accumulator_type=params_config.get("accumulator_type", torch.float32),
                device=device,
                saturation_m=params_config.get("saturation_m", 5000),
            )
            if isinstance(result[0], int) and result[0] != 0:
                return result[0]
        elif algo_mode == "cute_grouped":
            if not _HAS_CUTE_GROUPED:
                return -1
            result = _cute_grouped_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                requires_grad=(True, True),
                device=device,
                mma_tile=params_config.get("mma_tile", 3),
            )
            if isinstance(result[0], int) and result[0] != 0:
                return result[0]
        else:
            raise ValueError(f"Unsupported algo_mode in _execute_single_bwd_pass: {algo_mode}")

    params_to_use = custom_params if custom_params is not None else _get_filtered_backward_params()
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
        except (RuntimeError, Exception) as e:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (error: {e})")
            continue

        if isinstance(status, int) and status != 0:
            logger.debug(f"  [{idx}/{num_candidates}] {algo_mode} — skipped (unsupported)")
            continue

        # Benchmark runs
        current_algo_min_time_ms = float("inf")

        if benchmark_iters == 0:
            if warmup_iters == 0:
                continue
        else:
            try:
                for _ in range(benchmark_iters):
                    with timer:
                        _execute_single_bwd_pass(algo_mode, params_config)
                    current_algo_min_time_ms = min(current_algo_min_time_ms, timer.elapsed_time)
            except (RuntimeError, Exception) as e:
                logger.debug(
                    f"  [{idx}/{num_candidates}] {algo_mode} — failed during benchmark (error: {e})"
                )
                continue

        if current_algo_min_time_ms != float("inf"):
            all_benchmark_results.append((algo_mode, params_config, current_algo_min_time_ms))
            _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode}"
                + (f" ({_param_str})" if _param_str else "")
                + f" — {current_algo_min_time_ms:.2f}ms"
            )

    if not all_benchmark_results:
        logger.warning("No backward benchmark succeeded. Falling back to explicit_gemm.")
        with timer:
            _execute_single_bwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    _best_param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    logger.info(
        f"Auto-tune backward complete: {best_algo}"
        + (f" ({_best_param_str})" if _best_param_str else "")
        + f" — {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results
