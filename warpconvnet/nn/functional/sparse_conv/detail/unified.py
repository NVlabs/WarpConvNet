# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

from enum import Enum
import math

import torch
from torch import Tensor
from torch.autograd import Function

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult

from warpconvnet.utils.benchmark_cache import (
    SpatiallySparseConvConfig,
    generic_benchmark_get_namespace,
    generic_benchmark_update_entry,
)
from warpconvnet.utils.timer import CUDATimer
from warpconvnet.utils.ntuple import _pad_tuple
from warpconvnet.constants import (
    WARPCONVNET_FWD_ALGO_MODE,
    WARPCONVNET_BWD_ALGO_MODE,
)
from warpconvnet.utils.logger import get_logger

# Prevent circular import using relative imports
from .explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
)
from .implicit_direct import (
    _implicit_gemm_forward_logic,
    _implicit_gemm_backward_logic,
)
from .cutlass import (
    _cutlass_implicit_gemm_forward_logic,
    _cutlass_implicit_gemm_backward_logic,
)
from .implicit_wmma import (
    _wmma_implicit_gemm_forward_logic,
    _wmma_implicit_gemm_backward_logic,
)

logger = get_logger(__name__)

# Separate benchmark parameters for independent operations
_BENCHMARK_NUM_RUNS = 2

# Forward benchmark candidates: CUTLASS (autotuned per-call), IMPLICIT (block size), EXPLICIT
_BENCHMARK_FORWARD_PARAMS = [
    ("cutlass_implicit_gemm", {}),
    ("wmma_implicit_gemm", {}),
    *[("implicit_gemm", {"fwd_block_size": block_size}) for block_size in [4, 16, 32]],
    ("explicit_gemm", {}),
]

# Backward benchmark candidates: CUTLASS (autotuned per-call), IMPLICIT (configs), EXPLICIT
_BENCHMARK_BACKWARD_PARAMS = [
    ("cutlass_implicit_gemm", {}),
    ("wmma_implicit_gemm", {}),
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
]

_BENCHMARK_FORWARD_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}
_BENCHMARK_BACKWARD_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[str, Dict[str, Any], float]],
] = {}


# Enums for granular algorithm control
class SPARSE_CONV_FWD_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    CUTLASS_IMPLICIT_GEMM = "cutlass_implicit_gemm"
    WMMA_IMPLICIT_GEMM = "wmma_implicit_gemm"
    # EXPLICIT_GEMM_BATCHED = "explicit_gemm_batched" # TODO: Add if supporting
    AUTO = "auto"  # Benchmark and select the best algorithm


class SPARSE_CONV_BWD_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    CUTLASS_IMPLICIT_GEMM = "cutlass_implicit_gemm"
    WMMA_IMPLICIT_GEMM = "wmma_implicit_gemm"
    # EXPLICIT_GEMM_BATCHED = "explicit_gemm_batched" # TODO: Add if supporting
    AUTO = "auto"  # Benchmark and select the best algorithm


# ------------------------------
# Serialization helpers for cache
# ------------------------------


def _serialize_algo_value(algo: Any) -> str:
    if isinstance(algo, Enum):
        return str(algo.value)
    return str(algo)


def _serialize_benchmark_results(
    results: List[
        Tuple[
            Union[str, SPARSE_CONV_FWD_ALGO_MODE, SPARSE_CONV_BWD_ALGO_MODE], Dict[str, Any], float
        ]
    ],
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


# Load cached benchmark results at module initialization
def _initialize_benchmark_cache():
    """Load cached benchmark results and populate global dictionaries."""
    # Load from generic cache namespaces
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


def _filter_benchmark_params_by_env_config(
    all_params: List[Tuple[Union[str, Any], Dict[str, Any]]],
    env_config: Union[str, List[Union[str, Any]]],
    is_forward: bool = True,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Filter benchmark parameters based on environment variable configuration.

    Args:
        all_params: All available benchmark parameters
        env_config: Environment variable value (string or list of algorithm names)
        is_forward: Whether this is for forward pass (affects enum type)

    Returns:
        Filtered list of benchmark parameters
    """
    if env_config == "auto":
        # When "auto", use all available algorithms
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


def _get_filtered_forward_params() -> List[Tuple[str, Dict[str, Any]]]:
    """Get forward benchmark parameters filtered by environment variable."""
    return _filter_benchmark_params_by_env_config(
        _BENCHMARK_FORWARD_PARAMS, WARPCONVNET_FWD_ALGO_MODE, is_forward=True
    )


def _get_filtered_backward_params() -> List[Tuple[str, Dict[str, Any]]]:
    """Get backward benchmark parameters filtered by environment variable."""
    return _filter_benchmark_params_by_env_config(
        _BENCHMARK_BACKWARD_PARAMS, WARPCONVNET_BWD_ALGO_MODE, is_forward=False
    )


def _run_forward_benchmarks(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    warmup_iters: int = _BENCHMARK_NUM_RUNS // 2,
    benchmark_iters: int = _BENCHMARK_NUM_RUNS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> Tuple[str, Dict[str, Any], float]:
    """
    Benchmark different forward algorithms and return the best one with its parameters and runtime.
    The best is determined by the minimum execution time over benchmark_iters.
    """
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    logger.warning(
        "Using benchmarked forward algo. Until the algorithm finds the best parameters, forward performance will be slow."
    )
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
        elif algo_mode == "wmma_implicit_gemm":
            status = _wmma_implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
            )
            if isinstance(status, int) and status != 0:
                return status
        else:
            # Should not happen with current _BENCHMARK_FORWARD_PARAMS
            raise ValueError(f"Unsupported algo_mode in _execute_single_fwd_pass: {algo_mode}")

    params_to_use = custom_params if custom_params is not None else _get_filtered_forward_params()
    # Filter out IMPLICIT_GEMM when dtype is float64 (unsupported by kernels)
    dtype_to_check = compute_dtype if compute_dtype is not None else in_features.dtype
    if dtype_to_check == torch.float64:
        params_to_use = [(algo, cfg) for (algo, cfg) in params_to_use if algo != "implicit_gemm"]

    for algo_mode, params_config in params_to_use:
        # Warmup runs
        status = None
        for _ in range(warmup_iters):
            logger.debug(f"Warmup {algo_mode} {params_config}")
            status = _execute_single_fwd_pass(algo_mode, params_config)
            if isinstance(status, int) and status != 0:
                logger.warning(
                    f"Skipping {algo_mode} because it returned status {status} during warmup."
                )
                continue

        if status is not None and status != 0:
            continue

        # Benchmark runs
        current_algo_min_time_ms = float("inf")  # Min time for this specific algorithm config

        logger.debug(f"Benchmark {algo_mode} {params_config}")
        for _ in range(benchmark_iters):
            with timer:
                _execute_single_fwd_pass(algo_mode, params_config)
            current_algo_min_time_ms = min(current_algo_min_time_ms, timer.elapsed_time)

        logger.debug(
            f"Forward benchmark result: {str(algo_mode)} {params_config} {current_algo_min_time_ms:.2f}ms"
        )
        if current_algo_min_time_ms != float("inf"):
            all_benchmark_results.append((algo_mode, params_config, current_algo_min_time_ms))

    if not all_benchmark_results:
        logger.warning(
            "Warning: No forward benchmark was successful or no algorithms to test. Defaulting to EXPLICIT_GEMM."
        )
        # Return a default entry indicating failure or no successful benchmarks
        with timer:
            _execute_single_fwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]

    logger.debug(
        f"Best forward algo: {str(best_algo)} for log N_in={math.ceil(math.log2(in_features.shape[0])) if in_features.shape[0] > 0 else 0}, log N_out={math.ceil(math.log2(num_out_coords)) if num_out_coords > 0 else 0}, C_in={in_features.shape[1]}, C_out={weight.shape[2]}, K_vol={weight.shape[0]} {best_params} {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results  # Return the sorted list of all results


def _run_backward_benchmarks(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    warmup_iters: int = _BENCHMARK_NUM_RUNS // 2,
    benchmark_iters: int = _BENCHMARK_NUM_RUNS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> Tuple[str, Dict[str, Any], float]:
    """
    Benchmark different backward algorithms and return the best one with its parameters and runtime.
    The best is determined by the minimum execution time over benchmark_iters.
    """
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
        elif algo_mode == "wmma_implicit_gemm":
            status_or_tensor, _ = _wmma_implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                requires_grad=(True, True),
                device=device,
            )
            if isinstance(status_or_tensor, int) and status_or_tensor != 0:
                return status_or_tensor
        else:
            raise ValueError(f"Unsupported algo_mode in _execute_single_bwd_pass: {algo_mode}")

    params_to_use = custom_params if custom_params is not None else _get_filtered_backward_params()
    # Filter out IMPLICIT_GEMM when dtype is float64 (unsupported by kernels)
    dtype_to_check = compute_dtype if compute_dtype is not None else grad_output.dtype
    if dtype_to_check == torch.float64:
        params_to_use = [(algo, cfg) for (algo, cfg) in params_to_use if algo != "implicit_gemm"]

    for algo_mode, params_config in params_to_use:
        status = None
        for _ in range(warmup_iters):
            status = _execute_single_bwd_pass(algo_mode, params_config)
            if isinstance(status, int) and status != 0:
                logger.debug(
                    f"Error in _run_backward_benchmarks: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(status))}"
                )
                continue

        if status is not None and status != 0:
            continue

        # Benchmark runs
        current_algo_min_time_ms = float("inf")  # Min time for this specific algorithm config

        if benchmark_iters == 0:
            if warmup_iters == 0:
                continue
        else:
            for _ in range(benchmark_iters):
                with timer:
                    _execute_single_bwd_pass(algo_mode, params_config)
                current_algo_min_time_ms = min(current_algo_min_time_ms, timer.elapsed_time)

        logger.debug(
            f"Backward benchmark result: {str(algo_mode)} {params_config} {current_algo_min_time_ms:.2f}ms"
        )
        if current_algo_min_time_ms != float("inf"):
            all_benchmark_results.append((algo_mode, params_config, current_algo_min_time_ms))

    if not all_benchmark_results:
        logger.warning(
            "Warning: No backward benchmark was successful or no algorithms to test. Defaulting to EXPLICIT_GEMM."
        )
        with timer:
            _execute_single_bwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    logger.debug(
        f"Best backward algo: {str(best_algo)} for log N_in={math.ceil(math.log2(in_features.shape[0])) if in_features.shape[0] > 0 else 0}, log N_out={math.ceil(math.log2(num_out_coords)) if num_out_coords > 0 else 0}, C_in={in_features.shape[1]}, C_out={weight.shape[2]}, K_vol={weight.shape[0]} {best_params} {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results  # Return the sorted list of all results


class UnifiedSpatiallySparseConvFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        fwd_algo: Union[str, List[Union[str, SPARSE_CONV_FWD_ALGO_MODE]]],
        bwd_algo: Union[str, List[Union[str, SPARSE_CONV_BWD_ALGO_MODE]]],
        compute_dtype: Optional[torch.dtype],
        fwd_block_size: Optional[int],  # For implicit GEMM if not AUTO
        bwd_block_size: Optional[int],  # For implicit GEMM if not AUTO
    ) -> Float[Tensor, "M C_out"]:
        global _BENCHMARK_FORWARD_RESULTS  # noqa: F824
        output_feature_tensor = None

        # Normalize input algos to strings for benchmarking and caching
        def _to_algo_str_list(
            x: Union[str, List[Union[str, Enum]], Enum],
        ) -> Union[str, List[str]]:
            if isinstance(x, list):
                return [a.value if isinstance(a, Enum) else str(a) for a in x]
            return x.value if isinstance(x, Enum) else str(x)

        fwd_algo = _to_algo_str_list(fwd_algo)
        bwd_algo = _to_algo_str_list(bwd_algo)

        # UNIFIED APPROACH: Always benchmark within filtered algorithm space
        # Step 1: Determine algorithm filter set
        if isinstance(fwd_algo, list):
            algorithm_filter = fwd_algo
        elif fwd_algo == "auto":
            algorithm_filter = "auto"  # All algorithms
        else:
            # Single algorithm - create list for consistent processing
            algorithm_filter = [str(fwd_algo)]

        # Step 2: Generate configuration for caching
        config = SpatiallySparseConvConfig(
            num_in_coords=in_features.shape[0],
            num_out_coords=num_out_coords,
            in_channels=in_features.shape[1],
            out_channels=weight.shape[2],
            kernel_volume=weight.shape[0],
            in_dtype=in_features.dtype,
        )

        # Step 3: Check cache first
        cached_result = _BENCHMARK_FORWARD_RESULTS.get(config)
        if cached_result is not None:
            # Support tuple (best) or list-of-tuples (best-first)
            if isinstance(cached_result, tuple):
                best_tuple = cached_result
                best_list = [best_tuple]
            else:
                best_list = cached_result
            if algorithm_filter == "auto":
                chosen_fwd_algo, chosen_fwd_params, _ = best_list[0]
            else:
                filtered_cached_results = []
                for algo, params, time in best_list:
                    if algo in algorithm_filter:
                        filtered_cached_results.append((algo, params, time))

                if filtered_cached_results:
                    chosen_fwd_algo, chosen_fwd_params, _ = filtered_cached_results[0]
                else:
                    filtered_params = _filter_benchmark_params_by_env_config(
                        _BENCHMARK_FORWARD_PARAMS, algorithm_filter, is_forward=True
                    )
                    if not filtered_params and "explicit_gemm" in algorithm_filter:
                        chosen_fwd_algo, chosen_fwd_params = (
                            "explicit_gemm",
                            {},
                        )
                    else:
                        all_fwd_benchmark_results = _run_forward_benchmarks(
                            in_features,
                            weight,
                            kernel_map,
                            num_out_coords,
                            compute_dtype,
                            custom_params=filtered_params,
                        )
                        _BENCHMARK_FORWARD_RESULTS[config] = all_fwd_benchmark_results[0]
                        # Save a serialized copy (algo as string) to the generic cache
                        generic_benchmark_update_entry(
                            "sparse_conv_forward",
                            config,
                            _serialize_benchmark_results(all_fwd_benchmark_results),
                            force=False,
                        )
                        chosen_fwd_algo, chosen_fwd_params, _ = all_fwd_benchmark_results[0]
        else:
            # Step 4: No cache - always benchmark within filtered space
            if algorithm_filter == "auto":
                # Benchmark all algorithms
                filtered_params = _BENCHMARK_FORWARD_PARAMS
            else:
                # Filter benchmark parameters to only include algorithms in filter set
                filtered_params = _filter_benchmark_params_by_env_config(
                    _BENCHMARK_FORWARD_PARAMS, algorithm_filter, is_forward=True
                )

            # Always run benchmarks to find optimal parameters
            all_fwd_benchmark_results = _run_forward_benchmarks(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                custom_params=filtered_params,
            )
            _BENCHMARK_FORWARD_RESULTS[config] = all_fwd_benchmark_results[0]
            # Persist a serialized copy to generic cache
            generic_benchmark_update_entry(
                "sparse_conv_forward",
                config,
                _serialize_benchmark_results(all_fwd_benchmark_results),
                force=False,
            )
            chosen_fwd_algo, chosen_fwd_params, _ = all_fwd_benchmark_results[0]

        # Step 5: Execute with optimal algorithm and parameters
        if chosen_fwd_algo == "explicit_gemm":
            output_feature_tensor = _explicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
            )
        elif chosen_fwd_algo == "implicit_gemm":
            current_fwd_block_size = chosen_fwd_params.get("fwd_block_size")
            if current_fwd_block_size is None:  # Fallback if somehow not set
                current_fwd_block_size = fwd_block_size if fwd_block_size is not None else 16
                logger.warning(
                    f"fwd_block_size not found in chosen_fwd_params for IMPLICIT_GEMM, using fallback {current_fwd_block_size}"
                )
            output_feature_tensor = _implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                current_fwd_block_size,
            )
        elif chosen_fwd_algo == "cutlass_implicit_gemm":
            output_feature_tensor = _cutlass_implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                accumulator_type=chosen_fwd_params.get("accumulator_type", torch.float32),
            )
            if isinstance(output_feature_tensor, int) and output_feature_tensor != 0:
                raise RuntimeError(
                    f"Error in _cutlass_implicit_gemm_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(output_feature_tensor))}"
                )
        elif chosen_fwd_algo == "wmma_implicit_gemm":
            output_feature_tensor = _wmma_implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
            )
            if isinstance(output_feature_tensor, int) and output_feature_tensor != 0:
                raise RuntimeError(
                    f"Error in _wmma_implicit_gemm_forward_logic: status {output_feature_tensor}"
                )
        else:
            raise ValueError(f"Unsupported forward algorithm: {chosen_fwd_algo}")

        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map

        # For SpatiallySparseConvConfig in backward if bwd_algo is AUTO
        ctx.config_params_for_bwd = {
            "num_in_coords": in_features.shape[0],
            "num_out_coords": num_out_coords,
            "in_channels": in_features.shape[1],
            "out_channels": weight.shape[2],
            "kernel_volume": weight.shape[0],
            "implicit_matmul_fwd_block_size": chosen_fwd_params.get(
                "fwd_block_size", fwd_block_size
            ),  # from fwd decision
            "implicit_matmul_bwd_block_size": bwd_block_size,  # from user input for bwd
            "compute_dtype": compute_dtype,
            "device": in_features.device,
            "initial_bwd_algo": bwd_algo,
            "initial_bwd_block_size": bwd_block_size,
        }

        # Return structure for backward:
        # grads for: in_features, weight, kernel_map, num_out_coords, fwd_algo, bwd_algo, compute_dtype, fwd_block_size, bwd_block_size
        return output_feature_tensor

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Optional[Float[Tensor, "N C_in"]],
        Optional[Float[Tensor, "K C_in C_out"]],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        global _BENCHMARK_BACKWARD_RESULTS  # noqa: F824
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        config_params = ctx.config_params_for_bwd
        num_out_coords = config_params["num_out_coords"]
        compute_dtype = config_params["compute_dtype"]
        device = config_params["device"]
        initial_bwd_algo = config_params["initial_bwd_algo"]
        initial_bwd_block_size = config_params["initial_bwd_block_size"]
        # explicit_matmul_batch_size = ctx.explicit_matmul_batch_size # TODO

        # Normalize input to strings
        if isinstance(initial_bwd_algo, list):
            initial_bwd_algo = [
                str(a.value) if isinstance(a, Enum) else str(a) for a in initial_bwd_algo
            ]
        else:
            initial_bwd_algo = (
                str(initial_bwd_algo.value)
                if isinstance(initial_bwd_algo, Enum)
                else str(initial_bwd_algo)
            )

        grad_in_features, grad_weight = None, None

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 9)

        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        if (
            num_out_coords == 0
            or K == 0
            or C_in == 0
            or C_out == 0
            or N_in == 0
            or grad_output.shape[0] == 0
        ):
            grad_in_final = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_weight_final = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in_final, grad_weight_final, 9)

        # UNIFIED APPROACH: Always benchmark within filtered algorithm space
        # Step 1: Determine algorithm filter set
        if isinstance(initial_bwd_algo, list):
            algorithm_filter = initial_bwd_algo
        elif initial_bwd_algo == "auto":
            algorithm_filter = "auto"  # All algorithms
        else:
            # Single algorithm - create list for consistent processing
            algorithm_filter = [str(initial_bwd_algo)]

        # Step 2: Generate configuration for caching
        config_params = ctx.config_params_for_bwd
        config = SpatiallySparseConvConfig(
            num_in_coords=config_params["num_in_coords"],
            num_out_coords=config_params["num_out_coords"],
            in_channels=config_params["in_channels"],
            out_channels=config_params["out_channels"],
            kernel_volume=config_params["kernel_volume"],
            in_dtype=grad_output.dtype,
        )

        # Step 3: Check cache first
        cached_result = _BENCHMARK_BACKWARD_RESULTS.get(config)
        if cached_result is not None:
            if isinstance(cached_result, tuple):
                best_list = [cached_result]
            else:
                best_list = cached_result
            if algorithm_filter == "auto":
                chosen_bwd_algo, chosen_bwd_params, _ = best_list[0]
            else:
                filtered_cached_results = []
                for algo, params, time in best_list:
                    if algo in algorithm_filter:
                        filtered_cached_results.append((algo, params, time))

                if filtered_cached_results:
                    chosen_bwd_algo, chosen_bwd_params, _ = filtered_cached_results[0]
                else:
                    filtered_params = _filter_benchmark_params_by_env_config(
                        _BENCHMARK_BACKWARD_PARAMS, algorithm_filter, is_forward=False
                    )
                    if not filtered_params and "explicit_gemm" in algorithm_filter:
                        chosen_bwd_algo, chosen_bwd_params = (
                            "explicit_gemm",
                            {},
                        )
                    else:
                        all_bwd_benchmark_results = _run_backward_benchmarks(
                            grad_output,
                            in_features,
                            weight,
                            kernel_map,
                            num_out_coords,
                            compute_dtype,
                            device,
                            custom_params=filtered_params,
                        )
                        _BENCHMARK_BACKWARD_RESULTS[config] = all_bwd_benchmark_results[0]
                        generic_benchmark_update_entry(
                            "sparse_conv_backward",
                            config,
                            _serialize_benchmark_results(all_bwd_benchmark_results),
                            force=False,
                        )
                        chosen_bwd_algo, chosen_bwd_params, _ = all_bwd_benchmark_results[0]
        else:
            # Step 4: No cache - always benchmark within filtered space
            if algorithm_filter == "auto":
                # Benchmark all algorithms
                filtered_params = _BENCHMARK_BACKWARD_PARAMS
            else:
                # Filter benchmark parameters to only include algorithms in filter set
                filtered_params = _filter_benchmark_params_by_env_config(
                    _BENCHMARK_BACKWARD_PARAMS, algorithm_filter, is_forward=False
                )

            # Always run benchmarks to find optimal parameters
            all_bwd_benchmark_results = _run_backward_benchmarks(
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                device,
                custom_params=filtered_params,
            )
            _BENCHMARK_BACKWARD_RESULTS[config] = all_bwd_benchmark_results[0]
            # Persist a serialized copy to generic cache
            generic_benchmark_update_entry(
                "sparse_conv_backward",
                config,
                _serialize_benchmark_results(all_bwd_benchmark_results),
                force=False,
            )
            chosen_bwd_algo, chosen_bwd_params, _ = all_bwd_benchmark_results[0]

        if chosen_bwd_algo == "explicit_gemm":
            grad_in_features, grad_weight = _explicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                compute_dtype,
                device,
            )
        elif chosen_bwd_algo == "implicit_gemm":
            grad_in_features, grad_weight = _implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                gemm_block_size=chosen_bwd_params.get("bwd_block_size", 16),
                split_k_threads_per_block=chosen_bwd_params.get("split_k_threads_per_block", 256),
                split_k_factor=chosen_bwd_params.get("split_k_factor", 4),
                compute_dtype=compute_dtype,
            )
        elif chosen_bwd_algo == "cutlass_implicit_gemm":
            grad_in_features, grad_weight = _cutlass_implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                accumulator_type=chosen_bwd_params.get("accumulator_type", torch.float32),
                device=device,
            )
            if isinstance(grad_in_features, int) and grad_in_features != 0:
                raise RuntimeError(
                    f"Error in _cutlass_implicit_gemm_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(grad_in_features))}"
                )
        elif chosen_bwd_algo == "wmma_implicit_gemm":
            grad_in_features, grad_weight = _wmma_implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
                device=device,
            )
            if isinstance(grad_in_features, int) and grad_in_features != 0:
                raise RuntimeError(
                    f"Error in _wmma_implicit_gemm_backward_logic: status {grad_in_features}"
                )

        # elif chosen_bwd_algo == SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM_BATCHED: # TODO
        # if explicit_matmul_batch_size is None:
        #     raise ValueError("explicit_matmul_batch_size is required for batched explicit GEMM backward.")
        # grad_in_features, grad_weight = _batched_explicit_gemm_backward_logic(...)
        else:
            raise ValueError(f"Unsupported backward algorithm: {chosen_bwd_algo}")

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 9)
