# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union, Generator, Dict, Any, Sequence

import numpy as np
import torch
from jaxtyping import Float, Int
import logging
import cupy as cp
import math
import os
import itertools

from torch import Tensor
from torch.autograd import Function

from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.ntuple import ntuple
from warpconvnet.utils.cuda_utils import load_kernel
from warpconvnet.utils.logger import get_logger
from warpconvnet.utils.benchmark_cache import (
    load_dict_benchmark_cache,
    save_dict_benchmark_cache,
    mark_benchmark_cache_dirty,
)
from warpconvnet.nn.functional.sparse_conv import (
    SpatiallySparseConvConfig,
    _maybe_cast,
    _get_cuda_kernel,
    CUDATimer,
    _backward_return,
    _BENCHMARK_NUM_RUNS,
    _initialize_benchmark_cache,
)
from warpconvnet.utils.type_cast import _min_dtype, _max_dtype

logger = get_logger(__name__)
try:
    import warpconvnet._C as _C
except ImportError as e:
    logger.warning(
        f"Error importing warpconvnet._C: {e}. Using fallback implementation."
    )
    _C = None


# Depthwise convolution algorithm enums
class SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    AUTO = "auto"


class SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    AUTO = "auto"


# Depthwise convolution benchmark parameters
_BENCHMARK_DEPTHWISE_FORWARD_PARAMS = [
    (SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT, {}),
    (SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT, {}),
]

_BENCHMARK_DEPTHWISE_BACKWARD_PARAMS = [
    (SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT, {}),
    (SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT, {}),
]

# Depthwise convolution benchmark result caches
_BENCHMARK_DEPTHWISE_FORWARD_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE, Dict[str, Any], float]],
] = {}
_BENCHMARK_DEPTHWISE_BACKWARD_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE, Dict[str, Any], float]],
] = {}


# Load cached benchmark results at module initialization
def _initialize_depthwise_benchmark_cache():
    """Load cached depthwise benchmark results and populate global dictionaries."""
    try:
        cached_results = load_dict_benchmark_cache()
        _BENCHMARK_DEPTHWISE_FORWARD_RESULTS.update(
            cached_results.get("sparse_conv_depthwise_forward_results", {})
        )
        _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS.update(
            cached_results.get("sparse_conv_depthwise_backward_results", {})
        )
        if (
            _BENCHMARK_DEPTHWISE_FORWARD_RESULTS
            or _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS
        ):
            logger.info(
                f"Loaded {len(_BENCHMARK_DEPTHWISE_FORWARD_RESULTS)} depthwise forward and "
                f"{len(_BENCHMARK_DEPTHWISE_BACKWARD_RESULTS)} depthwise backward benchmark configurations from cache"
            )
    except Exception as e:
        logger.warning(f"Failed to initialize depthwise benchmark cache: {e}")


# Initialize cache on module load
_initialize_depthwise_benchmark_cache()


def _explicit_depthwise_forward_logic(
    in_features: Float[Tensor, "N C"],  # noqa: F821
    weight: Float[Tensor, "K C"],  # noqa: F821
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
) -> Float[Tensor, "M C"]:  # noqa: F821
    """Forward pass for depthwise convolution using explicit GEMM."""
    device = in_features.device
    comp_in_feats = _maybe_cast(in_features, compute_dtype)
    comp_weight = _maybe_cast(weight, compute_dtype)
    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        output_feature_tensor = comp_in_feats * comp_weight[iden_idx].unsqueeze(0)
    else:
        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=comp_in_feats.dtype
        )

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device)
        out_map = out_map.to(device)
        curr_out_features = comp_in_feats[in_map] * comp_weight[i].unsqueeze(0)
        output_feature_tensor[out_map] += curr_out_features.to(device=device)
    return output_feature_tensor.to(dtype=in_features.dtype)


def _explicit_depthwise_backward_logic(
    grad_output: Float[Tensor, "M C"],  # noqa: F821
    in_features: Float[Tensor, "N C"],  # noqa: F821
    weight: Float[Tensor, "K C"],  # noqa: F821
    kernel_map: IntSearchResult,
    compute_dtype: Optional[torch.dtype] = None,
    device: torch.device = None,
) -> Tuple[Float[Tensor, "N C"], Float[Tensor, "K C"]]:  # noqa: F821
    """Backward pass for explicit depthwise convolution."""
    if device is None:
        device = grad_output.device

    dtype_to_use = compute_dtype if compute_dtype is not None else in_features.dtype
    comp_in_feats = in_features.to(device=device, dtype=dtype_to_use)
    comp_weight = weight.to(device=device, dtype=dtype_to_use)
    comp_grad_output = grad_output.to(device=device, dtype=dtype_to_use)
    grad_weight = torch.zeros_like(comp_weight, device=device)

    # y = x * w
    # L = f(y)
    # dL/dx = dL/dy * dy/dx = dL/dy * w
    # dL/dw = dL/dy * dy/dw = dL/dy * x
    # dL/dy is grad_output
    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = comp_grad_output * comp_weight[iden_idx].unsqueeze(0)
        grad_weight[iden_idx] = torch.sum(comp_in_feats * comp_grad_output, dim=0)
    else:
        grad_in_features = torch.zeros_like(comp_in_feats, device=device)

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue

        curr_grad_output = comp_grad_output[out_map]
        curr_in_feats = comp_in_feats[in_map]
        curr_weight = comp_weight[i]
        grad_in_features[in_map] += curr_grad_output * curr_weight.unsqueeze(0)
        grad_weight[i] += torch.sum(curr_in_feats * curr_grad_output, dim=0)
    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )


def _implicit_depthwise_forward_logic(
    in_features: Float[Tensor, "N C"],  # noqa: F821
    weight: Float[Tensor, "K C"],  # noqa: F821
    kernel_map: IntSearchResult,
    num_out_coords: int,
) -> Float[Tensor, "M C"]:  # noqa: F821
    """Forward pass for depthwise convolution using implicit GEMM kernels."""
    if _C is None:
        raise ImportError(
            "warpconvnet._C is not available. Please install warpconvnet with CUDA support."
        )

    in_dtype = _min_dtype(in_features.dtype, weight.dtype)
    device = in_features.device
    comp_in_feats = _maybe_cast(in_features, in_dtype)
    comp_weight = _maybe_cast(weight, in_dtype)
    iden_idx = kernel_map.identity_map_index

    if iden_idx is not None:
        output_feature_tensor = comp_in_feats * comp_weight[iden_idx].unsqueeze(0)
    else:
        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=comp_in_feats.dtype
        )

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue

        in_map = in_map.to(device)
        out_map = out_map.to(device)

        # Use implicit_fma: c[out_index] += a[in_index] * b
        # output_feature_tensor[out_map] += comp_in_feats[in_map] * comp_weight[i]
        _C.fma.implicit_fma(
            comp_in_feats,
            comp_weight[i],
            output_feature_tensor,
            in_map,
            out_map,
            "basic",
        )

    return output_feature_tensor.to(dtype=in_features.dtype)


def _implicit_depthwise_backward_logic(
    grad_output: Float[Tensor, "M C"],  # noqa: F821
    in_features: Float[Tensor, "N C"],  # noqa: F821
    weight: Float[Tensor, "K C"],  # noqa: F821
    kernel_map: IntSearchResult,
    device: torch.device = None,
) -> Tuple[Float[Tensor, "N C"], Float[Tensor, "K C"]]:  # noqa: F821
    """Backward pass for depthwise convolution using implicit kernels."""
    if _C is None:
        raise ImportError(
            "warpconvnet._C is not available. Please install warpconvnet with CUDA support."
        )
    if device is None:
        device = grad_output.device

    dtype_to_use = _min_dtype(in_features.dtype, weight.dtype)
    comp_in_feats = in_features.to(device=device, dtype=dtype_to_use)
    comp_weight = weight.to(device=device, dtype=dtype_to_use)
    comp_grad_output = grad_output.to(device=device, dtype=dtype_to_use)
    grad_weight = torch.zeros_like(comp_weight, device=device)

    # y = x * w
    # L = f(y)
    # dL/dx = dL/dy * dy/dx = dL/dy * w
    # dL/dw = dL/dy * dy/dw = dL/dy * x
    # dL/dy is grad_output
    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = comp_grad_output * comp_weight[iden_idx].unsqueeze(0)
        grad_weight[iden_idx] = torch.sum(comp_in_feats * comp_grad_output, dim=0)
    else:
        grad_in_features = torch.zeros_like(comp_in_feats, device=device)

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue

        in_map = in_map.to(device)
        out_map = out_map.to(device)

        # Compute gradient w.r.t. input features
        # grad_in_features[in_map] += grad_output[out_map] * weight[i]
        _C.fma.implicit_fma(
            comp_grad_output, comp_weight[i], grad_in_features, out_map, in_map, "basic"
        )

        # Compute gradient w.r.t. weight
        # grad_weight[i] += sum(in_features[in_map] * grad_output[out_map])
        temp_result = torch.zeros_like(comp_weight[i], device=device)
        _C.fma.implicit_reduction(
            comp_in_feats, in_map, comp_grad_output, out_map, temp_result, "basic"
        )
        grad_weight[i] += temp_result

    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )


def _run_depthwise_forward_benchmarks(
    in_features: Float[Tensor, "N C"],
    weight: Float[Tensor, "K C"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    warmup_iters: int = _BENCHMARK_NUM_RUNS // 2,
    benchmark_iters: int = _BENCHMARK_NUM_RUNS,
) -> List[Tuple[SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE, Dict[str, Any], float]]:
    """
    Benchmark different depthwise forward algorithms and return the results sorted by runtime.
    """
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    logger.warn(
        "Using benchmarked depthwise forward algo. Until the algorithm finds the best parameters, forward performance will be slow."
    )
    all_benchmark_results: List[
        Tuple[SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE, Dict[str, Any], float]
    ] = []
    timer = CUDATimer()

    def _execute_single_depthwise_fwd_pass(
        algo_mode: SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE, params_config: Dict[str, Any]
    ) -> Optional[int]:
        if algo_mode == SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT:
            _ = _explicit_depthwise_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
            )
        elif algo_mode == SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT:
            _ = _implicit_depthwise_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
            )
        else:
            raise ValueError(
                f"Unsupported algo_mode in _execute_single_depthwise_fwd_pass: {algo_mode}"
            )

    for algo_mode, params_config in _BENCHMARK_DEPTHWISE_FORWARD_PARAMS:
        # Skip implicit if _C is not available
        if algo_mode == SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT and _C is None:
            continue

        # Warmup runs
        status = None
        for _ in range(warmup_iters):
            try:
                status = _execute_single_depthwise_fwd_pass(algo_mode, params_config)
                if isinstance(status, int) and status != 0:
                    break
            except Exception as e:
                logger.warning(f"Error in warmup for {algo_mode}: {e}")
                status = -1
                break

        if status is not None and status != 0:
            continue

        # Benchmark runs
        current_algo_min_time_ms = float("inf")

        if benchmark_iters > 0:
            for _ in range(benchmark_iters):
                try:
                    with timer:
                        _execute_single_depthwise_fwd_pass(algo_mode, params_config)
                    if timer.elapsed_time is not None:
                        current_algo_min_time_ms = min(
                            current_algo_min_time_ms, timer.elapsed_time
                        )
                except Exception as e:
                    logger.warning(f"Error in benchmark for {algo_mode}: {e}")
                    current_algo_min_time_ms = float("inf")
                    break

        logger.debug(
            f"Depthwise forward benchmark result: {algo_mode.value} {params_config} {current_algo_min_time_ms:.2f}ms"
        )
        if current_algo_min_time_ms != float("inf"):
            all_benchmark_results.append(
                (algo_mode, params_config, current_algo_min_time_ms)
            )

    if not all_benchmark_results:
        logger.warning(
            "Warning: No depthwise forward benchmark was successful. Defaulting to EXPLICIT."
        )
        with timer:
            _execute_single_depthwise_fwd_pass(
                SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT, {}
            )
        fallback_time = timer.elapsed_time if timer.elapsed_time is not None else 0.0
        all_benchmark_results.append(
            (SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT, {}, fallback_time)
        )

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    logger.debug(
        f"Best depthwise forward algo: {best_algo.value} for log N_in={math.ceil(math.log2(in_features.shape[0])) if in_features.shape[0] > 0 else 0}, log N_out={math.ceil(math.log2(num_out_coords)) if num_out_coords > 0 else 0}, C={in_features.shape[1]}, K_vol={weight.shape[0]} {best_params} {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results


def _run_depthwise_backward_benchmarks(
    grad_output: Float[Tensor, "M C"],
    in_features: Float[Tensor, "N C"],
    weight: Float[Tensor, "K C"],
    kernel_map: IntSearchResult,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    warmup_iters: int = _BENCHMARK_NUM_RUNS // 2,
    benchmark_iters: int = _BENCHMARK_NUM_RUNS,
) -> List[Tuple[SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE, Dict[str, Any], float]]:
    """
    Benchmark different depthwise backward algorithms and return the results sorted by runtime.
    """
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[
        Tuple[SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE, Dict[str, Any], float]
    ] = []
    timer = CUDATimer()

    def _execute_single_depthwise_bwd_pass(
        algo_mode: SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE, params_config: Dict[str, Any]
    ) -> Optional[int]:
        if algo_mode == SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT:
            _, _ = _explicit_depthwise_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                compute_dtype,
                device,
            )
        elif algo_mode == SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT:
            _, _ = _implicit_depthwise_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                device,
            )
        else:
            raise ValueError(
                f"Unsupported algo_mode in _execute_single_depthwise_bwd_pass: {algo_mode}"
            )

    for algo_mode, params_config in _BENCHMARK_DEPTHWISE_BACKWARD_PARAMS:
        # Skip implicit if _C is not available
        if algo_mode == SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT and _C is None:
            continue

        # Warmup runs
        status = None
        for _ in range(warmup_iters):
            try:
                status = _execute_single_depthwise_bwd_pass(algo_mode, params_config)
                if isinstance(status, int) and status != 0:
                    break
            except Exception as e:
                logger.warning(f"Error in warmup for {algo_mode}: {e}")
                status = -1
                break

        if status is not None and status != 0:
            continue

        # Benchmark runs
        current_algo_min_time_ms = float("inf")

        if benchmark_iters > 0:
            for _ in range(benchmark_iters):
                try:
                    with timer:
                        _execute_single_depthwise_bwd_pass(algo_mode, params_config)
                    if timer.elapsed_time is not None:
                        current_algo_min_time_ms = min(
                            current_algo_min_time_ms, timer.elapsed_time
                        )
                except Exception as e:
                    logger.warning(f"Error in benchmark for {algo_mode}: {e}")
                    current_algo_min_time_ms = float("inf")
                    break

        logger.debug(
            f"Depthwise backward benchmark result: {algo_mode.value} {params_config} {current_algo_min_time_ms:.2f}ms"
        )
        if current_algo_min_time_ms != float("inf"):
            all_benchmark_results.append(
                (algo_mode, params_config, current_algo_min_time_ms)
            )

    if not all_benchmark_results:
        logger.warning(
            "Warning: No depthwise backward benchmark was successful. Defaulting to EXPLICIT."
        )
        with timer:
            _execute_single_depthwise_bwd_pass(
                SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT, {}
            )
        fallback_time = timer.elapsed_time if timer.elapsed_time is not None else 0.0
        all_benchmark_results.append(
            (SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT, {}, fallback_time)
        )

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    logger.debug(
        f"Best depthwise backward algo: {best_algo.value} for log N_in={math.ceil(math.log2(in_features.shape[0])) if in_features.shape[0] > 0 else 0}, log N_out={math.ceil(math.log2(grad_output.shape[0])) if grad_output.shape[0] > 0 else 0}, C={in_features.shape[1]}, K_vol={weight.shape[0]} {best_params} {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results


class UnifiedSpatiallySparseDepthwiseConvFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C"],
        weight: Float[Tensor, "K C"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        fwd_algo: SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE,
        bwd_algo: SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE,
        compute_dtype: Optional[torch.dtype],
    ) -> Float[Tensor, "M C"]:
        output_feature_tensor = None

        chosen_fwd_algo = fwd_algo
        chosen_fwd_params = {}

        if fwd_algo == SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.AUTO:
            config = SpatiallySparseConvConfig(
                num_in_coords=in_features.shape[0],
                num_out_coords=num_out_coords,
                in_channels=in_features.shape[1],
                out_channels=weight.shape[
                    1
                ],  # For depthwise, in_channels == out_channels
                kernel_volume=weight.shape[0],
                in_dtype=in_features.dtype,
            )
            global _BENCHMARK_DEPTHWISE_FORWARD_RESULTS  # noqa: F824
            cached_result = _BENCHMARK_DEPTHWISE_FORWARD_RESULTS.get(config)
            if cached_result is not None:
                chosen_fwd_algo, chosen_fwd_params, _ = cached_result[
                    0
                ]  # Best is first
            else:
                all_fwd_benchmark_results = _run_depthwise_forward_benchmarks(
                    in_features,
                    weight,
                    kernel_map,
                    num_out_coords,
                    compute_dtype,
                )
                _BENCHMARK_DEPTHWISE_FORWARD_RESULTS[config] = all_fwd_benchmark_results
                chosen_fwd_algo, chosen_fwd_params, min_time = (
                    all_fwd_benchmark_results[0]
                )  # Best is first

                # Mark cache as dirty - background thread will save periodically
                mark_benchmark_cache_dirty()

        if chosen_fwd_algo == SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT:
            output_feature_tensor = _explicit_depthwise_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
            )
        elif chosen_fwd_algo == SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT:
            output_feature_tensor = _implicit_depthwise_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
            )
        else:
            raise ValueError(f"Unsupported forward algorithm: {chosen_fwd_algo}")

        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.config_params_for_bwd = {
            "num_in_coords": in_features.shape[0],
            "num_out_coords": num_out_coords,
            "in_channels": in_features.shape[1],
            "out_channels": weight.shape[1],
            "kernel_volume": weight.shape[0],
            "compute_dtype": compute_dtype,
            "device": in_features.device,
            "initial_bwd_algo": bwd_algo,
        }

        return output_feature_tensor

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C"]) -> Tuple[
        Optional[Float[Tensor, "N C"]],
        Optional[Float[Tensor, "K C"]],
        None,
        None,
        None,
        None,
        None,
    ]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        config_params = ctx.config_params_for_bwd
        compute_dtype = config_params["compute_dtype"]
        device = config_params["device"]
        initial_bwd_algo = config_params["initial_bwd_algo"]

        grad_in_features, grad_weight = None, None

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _backward_return(None, None, 7)

        N_in, C_in = in_features.shape
        K, C = weight.shape
        if (
            config_params["num_out_coords"] == 0
            or K == 0
            or C_in == 0
            or C == 0
            or N_in == 0
            or grad_output.shape[0] == 0
        ):
            grad_in_final = (
                torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            )
            grad_weight_final = (
                torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            )
            return _backward_return(grad_in_final, grad_weight_final, 7)

        chosen_bwd_algo = initial_bwd_algo
        chosen_bwd_params = {}

        if initial_bwd_algo == SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.AUTO:
            config = SpatiallySparseConvConfig(
                num_in_coords=config_params["num_in_coords"],
                num_out_coords=config_params["num_out_coords"],
                in_channels=config_params["in_channels"],
                out_channels=config_params["out_channels"],
                kernel_volume=config_params["kernel_volume"],
                in_dtype=grad_output.dtype,
            )
            global _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS  # noqa: F824
            cached_result = _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS.get(config)
            if cached_result is not None:
                chosen_bwd_algo, chosen_bwd_params, _ = cached_result[
                    0
                ]  # Best is first
            else:
                all_bwd_benchmark_results = _run_depthwise_backward_benchmarks(
                    grad_output,
                    in_features,
                    weight,
                    kernel_map,
                    compute_dtype,
                    device,
                )
                _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS[config] = (
                    all_bwd_benchmark_results
                )
                chosen_bwd_algo, chosen_bwd_params, min_time = (
                    all_bwd_benchmark_results[0]
                )  # Best is first

                # Mark cache as dirty - background thread will save periodically
                mark_benchmark_cache_dirty()

        if chosen_bwd_algo == SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT:
            grad_in_features, grad_weight = _explicit_depthwise_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                compute_dtype,
                device,
            )
        elif chosen_bwd_algo == SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT:
            grad_in_features, grad_weight = _implicit_depthwise_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                device,
            )
        else:
            raise ValueError(f"Unsupported backward algorithm: {chosen_bwd_algo}")

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _backward_return(grad_in_features, grad_weight, 7)


def spatially_sparse_depthwise_conv(
    in_features: Float[Tensor, "N C"],
    weight: Float[Tensor, "K C"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    fwd_algo: SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE = SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT,
    bwd_algo: SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE = SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
    compute_dtype: Optional[torch.dtype] = None,
) -> Float[Tensor, "M C"]:
    """
    Perform spatially sparse depthwise convolution.

    Args:
        in_features: Input features of shape (N, C)
        weight: Depthwise convolution weights of shape (K, C)
        kernel_map: Kernel mapping from IntSearchResult
        num_out_coords: Number of output coordinates
        fwd_algo: Forward algorithm to use
        bwd_algo: Backward algorithm to use
        compute_dtype: Computation dtype (defaults to input dtype)

    Returns:
        Output features of shape (M, C)
    """
    return UnifiedSpatiallySparseDepthwiseConvFunction.apply(
        in_features,
        weight,
        kernel_map,
        num_out_coords,
        fwd_algo,
        bwd_algo,
        compute_dtype,
    )
