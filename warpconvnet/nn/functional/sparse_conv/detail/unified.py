# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

from enum import Enum

import torch
from torch import Tensor
from torch.autograd import Function

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult

from warpconvnet.utils.benchmark_cache import (
    SpatiallySparseConvConfig,
    generic_benchmark_update_entry,
)
from warpconvnet.utils.ntuple import _pad_tuple
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
    _implicit_gemm_backward_grouped,
)
from .cutlass import (
    _cutlass_implicit_gemm_forward_logic,
    _cutlass_implicit_gemm_backward_logic,
    _cutlass_implicit_gemm_forward_grouped,
    _cutlass_implicit_gemm_backward_grouped,
)
from .mask_gemm import (
    _mask_implicit_gemm_forward_logic,
    _mask_implicit_gemm_backward_logic,
)
from .algo_params import (
    SPARSE_CONV_FWD_ALGO_MODE,
    SPARSE_CONV_BWD_ALGO_MODE,
    _HAS_CUTE_BACKEND,
    _HAS_CUTE_GROUPED,
    _HAS_CUTE_SM90,
    _HAS_CUTE_GROUPED_SM90,
    _BENCHMARK_BACKWARD_PARAMS,
    _ALL_BENCHMARK_FORWARD_PARAMS,
    _ALL_BENCHMARK_BACKWARD_PARAMS,
    _get_adaptive_forward_params,
    _get_adaptive_backward_params,
    _get_filtered_forward_params,
    _get_filtered_backward_params,
    _filter_benchmark_params_by_env_config,
)
from .autotune import (
    _BENCHMARK_FORWARD_RESULTS,
    _BENCHMARK_BACKWARD_RESULTS,
    _serialize_benchmark_results,
    _run_forward_benchmarks,
    _run_backward_benchmarks,
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

if _HAS_CUTE_SM90:
    from .cute_sm90 import (
        _cute_implicit_gemm_sm90_forward_logic,
        _cute_implicit_gemm_sm90_backward_logic,
    )

if _HAS_CUTE_GROUPED_SM90:
    from .cute_grouped_sm90 import (
        _cute_grouped_sm90_forward_logic,
        _cute_grouped_sm90_backward_logic,
    )

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "SPARSE_CONV_FWD_ALGO_MODE",
    "SPARSE_CONV_BWD_ALGO_MODE",
    "UnifiedSpatiallySparseConvFunction",
]


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
        voxel_size: Optional[Tuple[int, ...]] = None,
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
        elif fwd_algo in ("auto", "all"):
            algorithm_filter = fwd_algo
        else:
            # Single algorithm - create list for consistent processing
            algorithm_filter = [str(fwd_algo)]

        # Step 2: Generate configuration for caching
        C_in = in_features.shape[1]
        C_out = weight.shape[2]
        kv = weight.shape[0]
        adaptive_fwd_params = _get_adaptive_forward_params(
            C_in,
            C_out,
            kv,
            num_in_coords=in_features.shape[0],
            voxel_size=voxel_size,
        )

        config = SpatiallySparseConvConfig(
            num_in_coords=in_features.shape[0],
            num_out_coords=num_out_coords,
            in_channels=C_in,
            out_channels=C_out,
            kernel_volume=kv,
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
            if algorithm_filter in ("auto", "all"):
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
                        adaptive_fwd_params, algorithm_filter, is_forward=True
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
                        _BENCHMARK_FORWARD_RESULTS[config] = all_fwd_benchmark_results[
                            0
                        ]
                        # Save a serialized copy (algo as string) to the generic cache
                        generic_benchmark_update_entry(
                            "sparse_conv_forward",
                            config,
                            _serialize_benchmark_results(all_fwd_benchmark_results),
                            force=False,
                        )
                        chosen_fwd_algo, chosen_fwd_params, _ = (
                            all_fwd_benchmark_results[0]
                        )
        else:
            # Step 4: No cache - always benchmark within filtered space
            if algorithm_filter in ("auto", "all"):
                # Benchmark algorithms - "auto" uses adaptive set, "all" uses exhaustive set
                filtered_params = _filter_benchmark_params_by_env_config(
                    adaptive_fwd_params, algorithm_filter, is_forward=True
                )
            else:
                # Filter benchmark parameters to only include algorithms in filter set
                filtered_params = _filter_benchmark_params_by_env_config(
                    adaptive_fwd_params, algorithm_filter, is_forward=True
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

        # Step 5: Pre-cast weight once (avoids per-algorithm re-casting)
        if compute_dtype is not None:
            _weight_cast = weight.contiguous().to(dtype=compute_dtype)
        else:
            _weight_cast = weight.contiguous()

        logger.debug(
            f"[dispatch] FWD algo={chosen_fwd_algo} params={chosen_fwd_params} "
            f"N_in={in_features.shape[0]} N_out={num_out_coords} "
            f"C_in={in_features.shape[1]} C_out={weight.shape[2]} "
            f"kv={weight.shape[0]} dtype={in_features.dtype}"
        )
        try:
            output_feature_tensor = _execute_forward(
                chosen_fwd_algo,
                chosen_fwd_params,
                in_features,
                _weight_cast,
                kernel_map,
                num_out_coords,
                compute_dtype,
                fwd_block_size,
            )
        except (RuntimeError, Exception) as e:
            if chosen_fwd_algo == "explicit_gemm":
                raise  # No fallback for the fallback
            logger.warning(
                f"Forward algorithm '{chosen_fwd_algo}' failed at execution: {e}. "
                f"Falling back to explicit_gemm."
            )
            # Invalidate the cached result for this config
            _BENCHMARK_FORWARD_RESULTS.pop(config, None)
            output_feature_tensor = _execute_forward(
                "explicit_gemm",
                {},
                in_features,
                _weight_cast,
                kernel_map,
                num_out_coords,
                compute_dtype,
                fwd_block_size,
            )

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

        # Normalize input to strings
        if isinstance(initial_bwd_algo, list):
            initial_bwd_algo = [
                str(a.value) if isinstance(a, Enum) else str(a)
                for a in initial_bwd_algo
            ]
        else:
            initial_bwd_algo = (
                str(initial_bwd_algo.value)
                if isinstance(initial_bwd_algo, Enum)
                else str(initial_bwd_algo)
            )

        grad_in_features, grad_weight = None, None

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 10)

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
            grad_in_final = (
                torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            )
            grad_weight_final = (
                torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            )
            return _pad_tuple(grad_in_final, grad_weight_final, 10)

        # UNIFIED APPROACH: Always benchmark within filtered algorithm space
        # Step 1: Determine algorithm filter set
        if isinstance(initial_bwd_algo, list):
            algorithm_filter = initial_bwd_algo
        elif initial_bwd_algo in ("auto", "all"):
            algorithm_filter = initial_bwd_algo
        else:
            # Single algorithm - create list for consistent processing
            algorithm_filter = [str(initial_bwd_algo)]

        # Step 2: Generate configuration for caching
        config_params = ctx.config_params_for_bwd
        C_in_bwd = config_params["in_channels"]
        C_out_bwd = config_params["out_channels"]
        kv_bwd = config_params["kernel_volume"]
        N_in_bwd = config_params["num_in_coords"]
        config = SpatiallySparseConvConfig(
            num_in_coords=N_in_bwd,
            num_out_coords=config_params["num_out_coords"],
            in_channels=C_in_bwd,
            out_channels=C_out_bwd,
            kernel_volume=kv_bwd,
            in_dtype=grad_output.dtype,
        )

        # Get adaptive backward params based on dimensions
        adaptive_bwd_params = _get_adaptive_backward_params(
            C_in_bwd,
            C_out_bwd,
            kv_bwd,
            num_in_coords=N_in_bwd,
        )

        # Step 3: Check cache first
        cached_result = _BENCHMARK_BACKWARD_RESULTS.get(config)
        if cached_result is not None:
            if isinstance(cached_result, tuple):
                best_list = [cached_result]
            else:
                best_list = cached_result
            if algorithm_filter in ("auto", "all"):
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
                        adaptive_bwd_params, algorithm_filter, is_forward=False
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
                        _BENCHMARK_BACKWARD_RESULTS[config] = all_bwd_benchmark_results[
                            0
                        ]
                        generic_benchmark_update_entry(
                            "sparse_conv_backward",
                            config,
                            _serialize_benchmark_results(all_bwd_benchmark_results),
                            force=False,
                        )
                        chosen_bwd_algo, chosen_bwd_params, _ = (
                            all_bwd_benchmark_results[0]
                        )
        else:
            # Step 4: No cache - always benchmark within filtered space
            if algorithm_filter in ("auto", "all"):
                # Benchmark algorithms - "auto" uses adaptive set, "all" uses exhaustive set
                filtered_params = _filter_benchmark_params_by_env_config(
                    adaptive_bwd_params, algorithm_filter, is_forward=False
                )
            else:
                # Filter benchmark parameters to only include algorithms in filter set
                filtered_params = _filter_benchmark_params_by_env_config(
                    adaptive_bwd_params, algorithm_filter, is_forward=False
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

        logger.debug(
            f"[dispatch] BWD algo={chosen_bwd_algo} params={chosen_bwd_params} "
            f"N_in={in_features.shape[0]} N_out={num_out_coords} "
            f"C_in={in_features.shape[1]} C_out={weight.shape[2]} "
            f"kv={weight.shape[0]} dtype={in_features.dtype}"
        )
        try:
            grad_in_features, grad_weight = _execute_backward(
                chosen_bwd_algo,
                chosen_bwd_params,
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                device,
                ctx.needs_input_grad,
            )
        except (RuntimeError, Exception) as e:
            if chosen_bwd_algo == "explicit_gemm":
                raise
            logger.warning(
                f"Backward algorithm '{chosen_bwd_algo}' failed at execution: {e}. "
                f"Falling back to explicit_gemm."
            )
            _BENCHMARK_BACKWARD_RESULTS.pop(config, None)
            grad_in_features, grad_weight = _execute_backward(
                "explicit_gemm",
                {},
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                device,
                ctx.needs_input_grad,
            )

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 10)


# ---------------------------------------------------------------------------
# Algorithm execution dispatch (shared by forward/backward)
# ---------------------------------------------------------------------------


def _execute_forward(
    algo: str,
    params: Dict[str, Any],
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    fwd_block_size: Optional[int],
) -> Tensor:
    if algo == "explicit_gemm":
        return _explicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
        )
    elif algo == "implicit_gemm":
        current_fwd_block_size = params.get("fwd_block_size")
        if current_fwd_block_size is None:
            current_fwd_block_size = (
                fwd_block_size if fwd_block_size is not None else 16
            )
            logger.warning(
                f"fwd_block_size not found in chosen_fwd_params for IMPLICIT_GEMM, using fallback {current_fwd_block_size}"
            )
        return _implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            current_fwd_block_size,
        )
    elif algo == "cutlass_implicit_gemm":
        result = _cutlass_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            accumulator_type=params.get("accumulator_type", torch.float32),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"Error in _cutlass_implicit_gemm_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_implicit_gemm":
        result = _cute_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"Error in _cute_implicit_gemm_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_grouped":
        result = _cute_grouped_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            mma_tile=params.get("mma_tile", 3),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"Error in _cute_grouped_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "explicit_gemm_grouped":
        return _explicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "implicit_gemm_grouped":
        from .implicit_direct import _implicit_gemm_forward_grouped

        return _implicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            fwd_block_size=params.get("fwd_block_size", 16),
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "cutlass_grouped_hybrid":
        result = _cutlass_implicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            accumulator_type=params.get("accumulator_type", torch.float32),
            saturation_m=params.get("saturation_m", 5000),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"Error in _cutlass_implicit_gemm_forward_grouped: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_implicit_gemm_sm90":
        result = _cute_implicit_gemm_sm90_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            mma_tile=params.get("mma_tile", 100),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"Error in _cute_implicit_gemm_sm90_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_grouped_sm90":
        result = _cute_grouped_sm90_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            mma_tile=params.get("mma_tile", 100),
            use_cp_async=params.get("use_cp_async", True),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"Error in _cute_grouped_sm90_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "mask_implicit_gemm":
        return _mask_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            block_size=params.get("block_size", 16),
            mma_tile=params.get("mma_tile", 3),
        )
    else:
        raise ValueError(f"Unsupported forward algorithm: {algo}")


def _execute_backward(
    algo: str,
    params: Dict[str, Any],
    grad_output: Tensor,
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    needs_input_grad: Tuple[bool, ...],
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if algo == "explicit_gemm":
        return _explicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            compute_dtype,
            device,
        )
    elif algo == "implicit_gemm":
        return _implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            gemm_block_size=params.get("bwd_block_size", 16),
            split_k_threads_per_block=params.get("split_k_threads_per_block", 256),
            split_k_factor=params.get("split_k_factor", 4),
            compute_dtype=compute_dtype,
        )
    elif algo == "cutlass_implicit_gemm":
        result = _cutlass_implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            accumulator_type=params.get("accumulator_type", torch.float32),
            device=device,
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"Error in _cutlass_implicit_gemm_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "cute_implicit_gemm":
        result = _cute_implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(needs_input_grad[0], needs_input_grad[1]),
            device=device,
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"Error in _cute_implicit_gemm_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "explicit_gemm_grouped":
        return _explicit_gemm_backward_grouped(
            grad_output,
            in_features,
            weight,
            kernel_map,
            compute_dtype,
            device,
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "implicit_gemm_grouped":
        return _implicit_gemm_backward_grouped(
            grad_output,
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            gemm_block_size=params.get("gemm_block_size", 16),
            split_k_threads_per_block=params.get("split_k_threads_per_block", 256),
            split_k_factor=params.get("split_k_factor", 4),
            compute_dtype=compute_dtype,
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "cutlass_grouped_hybrid":
        result = _cutlass_implicit_gemm_backward_grouped(
            grad_output,
            in_features,
            weight,
            kernel_map,
            accumulator_type=params.get("accumulator_type", torch.float32),
            device=device,
            saturation_m=params.get("saturation_m", 5000),
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"Error in _cutlass_implicit_gemm_backward_grouped: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "cute_grouped":
        result = _cute_grouped_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(needs_input_grad[0], needs_input_grad[1]),
            device=device,
            mma_tile=params.get("mma_tile", 3),
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"Error in _cute_grouped_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "cute_implicit_gemm_sm90":
        result = _cute_implicit_gemm_sm90_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(needs_input_grad[0], needs_input_grad[1]),
            device=device,
            mma_tile=params.get("mma_tile", 100),
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"Error in _cute_implicit_gemm_sm90_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "cute_grouped_sm90":
        result = _cute_grouped_sm90_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(needs_input_grad[0], needs_input_grad[1]),
            device=device,
            mma_tile=params.get("mma_tile", 100),
            use_cp_async=params.get("use_cp_async", True),
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"Error in _cute_grouped_sm90_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "mask_implicit_gemm":
        return _mask_implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            needs_input_grad=needs_input_grad,
            block_size=params.get("block_size", 16),
        )
    else:
        raise ValueError(f"Unsupported backward algorithm: {algo}")
