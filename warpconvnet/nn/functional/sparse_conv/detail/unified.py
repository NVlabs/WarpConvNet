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

from .dispatch import _execute_forward, _execute_backward
from .algo_params import (
    SPARSE_CONV_AB_ALGO_MODE,
    SPARSE_CONV_ATB_ALGO_MODE,
    _HAS_CUTE_BACKEND,
    _HAS_CUTE_GROUPED,
    _HAS_CUTE_SM90,
    _HAS_CUTE_GROUPED_SM90,
    _ATB_PARAMS_AUTO,
    _get_adaptive_AB_params,
    _get_adaptive_AtB_params,
    _get_trimmed_AB_params,
    _get_trimmed_AtB_params,
    _filter_benchmark_params_by_env_config,
)
from .autotune import (
    _BENCHMARK_AB_RESULTS,
    _BENCHMARK_ATB_RESULTS,
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


class UnifiedSpatiallySparseConvFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        fwd_algo: Union[str, List[Union[str, SPARSE_CONV_AB_ALGO_MODE]]],
        dgrad_algo: Union[str, List[Union[str, SPARSE_CONV_AB_ALGO_MODE]]],
        wgrad_algo: Union[str, List[Union[str, SPARSE_CONV_ATB_ALGO_MODE]]],
        compute_dtype: Optional[torch.dtype],
        fwd_block_size: Optional[int],  # For implicit GEMM if not AUTO
        bwd_block_size: Optional[int],  # For implicit GEMM if not AUTO
        voxel_size: Optional[Tuple[int, ...]] = None,
    ) -> Float[Tensor, "M C_out"]:
        global _BENCHMARK_AB_RESULTS  # noqa: F824
        output_feature_tensor = None

        # Normalize input algos to strings for benchmarking and caching
        def _to_algo_str_list(
            x: Union[str, List[Union[str, Enum]], Enum],
        ) -> Union[str, List[str]]:
            if isinstance(x, list):
                return [a.value if isinstance(a, Enum) else str(a) for a in x]
            return x.value if isinstance(x, Enum) else str(x)

        fwd_algo = _to_algo_str_list(fwd_algo)
        dgrad_algo = _to_algo_str_list(dgrad_algo)
        wgrad_algo = _to_algo_str_list(wgrad_algo)

        # UNIFIED APPROACH: Always benchmark within filtered algorithm space
        # Step 1: Determine algorithm filter set
        if isinstance(fwd_algo, list):
            algorithm_filter = fwd_algo
        elif fwd_algo in ("auto", "all", "trimmed"):
            algorithm_filter = fwd_algo
        else:
            # Single algorithm - create list for consistent processing
            algorithm_filter = [str(fwd_algo)]

        # Step 2: Generate configuration for caching
        C_in = in_features.shape[1]
        C_out = weight.shape[2]
        kv = weight.shape[0]
        if algorithm_filter == "trimmed":
            adaptive_fwd_params = _get_trimmed_AB_params(
                C_in,
                C_out,
                kv,
                num_in_coords=in_features.shape[0],
            )
        else:
            adaptive_fwd_params = _get_adaptive_AB_params(
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
        cached_result = _BENCHMARK_AB_RESULTS.get(config)
        if cached_result is not None:
            # Support tuple (best) or list-of-tuples (best-first)
            if isinstance(cached_result, tuple):
                best_tuple = cached_result
                best_list = [best_tuple]
            else:
                best_list = cached_result
            if algorithm_filter in ("auto", "all", "trimmed"):
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
                        _BENCHMARK_AB_RESULTS[config] = all_fwd_benchmark_results[0]
                        # Save a serialized copy (algo as string) to the generic cache
                        generic_benchmark_update_entry(
                            "AB_gather_scatter",
                            config,
                            _serialize_benchmark_results(all_fwd_benchmark_results),
                            force=False,
                        )
                        chosen_fwd_algo, chosen_fwd_params, _ = all_fwd_benchmark_results[0]
        else:
            # Step 4: No cache - always benchmark within filtered space
            if algorithm_filter in ("auto", "all", "trimmed"):
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
            _BENCHMARK_AB_RESULTS[config] = all_fwd_benchmark_results[0]
            # Persist a serialized copy to generic cache
            generic_benchmark_update_entry(
                "AB_gather_scatter",
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
            _BENCHMARK_AB_RESULTS.pop(config, None)
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

        # Save backward state when any input requires gradients.
        # Note: torch.is_grad_enabled() is False inside Function.forward()
        # by design (PyTorch >= 2.1), so we check needs_input_grad instead.
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # Features and weight are pre-cast to compute_dtype in helper.py
            # before Function.apply(), so they are already fp16 under AMP.
            ctx.save_for_backward(in_features, weight)
            ctx.kernel_map = kernel_map
            ctx.config_params_for_bwd = {
                "num_in_coords": in_features.shape[0],
                "num_out_coords": num_out_coords,
                "in_channels": in_features.shape[1],
                "out_channels": weight.shape[2],
                "kernel_volume": weight.shape[0],
                "implicit_matmul_fwd_block_size": chosen_fwd_params.get(
                    "fwd_block_size", fwd_block_size
                ),
                "implicit_matmul_bwd_block_size": bwd_block_size,
                "compute_dtype": compute_dtype,
                "device": in_features.device,
                "initial_dgrad_algo": dgrad_algo,
                "initial_wgrad_algo": wgrad_algo,
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
        kernel_map = getattr(ctx, "kernel_map", None)
        config_params = getattr(ctx, "config_params_for_bwd", None)
        if kernel_map is None or config_params is None or len(ctx.saved_tensors) == 0:
            # Forward ran without grad (e.g., frozen backbone with torch.no_grad())
            return _pad_tuple(None, None, 11)

        in_features, weight = ctx.saved_tensors
        num_out_coords = config_params["num_out_coords"]
        compute_dtype = config_params["compute_dtype"]
        device = config_params["device"]
        initial_dgrad_algo = config_params["initial_dgrad_algo"]
        initial_wgrad_algo = config_params["initial_wgrad_algo"]

        grad_in_features, grad_weight = None, None

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 11)

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
            return _pad_tuple(grad_in_final, grad_weight_final, 11)

        # --- Split dgrad/wgrad auto-tuning ---
        # Each direction is auto-tuned independently so the best algorithm
        # for dgrad (same structure as forward) can differ from wgrad
        # (reduction over voxels).

        config_params = ctx.config_params_for_bwd
        C_in_bwd = config_params["in_channels"]
        C_out_bwd = config_params["out_channels"]
        kv_bwd = config_params["kernel_volume"]
        N_in_bwd = config_params["num_in_coords"]
        N_out_bwd = config_params["num_out_coords"]

        # Dgrad config: swapped perspective — the AB kernel iterates N_in rows,
        # gathers from N_out, reduces over C_out, outputs C_in.
        dgrad_config = SpatiallySparseConvConfig(
            num_in_coords=N_out_bwd,
            num_out_coords=N_in_bwd,
            in_channels=C_out_bwd,
            out_channels=C_in_bwd,
            kernel_volume=kv_bwd,
            in_dtype=grad_output.dtype,
        )
        # Wgrad config: reduction over gathered pairs, no swap needed.
        wgrad_config = SpatiallySparseConvConfig(
            num_in_coords=N_in_bwd,
            num_out_coords=N_out_bwd,
            in_channels=C_in_bwd,
            out_channels=C_out_bwd,
            kernel_volume=kv_bwd,
            in_dtype=grad_output.dtype,
        )

        def _normalize_algo(algo):
            if isinstance(algo, list):
                return [str(a.value) if isinstance(a, Enum) else str(a) for a in algo]
            if isinstance(algo, Enum):
                return str(algo.value)
            return str(algo)

        dgrad_filter = _normalize_algo(initial_dgrad_algo)
        wgrad_filter = _normalize_algo(initial_wgrad_algo)

        # Separate candidate lists for dgrad (AB) vs wgrad (AtB)
        if dgrad_filter == "trimmed":
            dgrad_adaptive = _get_trimmed_AB_params(
                C_out_bwd,
                C_in_bwd,
                kv_bwd,
                num_in_coords=N_in_bwd,
            )
        else:
            dgrad_adaptive = _get_adaptive_AB_params(
                C_out_bwd,
                C_in_bwd,
                kv_bwd,
                num_in_coords=N_in_bwd,
            )
        # Add dgrad-via-fwd candidates (fwd kernel with explicit weight transpose)
        from .algo_params import _AB_PRODUCTION_FWD_AS_DGRAD

        dgrad_adaptive = list(dgrad_adaptive) + list(_AB_PRODUCTION_FWD_AS_DGRAD)
        if wgrad_filter == "trimmed":
            wgrad_adaptive = _get_trimmed_AtB_params(
                C_in_bwd,
                C_out_bwd,
                kv_bwd,
                num_in_coords=N_in_bwd,
            )
        else:
            wgrad_adaptive = _get_adaptive_AtB_params(
                C_in_bwd,
                C_out_bwd,
                kv_bwd,
                num_in_coords=N_in_bwd,
            )
        filtered_dgrad_params = _filter_benchmark_params_by_env_config(
            dgrad_adaptive, dgrad_filter, is_forward=True
        )
        filtered_wgrad_params = _filter_benchmark_params_by_env_config(
            wgrad_adaptive, wgrad_filter, is_forward=False
        )

        # Helper to auto-tune one direction
        def _autotune_one_direction(
            cache_dict, cache_ns, needs_grad_tuple, params_for_direction, cfg
        ):
            cached = cache_dict.get(cfg)
            if cached is not None:
                best_list = [cached] if isinstance(cached, tuple) else cached
                return best_list[0][0], best_list[0][1]
            results = _run_backward_benchmarks(
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                device,
                custom_params=params_for_direction,
                needs_input_grad=needs_grad_tuple,
            )
            cache_dict[cfg] = results[0]
            generic_benchmark_update_entry(
                cache_ns,
                cfg,
                _serialize_benchmark_results(results),
                force=False,
            )
            return results[0][0], results[0][1]

        # Pre-cast tensors once so dgrad and wgrad don't duplicate work.
        # The kernel logic functions will detect matching dtype and skip
        # redundant .to() / .contiguous() / .detach() calls.
        from warpconvnet.utils.type_cast import _min_dtype

        _cast_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
        _min_dt = _min_dtype(_cast_dtype, weight.dtype)
        if _min_dt == torch.float64:
            _min_dt = torch.float32

        def _prepare(t: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
            if t.dtype == dt and t.is_contiguous() and not t.requires_grad:
                return t
            return t.contiguous().detach().to(dtype=dt)

        _grad_output = _prepare(grad_output, _min_dt)
        _in_features = _prepare(in_features, _min_dt)
        _weight = _prepare(weight, _min_dt)

        # Pre-compute weight transpose once for dgrad (both mask and
        # cute_grouped need [K, C_out, C_in] contiguous for the AB kernel).
        _weight_T = _weight.transpose(1, 2).contiguous() if ctx.needs_input_grad[0] else None

        # Auto-tune dgrad and wgrad independently
        grad_in_features = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            dgrad_algo, dgrad_params = _autotune_one_direction(
                _BENCHMARK_AB_RESULTS,
                "AB_gather_scatter",
                (True, False),
                filtered_dgrad_params,
                dgrad_config,
            )
            logger.debug(
                f"[dispatch] DGRAD algo={dgrad_algo} params={dgrad_params} "
                f"N_in={in_features.shape[0]} C_in={in_features.shape[1]} "
                f"C_out={weight.shape[2]} kv={weight.shape[0]}"
            )
            try:
                grad_in_features, _ = _execute_backward(
                    dgrad_algo,
                    dgrad_params,
                    _grad_output,
                    _in_features,
                    _weight,
                    kernel_map,
                    num_out_coords,
                    compute_dtype,
                    device,
                    needs_input_grad=(True, False),
                    weight_T=_weight_T,
                )
            except (RuntimeError, Exception) as e:
                logger.warning(f"DGRAD '{dgrad_algo}' failed: {e}. Falling back.")
                _BENCHMARK_AB_RESULTS.pop(dgrad_config, None)
                grad_in_features, _ = _execute_backward(
                    "explicit_gemm",
                    {},
                    _grad_output,
                    _in_features,
                    _weight,
                    kernel_map,
                    num_out_coords,
                    compute_dtype,
                    device,
                    needs_input_grad=(True, False),
                    weight_T=_weight_T,
                )

        if ctx.needs_input_grad[1]:
            wgrad_algo, wgrad_params = _autotune_one_direction(
                _BENCHMARK_ATB_RESULTS,
                "AtB_gather_gather",
                (False, True),
                filtered_wgrad_params,
                wgrad_config,
            )
            logger.debug(
                f"[dispatch] WGRAD algo={wgrad_algo} params={wgrad_params} "
                f"N_in={in_features.shape[0]} C_in={in_features.shape[1]} "
                f"C_out={weight.shape[2]} kv={weight.shape[0]}"
            )
            try:
                _, grad_weight = _execute_backward(
                    wgrad_algo,
                    wgrad_params,
                    _grad_output,
                    _in_features,
                    _weight,
                    kernel_map,
                    num_out_coords,
                    compute_dtype,
                    device,
                    needs_input_grad=(False, True),
                )
            except (RuntimeError, Exception) as e:
                logger.warning(f"WGRAD '{wgrad_algo}' failed: {e}. Falling back.")
                _BENCHMARK_ATB_RESULTS.pop(wgrad_config, None)
                _, grad_weight = _execute_backward(
                    "explicit_gemm",
                    {},
                    _grad_output,
                    _in_features,
                    _weight,
                    kernel_map,
                    num_out_coords,
                    compute_dtype,
                    device,
                    needs_input_grad=(False, True),
                )

        # Free pre-cast tensors eagerly
        del _grad_output, _in_features, _weight, _weight_T

        # Release kernel_map GPU tensors (in_maps, out_maps, _pair_table)
        # eagerly. ctx attributes are not managed by save_for_backward and
        # can persist until the autograd graph is garbage collected.
        ctx.kernel_map = None
        ctx.config_params_for_bwd = None

        return _pad_tuple(grad_in_features, grad_weight, 11)


# Algorithm execution dispatch moved to dispatch.py
# _execute_forward and _execute_backward are imported from there.
# This comment replaces ~300 lines of dispatch code that was extracted.
_DISPATCH_MOVED = True  # sentinel to confirm extraction
