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
    candidate_pool,
    _filter_benchmark_params_by_env_config,
    _AB_MASK_GEMM_STRIDED_F32ACC,
)
from .autotune import (
    _BENCHMARK_AB_RESULTS,
    _BENCHMARK_ABT_RESULTS,
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


# Algorithms that tolerate a non-16B-aligned weight (operand B) base:
#   - explicit_gemm / explicit_gemm_grouped : torch.matmul -> cuBLAS, which handles
#     2-byte-aligned bf16/fp16 operands internally (non-128-bit path).
#   - mask_gemm, cute_implicit_gemm, cute_grouped : dense-B loader guards on
#     ((ptr | N*sizeof) & 15) and falls back to scalar ld.global + STS.128.
# The CUTLASS GemmUniversal path (cutlass_implicit_gemm), the SM90 CuTe kernels, and the
# custom implicit_gemm kernel do NOT self-handle (or are unverified) — they issue 128-bit
# loads assuming an aligned base and hard-fault, so a pool with them forces a weight clone.
_ALIGN_SELF_HANDLING = frozenset(
    {
        "explicit_gemm",
        "explicit_gemm_grouped",
        "mask_gemm",
        "cute_implicit_gemm",
        "cute_grouped",
    }
)


def _pool_self_handles_alignment(algo_filter: Any) -> bool:
    """True if every candidate algo handles misaligned operand bases in-kernel.

    Only an explicit mask_gemm-only filter qualifies; the auto/all/trimmed pools
    include cutlass/cute GEMMs that fault on a misaligned base.
    """
    if isinstance(algo_filter, str):
        if algo_filter in ("auto", "all", "trimmed"):
            return False
        algo_filter = [algo_filter]
    if isinstance(algo_filter, list):
        return len(algo_filter) > 0 and all(a in _ALIGN_SELF_HANDLING for a in algo_filter)
    return False


def _ensure_aligned(t: Optional[Tensor]) -> Optional[Tensor]:
    """Return a 16-byte-aligned-base view of ``t`` (clone if misaligned).

    A weight or feature tensor that is a contiguous view into a flat parameter
    buffer (e.g. DeepSpeed/ZeRO) can sit at a non-16B offset; ``.contiguous()``
    and ``.to(dtype)`` are no-ops there and preserve the misaligned pointer, so a
    CUTLASS/CuTe kernel hard-faults. This must run BEFORE autotune, which
    benchmarks the real tensors. clone() forces an allocator-aligned base. Skip
    the clone when the candidate pool is mask_gemm-only (see _ALIGN_SELF_HANDLING).
    """
    if t is not None and t.data_ptr() % 16 != 0:
        return t.clone()
    return t


_STRIDED_FWD_TILE_IDS = frozenset(range(300, 308))


def _results_include_tile(results: Any, tile_ids: frozenset[int]) -> bool:
    if isinstance(results, tuple):
        result_iter = [results]
    else:
        result_iter = results or []
    for item in result_iter:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        algo, params = item[0], item[1]
        if algo == "mask_gemm" and isinstance(params, dict) and params.get("tile_id") in tile_ids:
            return True
    return False


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
        conv_cache_metadata: Optional[Dict[str, Any]] = None,
        groups: int = 1,
        use_fp16_accum: bool = False,
    ) -> Float[Tensor, "M C_out"]:
        global _BENCHMARK_AB_RESULTS  # noqa: F824
        output_feature_tensor = None

        # Normalize input algos to strings for benchmarking and caching
        def _to_algo_str_list(
            x: Union[str, List[Union[str, Enum]], Enum],
        ) -> Union[str, List[str]]:
            def _canonical_algo(a: Union[str, Enum]) -> str:
                return a.value if isinstance(a, Enum) else str(a)

            if isinstance(x, list):
                return [_canonical_algo(a) for a in x]
            return _canonical_algo(x)

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

        # Guarantee a 16B-aligned weight base before autotune, unless the pool is
        # mask_gemm-only (which self-handles misalignment). Only the weight can be
        # misaligned in practice (a DeepSpeed/ZeRO flat-buffer view); activations are
        # fresh allocator-aligned tensors. See _ensure_aligned.
        if not _pool_self_handles_alignment(algorithm_filter):
            weight = _ensure_aligned(weight)

        # Step 2: Generate configuration for caching
        C_in = in_features.shape[1]
        C_out = weight.shape[-1] * groups if groups > 1 else weight.shape[2]
        kv = weight.shape[0]

        cache_metadata = conv_cache_metadata or {}
        config = SpatiallySparseConvConfig(
            num_in_coords=in_features.shape[0],
            num_out_coords=num_out_coords,
            in_channels=C_in,
            out_channels=C_out,
            kernel_volume=kv,
            in_dtype=in_features.dtype,
            groups=groups,
            use_fp16_accum=use_fp16_accum,
            **cache_metadata,
        )

        # Lazily build the adaptive/trimmed candidate pool. On the warm path
        # (cache hit in auto/all/trimmed mode) this is never called, so a
        # steady-state training loop pays only a dict lookup per forward instead
        # of rebuilding the N-candidate pool every call.
        def _build_adaptive_fwd_params():
            params = candidate_pool(
                "AB",
                "trimmed" if algorithm_filter == "trimmed" else "auto",
                C_in,
                C_out,
                kv,
                num_in_coords=in_features.shape[0],
                use_fp16_accum=use_fp16_accum,
                voxel_size=voxel_size,
            )
            if num_out_coords != in_features.shape[0]:
                params = list(_AB_MASK_GEMM_STRIDED_F32ACC) + list(params)
            return params

        # Step 3: Check cache first
        cached_result = _BENCHMARK_AB_RESULTS.get(config)
        # Strided-conv staleness: an entry cached before native strided tiles
        # (300-307) existed lacks those candidates and must be re-tuned. Strided
        # tiles enter the pool iff this is a strided conv (num_out != N_in) AND
        # the strided mask_gemm pool is non-empty — test that directly instead of
        # building the pool just to inspect it (equivalent to the old
        # _params_include_tile(adaptive_fwd_params, ...) check).
        _strided_candidates_expected = num_out_coords != in_features.shape[0] and bool(
            _AB_MASK_GEMM_STRIDED_F32ACC
        )
        if (
            cached_result is not None
            and _strided_candidates_expected
            and not _results_include_tile(cached_result, _STRIDED_FWD_TILE_IDS)
        ):
            logger.info(
                "Ignoring stale strided forward autotune cache entry without "
                "tile_id 300..307 candidates"
            )
            _BENCHMARK_AB_RESULTS.pop(config, None)
            cached_result = None

        def _benchmark_and_cache_fwd(filtered_params):
            results = _run_forward_benchmarks(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                custom_params=filtered_params,
                groups=groups,
            )
            _BENCHMARK_AB_RESULTS[config] = results
            generic_benchmark_update_entry(
                "AB_gather_scatter",
                config,
                _serialize_benchmark_results(results),
                force=False,
            )
            return results

        if cached_result is not None:
            # Support tuple (best) or list-of-tuples (best-first)
            if isinstance(cached_result, tuple):
                best_list = [cached_result]
            else:
                best_list = cached_result
            if algorithm_filter in ("auto", "all", "trimmed"):
                chosen_fwd_algo, chosen_fwd_params, _ = best_list[0]
            else:
                filtered_cached_results = [
                    (algo, params, time)
                    for algo, params, time in best_list
                    if algo in algorithm_filter
                ]
                if filtered_cached_results:
                    chosen_fwd_algo, chosen_fwd_params, _ = filtered_cached_results[0]
                else:
                    filtered_params = _filter_benchmark_params_by_env_config(
                        _build_adaptive_fwd_params(), algorithm_filter, is_forward=True
                    )
                    if not filtered_params and "explicit_gemm" in algorithm_filter:
                        chosen_fwd_algo, chosen_fwd_params = (
                            "explicit_gemm",
                            {},
                        )
                    else:
                        all_fwd_benchmark_results = _benchmark_and_cache_fwd(filtered_params)
                        chosen_fwd_algo, chosen_fwd_params, _ = all_fwd_benchmark_results[0]
        else:
            # Step 4: No cache - benchmark within filtered space. "auto"/"all"/
            # "trimmed" and an explicit algo list all funnel through the same
            # filter call.
            filtered_params = _filter_benchmark_params_by_env_config(
                _build_adaptive_fwd_params(), algorithm_filter, is_forward=True
            )
            all_fwd_benchmark_results = _benchmark_and_cache_fwd(filtered_params)
            chosen_fwd_algo, chosen_fwd_params, _ = all_fwd_benchmark_results[0]

        # Step 5: Pre-cast weight once (avoids per-algorithm re-casting).
        # weight is already 16B-aligned (guarded at function entry).
        if compute_dtype is not None:
            _weight_cast = weight.contiguous().to(dtype=compute_dtype)
        else:
            _weight_cast = weight.contiguous()

        logger.debug(
            f"[dispatch] FWD algo={chosen_fwd_algo} params={chosen_fwd_params} "
            f"N_in={in_features.shape[0]} N_out={num_out_coords} "
            f"C_in={in_features.shape[1]} C_out={C_out} "
            f"kv={kv} dtype={in_features.dtype}"
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
                groups=groups,
                use_fp16_accum=use_fp16_accum,
            )
        except (RuntimeError, Exception) as e:
            if chosen_fwd_algo == "explicit_gemm":
                raise  # No fallback for the fallback
            logger.warning(
                f"Forward algorithm '{chosen_fwd_algo}' failed at execution: {e}. "
                f"Falling back to explicit_gemm."
            )
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
                groups=groups,
                use_fp16_accum=use_fp16_accum,
            )

        # Save backward state when any input requires gradients.
        # Note: torch.is_grad_enabled() is False inside Function.forward()
        # by design (PyTorch >= 2.1), so we check needs_input_grad instead.
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # Features and weight are pre-cast to compute_dtype in helper.py
            # before Function.apply(), so they are already fp16 under AMP.
            ctx.save_for_backward(in_features, weight)
            ctx.kernel_map = kernel_map
            ctx.groups = groups
            ctx.use_fp16_accum = use_fp16_accum
            ctx.config_params_for_bwd = {
                "num_in_coords": in_features.shape[0],
                "num_out_coords": num_out_coords,
                "in_channels": in_features.shape[1],
                "out_channels": C_out,
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
                "conv_cache_metadata": cache_metadata,
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
        None,
        None,
        None,
        None,
    ]:
        kernel_map = getattr(ctx, "kernel_map", None)
        config_params = getattr(ctx, "config_params_for_bwd", None)
        if kernel_map is None or config_params is None or len(ctx.saved_tensors) == 0:
            # Forward ran without grad (e.g., frozen backbone with torch.no_grad())
            return _pad_tuple(None, None, 14)

        in_features, weight = ctx.saved_tensors
        num_out_coords = config_params["num_out_coords"]
        compute_dtype = config_params["compute_dtype"]
        device = config_params["device"]
        initial_dgrad_algo = config_params["initial_dgrad_algo"]
        initial_wgrad_algo = config_params["initial_wgrad_algo"]

        grad_in_features, grad_weight = None, None

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 14)

        groups = getattr(ctx, "groups", 1)
        use_fp16_accum = getattr(ctx, "use_fp16_accum", False)
        N_in, C_in = in_features.shape
        K = weight.shape[0]
        C_out = weight.shape[-1] * groups if groups > 1 else weight.shape[2]
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
            return _pad_tuple(grad_in_final, grad_weight_final, 14)

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

        use_fp16_accum_bwd = getattr(ctx, "use_fp16_accum", False)
        cache_metadata = config_params.get("conv_cache_metadata", {})
        # Dgrad config: swapped perspective — the ABt kernel iterates N_in rows,
        # gathers from N_out, reduces over C_out, outputs C_in.
        dgrad_config = SpatiallySparseConvConfig(
            num_in_coords=N_out_bwd,
            num_out_coords=N_in_bwd,
            in_channels=C_out_bwd,
            out_channels=C_in_bwd,
            kernel_volume=kv_bwd,
            in_dtype=grad_output.dtype,
            groups=groups,
            use_fp16_accum=use_fp16_accum_bwd,
            **cache_metadata,
        )
        # Wgrad config: reduction over gathered pairs, no swap needed.
        wgrad_config = SpatiallySparseConvConfig(
            num_in_coords=N_in_bwd,
            num_out_coords=N_out_bwd,
            in_channels=C_in_bwd,
            out_channels=C_out_bwd,
            kernel_volume=kv_bwd,
            in_dtype=grad_output.dtype,
            groups=groups,
            use_fp16_accum=use_fp16_accum_bwd,
            **cache_metadata,
        )

        def _normalize_algo(algo):
            def _canonical_algo(a):
                return str(a.value) if isinstance(a, Enum) else str(a)

            if isinstance(algo, list):
                return [_canonical_algo(a) for a in algo]
            return _canonical_algo(algo)

        dgrad_filter = _normalize_algo(initial_dgrad_algo)
        wgrad_filter = _normalize_algo(initial_wgrad_algo)

        # Guarantee a 16B-aligned weight base before autotune, unless both the dgrad
        # and wgrad pools are mask_gemm-only (which self-handle). Only the weight can be
        # misaligned in practice (DeepSpeed/ZeRO flat-buffer view). See _ensure_aligned.
        if not (
            _pool_self_handles_alignment(dgrad_filter)
            and _pool_self_handles_alignment(wgrad_filter)
        ):
            weight = _ensure_aligned(weight)

        # Separate candidate lists for dgrad (AB) vs wgrad (AtB). Both pools are
        # built lazily (only on a cache miss) so a warm steady-state backward
        # pays a dict lookup per direction instead of rebuilding both pools.
        use_fp16_accum = getattr(ctx, "use_fp16_accum", False)

        def _build_filtered_dgrad_params():
            # dgrad is AB gather-scatter from the swapped (C_out, C_in) perspective.
            dgrad_adaptive = candidate_pool(
                "AB",
                "trimmed" if dgrad_filter == "trimmed" else "auto",
                C_out_bwd,
                C_in_bwd,
                kv_bwd,
                num_in_coords=N_in_bwd,
                use_fp16_accum=use_fp16_accum,
            )
            # Add dgrad-via-fwd candidates (fwd kernel with explicit weight
            # transpose). F32Acc always included; F16Acc added when
            # use_fp16_accum=True. PCOFF aliases require mask_words==1
            # (kv_bwd <= 32) — dispatch raises for K>32, so we gate at pool
            # construction. Within the kv<=32 band:
            #   - F32-accum pcoff (909/910/911 fwd_as_dgrad, 68/69 native) always
            #     in pool — fp32 accumulator matches baseline precision.
            #   - F16-accum pcoff (905/906/907/908 fwd_as_dgrad, 64/65/66/67
            #     native) require WARPCONVNET_USE_FP16_ACCUM=true. Bare default
            #     selecting fp16-accum dgrad kernels degrades MinkUNet ScanNet
            #     AMP training (verified via bisect 2026-05-20).
            from .algo_params import (
                _AB_MASK_GEMM_FWD_AS_DGRAD_F16ACC,
                _AB_MASK_GEMM_FWD_AS_DGRAD_F32ACC,
                _AB_MASK_GEMM_FWD_AS_DGRAD_PCOFF_F16ACC,
                _AB_MASK_GEMM_FWD_AS_DGRAD_PCOFF_F32ACC,
                _AB_MASK_GEMM_DGRAD_PCOFF_F16ACC,
                _AB_MASK_GEMM_DGRAD_PCOFF_F32ACC,
            )

            # Small-channel F16-accum pcoff allowance: at max(C_in_bwd, C_out_bwd)
            # <= WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL the per-row K*C
            # accumulation is short enough that fp16-accum rounding drift stays
            # under AMP gradient noise floor. Mirrors _get_adaptive_AB_params gate.
            from warpconvnet.constants import WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL

            _max_ch_bwd = max(C_in_bwd, C_out_bwd)
            _allow_small_ch_pcoff_f16_bwd = (
                WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL > 0
                and _max_ch_bwd <= WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL
            )

            dgrad_adaptive = list(dgrad_adaptive) + list(_AB_MASK_GEMM_FWD_AS_DGRAD_F32ACC)
            if use_fp16_accum:
                dgrad_adaptive += list(_AB_MASK_GEMM_FWD_AS_DGRAD_F16ACC)
            if kv_bwd <= 32:
                dgrad_adaptive += list(_AB_MASK_GEMM_FWD_AS_DGRAD_PCOFF_F32ACC)
                dgrad_adaptive += list(_AB_MASK_GEMM_DGRAD_PCOFF_F32ACC)
                if use_fp16_accum or _allow_small_ch_pcoff_f16_bwd:
                    dgrad_adaptive += list(_AB_MASK_GEMM_FWD_AS_DGRAD_PCOFF_F16ACC)
                    dgrad_adaptive += list(_AB_MASK_GEMM_DGRAD_PCOFF_F16ACC)
            return _filter_benchmark_params_by_env_config(
                dgrad_adaptive, dgrad_filter, is_forward=True
            )

        def _build_filtered_wgrad_params():
            wgrad_adaptive = candidate_pool(
                "AtB",
                "trimmed" if wgrad_filter == "trimmed" else "auto",
                C_in_bwd,
                C_out_bwd,
                kv_bwd,
                num_in_coords=N_in_bwd,
                use_fp16_accum=use_fp16_accum,
            )
            return _filter_benchmark_params_by_env_config(
                wgrad_adaptive, wgrad_filter, is_forward=False
            )

        # Helper to auto-tune one direction. ``build_params`` is a thunk invoked
        # only on a cache miss, so the candidate pool is not built on the warm path.
        def _autotune_one_direction(cache_dict, cache_ns, needs_grad_tuple, build_params, cfg):
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
                custom_params=build_params(),
                needs_input_grad=needs_grad_tuple,
                groups=groups,
            )
            cache_dict[cfg] = results
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
            # Operands already 16B-aligned (guarded at function entry).
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
                _BENCHMARK_ABT_RESULTS,
                "ABt_gather_scatter",
                (True, False),
                _build_filtered_dgrad_params,
                dgrad_config,
            )
            logger.debug(
                f"[dispatch] DGRAD algo={dgrad_algo} params={dgrad_params} "
                f"N_in={in_features.shape[0]} C_in={in_features.shape[1]} "
                f"C_out={C_out} kv={weight.shape[0]}"
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
                    groups=groups,
                )
            except (RuntimeError, Exception) as e:
                logger.warning(f"DGRAD '{dgrad_algo}' failed: {e}. Falling back.")
                _BENCHMARK_ABT_RESULTS.pop(dgrad_config, None)
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
                    groups=groups,
                )

        if ctx.needs_input_grad[1]:
            wgrad_algo, wgrad_params = _autotune_one_direction(
                _BENCHMARK_ATB_RESULTS,
                "AtB_gather_gather",
                (False, True),
                _build_filtered_wgrad_params,
                wgrad_config,
            )
            logger.debug(
                f"[dispatch] WGRAD algo={wgrad_algo} params={wgrad_params} "
                f"N_in={in_features.shape[0]} C_in={in_features.shape[1]} "
                f"C_out={C_out} kv={weight.shape[0]}"
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
                    groups=groups,
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
                    groups=groups,
                )

        # Free pre-cast tensors eagerly
        del _grad_output, _in_features, _weight, _weight_T

        # Release kernel_map GPU tensors (in_maps, out_maps, _pair_table)
        # eagerly. ctx attributes are not managed by save_for_backward and
        # can persist until the autograd graph is garbage collected.
        ctx.kernel_map = None
        ctx.config_params_for_bwd = None

        return _pad_tuple(grad_in_features, grad_weight, 14)
