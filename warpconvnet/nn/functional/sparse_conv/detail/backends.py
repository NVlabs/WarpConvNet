# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Single source-of-truth backend registry for sparse-convolution dispatch.

Every algorithm is wired exactly once here, as an adapter that unpacks a
``FwdCtx`` / ``BwdCtx`` and calls the underlying kernel-logic
function. Both the execution path (``dispatch.py``) and the autotuner
(``autotune.py``) route through these same adapters, so the kernel that gets
benchmarked is provably the kernel that gets executed — they can no longer
drift (previously each maintained its own ``if algo == ...`` ladder with
subtly different argument defaults).

Adapters return the backend's *raw* result:
  - forward:  a ``Tensor`` on success, or an ``int`` GEMM status on failure.
  - backward: a ``(grad_in, grad_weight)`` tuple whose first element is a
              ``Tensor``/``None`` on success or an ``int`` GEMM status on
              failure.

Callers convert that raw result to their convention via the helpers at the
bottom (``run_forward``/``run_backward`` for execution, ``benchmark_forward``/
``benchmark_backward`` for autotune).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult

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
from .mask_gemm import (
    _mask_gemm_forward_logic,
    _mask_gemm_backward_logic,
)
from .algo_params import (
    _HAS_CUTE_BACKEND,
    _HAS_CUTE_GROUPED,
    _HAS_CUTE_SM90,
    _HAS_CUTE_GROUPED_SM90,
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


# ---------------------------------------------------------------------------
# Invocation contexts
# ---------------------------------------------------------------------------


@dataclass
class FwdCtx:
    in_features: Tensor
    weight: Tensor
    kernel_map: IntSearchResult
    num_out_coords: int
    compute_dtype: Optional[torch.dtype]
    params: Dict[str, Any]
    fwd_block_size: Optional[int] = None
    groups: int = 1
    use_fp16_accum: bool = False


@dataclass
class BwdCtx:
    grad_output: Tensor
    in_features: Tensor
    weight: Tensor
    kernel_map: IntSearchResult
    num_out_coords: int
    compute_dtype: Optional[torch.dtype]
    device: torch.device
    needs_input_grad: Tuple[bool, ...]
    params: Dict[str, Any]
    weight_T: Optional[Tensor] = None
    groups: int = 1
    use_fp16_accum: bool = False


# Type aliases for the adapter signatures.
FwdResult = Any  # Tensor | int
BwdResult = Tuple[Any, Any]  # (Tensor|int|None, Tensor|None)
FwdFn = Callable[[FwdCtx], FwdResult]
BwdFn = Callable[[BwdCtx], BwdResult]


# ---------------------------------------------------------------------------
# Forward adapters
# ---------------------------------------------------------------------------


def _fwd_explicit(ctx: FwdCtx) -> FwdResult:
    return _explicit_gemm_forward_logic(
        ctx.in_features, ctx.weight, ctx.kernel_map, ctx.num_out_coords, ctx.compute_dtype
    )


def _fwd_implicit(ctx: FwdCtx) -> FwdResult:
    bs = ctx.params.get("fwd_block_size", ctx.fwd_block_size or 16)
    return _implicit_gemm_forward_logic(
        ctx.in_features, ctx.weight, ctx.kernel_map, ctx.num_out_coords, ctx.compute_dtype, bs
    )


def _fwd_cutlass(ctx: FwdCtx) -> FwdResult:
    return _cutlass_implicit_gemm_forward_logic(
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        accumulator_type=ctx.params.get("accumulator_type", torch.float32),
    )


def _fwd_explicit_grouped(ctx: FwdCtx) -> FwdResult:
    return _explicit_gemm_forward_grouped(
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        ctx.compute_dtype,
        saturation_m=ctx.params.get("saturation_m", 5000),
    )


def _fwd_implicit_grouped(ctx: FwdCtx) -> FwdResult:
    return _implicit_gemm_forward_grouped(
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        ctx.compute_dtype,
        fwd_block_size=ctx.params.get("fwd_block_size", 16),
        saturation_m=ctx.params.get("saturation_m", 5000),
    )


def _fwd_cutlass_grouped(ctx: FwdCtx) -> FwdResult:
    return _cutlass_implicit_gemm_forward_grouped(
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        accumulator_type=ctx.params.get("accumulator_type", torch.float32),
        saturation_m=ctx.params.get("saturation_m", 5000),
    )


def _fwd_mask(ctx: FwdCtx) -> FwdResult:
    return _mask_gemm_forward_logic(
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        ctx.params,
        groups=ctx.groups,
    )


def _fwd_cute(ctx: FwdCtx) -> FwdResult:
    return _cute_implicit_gemm_forward_logic(
        ctx.in_features, ctx.weight, ctx.kernel_map, ctx.num_out_coords
    )


def _fwd_cute_grouped(ctx: FwdCtx) -> FwdResult:
    return _cute_grouped_forward_logic(
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        mma_tile=ctx.params.get("mma_tile", 3),
    )


def _fwd_cute_sm90(ctx: FwdCtx) -> FwdResult:
    return _cute_implicit_gemm_sm90_forward_logic(
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        backend=ctx.params["backend"],
        tile_id=ctx.params["tile_id"],
    )


def _fwd_cute_grouped_sm90(ctx: FwdCtx) -> FwdResult:
    return _cute_grouped_sm90_forward_logic(
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        backend=ctx.params["backend"],
        tile_id=ctx.params["tile_id"],
        use_cp_async=ctx.params.get("use_cp_async", True),
    )


# ---------------------------------------------------------------------------
# Backward adapters
# ---------------------------------------------------------------------------


def _bwd_explicit(ctx: BwdCtx) -> BwdResult:
    return _explicit_gemm_backward_logic(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.compute_dtype,
        ctx.device,
    )


def _bwd_implicit(ctx: BwdCtx) -> BwdResult:
    # The candidate pools key the block size as "gemm_block_size"; the old
    # execution path read "bwd_block_size" (which the pools never set, so it
    # silently always defaulted to 16) while autotune read "gemm_block_size".
    # Reading "gemm_block_size" here (with "bwd_block_size" as legacy fallback)
    # makes execution honour the benchmarked value and unifies both paths.
    return _implicit_gemm_backward_logic(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        gemm_block_size=ctx.params.get("gemm_block_size", ctx.params.get("bwd_block_size", 16)),
        split_k_threads_per_block=ctx.params.get("split_k_threads_per_block", 256),
        split_k_factor=ctx.params.get("split_k_factor", 4),
        compute_dtype=ctx.compute_dtype,
    )


def _bwd_cutlass(ctx: BwdCtx) -> BwdResult:
    return _cutlass_implicit_gemm_backward_logic(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        accumulator_type=ctx.params.get("accumulator_type", torch.float32),
        device=ctx.device,
    )


def _bwd_explicit_grouped(ctx: BwdCtx) -> BwdResult:
    return _explicit_gemm_backward_grouped(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.compute_dtype,
        ctx.device,
        saturation_m=ctx.params.get("saturation_m", 5000),
    )


def _bwd_implicit_grouped(ctx: BwdCtx) -> BwdResult:
    return _implicit_gemm_backward_grouped(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        gemm_block_size=ctx.params.get("gemm_block_size", 16),
        split_k_threads_per_block=ctx.params.get("split_k_threads_per_block", 256),
        split_k_factor=ctx.params.get("split_k_factor", 4),
        compute_dtype=ctx.compute_dtype,
        saturation_m=ctx.params.get("saturation_m", 5000),
    )


def _bwd_cutlass_grouped(ctx: BwdCtx) -> BwdResult:
    return _cutlass_implicit_gemm_backward_grouped(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        accumulator_type=ctx.params.get("accumulator_type", torch.float32),
        device=ctx.device,
        saturation_m=ctx.params.get("saturation_m", 5000),
    )


def _bwd_cute(ctx: BwdCtx) -> BwdResult:
    return _cute_implicit_gemm_backward_logic(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
        device=ctx.device,
    )


def _bwd_cute_grouped(ctx: BwdCtx) -> BwdResult:
    # Fast path: fused C++ wgrad when available and only wgrad is requested
    # (works for any stride). This is the path that executes, so autotune now
    # times it too instead of the generic cute_grouped backward.
    _has_fused = hasattr(_C.gemm, "sparse_conv_wgrad")
    if _has_fused and ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]:
        km = ctx.kernel_map
        iden = km.identity_map_index if km.identity_map_index is not None else -1
        grad_weight = _C.gemm.sparse_conv_wgrad(
            ctx.in_features,
            ctx.grad_output,
            km.in_maps,
            km.out_maps,
            km.offsets,
            iden,
            len(km),
            ctx.params.get("mma_tile", 3),
        )
        return None, grad_weight
    return _cute_grouped_backward_logic(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
        device=ctx.device,
        mma_tile=ctx.params.get("mma_tile", 3),
        weight_T=ctx.weight_T,
        splits=ctx.params.get("splits", 1),
    )


def _bwd_cute_sm90(ctx: BwdCtx) -> BwdResult:
    return _cute_implicit_gemm_sm90_backward_logic(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
        device=ctx.device,
        backend=ctx.params["backend"],
        tile_id=ctx.params["tile_id"],
    )


def _bwd_cute_grouped_sm90(ctx: BwdCtx) -> BwdResult:
    return _cute_grouped_sm90_backward_logic(
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
        device=ctx.device,
        backend=ctx.params["backend"],
        tile_id=ctx.params["tile_id"],
        use_cp_async=ctx.params.get("use_cp_async", True),
    )


def _bwd_mask_impl(algo: str, ctx: BwdCtx) -> BwdResult:
    return _mask_gemm_backward_logic(
        algo,
        ctx.grad_output,
        ctx.in_features,
        ctx.weight,
        ctx.kernel_map,
        ctx.num_out_coords,
        ctx.device,
        ctx.needs_input_grad,
        ctx.params,
        weight_T=ctx.weight_T,
        groups=ctx.groups,
        use_fp16_accum=ctx.use_fp16_accum,
    )


def _bwd_mask(ctx: BwdCtx) -> BwdResult:
    # Native dgrad (W read with an in-shared-memory stride transpose).
    return _bwd_mask_impl("mask_gemm", ctx)


def _bwd_mask_fwd_as_dgrad(ctx: BwdCtx) -> BwdResult:
    # dgrad via the fwd kernel after the caller pre-transposes the weight.
    return _bwd_mask_impl("mask_gemm_fwd_as_dgrad", ctx)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

FORWARD_BACKENDS: Dict[str, FwdFn] = {
    "explicit_gemm": _fwd_explicit,
    "implicit_gemm": _fwd_implicit,
    "cutlass_implicit_gemm": _fwd_cutlass,
    "explicit_gemm_grouped": _fwd_explicit_grouped,
    "implicit_gemm_grouped": _fwd_implicit_grouped,
    "cutlass_grouped_hybrid": _fwd_cutlass_grouped,
    "mask_gemm": _fwd_mask,
}

BACKWARD_BACKENDS: Dict[str, BwdFn] = {
    "explicit_gemm": _bwd_explicit,
    "implicit_gemm": _bwd_implicit,
    "cutlass_implicit_gemm": _bwd_cutlass,
    "explicit_gemm_grouped": _bwd_explicit_grouped,
    "implicit_gemm_grouped": _bwd_implicit_grouped,
    "cutlass_grouped_hybrid": _bwd_cutlass_grouped,
    "cute_grouped": _bwd_cute_grouped,
    "mask_gemm": _bwd_mask,
    "mask_gemm_fwd_as_dgrad": _bwd_mask_fwd_as_dgrad,
}

if _HAS_CUTE_BACKEND:
    FORWARD_BACKENDS["cute_implicit_gemm"] = _fwd_cute
    BACKWARD_BACKENDS["cute_implicit_gemm"] = _bwd_cute
if _HAS_CUTE_GROUPED:
    FORWARD_BACKENDS["cute_grouped"] = _fwd_cute_grouped
    # backward cute_grouped registered unconditionally above (fused-wgrad path
    # only needs _C.gemm.sparse_conv_wgrad); the generic logic is gated here.
if _HAS_CUTE_SM90:
    FORWARD_BACKENDS["cute_implicit_gemm_sm90"] = _fwd_cute_sm90
    BACKWARD_BACKENDS["cute_implicit_gemm_sm90"] = _bwd_cute_sm90
if _HAS_CUTE_GROUPED_SM90:
    FORWARD_BACKENDS["cute_grouped_sm90"] = _fwd_cute_grouped_sm90
    BACKWARD_BACKENDS["cute_grouped_sm90"] = _bwd_cute_grouped_sm90


# ---------------------------------------------------------------------------
# Result-convention helpers
# ---------------------------------------------------------------------------


def _gemm_status_str(status: int) -> str:
    return _C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(status))


def run_forward(algo: str, ctx: FwdCtx) -> FwdResult:
    """Execute a forward backend, raising on a non-zero GEMM status."""
    try:
        fn = FORWARD_BACKENDS[algo]
    except KeyError:
        raise ValueError(f"Unsupported forward algorithm: {algo}")
    result = fn(ctx)
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"{algo} fwd error: {_gemm_status_str(result)}")
    return result


def run_backward(algo: str, ctx: BwdCtx) -> BwdResult:
    """Execute a backward backend, raising on a non-zero GEMM status."""
    try:
        fn = BACKWARD_BACKENDS[algo]
    except KeyError:
        raise ValueError(f"Unsupported backward algorithm: {algo}")
    result = fn(ctx)
    if isinstance(result[0], int) and result[0] != 0:
        raise RuntimeError(f"{algo} bwd error: {_gemm_status_str(result[0])}")
    return result


def benchmark_forward(algo: str, ctx: FwdCtx) -> Optional[int]:
    """Run a forward backend for autotuning; return a non-zero status int when
    the candidate is unsupported, else ``None``."""
    fn = FORWARD_BACKENDS.get(algo)
    if fn is None:
        raise ValueError(f"Unsupported algo_mode in benchmark_forward: {algo}")
    result = fn(ctx)
    if isinstance(result, int) and result != 0:
        return result
    return None


def benchmark_backward(algo: str, ctx: BwdCtx) -> Optional[int]:
    """Run a backward backend for autotuning; return a non-zero status int when
    the candidate is unsupported, else ``None``."""
    fn = BACKWARD_BACKENDS.get(algo)
    if fn is None:
        raise ValueError(f"Unsupported algo_mode in benchmark_backward: {algo}")
    result = fn(ctx)
    if isinstance(result[0], int) and result[0] != 0:
        return result[0]
    return None
