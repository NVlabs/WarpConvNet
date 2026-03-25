# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm dispatch for sparse convolution forward and backward.

Extracted from unified.py for cleaner separation of concerns.
The backward dispatch supports split dgrad/wgrad execution via
needs_input_grad=(bool, bool).
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
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
    _HAS_CUTE_BACKEND,
    _HAS_CUTE_GROUPED,
    _HAS_CUTE_SM90,
    _HAS_CUTE_GROUPED_SM90,
)

logger = get_logger(__name__)

# Lazy imports for optional backends
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
    """Dispatch forward pass to the selected algorithm."""
    if algo == "explicit_gemm":
        return _explicit_gemm_forward_logic(
            in_features, weight, kernel_map, num_out_coords, compute_dtype
        )
    elif algo == "implicit_gemm":
        bs = params.get("fwd_block_size", fwd_block_size or 16)
        return _implicit_gemm_forward_logic(
            in_features, weight, kernel_map, num_out_coords, compute_dtype, bs
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
                f"cutlass fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_implicit_gemm":
        result = _cute_implicit_gemm_forward_logic(
            in_features, weight, kernel_map, num_out_coords
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"cute fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
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
                f"cute_grouped fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
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
                f"cutlass_grouped fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
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
                f"cute_sm90 fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
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
                f"cute_grouped_sm90 fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
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
    """Dispatch backward pass to the selected algorithm.

    Returns (grad_in_features, grad_weight). Either can be None if the
    corresponding needs_input_grad flag is False AND the algorithm supports it.
    """
    if algo == "explicit_gemm":
        return _explicit_gemm_backward_logic(
            grad_output, in_features, weight, kernel_map, compute_dtype, device
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
                f"cutlass bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
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
                f"cute bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
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
                f"cutlass_grouped bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
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
                f"cute_grouped bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
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
                f"cute_sm90 bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
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
                f"cute_grouped_sm90 bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
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
