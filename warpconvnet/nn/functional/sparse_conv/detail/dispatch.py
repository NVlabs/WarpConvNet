# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm dispatch for sparse convolution forward and backward.

This is a thin shim over the backend registry in ``backends.py``: it builds the
invocation context, routes through the single registry, and converts a non-zero
GEMM status into a ``RuntimeError``. The per-algorithm wiring lives in
``backends.py`` (shared with the autotuner so the benchmarked kernel is exactly
the executed kernel). The backward dispatch supports split dgrad/wgrad execution
via ``needs_input_grad=(bool, bool)``.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from warpconvnet.geometry.coords.search.search_results import IntSearchResult

from .backends import BwdCtx, FwdCtx, run_backward, run_forward


def _execute_forward(
    algo: str,
    params: Dict[str, Any],
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    fwd_block_size: Optional[int],
    groups: int = 1,
    use_fp16_accum: bool = False,
) -> Tensor:
    """Dispatch forward pass to the selected algorithm."""
    if groups > 1 and algo != "mask_gemm":
        raise ValueError(
            f"Group convolution (groups={groups}) only supported with algo='mask_gemm', "
            f"got '{algo}'"
        )
    if groups > 1:
        C_in_g = weight.shape[2]
        C_out_g = weight.shape[3]
        if C_in_g < 8 or C_out_g < 8:
            raise ValueError(
                f"Group convolution requires per-group channels >= 8 "
                f"(got C_in/G={C_in_g}, C_out/G={C_out_g}). "
                f"Reduce groups or increase channels."
            )

    ctx = FwdCtx(
        in_features=in_features,
        weight=weight,
        kernel_map=kernel_map,
        num_out_coords=num_out_coords,
        compute_dtype=compute_dtype,
        params=params,
        fwd_block_size=fwd_block_size,
        groups=groups,
        use_fp16_accum=use_fp16_accum,
    )
    return run_forward(algo, ctx)


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
    weight_T: Optional[Tensor] = None,
    groups: int = 1,
    use_fp16_accum: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Dispatch backward pass to the selected algorithm.

    Args:
        weight_T: Pre-computed weight.transpose(1,2).contiguous() to avoid
            redundant copies when dgrad and wgrad are dispatched separately.

    Returns (grad_in_features, grad_weight). Either can be None if the
    corresponding needs_input_grad flag is False AND the algorithm supports it.
    """
    ctx = BwdCtx(
        grad_output=grad_output,
        in_features=in_features,
        weight=weight,
        kernel_map=kernel_map,
        num_out_coords=num_out_coords,
        compute_dtype=compute_dtype,
        device=device,
        needs_input_grad=needs_input_grad,
        params=params,
        weight_T=weight_T,
        groups=groups,
        use_fp16_accum=use_fp16_accum,
    )
    return run_backward(algo, ctx)
