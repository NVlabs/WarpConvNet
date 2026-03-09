# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 WGMMA single GEMM dispatch for sparse convolution forward/backward."""

from typing import Optional, Tuple, Union
from jaxtyping import Float

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.type_cast import _min_dtype


def _cute_implicit_gemm_sm90_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    mma_tile: int = 100,
) -> Union[Float[Tensor, "M C_out"], int]:
    """Forward pass using SM90 WGMMA CuTe GEMM kernels."""
    device = in_features.device
    iden_idx = kernel_map.identity_map_index
    min_dtype = _min_dtype(in_features.dtype, weight.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)

    out_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float32

    if iden_idx is not None:
        output_feature_tensor = torch.matmul(_in_features_detached, _weight_detached[iden_idx]).to(
            dtype=out_dtype
        )
    else:
        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=out_dtype
        )

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device).int()
        out_map = out_map.to(device).int()
        status = _C.gemm.cute_gemm_sm90_AD_gather_scatter(
            _in_features_detached,
            _weight_detached[i],
            output_feature_tensor,
            output_feature_tensor,
            in_map,
            out_map,
            mma_tile=mma_tile,
            alpha=1.0,
            beta=1.0,
        )
        if status != 0:
            return status
    return output_feature_tensor


def _cute_implicit_gemm_sm90_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    requires_grad: Tuple[bool, bool] = (True, True),
    device: torch.device = None,
    mma_tile: int = 100,
) -> Union[Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]], Tuple[int, int]]:
    """Backward pass using SM90 WGMMA CuTe GEMM kernels."""
    if device is None:
        device = in_features.device

    min_dtype = _min_dtype(in_features.dtype, weight.dtype, grad_output.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _grad_output_detached = grad_output.contiguous().detach().to(dtype=min_dtype)
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)

    out_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float32
    grad_weight = torch.zeros_like(weight, dtype=out_dtype, device=device)

    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = torch.matmul(_grad_output_detached, _weight_detached[iden_idx].T).to(
            dtype=out_dtype
        )
        grad_weight[iden_idx] = torch.matmul(_in_features_detached.T, _grad_output_detached).to(
            dtype=out_dtype
        )
    else:
        grad_in_features = torch.zeros(
            _in_features_detached.shape[0],
            _in_features_detached.shape[1],
            device=device,
            dtype=out_dtype,
        )

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device).int()
        out_map = out_map.to(device).int()

        if requires_grad[0]:
            status = _C.gemm.cute_gemm_sm90_AD_gather_scatter(
                _grad_output_detached,
                _weight_detached[i].T.contiguous(),
                grad_in_features,
                grad_in_features,
                out_map,
                in_map,
                mma_tile=mma_tile,
                alpha=1.0,
                beta=1.0,
            )
            if status != 0:
                return status, i

        if requires_grad[1]:
            # Weight gradient: use SM80 TrAB (SM90 TrAB with dual gather not yet implemented)
            status = _C.gemm.cute_gemm_trAB_gather(
                _in_features_detached,
                _grad_output_detached,
                grad_weight[i],
                grad_weight[i],
                in_map,
                out_map,
                mma_tile=0,  # SM80 Tile128x128x32 default
                alpha=1.0,
                beta=0.0,
            )
            if status != 0:
                return status, i

    return (
        grad_in_features,
        grad_weight,
    )
