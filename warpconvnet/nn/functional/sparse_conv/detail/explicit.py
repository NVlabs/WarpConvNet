# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from jaxtyping import Float

import torch
from torch import Tensor
from torch.autograd import Function

from warpconvnet.geometry.coords.search.search_results import IntSearchResult

from warpconvnet.utils.type_cast import _maybe_cast
from warpconvnet.utils.ntuple import _pad_tuple


def _explicit_gemm_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
) -> Float[Tensor, "M C_out"]:
    """Reference implementation of forward pass using explicit GEMM."""
    device = in_features.device
    comp_in_feats = _maybe_cast(in_features, compute_dtype)
    comp_weight = _maybe_cast(weight, compute_dtype)
    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        output_feature_tensor = torch.matmul(comp_in_feats, comp_weight[iden_idx])
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
        curr_out_features = torch.matmul(comp_in_feats[in_map], comp_weight[i])
        output_feature_tensor[out_map] += curr_out_features.to(device=device)
    return output_feature_tensor.to(dtype=in_features.dtype)


def _explicit_gemm_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    compute_dtype: Optional[torch.dtype] = None,
    device: torch.device = None,
) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
    """Backward pass for explicit GEMM convolution."""
    if device is None:
        device = grad_output.device

    dtype_to_use = compute_dtype if compute_dtype is not None else in_features.dtype
    comp_in_feats = in_features.to(device=device, dtype=dtype_to_use)
    comp_weight = weight.to(device=device, dtype=dtype_to_use)
    comp_grad_output = grad_output.to(device=device, dtype=dtype_to_use)
    grad_weight = torch.zeros_like(comp_weight, device=device)

    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = torch.matmul(comp_grad_output, comp_weight[iden_idx].T)
        grad_weight[iden_idx] = torch.matmul(comp_in_feats.T, comp_grad_output)
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
        grad_in_features[in_map] += torch.matmul(curr_grad_output, curr_weight.T)
        grad_weight[i] += torch.matmul(curr_in_feats.T, curr_grad_output)
    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )


class SpatiallySparseConvExplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> Float[Tensor, "M C_out"]:
        output_feature_tensor = _explicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
        )
        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.compute_dtype = compute_dtype
        ctx.device = in_features.device
        return output_feature_tensor

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Optional[Float[Tensor, "N C_in"]],
        Optional[Float[Tensor, "K C_in C_out"]],
        None,
        None,
        None,
    ]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        compute_dtype = ctx.compute_dtype
        device = ctx.device

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 5)

        # Basic check for empty inputs, similar to how it was in Unified Function
        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        # Assuming num_out_coords was implicitly handled by grad_output.shape[0] in original explicit backward
        if K == 0 or C_in == 0 or C_out == 0 or N_in == 0 or grad_output.shape[0] == 0:
            grad_in_final = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_weight_final = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in_final, grad_weight_final, 5)

        grad_in_features, grad_weight = _explicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            compute_dtype,
            device,
        )

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 5)
