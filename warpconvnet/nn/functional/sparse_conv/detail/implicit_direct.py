# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from jaxtyping import Float

import torch
from torch import Tensor
from torch.autograd import Function

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult

from warpconvnet.utils.type_cast import _min_dtype
from warpconvnet.utils.ntuple import _pad_tuple


def _implicit_gemm_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    fwd_block_size: int,
) -> Float[Tensor, "M C_out"]:
    """Forward pass using implicit GEMM with a custom CUDA kernel."""
    device = in_features.device
    feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
    min_dtype = _min_dtype(feature_dtype, weight.dtype)

    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)

    N_in, C_in = _in_features_detached.shape
    K, _, C_out = _weight_detached.shape
    iden_idx = kernel_map.identity_map_index

    if iden_idx is not None:
        output_feature_tensor = torch.matmul(_in_features_detached, _weight_detached[iden_idx]).to(
            dtype=min_dtype
        )
    else:
        output_feature_tensor = torch.zeros(
            (num_out_coords, C_out), dtype=min_dtype, device=device
        )

    if (
        num_out_coords == 0
        or K == 0
        or C_in == 0
        or C_out == 0
        or _in_features_detached.shape[0] == 0
    ):
        return output_feature_tensor.to(dtype=in_features.dtype)

    for k_idx in range(len(kernel_map)):
        if k_idx == iden_idx:
            continue

        in_map_k, out_map_k = kernel_map[k_idx]
        num_active_pairs = in_map_k.shape[0]
        if num_active_pairs == 0:
            continue

        _C.gemm.implicit_gemm(
            _in_features_detached,
            _weight_detached[k_idx],
            output_feature_tensor,
            in_map_k,
            out_map_k,
            "basic",
            fwd_block_size,
        )

    return output_feature_tensor.to(dtype=in_features.dtype)


def _implicit_gemm_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    gemm_block_size: int,
    split_k_threads_per_block: int,
    split_k_factor: int,
    compute_dtype: Optional[torch.dtype],
) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
    """Backward pass using implicit GEMM kernels."""
    feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
    min_dtype = _min_dtype(feature_dtype, weight.dtype)
    device = in_features.device

    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)
    _grad_output_detached = grad_output.contiguous().detach().to(dtype=min_dtype)

    N_in, C_in = _in_features_detached.shape
    K, _, C_out = _weight_detached.shape
    iden_idx = kernel_map.identity_map_index

    grad_weight_tensor = torch.zeros((K, C_in, C_out), dtype=min_dtype, device=device)
    if iden_idx is not None:
        grad_in_features = torch.matmul(_grad_output_detached, _weight_detached[iden_idx].T)
        grad_weight_tensor[iden_idx] = torch.matmul(
            _in_features_detached.T, _grad_output_detached
        ).to(dtype=min_dtype)
    else:
        grad_in_features = torch.zeros((N_in, C_in), dtype=min_dtype, device=device)

    if (
        num_out_coords == 0
        or K == 0
        or C_in == 0
        or C_out == 0
        or N_in == 0
        or _grad_output_detached.shape[0] == 0
    ):
        return grad_in_features.to(dtype=in_features.dtype), grad_weight_tensor.to(
            dtype=weight.dtype
        )

    for k_idx in range(len(kernel_map)):
        if k_idx == iden_idx:
            continue

        in_map_k, out_map_k = kernel_map[k_idx]
        num_active_pairs = in_map_k.shape[0]
        if num_active_pairs == 0:
            continue

        _C.gemm.implicit_gemm(
            _grad_output_detached,
            _weight_detached[k_idx].T,
            grad_in_features,
            out_map_k,
            in_map_k,
            "basic",
            gemm_block_size,
        )

        _C.gemm.split_k_implicit_gemm(
            _in_features_detached,
            _grad_output_detached,
            grad_weight_tensor[k_idx],
            in_map_k,
            out_map_k,
            split_k_factor=split_k_factor,
            block_size=split_k_threads_per_block,
        )

    return grad_in_features.to(dtype=in_features.dtype), grad_weight_tensor.to(dtype=weight.dtype)


class SpatiallySparseConvImplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        gemm_block_size: int = 16,
        split_k_threads_per_block: int = 128,
        split_k_factor: int = 4,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> Float[Tensor, "M C_out"]:
        output_feature_tensor = _implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            gemm_block_size,
        )
        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.gemm_params = {
            "compute_dtype": compute_dtype,
            "gemm_block_size": gemm_block_size,
            "split_k_threads_per_block": split_k_threads_per_block,
            "split_k_factor": split_k_factor,
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
    ]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        num_out_coords = grad_output.shape[0]
        gemm_params = ctx.gemm_params

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 7)

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
            return _pad_tuple(grad_in_final, grad_weight_final, 7)

        grad_in_features, grad_weight = _implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            gemm_params["gemm_block_size"],
            gemm_params["split_k_threads_per_block"],
            gemm_params["split_k_factor"],
            gemm_params["compute_dtype"],
        )

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 7)
