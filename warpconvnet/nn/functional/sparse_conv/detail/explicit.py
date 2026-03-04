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

from warpconvnet.nn.functional.sparse_conv.detail.grouping import (
    GroupedKernelMap,
    prepare_grouped_kernel_map,
)


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


def _explicit_gemm_forward_grouped(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
    grouping_threshold: float = 0.1,
    saturation_m: int = 5000,
) -> Float[Tensor, "M C_out"]:
    """Forward pass using explicit GEMM with adaptive offset grouping.

    Large offsets (M_k >= saturation_m) are processed individually.
    Small offsets are grouped into buckets and executed as batched GEMMs.
    """
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

    grouped = prepare_grouped_kernel_map(
        kernel_map,
        grouping_threshold=grouping_threshold,
        saturation_m=saturation_m,
    )

    # Process large offsets individually (existing path)
    for k_idx in grouped.large_offset_indices:
        in_map, out_map = kernel_map[k_idx]
        in_map = in_map.to(device)
        out_map = out_map.to(device)
        curr_out = torch.matmul(comp_in_feats[in_map], comp_weight[k_idx])
        output_feature_tensor[out_map] += curr_out.to(device=device)

    # Process small offset buckets as batched GEMMs with vectorized gather/scatter
    C_in = weight.shape[1]
    C_out = weight.shape[2]
    for bucket_offsets, cat_in, cat_out, flat_idx, max_m in zip(
        grouped.buckets,
        grouped.bucket_cat_in_maps,
        grouped.bucket_cat_out_maps,
        grouped.bucket_gather_flat_idx,
        grouped.bucket_max_m,
    ):
        B = len(bucket_offsets)
        total_pairs = cat_in.shape[0]

        # Vectorized gather: scatter features into padded buffer via flat indices
        gathered_flat = torch.zeros(B * max_m, C_in, device=device, dtype=comp_in_feats.dtype)
        gathered_flat[flat_idx] = comp_in_feats[cat_in]
        gathered = gathered_flat.view(B, max_m, C_in)

        # Batched GEMM: [B, max_m, C_in] @ [B, C_in, C_out] -> [B, max_m, C_out]
        bucket_w = comp_weight[bucket_offsets].contiguous()
        result = torch.bmm(gathered, bucket_w)

        # Vectorized scatter: extract valid results and scatter-add to output
        result_flat = result.view(B * max_m, C_out)
        output_feature_tensor.scatter_add_(
            0,
            cat_out.unsqueeze(1).expand(-1, C_out),
            result_flat[flat_idx],
        )

    return output_feature_tensor.to(dtype=in_features.dtype)


def _explicit_gemm_backward_grouped(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    compute_dtype: Optional[torch.dtype] = None,
    device: torch.device = None,
    grouping_threshold: float = 0.1,
    saturation_m: int = 5000,
) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
    """Backward pass for explicit GEMM with adaptive offset grouping."""
    if device is None:
        device = grad_output.device

    dtype_to_use = compute_dtype if compute_dtype is not None else in_features.dtype
    comp_in_feats = in_features.to(device=device, dtype=dtype_to_use)
    comp_weight = weight.to(device=device, dtype=dtype_to_use)
    comp_grad_output = grad_output.to(device=device, dtype=dtype_to_use)
    grad_weight = torch.zeros_like(comp_weight, device=device)
    C_in = weight.shape[1]
    C_out = weight.shape[2]

    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = torch.matmul(comp_grad_output, comp_weight[iden_idx].T)
        grad_weight[iden_idx] = torch.matmul(comp_in_feats.T, comp_grad_output)
    else:
        grad_in_features = torch.zeros_like(comp_in_feats, device=device)

    grouped = prepare_grouped_kernel_map(
        kernel_map,
        grouping_threshold=grouping_threshold,
        saturation_m=saturation_m,
    )

    # Large offsets: individual (existing path)
    for k_idx in grouped.large_offset_indices:
        in_map, out_map = kernel_map[k_idx]
        curr_grad_output = comp_grad_output[out_map]
        curr_in_feats = comp_in_feats[in_map]
        curr_weight = comp_weight[k_idx]
        grad_in_features[in_map] += torch.matmul(curr_grad_output, curr_weight.T)
        grad_weight[k_idx] += torch.matmul(curr_in_feats.T, curr_grad_output)

    # Small offset buckets: batched with vectorized gather/scatter
    for bucket_offsets, cat_in, cat_out, flat_idx, max_m in zip(
        grouped.buckets,
        grouped.bucket_cat_in_maps,
        grouped.bucket_cat_out_maps,
        grouped.bucket_gather_flat_idx,
        grouped.bucket_max_m,
    ):
        B = len(bucket_offsets)

        # Vectorized gather of grad_output and in_features
        gathered_grad_flat = torch.zeros(B * max_m, C_out, device=device, dtype=dtype_to_use)
        gathered_in_flat = torch.zeros(B * max_m, C_in, device=device, dtype=dtype_to_use)
        gathered_grad_flat[flat_idx] = comp_grad_output[cat_out]
        gathered_in_flat[flat_idx] = comp_in_feats[cat_in]

        gathered_grad = gathered_grad_flat.view(B, max_m, C_out)
        gathered_in = gathered_in_flat.view(B, max_m, C_in)

        # grad_in: [B, max_m, C_out] @ [B, C_out, C_in] -> [B, max_m, C_in]
        bucket_w_T = comp_weight[bucket_offsets].transpose(-1, -2).contiguous()
        grad_in_result = torch.bmm(gathered_grad, bucket_w_T)

        # grad_weight: [B, C_in, max_m] @ [B, max_m, C_out] -> [B, C_in, C_out]
        grad_w_result = torch.bmm(gathered_in.transpose(1, 2), gathered_grad)

        # Vectorized scatter for grad_in
        grad_in_result_flat = grad_in_result.view(B * max_m, C_in)
        grad_in_features.scatter_add_(
            0,
            cat_in.unsqueeze(1).expand(-1, C_in),
            grad_in_result_flat[flat_idx],
        )

        # Assign grad_weight per offset
        grad_weight[bucket_offsets] = grad_w_result

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
