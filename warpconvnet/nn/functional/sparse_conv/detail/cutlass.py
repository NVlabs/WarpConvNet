# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
from jaxtyping import Float

import torch
from torch import Tensor
from torch.autograd import Function

import warpconvnet._C as _C
from warpconvnet.csrc.autotuned_ops import (
    cutlass_gemm_AD_gather_scatter_autotuned,
    cutlass_gemm_trAB_gather_autotuned,
)
from warpconvnet.geometry.coords.search.search_results import IntSearchResult

from warpconvnet.utils.type_cast import _min_dtype
from warpconvnet.utils.ntuple import _pad_tuple

from warpconvnet.nn.functional.sparse_conv.detail.grouping import (
    prepare_grouped_kernel_map,
)


_CUTLASS_ALIGNMENT = 8  # CUTLASS tensor ops require channels aligned to 8


def _align_to(x: int, alignment: int) -> int:
    """Round up x to the next multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment


def _pad_features(features: Tensor, target_channels: int) -> Tensor:
    """Zero-pad features along the channel dimension."""
    if features.shape[-1] >= target_channels:
        return features
    pad_size = target_channels - features.shape[-1]
    return torch.nn.functional.pad(features, (0, pad_size))


def _pad_weight(weight: Tensor, target_cin: int, target_cout: int) -> Tensor:
    """Zero-pad weight [K, C_in, C_out] along both channel dimensions."""
    K, C_in, C_out = weight.shape
    if C_in >= target_cin and C_out >= target_cout:
        return weight
    pad_cin = target_cin - C_in
    pad_cout = target_cout - C_out
    # Pad: (C_out_left, C_out_right, C_in_left, C_in_right)
    return torch.nn.functional.pad(weight, (0, pad_cout, 0, pad_cin))


def _cutlass_implicit_gemm_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    accumulator_type: torch.dtype = torch.float32,
) -> Union[Float[Tensor, "M C_out"], int]:
    """Forward pass leveraging CUTLASS implicit GEMM kernels with inner autotune."""
    assert (
        _C is not None and cutlass_gemm_AD_gather_scatter_autotuned is not None
    ), "CUTLASS autotuned ops are not available. Please install warpconvnet with cutlass support."

    # Pad unaligned channels so CUTLASS tensor ops can be used
    C_in, C_out = weight.shape[1], weight.shape[2]
    needs_padding = (C_in % _CUTLASS_ALIGNMENT != 0) or (C_out % _CUTLASS_ALIGNMENT != 0)
    orig_C_out = C_out
    if needs_padding:
        target_cin = _align_to(C_in, _CUTLASS_ALIGNMENT)
        target_cout = _align_to(C_out, _CUTLASS_ALIGNMENT)
        in_features = _pad_features(in_features, target_cin)
        weight = _pad_weight(weight, target_cin, target_cout)
        C_in, C_out = target_cin, target_cout

    device = in_features.device
    iden_idx = kernel_map.identity_map_index
    min_dtype = _min_dtype(in_features.dtype, weight.dtype)
    # CUTLASS kernels do not support float64; downcast compute to float32 when needed
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)
    if iden_idx is not None:
        output_feature_tensor = torch.matmul(
            _in_features_detached, _weight_detached[iden_idx]
        )
    else:
        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=min_dtype
        )

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device).int()
        out_map = out_map.to(device).int()
        status = cutlass_gemm_AD_gather_scatter_autotuned(
            _in_features_detached,
            _weight_detached[i],
            output_feature_tensor,
            output_feature_tensor,
            in_map,
            out_map,
            accumulator_type=accumulator_type,
            alpha=1.0,
            beta=1.0,
        )
        if status != 0:
            return status
    # Slice off padding if channels were padded
    if needs_padding:
        output_feature_tensor = output_feature_tensor[:, :orig_C_out]
    return output_feature_tensor.to(dtype=in_features.dtype)


def _cutlass_implicit_gemm_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    accumulator_type: torch.dtype = torch.float32,
    requires_grad: Tuple[bool, bool] = (True, True),
    device: torch.device = None,
) -> Union[
    Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]], Tuple[int, int]
]:
    """Backward pass leveraging CUTLASS implicit GEMM kernels with inner autotune."""
    assert (
        _C is not None and cutlass_gemm_AD_gather_scatter_autotuned is not None
    ), "CUTLASS autotuned ops are not available. Please install warpconvnet with cutlass support."

    C_in, C_out = weight.shape[1], weight.shape[2]
    orig_C_in, orig_C_out = C_in, C_out
    needs_padding_bwd = (C_in % _CUTLASS_ALIGNMENT != 0) or (C_out % _CUTLASS_ALIGNMENT != 0)
    if needs_padding_bwd:
        target_cin = _align_to(C_in, _CUTLASS_ALIGNMENT)
        target_cout = _align_to(C_out, _CUTLASS_ALIGNMENT)
        in_features = _pad_features(in_features, target_cin)
        grad_output = _pad_features(grad_output, target_cout)
        weight = _pad_weight(weight, target_cin, target_cout)
        C_in, C_out = target_cin, target_cout

    if device is None:
        device = in_features.device

    min_dtype = _min_dtype(in_features.dtype, weight.dtype, grad_output.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _grad_output_detached = grad_output.contiguous().detach().to(dtype=min_dtype)
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)
    grad_weight = torch.zeros_like(weight, device=device)

    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = torch.matmul(
            _grad_output_detached, _weight_detached[iden_idx].T
        )
        grad_weight[iden_idx] = torch.matmul(
            _in_features_detached.T, _grad_output_detached
        )
    else:
        grad_in_features = torch.zeros_like(_in_features_detached, device=device)

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device).int()
        out_map = out_map.to(device).int()

        if requires_grad[0]:
            status = cutlass_gemm_AD_gather_scatter_autotuned(
                _grad_output_detached,
                _weight_detached[i].T.contiguous(),
                grad_in_features,
                grad_in_features,
                out_map,
                in_map,
                accumulator_type=accumulator_type,
                alpha=1.0,
                beta=1.0,
            )
            if status != 0:
                return status, i

        if requires_grad[1]:
            status = cutlass_gemm_trAB_gather_autotuned(
                _in_features_detached,
                _grad_output_detached,
                grad_weight[i],
                grad_weight[i],
                in_map,
                out_map,
                alpha=1.0,
                beta=0.0,
                accumulator_type=accumulator_type,
            )
            if status != 0:
                return status, i

    # Slice off padding from gradients
    if needs_padding_bwd:
        grad_in_features = grad_in_features[:, :orig_C_in]
        grad_weight = grad_weight[:, :orig_C_in, :orig_C_out]
    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )


def _cutlass_implicit_gemm_forward_grouped(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    accumulator_type: torch.dtype = torch.float32,
    grouping_threshold: float = 0.1,
    saturation_m: int = 5000,
) -> Union[Float[Tensor, "M C_out"], int]:
    """Forward pass: CUTLASS for large offsets, torch.bmm for grouped small offsets."""
    assert (
        _C is not None and cutlass_gemm_AD_gather_scatter_autotuned is not None
    ), "CUTLASS autotuned ops are not available."

    C_in, C_out = weight.shape[1], weight.shape[2]
    orig_C_out_grp = C_out
    needs_padding_grp = (C_in % _CUTLASS_ALIGNMENT != 0) or (C_out % _CUTLASS_ALIGNMENT != 0)
    if needs_padding_grp:
        target_cin = _align_to(C_in, _CUTLASS_ALIGNMENT)
        target_cout = _align_to(C_out, _CUTLASS_ALIGNMENT)
        in_features = _pad_features(in_features, target_cin)
        weight = _pad_weight(weight, target_cin, target_cout)
        C_in, C_out = target_cin, target_cout

    device = in_features.device
    iden_idx = kernel_map.identity_map_index
    min_dtype = _min_dtype(in_features.dtype, weight.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)

    if iden_idx is not None:
        output_feature_tensor = torch.matmul(
            _in_features_detached, _weight_detached[iden_idx]
        )
    else:
        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=min_dtype
        )

    grouped = prepare_grouped_kernel_map(
        kernel_map,
        grouping_threshold=grouping_threshold,
        saturation_m=saturation_m,
    )

    # Large offsets: CUTLASS with fused gather/scatter (existing path)
    for k_idx in grouped.large_offset_indices:
        in_map, out_map = kernel_map[k_idx]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device).int()
        out_map = out_map.to(device).int()
        status = cutlass_gemm_AD_gather_scatter_autotuned(
            _in_features_detached,
            _weight_detached[k_idx],
            output_feature_tensor,
            output_feature_tensor,
            in_map,
            out_map,
            accumulator_type=accumulator_type,
            alpha=1.0,
            beta=1.0,
        )
        if status != 0:
            return status

    # Small offset buckets: vectorized gather + torch.bmm + vectorized scatter
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

        # Vectorized gather
        gathered_flat = torch.zeros(B * max_m, C_in, device=device, dtype=min_dtype)
        gathered_flat[flat_idx] = _in_features_detached[cat_in]
        gathered = gathered_flat.view(B, max_m, C_in)

        bucket_w = _weight_detached[bucket_offsets].contiguous()
        result = torch.bmm(gathered, bucket_w)

        # Vectorized scatter
        result_flat = result.view(B * max_m, C_out)
        output_feature_tensor.scatter_add_(
            0,
            cat_out.unsqueeze(1).expand(-1, C_out).long(),
            result_flat[flat_idx].to(dtype=output_feature_tensor.dtype),
        )

    if needs_padding_grp:
        output_feature_tensor = output_feature_tensor[:, :orig_C_out_grp]
    return output_feature_tensor.to(dtype=in_features.dtype)


def _cutlass_implicit_gemm_backward_grouped(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    accumulator_type: torch.dtype = torch.float32,
    requires_grad: Tuple[bool, bool] = (True, True),
    device: torch.device = None,
    grouping_threshold: float = 0.1,
    saturation_m: int = 5000,
) -> Union[
    Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]], Tuple[int, int]
]:
    """Backward: CUTLASS for large offsets, torch.bmm for grouped small offsets."""
    assert (
        _C is not None and cutlass_gemm_AD_gather_scatter_autotuned is not None
    ), "CUTLASS autotuned ops are not available."

    C_in, C_out = weight.shape[1], weight.shape[2]
    orig_C_in_grpb, orig_C_out_grpb = C_in, C_out
    needs_padding_grpb = (C_in % _CUTLASS_ALIGNMENT != 0) or (C_out % _CUTLASS_ALIGNMENT != 0)
    if needs_padding_grpb:
        target_cin = _align_to(C_in, _CUTLASS_ALIGNMENT)
        target_cout = _align_to(C_out, _CUTLASS_ALIGNMENT)
        in_features = _pad_features(in_features, target_cin)
        grad_output = _pad_features(grad_output, target_cout)
        weight = _pad_weight(weight, target_cin, target_cout)
        C_in, C_out = target_cin, target_cout

    if device is None:
        device = in_features.device

    min_dtype = _min_dtype(in_features.dtype, weight.dtype, grad_output.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _grad_output_detached = grad_output.contiguous().detach().to(dtype=min_dtype)
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)
    grad_weight = torch.zeros_like(weight, device=device)
    C_in = weight.shape[1]
    C_out = weight.shape[2]

    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = torch.matmul(
            _grad_output_detached, _weight_detached[iden_idx].T
        )
        grad_weight[iden_idx] = torch.matmul(
            _in_features_detached.T, _grad_output_detached
        )
    else:
        grad_in_features = torch.zeros_like(_in_features_detached, device=device)

    grouped = prepare_grouped_kernel_map(
        kernel_map,
        grouping_threshold=grouping_threshold,
        saturation_m=saturation_m,
    )

    # Large offsets: CUTLASS
    for k_idx in grouped.large_offset_indices:
        in_map, out_map = kernel_map[k_idx]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device).int()
        out_map = out_map.to(device).int()

        if requires_grad[0]:
            status = cutlass_gemm_AD_gather_scatter_autotuned(
                _grad_output_detached,
                _weight_detached[k_idx].T.contiguous(),
                grad_in_features,
                grad_in_features,
                out_map,
                in_map,
                accumulator_type=accumulator_type,
                alpha=1.0,
                beta=1.0,
            )
            if status != 0:
                return status, k_idx

        if requires_grad[1]:
            status = cutlass_gemm_trAB_gather_autotuned(
                _in_features_detached,
                _grad_output_detached,
                grad_weight[k_idx],
                grad_weight[k_idx],
                in_map,
                out_map,
                alpha=1.0,
                beta=0.0,
                accumulator_type=accumulator_type,
            )
            if status != 0:
                return status, k_idx

    # Small offset buckets: vectorized gather + bmm + vectorized scatter
    for bucket_offsets, cat_in, cat_out, flat_idx, max_m in zip(
        grouped.buckets,
        grouped.bucket_cat_in_maps,
        grouped.bucket_cat_out_maps,
        grouped.bucket_gather_flat_idx,
        grouped.bucket_max_m,
    ):
        B = len(bucket_offsets)

        # Vectorized gather of grad_output and in_features
        gathered_grad_flat = torch.zeros(
            B * max_m, C_out, device=device, dtype=min_dtype
        )
        gathered_in_flat = torch.zeros(B * max_m, C_in, device=device, dtype=min_dtype)
        gathered_grad_flat[flat_idx] = _grad_output_detached[cat_out]
        gathered_in_flat[flat_idx] = _in_features_detached[cat_in]

        gathered_grad = gathered_grad_flat.view(B, max_m, C_out)
        gathered_in = gathered_in_flat.view(B, max_m, C_in)

        if requires_grad[0]:
            bucket_w_T = _weight_detached[bucket_offsets].transpose(-1, -2).contiguous()
            grad_in_result = torch.bmm(gathered_grad, bucket_w_T)
            grad_in_result_flat = grad_in_result.view(B * max_m, C_in)
            grad_in_features.scatter_add_(
                0,
                cat_in.unsqueeze(1).expand(-1, C_in).long(),
                grad_in_result_flat[flat_idx].to(dtype=grad_in_features.dtype),
            )

        if requires_grad[1]:
            grad_w_result = torch.bmm(gathered_in.transpose(1, 2), gathered_grad)
            grad_weight[bucket_offsets] = grad_w_result.to(dtype=grad_weight.dtype)

    if needs_padding_grpb:
        grad_in_features = grad_in_features[:, :orig_C_in_grpb]
        grad_weight = grad_weight[:, :orig_C_in_grpb, :orig_C_out_grpb]
    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )


class SpatiallySparseConvCutlassImplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        accumulator_type: torch.dtype = torch.float32,
    ) -> Union[Float[Tensor, "M C_out"], int]:
        output_feature_tensor = _cutlass_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            accumulator_type,
        )
        if isinstance(output_feature_tensor, int):
            raise RuntimeError(
                f"Error in _cutlass_implicit_gemm_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(output_feature_tensor))}"
            )

        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.cutlass_params = {
            "accumulator_type": accumulator_type,
        }
        ctx.device = in_features.device
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
    ]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        cutlass_params = ctx.cutlass_params
        device = ctx.device

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 9)

        # Basic check for empty inputs, similar to how it was in Unified Function
        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        # Assuming num_out_coords was implicitly handled by grad_output.shape[0] in original explicit backward
        if K == 0 or C_in == 0 or C_out == 0 or N_in == 0 or grad_output.shape[0] == 0:
            grad_in_final = (
                torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            )
            grad_weight_final = (
                torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            )
            return _pad_tuple(grad_in_final, grad_weight_final, 9)

        grad_in_features, grad_weight = _cutlass_implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            accumulator_type=cutlass_params["accumulator_type"],
            requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
            device=device,
        )
        if isinstance(grad_in_features, int):
            raise RuntimeError(
                f"Error in _cutlass_implicit_gemm_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(grad_in_features))}"
            )
        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 9)
