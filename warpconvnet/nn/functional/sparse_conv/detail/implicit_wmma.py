from typing import Optional, Tuple, Union
from jaxtyping import Float

import torch
from torch import Tensor
from torch.autograd import Function

import warpconvnet._C as _C
from warpconvnet.csrc.autotuned_ops import (
    wmma_implicit_gemm_sm80_autotuned,
    wmma_split_k_implicit_gemm_sm80_autotuned,
)
from warpconvnet.geometry.coords.search.search_results import IntSearchResult

from warpconvnet.utils.type_cast import _min_dtype
from warpconvnet.utils.ntuple import _pad_tuple


def _wmma_implicit_gemm_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    accumulator_type: torch.dtype = torch.float32,
) -> Union[Float[Tensor, "M C_out"], int]:
    assert (
        _C is not None and wmma_implicit_gemm_sm80_autotuned is not None
    ), "WMMA implicit GEMM ops are not available. Please build warpconvnet with WMMA support."

    device = in_features.device
    iden_idx = kernel_map.identity_map_index
    min_dtype = _min_dtype(in_features.dtype, weight.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)
    if iden_idx is not None:
        output_feature_tensor = torch.matmul(_in_features_detached, _weight_detached[iden_idx])
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

        status = wmma_implicit_gemm_sm80_autotuned(
            _in_features_detached,
            _weight_detached[i],
            output_feature_tensor,
            output_feature_tensor,
            in_map,
            out_map,
            alpha=1.0,
            beta=1.0,
        )
        if status != 0:
            return status
    return output_feature_tensor.to(dtype=in_features.dtype)


def _wmma_implicit_gemm_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    accumulator_type: torch.dtype = torch.float32,
    requires_grad: Tuple[bool, bool] = (True, True),
    device: torch.device = None,
) -> Union[Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]], Tuple[int, int]]:
    assert (
        _C is not None
        and wmma_implicit_gemm_sm80_autotuned is not None
        and wmma_split_k_implicit_gemm_sm80_autotuned is not None
    ), "WMMA implicit GEMM ops are not available. Please build warpconvnet with WMMA support."

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
        grad_in_features = torch.matmul(_grad_output_detached, _weight_detached[iden_idx].T)
        grad_weight[iden_idx] = torch.matmul(_in_features_detached.T, _grad_output_detached)
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
            status = wmma_implicit_gemm_sm80_autotuned(
                _grad_output_detached,
                _weight_detached[i].T.contiguous(),
                grad_in_features,
                grad_in_features,
                out_map,
                in_map,
                alpha=1.0,
                beta=1.0,
            )
            if status != 0:
                return status, i

        if requires_grad[1]:
            status = wmma_split_k_implicit_gemm_sm80_autotuned(
                _in_features_detached,
                _grad_output_detached,
                grad_weight[i],
                in_map,
                out_map,
            )
            if status != 0:
                return status, i

    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )


class SpatiallySparseConvWMMAImplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        accumulator_type: torch.dtype = torch.float32,
    ) -> Union[Float[Tensor, "M C_out"], int]:
        output_feature_tensor = _wmma_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            accumulator_type,
        )
        if isinstance(output_feature_tensor, int):
            raise RuntimeError(
                f"Error in _wmma_implicit_gemm_forward_logic: status {output_feature_tensor}"
            )

        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.wmma_params = {
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
        params = ctx.wmma_params
        device = ctx.device

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 9)

        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        if K == 0 or C_in == 0 or C_out == 0 or N_in == 0 or grad_output.shape[0] == 0:
            grad_in_final = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_weight_final = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in_final, grad_weight_final, 9)

        grad_in_features, grad_weight = _wmma_implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            accumulator_type=params["accumulator_type"],
            requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
            device=device,
        )
        if isinstance(grad_in_features, int):
            raise RuntimeError(
                f"Error in _wmma_implicit_gemm_backward_logic: status {grad_in_features}"
            )
        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 9)
