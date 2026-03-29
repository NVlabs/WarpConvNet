# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
from jaxtyping import Float

import torch
from torch import Tensor
from torch.autograd import Function

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.type_cast import _min_dtype
from warpconvnet.utils.ntuple import _pad_tuple

# torch.dtype to pybind11 ScalarType int (matches torch C++ enum)
_DTYPE_TO_SCALAR_TYPE_INT = {
    torch.float16: 5,
    torch.float32: 6,
    torch.bfloat16: 15,
}


def _get_group_indices(offsets_cpu: Tensor, identity_map_index: Optional[int]) -> Tensor:
    """Return indices of non-identity offsets with count > 0 (vectorized)."""
    counts = offsets_cpu[1:] - offsets_cpu[:-1]
    valid = counts > 0
    if identity_map_index is not None:
        valid[identity_map_index] = False
    return torch.where(valid)[0]


def _prepare_grouped_params(
    kernel_map: IntSearchResult,
    weight: Tensor,
    identity_map_index: Optional[int],
    tile_m: int,
    device: torch.device,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, list, int]]:
    """Build device-side arrays for fused grouped GEMM (vectorized).

    Returns (weight_ptrs, tile_offsets, group_sizes, map_offsets, group_indices,
             total_m_tiles) or None if no non-identity groups exist.
    """
    offsets_cpu = kernel_map.offsets  # [K+1] int32, on CPU
    group_indices = _get_group_indices(offsets_cpu, identity_map_index)

    if len(group_indices) == 0:
        return None

    # group_sizes: M_g per group — vectorized diff
    group_sizes = (offsets_cpu[group_indices + 1] - offsets_cpu[group_indices]).to(
        dtype=torch.int32, device=device
    )

    # map_offsets: start offset per group + end of last group
    map_offsets = torch.cat(
        [
            offsets_cpu[group_indices],
            offsets_cpu[group_indices[-1] + 1 : group_indices[-1] + 2],
        ]
    ).to(dtype=torch.int32, device=device)

    # tile_offsets: prefix sum of ceil(M_g / tile_m)
    m_tiles = (group_sizes + tile_m - 1) // tile_m
    tile_offsets = torch.zeros(len(group_indices) + 1, dtype=torch.int32, device=device)
    torch.cumsum(m_tiles, dim=0, out=tile_offsets[1:])
    total_m_tiles = int(tile_offsets[-1])

    # weight_ptrs: stride arithmetic instead of per-element data_ptr()
    base_ptr = weight.data_ptr()
    stride_bytes = weight.stride(0) * weight.element_size()
    weight_ptrs = (
        torch.tensor(base_ptr, dtype=torch.int64, device=device)
        + group_indices.to(dtype=torch.int64, device=device) * stride_bytes
    )

    return (
        weight_ptrs,
        tile_offsets,
        group_sizes,
        map_offsets,
        group_indices.tolist(),
        total_m_tiles,
    )


def _prepare_grouped_trAB_params(
    kernel_map: IntSearchResult,
    grad_weight: Tensor,
    identity_map_index: Optional[int],
    device: torch.device,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Build device-side arrays for fused grouped TrAB GEMM (vectorized).

    Returns (output_ptrs, gather_sizes, map_offsets, in_map, out_map)
    or None if no non-identity groups exist.
    """
    offsets_cpu = kernel_map.offsets  # [K+1] int32, on CPU
    group_indices = _get_group_indices(offsets_cpu, identity_map_index)

    if len(group_indices) == 0:
        return None

    # gather_sizes: vectorized diff
    gather_sizes = (offsets_cpu[group_indices + 1] - offsets_cpu[group_indices]).to(
        dtype=torch.int32, device=device
    )

    # map_offsets: start offset per group
    map_offsets = offsets_cpu[group_indices].to(dtype=torch.int32, device=device)

    # output_ptrs: stride arithmetic
    base_ptr = grad_weight.data_ptr()
    stride_bytes = grad_weight.stride(0) * grad_weight.element_size()
    output_ptrs = (
        torch.tensor(base_ptr, dtype=torch.int64, device=device)
        + group_indices.to(dtype=torch.int64, device=device) * stride_bytes
    )

    in_map_dev = kernel_map.in_maps.to(device).int().contiguous()
    out_map_dev = kernel_map.out_maps.to(device).int().contiguous()

    return output_ptrs, gather_sizes, map_offsets, in_map_dev, out_map_dev


# Map from tile tag index to tM dimension
_TILE_M_SIZES = {
    0: 128,  # Tile128x128x32
    1: 128,  # Tile128x64x32
    2: 64,  # Tile64x128x32
    3: 64,  # Tile64x64x32
    4: 64,  # Tile64x64x64
    5: 128,  # Tile128x64x64
    6: 64,  # Tile64x128x64
    7: 128,  # Tile128x128x64
    8: 256,  # Tile256x64x32
    9: 64,  # Tile64x256x32
}


def _cute_grouped_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    mma_tile: int = 3,
) -> Union[Float[Tensor, "M C_out"], int]:
    """Forward pass using fused multi-offset CuTe GEMM."""
    device = in_features.device
    iden_idx = kernel_map.identity_map_index
    min_dtype = _min_dtype(in_features.dtype, weight.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32

    _in_features = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight = weight.contiguous().detach().to(dtype=min_dtype)

    out_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float32

    # The C++ binding downcasts float32 inputs to float16 for the CuTe kernel.
    # Weight data is passed as raw device pointers, so we must cast weights
    # to the same compute dtype here to avoid type mismatch (root cause of
    # garbage output / NaN with float32 inputs).
    compute_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float16
    _weight_compute = _weight.to(dtype=compute_dtype).contiguous()

    # Initialize output
    if iden_idx is not None:
        output = torch.matmul(_in_features, _weight[iden_idx]).to(dtype=out_dtype)
    else:
        output = torch.zeros(num_out_coords, weight.shape[-1], device=device, dtype=out_dtype)

    tile_m = _TILE_M_SIZES.get(mma_tile, 64)
    params = _prepare_grouped_params(kernel_map, _weight_compute, iden_idx, tile_m, device)

    if params is None:
        return output

    (
        weight_ptrs,
        tile_offsets,
        group_sizes,
        map_offsets,
        group_indices,
        total_m_tiles,
    ) = params

    in_map = kernel_map.in_maps.to(device).int().contiguous()
    out_map = kernel_map.out_maps.to(device).int().contiguous()

    status = _C.gemm.cute_gemm_grouped_AD_gather_scatter(
        _in_features,
        output,
        in_map,
        out_map,
        weight_ptrs,
        tile_offsets,
        group_sizes,
        map_offsets,
        total_m_tiles,
        mma_tile,
        1.0,
    )

    if status != 0:
        return status
    return output


def _cute_grouped_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    requires_grad: Tuple[bool, bool] = (True, True),
    device: torch.device = None,
    mma_tile: int = 3,
) -> Union[
    Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]],
    Tuple[int, int],
]:
    """Backward pass using fused multi-offset CuTe GEMM for input grad,
    and fused grouped TrAB for weight grad."""
    if device is None:
        device = in_features.device

    min_dtype = _min_dtype(in_features.dtype, weight.dtype, grad_output.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _grad_output = grad_output.contiguous().detach().to(dtype=min_dtype)
    _in_features = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight = weight.contiguous().detach().to(dtype=min_dtype)

    out_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float32

    # Match the C++ binding's float32→float16 downcast for weight pointers
    compute_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float16

    iden_idx = kernel_map.identity_map_index
    grad_weight = torch.zeros_like(weight, dtype=out_dtype, device=device)

    # --- Input gradient: fused grouped GEMM ---
    if requires_grad[0]:
        if iden_idx is not None:
            grad_in_features = torch.matmul(_grad_output, _weight[iden_idx].T).to(dtype=out_dtype)
        else:
            grad_in_features = torch.zeros(
                _in_features.shape[0],
                _in_features.shape[1],
                device=device,
                dtype=out_dtype,
            )

        # For input grad: A=grad_output gathered by out_map, B=weight.T, scatter to in_map
        # We need transposed weights in compute dtype
        weight_t = (
            _weight.to(dtype=compute_dtype).transpose(-1, -2).contiguous()
        )  # [K, C_out, C_in]

        tile_m = _TILE_M_SIZES.get(mma_tile, 64)
        params = _prepare_grouped_params(kernel_map, weight_t, iden_idx, tile_m, device)

        if params is not None:
            (
                weight_ptrs,
                tile_offsets,
                group_sizes,
                map_offsets,
                group_indices,
                total_m_tiles,
            ) = params
            out_map_dev = kernel_map.out_maps.to(device).int().contiguous()
            in_map_dev = kernel_map.in_maps.to(device).int().contiguous()

            status = _C.gemm.cute_gemm_grouped_AD_gather_scatter(
                _grad_output,
                grad_in_features,
                out_map_dev,  # gather from grad_output
                in_map_dev,  # scatter to grad_input
                weight_ptrs,
                tile_offsets,
                group_sizes,
                map_offsets,
                total_m_tiles,
                mma_tile,
                1.0,
            )
            if status != 0:
                return status, -1
    else:
        grad_in_features = torch.zeros(
            _in_features.shape[0],
            _in_features.shape[1],
            device=device,
            dtype=out_dtype,
        )

    # --- Weight gradient: fused grouped TrAB ---
    if requires_grad[1]:
        if iden_idx is not None:
            grad_weight[iden_idx] = torch.matmul(_in_features.T, _grad_output).to(dtype=out_dtype)

        trAB_params = _prepare_grouped_trAB_params(kernel_map, grad_weight, iden_idx, device)

        if trAB_params is not None:
            output_ptrs, gather_sizes, map_offsets_t, in_map_dev, out_map_dev = trAB_params
            C_in = _in_features.shape[1]
            C_out = _grad_output.shape[1]

            status = _C.gemm.cute_gemm_grouped_trAB_gather(
                _in_features,
                _grad_output,
                in_map_dev,
                out_map_dev,
                output_ptrs,
                gather_sizes,
                map_offsets_t,
                C_in,
                C_out,
                mma_tile,
                1.0,
                _DTYPE_TO_SCALAR_TYPE_INT[out_dtype],
            )
            if status != 0:
                return status, -2

    return grad_in_features, grad_weight


class CuteGroupedSpatiallySparseConvFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        mma_tile: int = 3,
    ) -> Float[Tensor, "M C_out"]:
        output = _cute_grouped_forward_logic(
            in_features, weight, kernel_map, num_out_coords, mma_tile
        )
        if isinstance(output, int):
            raise RuntimeError(
                f"CuTe grouped forward failed: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(output))}"
            )
        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.device = in_features.device
        ctx.mma_tile = mma_tile
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 5)

        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        if K == 0 or C_in == 0 or C_out == 0 or N_in == 0 or grad_output.shape[0] == 0:
            grad_in = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_w = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in, grad_w, 5)

        result = _cute_grouped_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
            device=ctx.device,
            mma_tile=ctx.mma_tile,
        )
        if isinstance(result[0], int):
            raise RuntimeError(
                f"CuTe grouped backward failed: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        grad_in_features, grad_weight = result

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 5)
