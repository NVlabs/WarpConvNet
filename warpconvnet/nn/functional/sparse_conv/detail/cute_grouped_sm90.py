# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 WGMMA grouped GEMM dispatch for sparse convolution forward/backward."""

from typing import Optional, Tuple, Union
from jaxtyping import Float

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.type_cast import _min_dtype

from .cute_grouped import (
    _prepare_grouped_params,
    _prepare_grouped_trAB_params,
    _DTYPE_TO_SCALAR_TYPE_INT,
)


# SM90 tile tag index to tM dimension mapping
_SM90_TILE_M_SIZES = {
    100: 64,  # SM90_Tile64x128x64
    101: 128,  # SM90_Tile128x128x64
    102: 128,  # SM90_Tile128x256x64
    103: 256,  # SM90_Tile256x128x64
    104: 64,  # SM90_Tile64x64x64
}


def _cute_grouped_sm90_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    mma_tile: int = 100,
    use_cp_async: bool = True,
) -> Union[Float[Tensor, "M C_out"], int]:
    """Forward pass using SM90 WGMMA fused multi-offset CuTe GEMM."""
    device = in_features.device
    iden_idx = kernel_map.identity_map_index
    min_dtype = _min_dtype(in_features.dtype, weight.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32

    _in_features = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight = weight.contiguous().detach().to(dtype=min_dtype)

    out_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float32

    # Weight data passed as raw device pointers; cast to compute dtype
    compute_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float16
    _weight_compute = _weight.to(dtype=compute_dtype).contiguous()

    # Initialize output
    if iden_idx is not None:
        output = torch.matmul(_in_features, _weight[iden_idx]).to(dtype=out_dtype)
    else:
        output = torch.zeros(num_out_coords, weight.shape[-1], device=device, dtype=out_dtype)

    tile_m = _SM90_TILE_M_SIZES.get(mma_tile, 64)
    params = _prepare_grouped_params(kernel_map, _weight_compute, iden_idx, tile_m, device)

    if params is None:
        return output

    weight_ptrs, tile_offsets, group_sizes, map_offsets, group_indices, total_m_tiles = params

    in_map = kernel_map.in_maps.to(device).int().contiguous()
    out_map = kernel_map.out_maps.to(device).int().contiguous()

    status = _C.gemm.cute_gemm_sm90_grouped_AD_gather_scatter(
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
        True,  # use_atomic=True: HW-coalesced atomicAdd is faster than direct store on SM90
        use_cp_async,
    )

    if status != 0:
        return status
    return output


def _cute_grouped_sm90_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    requires_grad: Tuple[bool, bool] = (True, True),
    device: torch.device = None,
    mma_tile: int = 100,
    use_cp_async: bool = True,
) -> Union[
    Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]],
    Tuple[int, int],
]:
    """Backward pass using SM90 WGMMA fused grouped GEMM for input grad,
    and SM90 TrAB for weight grad."""
    if device is None:
        device = in_features.device

    min_dtype = _min_dtype(in_features.dtype, weight.dtype, grad_output.dtype)
    if min_dtype == torch.float64:
        min_dtype = torch.float32
    _grad_output = grad_output.contiguous().detach().to(dtype=min_dtype)
    _in_features = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight = weight.contiguous().detach().to(dtype=min_dtype)

    out_dtype = min_dtype if min_dtype in (torch.float16, torch.bfloat16) else torch.float32

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

        weight_t = _weight.to(dtype=compute_dtype).transpose(-1, -2).contiguous()

        tile_m = _SM90_TILE_M_SIZES.get(mma_tile, 64)
        params = _prepare_grouped_params(kernel_map, weight_t, iden_idx, tile_m, device)

        if params is not None:
            weight_ptrs, tile_offsets, group_sizes, map_offsets, group_indices, total_m_tiles = (
                params
            )
            out_map_dev = kernel_map.out_maps.to(device).int().contiguous()
            in_map_dev = kernel_map.in_maps.to(device).int().contiguous()

            status = _C.gemm.cute_gemm_sm90_grouped_AD_gather_scatter(
                _grad_output,
                grad_in_features,
                out_map_dev,
                in_map_dev,
                weight_ptrs,
                tile_offsets,
                group_sizes,
                map_offsets,
                total_m_tiles,
                mma_tile,
                1.0,
                True,  # use_atomic=True: backward input grad has overlapping output rows
                use_cp_async,
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
                3,  # mma_tile=3: SM80 Tile64x64x32 for TrAB (best on H200)
                1.0,
                _DTYPE_TO_SCALAR_TYPE_INT[out_dtype],
            )
            if status != 0:
                return status, -2

    return grad_in_features, grad_weight
