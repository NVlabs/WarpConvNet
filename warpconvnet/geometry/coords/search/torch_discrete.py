# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import os
from typing import Literal, Optional, Sequence, Tuple, Dict

import numpy as np
import torch

from jaxtyping import Int
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.packed_hashmap import PackedHashTable
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.ntuple import ntuple

logger = logging.getLogger(__name__)


@torch.no_grad()
def kernel_offsets_from_size(
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    center_offset: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None,  # Added device argument
) -> Int[Tensor, "K D+1"]:
    """
    Generate the kernel offsets for the spatially sparse convolution.
    Supports arbitrary number of spatial dimensions.
    Returns a PyTorch Tensor.
    """
    assert len(kernel_size) == len(kernel_dilation)
    num_spatial_dims = len(kernel_size)

    # Create meshgrid for arbitrary dimensions
    ranges = [torch.arange(size, dtype=torch.int32, device="cpu") for size in kernel_size]
    grids = torch.meshgrid(*ranges, indexing="ij")
    flattened_grids = [grid.flatten() for grid in grids]

    if center_offset is None:
        # center odd-sized kernels and 0 for even-sized kernels
        center_offset = [(s - 1) // 2 if s % 2 == 1 else 0 for s in kernel_size]
    assert len(center_offset) == num_spatial_dims

    # Create offsets for each dimension
    offsets = [
        (grid - center_offset[i]) * kernel_dilation[i] for i, grid in enumerate(flattened_grids)
    ]

    # Add batch dimension (zeros)
    offsets = [torch.zeros_like(offsets[0])] + offsets

    return torch.stack(offsets, dim=1).contiguous().to(device)


@torch.no_grad()
def _kernel_map_search_to_result(
    found_in_coord_index: Int[Tensor, "K M"],
    identity_map_index: Optional[int] = None,
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K M"] | IntSearchResult:
    """Processes the raw found_in_coord_index tensor into compacted kernel maps.

    The found_in_coord_index is a tensor of shape (K, M) where K is the number of
    kernel offsets and M is the number of query coordinates. The value is the index
    of the input coordinate matched at that (offset, query) position, or -1 if no match.
    """
    if return_type == "indices":
        return found_in_coord_index

    assert return_type == "offsets"
    target_device = found_in_coord_index.device
    K, M = found_in_coord_index.shape

    counts = torch.zeros(K, dtype=torch.int32, device=target_device)
    _C.cuhash.postprocess_count(found_in_coord_index, counts, K, M)

    offsets = torch.zeros(K + 1, dtype=torch.int32, device=target_device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    num_total_maps = offsets[-1].item()

    in_maps = torch.empty(num_total_maps, dtype=torch.int32, device=target_device)
    out_maps = torch.empty(num_total_maps, dtype=torch.int32, device=target_device)

    if num_total_maps > 0:
        scatter_counters = torch.zeros(K, dtype=torch.int32, device=target_device)
        _C.cuhash.postprocess_scatter(
            found_in_coord_index,
            offsets,
            scatter_counters,
            in_maps,
            out_maps,
            K,
            M,
        )

    return IntSearchResult(
        in_maps,
        out_maps,
        offsets,
        identity_map_index=identity_map_index,
    )


@torch.no_grad()
def _kernel_map_from_offsets(
    hashtable,
    batched_query_coords: Int[Tensor, "N D_1"],
    kernel_offsets: Int[Tensor, "K D_1"],
    identity_map_index: Optional[int] = None,
    return_type: Literal["indices", "offsets"] = "offsets",
    threads_per_block_x: int = 64,
    threads_per_block_y: int = 8,
) -> Int[Tensor, "K N"] | IntSearchResult:
    """
    Compute the kernel map (input index, output index) for each kernel offset.
    Assumes D_1 includes batch dimension (e.g., 4 for 3D spatial + batch).
    """
    target_device = hashtable.device
    assert (
        target_device == batched_query_coords.device
    ), f"{target_device} != {batched_query_coords.device}"
    assert target_device == kernel_offsets.device, f"{target_device} != {kernel_offsets.device}"
    assert batched_query_coords.shape[1] == kernel_offsets.shape[1]
    assert batched_query_coords.ndim == 2
    assert kernel_offsets.ndim == 2
    assert batched_query_coords.dtype == torch.int32
    assert kernel_offsets.dtype == torch.int32

    assert isinstance(
        hashtable, PackedHashTable
    ), "Only PackedHashTable is supported. TorchHashTable legacy path has been removed."

    if identity_map_index is not None:
        assert (
            identity_map_index < kernel_offsets.shape[0]
        ), "Identity map index must be less than the number of kernel offsets"
        iden_offset = kernel_offsets[identity_map_index]
        assert torch.all(iden_offset == 0), "Identity map offset must be all zeros"

    num_query_coords = batched_query_coords.shape[0]
    num_kernel_offsets = kernel_offsets.shape[0]

    # Allocate output tensor
    found_in_coord_index = torch.empty(
        (num_kernel_offsets, num_query_coords),
        dtype=torch.int32,
        device=target_device,
    )

    _C.cuhash.packed_kernel_map_offset(
        hashtable.keys_tensor,
        hashtable.values_tensor,
        batched_query_coords.contiguous(),
        kernel_offsets.contiguous(),
        found_in_coord_index,
        num_query_coords,
        num_kernel_offsets,
        hashtable.capacity,
        threads_per_block_x,
        threads_per_block_y,
    )

    return _kernel_map_search_to_result(
        found_in_coord_index,
        identity_map_index=identity_map_index,
        return_type=return_type,
    )


@torch.no_grad()
def _kernel_map_from_size(
    hashtable,
    batched_query_coords: Int[Tensor, "N D_1"],
    kernel_sizes: Tuple[int, ...],
    identity_map_index: Optional[int] = None,
    return_type: Literal["indices", "offsets"] = "offsets",
    threads_per_block_x: int = 64,
    threads_per_block_y: int = 8,
    skip_symmetric_kernel_map: bool = False,
) -> Int[Tensor, "K N"] | IntSearchResult:
    """
    Compute the kernel map using kernel_size. Uses _kernel_map_from_offsets internally,
    or a specialized kernel if coordinates are 4D.
    Assumes D_1 includes batch dimension.

    Args:
        skip_symmetric_kernel_map: If True, skip symmetric parts of the kernel map
            for odd-sized kernels (e.g., for 3x3x3 kernels, only use half of the kernel positions). You can only use this if the input coordinates and output coordinates are the same.
    """
    target_device = hashtable.device
    assert str(target_device) == str(batched_query_coords.device)
    assert batched_query_coords.dtype == torch.int32

    assert isinstance(
        hashtable, PackedHashTable
    ), "Only PackedHashTable is supported. TorchHashTable legacy path has been removed."

    num_dims = batched_query_coords.shape[1]
    assert (
        len(kernel_sizes) == num_dims - 1
    ), f"kernel_size ({len(kernel_sizes)}) must match spatial dims ({num_dims - 1})"

    if skip_symmetric_kernel_map:
        assert all(
            k % 2 == 1 for k in kernel_sizes
        ), f"Kernel sizes must be odd for symmetric skipping. Got {kernel_sizes}"

    num_offsets = np.prod(kernel_sizes).item()
    assert num_dims == 4, f"Expected 4D batch-indexed coords, got {num_dims}D"

    num_query_coords = batched_query_coords.shape[0]

    if skip_symmetric_kernel_map:
        num_offsets = num_offsets // 2
        if identity_map_index is not None:
            assert identity_map_index == num_offsets

    kernel_size_tensor = torch.tensor(kernel_sizes, dtype=torch.int32, device=target_device)
    query_coords = batched_query_coords.contiguous()
    capacity = hashtable.capacity
    keys = hashtable.keys_tensor
    values = hashtable.values_tensor

    if return_type == "indices":
        found_in_coord_index = torch.empty(
            (num_offsets, num_query_coords),
            dtype=torch.int32,
            device=target_device,
        )
        _C.cuhash.packed_kernel_map_size(
            keys,
            values,
            query_coords,
            kernel_size_tensor,
            found_in_coord_index,
            num_query_coords,
            num_offsets,
            capacity,
            threads_per_block_x,
            threads_per_block_y,
        )
        return found_in_coord_index

    # Search-once + postprocess pipeline (single hash table pass)
    found_in_coord_index = torch.empty(
        (num_offsets, num_query_coords),
        dtype=torch.int32,
        device=target_device,
    )
    _C.cuhash.packed_kernel_map_size(
        keys,
        values,
        query_coords,
        kernel_size_tensor,
        found_in_coord_index,
        num_query_coords,
        num_offsets,
        capacity,
        threads_per_block_x,
        threads_per_block_y,
    )

    counts = torch.zeros(num_offsets, dtype=torch.int32, device=target_device)
    _C.cuhash.postprocess_count(found_in_coord_index, counts, num_offsets, num_query_coords)

    offsets = torch.zeros(num_offsets + 1, dtype=torch.int32, device=target_device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    num_total_maps = offsets[-1].item()

    in_maps = torch.empty(num_total_maps, dtype=torch.int32, device=target_device)
    out_maps = torch.empty(num_total_maps, dtype=torch.int32, device=target_device)

    if num_total_maps > 0:
        scatter_counters = torch.zeros(num_offsets, dtype=torch.int32, device=target_device)
        _C.cuhash.postprocess_scatter(
            found_in_coord_index,
            offsets,
            scatter_counters,
            in_maps,
            out_maps,
            num_offsets,
            num_query_coords,
        )

    result = IntSearchResult(in_maps, out_maps, offsets, identity_map_index=identity_map_index)
    result._pair_table = found_in_coord_index
    return result


@torch.compiler.disable
@torch.no_grad()
def generate_kernel_map(
    batch_indexed_in_coords: Int[Tensor, "N D_1"],
    batch_indexed_out_coords: Int[Tensor, "M D_1"],
    in_to_out_stride_ratio: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Optional[Tuple[int, ...]] = None,
    kernel_center_offset: Optional[Tuple[int, ...]] = None,
    method: Literal["offset", "size"] = "size",
    skip_symmetric_kernel_map: bool = False,
    **kwargs,
) -> IntSearchResult:
    """
    Generate the kernel map for the spatially sparse convolution using PackedHashTable.

    in_to_out_stride_ratio: the ratio of the input stride to the output stride. This will be multiplied to output coordinates to find matching input coordinates.
    method: 'offset' pre-calculates all kernel offsets and uses a custom kernel to find matches.
            'offset' pre-calculates all kernel offsets and uses a custom kernel to find matches (generally faster).
            'size' uses a specialized kernel for 4D coordinates if applicable, otherwise falls back to 'offset'.
    skip_symmetric_kernel_map: If True, skip symmetric parts of the kernel map for odd-sized kernels.
    """
    target_device = batch_indexed_in_coords.device
    assert target_device == batch_indexed_out_coords.device
    assert batch_indexed_in_coords.dtype == torch.int32
    assert batch_indexed_out_coords.dtype == torch.int32
    if skip_symmetric_kernel_map:
        assert len(batch_indexed_in_coords) == len(
            batch_indexed_out_coords
        ), "You can only skip symmetric kernel map if the input and output coordinates are the same."
        assert all(
            k % 2 == 1 for k in kernel_size
        ), "Kernel size must be odd for symmetric skipping."

    # Pad 3D coords (2D sparse conv) to 4D for PackedHashTable compatibility.
    # [batch, x, y] → [batch, x, y, 0] with kernel_size/stride/dilation extended by 1.
    num_dims = batch_indexed_in_coords.shape[1]
    if num_dims == 3:
        batch_indexed_in_coords = torch.nn.functional.pad(batch_indexed_in_coords, (0, 1), value=0)
        batch_indexed_out_coords = torch.nn.functional.pad(
            batch_indexed_out_coords, (0, 1), value=0
        )
        kernel_size = tuple(kernel_size) + (1,)
        in_to_out_stride_ratio = tuple(in_to_out_stride_ratio) + (1,)
        if kernel_dilation is not None:
            kernel_dilation = tuple(kernel_dilation) + (1,)
        if kernel_center_offset is not None:
            kernel_center_offset = tuple(kernel_center_offset) + (0,)
        num_dims = 4

    # Create hash table for the input coordinates (always 4D after padding)
    assert num_dims == 4, f"Expected 4D coords after padding, got {num_dims}D"
    hashtable = PackedHashTable.from_coords(batch_indexed_in_coords, device=target_device)

    num_spatial_dims = batch_indexed_out_coords.shape[1] - 1
    assert len(in_to_out_stride_ratio) == num_spatial_dims

    # Apply stride ratio to output coordinates
    if not all(s == 1 for s in in_to_out_stride_ratio):
        stride_tensor = torch.tensor(
            [1] + list(ntuple(in_to_out_stride_ratio, ndim=num_spatial_dims)),
            dtype=torch.int32,
            device=target_device,
        )
        # Ensure broadcasting works: coords [M, D+1], stride [D+1]
        strided_out_coords = batch_indexed_out_coords * stride_tensor
    else:
        strided_out_coords = batch_indexed_out_coords

    identity_map_index = None
    # Check if kernel is odd and potentially symmetric
    is_odd_kernel = all(k % 2 == 1 for k in kernel_size)
    same_in_out_coords = batch_indexed_in_coords.shape[0] == batch_indexed_out_coords.shape[0]
    if is_odd_kernel and same_in_out_coords:
        total_kernels = int(np.prod(kernel_size))
        center_idx = total_kernels // 2
        identity_map_index = center_idx

    # Force the symmetric kernel skipping to be False if the kernel is not odd
    if skip_symmetric_kernel_map and not is_odd_kernel:
        skip_symmetric_kernel_map = False

    if method == "offset":
        # This method generates offsets and launches the custom kernel_map_offset kernel
        if kernel_dilation is None:
            kernel_dilation = (1,) * num_spatial_dims

        kernel_offsets_tensor = kernel_offsets_from_size(
            kernel_size,
            kernel_dilation,
            center_offset=kernel_center_offset,
            device=target_device,
        )
        if identity_map_index is not None:
            kernel_offsets_tensor = kernel_offsets_tensor[:center_idx]

        return _kernel_map_from_offsets(
            hashtable,
            strided_out_coords,  # Use strided coordinates
            kernel_offsets_tensor,
            return_type="offsets",
            identity_map_index=identity_map_index,
        )
    elif method == "size":
        assert kernel_dilation is None or all(
            s == 1 for s in kernel_dilation
        ), "Kernel dilation is not supported with method='size'. Use method='offset' instead."
        assert (
            kernel_center_offset is None
        ), "Custom kernel_center_offset is not supported with method='size'. Use method='offset' instead."

        # For large odd kernels (K >= 125, i.e. 5^3+), use hierarchical
        # search: coarse probe prunes fine probes via bitmask.
        _K = int(np.prod(kernel_size))
        is_odd_kernel_all = all(k % 2 == 1 for k in kernel_size)
        if _K >= 125 and is_odd_kernel_all and not skip_symmetric_kernel_map:
            from warpconvnet.geometry.coords.search.hierarchical_search import (
                kernel_map_from_size_hierarchical,
            )

            result = kernel_map_from_size_hierarchical(
                hashtable,
                strided_out_coords,
                kernel_size,
                identity_map_index=identity_map_index,
            )
            return result

        result = _kernel_map_from_size(
            hashtable,
            strided_out_coords,
            kernel_size,
            return_type="offsets",
            skip_symmetric_kernel_map=skip_symmetric_kernel_map,
            identity_map_index=identity_map_index,
        )
        return result
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'offset', or 'size'.")


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = (x * 31 + hash(arr[i])) & 0xFFFFFFFF  # Keep it within 32-bit range
    return x


def string_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) & 0xFFFFFFFF
