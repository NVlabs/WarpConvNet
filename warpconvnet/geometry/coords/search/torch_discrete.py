# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import math
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Dict

import numpy as np
import torch

import cupy as cp
from jaxtyping import Int
from torch import Tensor

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.cuda_utils import load_kernel
from warpconvnet.utils.ntuple import ntuple

logger = logging.getLogger(__name__)

# Path to the new CUDA kernel file
KERNEL_FILE = Path(__file__).parent / "cuda" / "discrete_kernels.cu"


def _get_kernel_map_offset_kernel(hash_method: HashMethod) -> cp.RawKernel:
    suffix = hash_method.kernel_suffix()
    return load_kernel(f"kernel_map_offset_{suffix}", str(KERNEL_FILE))


def _get_map_results_kernel() -> cp.RawKernel:
    return load_kernel("map_found_indices_to_maps_cuda", str(KERNEL_FILE))


def _get_kernel_map_size_4d_kernel(hash_method: HashMethod) -> cp.RawKernel:
    suffix = hash_method.kernel_suffix()
    return load_kernel(f"kernel_map_size_4d_{suffix}", str(KERNEL_FILE))


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
    threads_per_block: int = 256,
) -> Int[Tensor, "K M"] | IntSearchResult:
    """Processes the raw found_in_coord_index tensor into the desired format.

    The found_in_coord_index is a tensor of shape (K, M) where K is the number of kernel offsets and M is the number of query coordinates. The value is the index of the query coordinate if the kernel offset is found in the query coordinate, otherwise -1. We remove the -1 values and return valid indices for each kernel offset.

    The return_type offset means the offsets will contain [0, K_0, K_0 + K_1, ...] where K_i is the number of valid maps for the i-th kernel offset.
    """
    # assert found_in_coord_index_wp.shape[0] == kernel_offsets.shape[0]
    # assert found_in_coord_index_wp.shape[1] == batched_query_coords.shape[0]
    target_device = found_in_coord_index.device
    K, M = found_in_coord_index.shape

    if return_type == "indices":
        return found_in_coord_index

    assert return_type == "offsets"

    found_in_coord_index_bool = found_in_coord_index >= 0

    # get the index of the non zero elements
    mapped_indices = (
        torch.cumsum(found_in_coord_index_bool.to(torch.int32), dim=1, dtype=torch.int32) - 1
    )
    # Need to handle rows with zero valid maps correctly (cumsum results in -1)
    # Clamp minimum value to 0 after subtracting 1
    mapped_indices = torch.clamp(mapped_indices, min=-1)  # Keep -1 for rows with no hits

    # Count valid maps per kernel offset row
    # If mapped_indices is -1 everywhere in a row, max will be -1, add 1 -> 0 count.
    num_valid_maps = mapped_indices.max(dim=1).values + 1

    # Calculate offsets
    offsets = torch.cumsum(num_valid_maps, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=target_device), offsets], dim=0)
    num_total_maps = offsets[-1].item()

    # Allocate output tensors
    in_maps = cp.empty(num_total_maps, dtype=cp.int32)
    out_maps = cp.empty(num_total_maps, dtype=cp.int32)

    if num_total_maps > 0:
        # Launch CUDA kernel to gather results
        map_results_kernel = _get_map_results_kernel()
        grid_size = math.ceil(found_in_coord_index.numel() / threads_per_block)

        # Ensure tensors are contiguous for kernel launch if necessary (CuPy might handle non-contiguous? Check docs)
        # Let's assume contiguous for safety for now.
        found_in_coord_index_cont = cp.from_dlpack(found_in_coord_index.contiguous())
        mapped_indices_cont = cp.from_dlpack(mapped_indices.contiguous())
        offsets_cont = cp.from_dlpack(offsets.contiguous())

        map_results_kernel(
            (grid_size,),
            (threads_per_block,),
            (
                found_in_coord_index_cont,
                mapped_indices_cont,
                offsets_cont,
                in_maps,
                out_maps,
                K,  # num_kernel_offsets
                M,  # num_query_coords
            ),
        )
        # torch.cuda.synchronize(target_device) # Optional sync needed?

    return IntSearchResult(
        torch.from_dlpack(in_maps),
        torch.from_dlpack(out_maps),
        offsets,
        identity_map_index=identity_map_index,
    )


@torch.no_grad()
def _kernel_map_from_offsets(
    hashtable: TorchHashTable,  # Use TorchHashTable
    batched_query_coords: Int[Tensor, "N D_1"],
    kernel_offsets: Int[Tensor, "K D_1"],
    identity_map_index: Optional[int] = None,
    return_type: Literal["indices", "offsets"] = "offsets",
    threads_per_block_x: int = 128,
    threads_per_block_y: int = 8,
) -> Int[Tensor, "K N"] | IntSearchResult:
    """
    Compute the kernel map (input index, output index) for each kernel offset using TorchHashTable.
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

    if hashtable._table_kvs is None or hashtable._vector_keys is None:
        raise RuntimeError(
            "Input TorchHashTable must be populated before calling kernel map functions."
        )

    if identity_map_index is not None:
        # Assert that the number of elements in the hashtable and the query coordinates are the same
        assert (
            identity_map_index < kernel_offsets.shape[0]
        ), "Identity map index must be less than the number of kernel offsets"
        iden_offset = kernel_offsets[identity_map_index]
        # assert that iden_offset is all zeros
        assert torch.all(iden_offset == 0), "Identity map offset must be all zeros"

    num_query_coords = batched_query_coords.shape[0]
    key_dim = batched_query_coords.shape[1]
    num_kernel_offsets = kernel_offsets.shape[0]

    # Get the appropriate kernel based on hash method
    kernel = _get_kernel_map_offset_kernel(hashtable.hash_method)

    # Calculate 2D grid size
    grid_size_x = math.ceil(num_query_coords / threads_per_block_x)
    grid_size_y = math.ceil(num_kernel_offsets / threads_per_block_y)

    # Ensure contiguous tensors for kernel launch
    table_kvs_cont = cp.from_dlpack(hashtable._table_kvs.contiguous())
    vector_keys_cont = cp.from_dlpack(hashtable._vector_keys.contiguous())
    query_coords_cont = cp.from_dlpack(batched_query_coords.contiguous())
    kernel_offsets_cont = cp.from_dlpack(kernel_offsets.contiguous())

    # Allocate output tensor
    found_in_coord_index = cp.empty(
        (num_kernel_offsets, num_query_coords),
        dtype=table_kvs_cont.dtype,
    )

    # Launch the kernel with 2D grid (x: query coords, y: kernel offsets)
    kernel(
        (grid_size_x, grid_size_y),
        (threads_per_block_x, threads_per_block_y),
        (
            table_kvs_cont,
            vector_keys_cont,
            query_coords_cont,
            kernel_offsets_cont,
            found_in_coord_index,  # Output
            num_query_coords,  # N
            key_dim,  # D+1
            num_kernel_offsets,  # K (effective count after symmetric reduction)
            hashtable.capacity,
        ),
    )
    # torch.cuda.synchronize(target_device) # Optional

    return _kernel_map_search_to_result(
        torch.from_dlpack(found_in_coord_index), return_type, identity_map_index=identity_map_index
    )


@torch.no_grad()
def _kernel_map_from_size(
    hashtable: TorchHashTable,  # Use TorchHashTable
    batched_query_coords: Int[Tensor, "N D_1"],
    kernel_sizes: Tuple[int, ...],
    identity_map_index: Optional[int] = None,
    return_type: Literal["indices", "offsets"] = "offsets",
    threads_per_block_x: int = 128,
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

    if hashtable._table_kvs is None or hashtable._vector_keys is None:
        raise RuntimeError(
            "Input TorchHashTable must be populated before calling kernel map functions."
        )

    num_dims = batched_query_coords.shape[1]
    assert (
        len(kernel_sizes) == num_dims - 1
    ), f"kernel_size ({len(kernel_sizes)}) must match spatial dims ({num_dims - 1})"

    # Check if we should skip symmetric kernel parts
    if skip_symmetric_kernel_map:
        assert all(
            k % 2 == 1 for k in kernel_sizes
        ), f"Kernel sizes must be odd for symmetric skipping. Got {kernel_sizes}"
        assert (
            identity_map_index is not None
        ), "Identity map index must be provided for symmetric skipping."

    num_offsets = np.prod(kernel_sizes)

    # --- Specialized 4D Case ---
    if num_dims == 4:
        num_query_coords = batched_query_coords.shape[0]
        num_offsets = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2]

        if skip_symmetric_kernel_map:
            # For symmetric kernels, only use the first half (excluding center)
            num_offsets = num_offsets // 2

        # Get the appropriate kernel based on symmetric skipping
        kernel = _get_kernel_map_size_4d_kernel(hashtable.hash_method)

        # Calculate 2D grid size
        grid_size_x = math.ceil(num_query_coords / threads_per_block_x)
        grid_size_y = math.ceil(num_offsets / threads_per_block_y)

        # Prepare kernel arguments
        table_kvs_cont = cp.from_dlpack(hashtable._table_kvs.contiguous())
        vector_keys_cont = cp.from_dlpack(hashtable._vector_keys.contiguous())
        query_coords_cont = cp.from_dlpack(batched_query_coords.contiguous())
        kernel_size_arg = cp.array(kernel_sizes, dtype=cp.int32)

        # Allocate output tensor
        found_in_coord_index = cp.empty(
            (num_offsets, num_query_coords),  # Shape K x N
            dtype=table_kvs_cont.dtype,
        )

        # Launch the kernel with 2D grid (x: query coords, y: kernel positions)
        kernel(
            (grid_size_x, grid_size_y),
            (threads_per_block_x, threads_per_block_y),
            (
                table_kvs_cont,
                vector_keys_cont,
                query_coords_cont,
                kernel_size_arg,  # Pass cp.int3
                found_in_coord_index,
                num_query_coords,
                hashtable.capacity,
                num_offsets,
            ),
        )
        torch.cuda.synchronize(target_device)

        return _kernel_map_search_to_result(
            torch.from_dlpack(found_in_coord_index),
            return_type=return_type,
            identity_map_index=identity_map_index,
        )

    # --- Generic Case (Fallback to offset method) ---
    else:
        logger.warning(
            f"Using generic offset-based kernel map for {num_dims}D coords when method='size'. "
            f"Consider implementing a specialized kernel or using method='offset' directly for potential performance gains."
        )  # Log a warning for non-4D case using size method
        # Generate kernel offsets on the correct device
        kernel_offsets_tensor = kernel_offsets_from_size(
            kernel_sizes, (1,) * len(kernel_sizes), device=target_device
        )

        # If skipping symmetric parts, reduce the kernel offsets
        if skip_symmetric_kernel_map:
            num_offsets = num_offsets // 2
            kernel_offsets_tensor = kernel_offsets_tensor[:num_offsets]

        # Call the offset-based function
        return _kernel_map_from_offsets(
            hashtable,
            batched_query_coords,
            kernel_offsets_tensor,
            return_type=return_type,
            identity_map_index=identity_map_index,
        )


@torch.no_grad()
def generate_kernel_map(
    batch_indexed_in_coords: Int[Tensor, "N D_1"],
    batch_indexed_out_coords: Int[Tensor, "M D_1"],
    in_to_out_stride_ratio: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Optional[Tuple[int, ...]] = None,
    kernel_center_offset: Optional[Tuple[int, ...]] = None,
    method: Literal["offset", "size"] = "size",  # Size is the default fastest method
    hash_method: HashMethod = HashMethod.CITY,  # Allow selecting hash method
    skip_symmetric_kernel_map: bool = False,
) -> IntSearchResult:
    """
    Generate the kernel map for the spatially sparse convolution using TorchHashTable.

    in_to_out_stride_ratio: the ratio of the input stride to the output stride. This will be multiplied to output coordinates to find matching input coordinates.
    method: 'query' directly queries the hash table for each offset point (can be slower for large kernels but flexible).
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

    # Create a TorchHashTable for the input coordinates
    hashtable = TorchHashTable.from_keys(
        batch_indexed_in_coords, hash_method=hash_method, device=target_device
    )

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
    # Apply symmetric kernel skipping if requested
    if skip_symmetric_kernel_map:
        # Check if kernel is odd and potentially symmetric
        is_odd_kernel = all(k % 2 == 1 for k in kernel_size)
        if is_odd_kernel:
            total_kernels = int(np.prod(kernel_size))
            center_idx = total_kernels // 2
            identity_map_index = center_idx
        else:
            # Force the symmetric kernel skipping to be False if the kernel is not odd
            skip_symmetric_kernel_map = False

    if method == "offset":
        # This method generates offsets and launches the custom kernel_map_offset kernel
        if kernel_dilation is None:
            kernel_dilation = (1,) * num_spatial_dims

        kernel_offsets_tensor = kernel_offsets_from_size(
            kernel_size, kernel_dilation, center_offset=kernel_center_offset, device=target_device
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
        # This method uses _kernel_map_from_size, which has the 4D specialization
        assert kernel_dilation is None or all(
            s == 1 for s in kernel_dilation
        ), "Kernel dilation is not supported with method='size'. Use method='offset' instead."
        assert (
            kernel_center_offset is None
        ), "Custom kernel_center_offset is not supported with method='size'. Use method='offset' instead."
        return _kernel_map_from_size(
            hashtable,
            strided_out_coords,
            kernel_size,
            return_type="offsets",
            skip_symmetric_kernel_map=skip_symmetric_kernel_map,
            identity_map_index=identity_map_index,
        )
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'query', 'offset', or 'size'.")


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = (x * 31 + hash(arr[i])) & 0xFFFFFFFF  # Keep it within 32-bit range
    return x


def string_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) & 0xFFFFFFFF
