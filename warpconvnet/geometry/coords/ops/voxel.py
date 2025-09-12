# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import warp as wp
from jaxtyping import Bool, Float, Int
from torch import Tensor

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable
from warpconvnet.geometry.coords.search.knn import batched_knn_search
from warpconvnet.geometry.coords.ops.batch_index import (
    batch_index_from_indices,
    batch_index_from_offset,
    batch_indexed_coordinates,
    offsets_from_batch_index,
)
from warpconvnet.geometry.utils.list_to_batch import list_to_cat_tensor
from warpconvnet.utils.ravel import ravel_multi_index_auto_shape
from warpconvnet.utils.unique import ToUnique, unique_hashmap, unique_torch

__all__ = [
    "voxel_downsample_csr_mapping",
    "voxel_downsample_random_indices",
    "voxel_downsample_mapping",
    "voxel_downsample_ravel",
    "voxel_downsample_hashmap",
    "voxel_downsample_np",
]


@torch.no_grad()
def voxel_downsample_hashmap(
    coords: Int[Tensor, "N D"],
):
    """
    Args:
        coords: Int[Tensor, "N D"] - coordinates

    Returns:
        unique_indices: sorted indices of unique voxels.
    """
    hash_table = TorchHashTable.from_keys(coords)
    unique_indices = hash_table.unique_index
    return unique_indices


# Voxel downsample
@torch.no_grad()
def voxel_downsample_csr_mapping(
    batched_points: Float[Tensor, "N 3"],  # noqa: F722,F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F722,F821
    voxel_size: float,
    unique_method: Literal["torch", "ravel", "morton"] = "torch",
) -> Tuple[
    Int[Tensor, "M 3"],  # noqa: F821
    Int[Tensor, "B+1"],  # noqa: F821
    Int[Tensor, "N"],  # noqa: F821
    Int[Tensor, "M+1"],  # noqa: F821
    ToUnique,
]:
    """
    Voxel downsample the coordinates

    - floor the points to the voxel coordinates
    - concat batch index to the voxel coordinates to create batched coordinates
    - hash the batched coordinates
    - get the unique hash values
    - get the unique voxel centers

    Args:
        batched_points: Float[Tensor, "N 3"] - batched points
        offsets: Int[Tensor, "B + 1"] - offsets for each batch
        voxel_size: float - voxel size

    Returns:
        unique_coords: Int[Tensor, "M 3"] - unique coordinates
        unique_offsets: Int[Tensor, "B + 1"] - unique offsets
        to_csr_indices: Int[Tensor, "N"] - indices to csr
        to_csr_offsets: Int[Tensor, "M+1"] - offsets to csr
        to_unique: ToUnique - ToUnique object
    """
    # Floor the points to the voxel coordinates
    N = len(batched_points)
    B = len(offsets) - 1
    device = str(batched_points.device)
    assert offsets[-1] == N, f"Offsets {offsets} does not match the number of points {N}"

    voxel_coords = torch.floor(batched_points / voxel_size).int()
    if B > 1:
        batch_index = batch_index_from_offset(offsets).to(device)
        voxel_coords = torch.cat([batch_index.unsqueeze(1), voxel_coords], dim=1)

    to_unique = ToUnique(return_to_unique_indices=True, unique_method=unique_method)
    unique_coords, to_csr_indices, to_csr_offsets = to_unique.to_unique_csr(voxel_coords, dim=0)

    if B == 1:
        unique_offsets = torch.IntTensor([0, len(unique_coords)])
    else:
        _, batch_counts = torch.unique(
            batch_index[to_unique.to_unique_indices], return_counts=True
        )
        batch_counts = batch_counts.cpu()
        unique_offsets = torch.cat((batch_counts.new_zeros(1), batch_counts.cumsum(dim=0)))
    assert len(unique_offsets) == B + 1

    return unique_coords, unique_offsets, to_csr_indices, to_csr_offsets, to_unique


@torch.no_grad()
def voxel_downsample_random_indices(
    batched_points: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
    voxel_size: Optional[float] = None,
) -> Tuple[Int[Tensor, "M"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Args:
        batched points: Float[Tensor, "N 3"] - batched points
        offsets: Int[Tensor, "B + 1"] - offsets for each batch
        voxel_size: Optional[float] - voxel size. Will quantize the points if voxel_size is provided.

    Returns:
        unique_indices: sorted indices of unique voxels.
        batch_offsets: Batch offsets.
    """

    # Floor the points to the voxel coordinates
    N = len(batched_points)
    B = len(offsets) - 1
    device = str(batched_points.device)
    assert offsets[-1] == N, f"Offsets {offsets} does not match the number of points {N}"

    if voxel_size is not None:
        voxel_coords = torch.floor(batched_points / voxel_size).int()
    else:
        voxel_coords = batched_points.int()
    batch_index = batch_index_from_offset(offsets).to(device)
    voxel_coords = torch.cat([batch_index.unsqueeze(1), voxel_coords], dim=1)

    unique_indices, hash_table = unique_hashmap(voxel_coords)
    # unique_indices is sorted

    if B == 1:
        batch_offsets = torch.IntTensor([0, len(unique_indices)])
    else:
        _, batch_counts = torch.unique(batch_index[unique_indices], return_counts=True)
        batch_counts = batch_counts.cpu()
        batch_offsets = torch.cat((batch_counts.new_zeros(1), batch_counts.cumsum(dim=0)))

    return unique_indices, batch_offsets


def voxel_downsample_ravel(
    batch_indexed_coords: Float[Tensor, "N D+1"],  # noqa: F821
    voxel_size: float,
):
    """
    Args:
        batch_indexed_coords: Float[Tensor, "N D+1"] - batch indexed coordinates
        voxel_size: float - voxel size

    Returns:
        unique_indices: sorted indices of unique voxels.
    """
    batch_indexed_coords[:, 1:] = torch.floor(batch_indexed_coords[:, 1:] / voxel_size).int()
    raveled_coords = ravel_multi_index_auto_shape(batch_indexed_coords)
    _, _, _, _, perm = unique_torch(raveled_coords, dim=0)
    return perm


@torch.no_grad()
def voxel_downsample_random_indices_list_of_coords(
    list_of_coords: List[Float[Tensor, "N 3"]],
    voxel_size: float,
    device: str,
) -> Tuple[Int[Tensor, "M"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Args:
        list_of_coords: List[Float[Tensor, "N 3"]] - list of batched coordinates
        voxel_size: float - voxel size

    Returns:
        unique_indices: sorted indices of unique voxels.
        batch_offsets: Batch offsets.
    """
    batched_coords, offsets, _ = list_to_cat_tensor(list_of_coords)
    return voxel_downsample_random_indices(batched_coords.to(device), offsets, voxel_size)


@torch.no_grad()
def voxel_downsample_mapping(
    up_batched_points: Float[Tensor, "N 3"],  # noqa: F821
    up_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    down_batched_points: Float[Tensor, "M 3"],  # noqa: F821
    down_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    voxel_size: Optional[float] = None,
    find_nearest_for_invalid: bool = False,
) -> Tuple[Int[Tensor, "L"], Int[Tensor, "L"], Bool[Tensor, "N"]]:  # noqa: F821
    """
    Find the mapping that select points in the up_batched_points that are in the down_batched_points up to voxel_size.
    The mapping is random up to voxel_size. If there is a corresponding point in the down_batched_points, the mapping
    will find a point in random within a voxel_size that is in the up_batched_points.

    up_batched_points[up_map] ~= down_batched_points[down_map]
    """
    # Only support CUDA, must be on the same device
    device = str(up_batched_points.device)
    assert "cuda" in device, "voxel_downsample_mapping only supports CUDA device"
    assert device == str(
        down_batched_points.device
    ), "up_batched_points and down_batched_points must be on the same device"

    # Convert the batched points to voxel coordinates
    if voxel_size is not None:
        up_batched_points = torch.floor(up_batched_points / voxel_size).int()
        down_batched_points = torch.floor(down_batched_points / voxel_size).int()
    else:
        up_batched_points = up_batched_points.int()
        down_batched_points = down_batched_points.int()

    # Get the batch index
    up_bcoords = batch_indexed_coordinates(up_batched_points, up_offsets)
    down_bcoords = batch_indexed_coordinates(down_batched_points, down_offsets)

    down_table = TorchHashTable.from_keys(down_bcoords)
    # Get the map that maps up_batched_points[up_map] ~= down_batched_points.
    down_map = down_table.search(up_bcoords)
    # remove invalid mappings (i.e. i < 0)
    valid = down_map >= 0
    if find_nearest_for_invalid and not valid.all():
        # Find the nearest valid point
        invalid_idx = torch.nonzero(~valid).squeeze()
        # get batch index
        # The invalid batch index is sorted as it is initialized from torch.nonzero.
        invalid_up_points = up_batched_points[invalid_idx]
        invalid_batch_index = batch_index_from_indices(invalid_idx, up_offsets, device=device)
        invalid_offsets = offsets_from_batch_index(invalid_batch_index)
        nearest_down = batched_knn_search(
            down_batched_points.float(),
            down_offsets,
            invalid_up_points.float(),
            invalid_offsets,
            k=1,
        )
        # Create maps
        up_map = torch.arange(0, len(up_batched_points))
        down_map[invalid_idx] = nearest_down.squeeze().int()
    else:
        down_map = down_map[valid]
        # Get the index of true values
        up_map = torch.nonzero(valid).squeeze(1)
    return up_map, down_map, valid


def voxel_downsample_np(
    coords: np.ndarray,
    voxel_size: Optional[float] = None,
):
    """
    Numpy based voxel downsampling function. Use it for data processing in data loading.

    Args:
        coords: np.ndarray - coordinates
        voxel_size: float - voxel size

    Returns:
        unique_coords: np.ndarray - unique coordinates
        unique_indices: np.ndarray - indices of unique coordinates
    """
    if voxel_size is not None:
        coords = np.floor(coords / voxel_size).astype(np.int32)
    else:
        coords = coords.astype(np.int32)
    unique_coords, unique_indices = np.unique(coords, axis=0, return_index=True)
    return unique_coords, unique_indices
