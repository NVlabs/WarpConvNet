from typing import Optional, Tuple

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

import warp as wp
from warp.convnet.core.hashmap import VectorHashTable
from warp.convnet.geometry.ops.neighbor_search_discrete import kernel_offsets_from_size
from warp.convnet.utils.batch_index import offsets_from_batch_index
from warp.convnet.utils.ntuple import ntuple


@torch.no_grad()
def generate_output_coords(
    batch_indexed_coords: Int[Tensor, "N D+1"],
    stride: Tuple[int, ...],
) -> Tuple[Int[Tensor, "M D+1"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Downsample the coordinates by the stride.
    """
    num_spatial_dims = batch_indexed_coords.shape[1] - 1
    assert (
        len(stride) == num_spatial_dims
    ), f"Stride must match the number of spatial dimensions. Got {len(stride)} spatial dimensions for but coordinates with {num_spatial_dims} spatial dimensions."

    if all(s == 1 for s in stride):
        # Assume that the batch index is already sorted
        return batch_indexed_coords, offsets_from_batch_index(
            batch_indexed_coords[:, 0], backend="torch"
        )

    # convert to wp array
    device = batch_indexed_coords.device
    batched_stride = torch.tensor(
        [1, *ntuple(stride, ndim=num_spatial_dims)], dtype=torch.int32, device=device
    )
    # discretize the coordinates by floor division
    discretized_coords = torch.floor(batch_indexed_coords / batched_stride).int()
    # Get unique coordinates
    unique_coords = torch.unique(discretized_coords, dim=0, sorted=True)

    out_batch_index = unique_coords[:, 0]
    out_offsets = offsets_from_batch_index(out_batch_index, backend="torch")

    return unique_coords, out_offsets


@torch.no_grad()
def expand_coords(
    batch_indexed_coords: Int[Tensor, "N 4"],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    kernel_batch: Optional[int] = None,
) -> Tuple[Int[Tensor, "M 4"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Expand the coordinates by the kernel size
    """
    num_total_kernels = np.prod(kernel_size)
    if kernel_batch is None:
        kernel_batch = num_total_kernels // kernel_size[0]
    # coords to batched coordinates
    batch_indexed_coords_wp = wp.from_torch(batch_indexed_coords)
    # Create a vector hashtable for the batched coordinates
    hashtable = VectorHashTable.from_keys(batch_indexed_coords_wp)
    # Initialize the unique coordinates with the batched coordinates
    unique_coords = batch_indexed_coords

    offsets = kernel_offsets_from_size(kernel_size, kernel_dilation).to(
        batch_indexed_coords.device
    )

    for batch_start in range(0, num_total_kernels, kernel_batch):
        batch_end = min(batch_start + kernel_batch, num_total_kernels)
        # Calculate offsets
        curr_offsets = offsets[batch_start:batch_end]

        # Apply offsets in batch
        new_batched_coords = batch_indexed_coords.unsqueeze(0) + curr_offsets.unsqueeze(1)
        new_batched_coords = new_batched_coords.view(-1, 4)
        new_batched_coords_wp = wp.from_torch(new_batched_coords)

        # Query the hashtable for all new coordinates at once
        indices_wp = hashtable.search(new_batched_coords_wp)
        not_in_hashtable = wp.to_torch(indices_wp) < 0

        # Add unique coordinates
        unique_coords = torch.cat([unique_coords, new_batched_coords[not_in_hashtable]], dim=0)
        # Update hashtable with new unique coordinates
        hashtable = VectorHashTable.from_keys(wp.from_torch(unique_coords))

    # sort the coordinates and return the coordinate and offset
    # sort the batch index
    out_coords = unique_coords[torch.argsort(unique_coords[:, 0])]
    out_batch_index = out_coords[:, 0]
    out_offsets = offsets_from_batch_index(out_batch_index, backend="torch")
    return out_coords, out_offsets
