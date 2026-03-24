# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Tuple
from jaxtyping import Int

import torch
from torch import Tensor

from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING, encode
from warpconvnet.utils.ntuple import ntuple
from warpconvnet.utils.ravel import ravel_multi_index_auto_shape
from warpconvnet.utils.unique import unique_hashmap, unique_inverse


@torch.no_grad()
def stride_coords(
    batch_indexed_coords: Int[Tensor, "N D+1"],
    stride: Tuple[int, ...],
    order: POINT_ORDERING | str = POINT_ORDERING.RANDOM,
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
            batch_indexed_coords[:, 0]
        )

    # convert to wp array
    device = batch_indexed_coords.device
    batched_stride = torch.tensor(
        [1, *ntuple(stride, ndim=num_spatial_dims)], dtype=torch.int32, device=device
    )
    # discretize the coordinates by floor division
    discretized_coords = torch.floor(batch_indexed_coords / batched_stride).int()
    unique_indices, _ = unique_hashmap(discretized_coords)
    unique_coords = discretized_coords[unique_indices]
    if order == POINT_ORDERING.RANDOM:
        # sort the batch index for the offset
        out_batch_index = unique_coords[:, 0]
        perm = torch.argsort(out_batch_index)
        unique_coords = unique_coords[perm]
    else:
        code_result = encode(unique_coords, order=order, return_perm=True)
        unique_coords = unique_coords[code_result.perm]

    out_offsets = offsets_from_batch_index(unique_coords[:, 0])
    return unique_coords, out_offsets
