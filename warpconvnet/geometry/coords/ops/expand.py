# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Optional
from jaxtyping import Int

import numpy as np
import torch
import warp as wp
from torch import Tensor

from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index
from warpconvnet.geometry.coords.search.torch_discrete import kernel_offsets_from_size
from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable


@torch.no_grad()
def expand_coords(
    batch_indexed_coords: Int[Tensor, "N D+1"],  # noqa: F821
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    kernel_batch: Optional[int] = None,
) -> Tuple[Int[Tensor, "M D+1"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Expand the coordinates by the kernel size

    TODOs(cchoy@2025-11-21):
      - Fix inefficiency from generating the new coords in a loop
      - Adding coords in a GPU kernel.
        - instead of creating coords explicitly, generate the new coords in shared memory and add them to the hashtable directly.
    """
    num_total_kernels = np.prod(kernel_size)
    if kernel_batch is None:
        kernel_batch = num_total_kernels // kernel_size[0]
    # Create a vector hashtable for the batched coordinates
    hashtable = TorchHashTable.from_keys(batch_indexed_coords)
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

        # Query the hashtable for all new coordinates at once
        indices = hashtable.search(new_batched_coords)
        not_in_hashtable = indices < 0

        # Add unique coordinates
        unique_coords = torch.cat([unique_coords, new_batched_coords[not_in_hashtable]], dim=0)
        # Update hashtable with new unique coordinates
        hashtable = TorchHashTable.from_keys(unique_coords)
        # Get the unique coordinates
        unique_coords = hashtable.unique_vector_keys

    # sort the coordinates and return the coordinate and offset
    # sort the batch index
    out_coords = unique_coords[torch.argsort(unique_coords[:, 0])]
    out_batch_index = out_coords[:, 0]
    out_offsets = offsets_from_batch_index(out_batch_index)
    return out_coords, out_offsets
