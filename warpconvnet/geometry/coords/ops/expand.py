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
    target_device = batch_indexed_coords.device
    if target_device.type != "cuda":
        raise ValueError(f"expand_coords requires CUDA tensors, got {target_device}")

    coords = batch_indexed_coords.to(dtype=torch.int32, device=target_device).contiguous()
    num_input = coords.shape[0]

    num_total_kernels = int(np.prod(kernel_size))
    if kernel_batch is None:
        kernel_batch = max(1, num_total_kernels // kernel_size[0])
    else:
        kernel_batch = max(1, min(kernel_batch, num_total_kernels))

    offsets = kernel_offsets_from_size(kernel_size, kernel_dilation, device=target_device)
    offsets = offsets.to(dtype=torch.int32).contiguous()

    # Start with a moderate load factor and grow as needed.
    table_capacity = max(16, int(max(1, num_input) * 4))
    hashtable = TorchHashTable.from_keys(
        coords,
        device=target_device,
        capacity=table_capacity,
        vector_capacity=num_input,
    )

    for batch_start in range(0, num_total_kernels, kernel_batch):
        batch_end = min(batch_start + kernel_batch, num_total_kernels)
        curr_offsets = offsets[batch_start:batch_end]
        if curr_offsets.numel() == 0:
            continue
        # Ensure the table has enough free slots for the upcoming insertions.
        potential_entries = hashtable.num_entries + num_input * curr_offsets.shape[0]
        if potential_entries > hashtable.capacity // 2:
            new_capacity = max(potential_entries * 2, hashtable.capacity * 2)
            new_vector_capacity = max(potential_entries, hashtable.num_entries * 2)
            hashtable = TorchHashTable.from_keys(
                hashtable.vector_keys,
                hash_method=hashtable.hash_method,
                device=target_device,
                capacity=new_capacity,
                vector_capacity=new_vector_capacity,
            )
        hashtable.expand_with_offsets(coords, curr_offsets)

    unique_coords = hashtable.vector_keys.contiguous()
    out_coords = unique_coords[torch.argsort(unique_coords[:, 0])]
    out_batch_index = out_coords[:, 0]
    out_offsets = offsets_from_batch_index(out_batch_index)
    return out_coords, out_offsets
