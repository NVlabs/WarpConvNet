# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional
from jaxtyping import Float, Int

import numpy as np
import cupy as cp
import math
import os

import torch
from torch import Tensor

from warpconvnet.utils.cuda_utils import load_kernel


# cuda_utils.py automatically handles the csrc path for just filename
_bsearch_kernel = load_kernel(kernel_file="find_first_gt_bsearch.cu", kernel_name="find_first_gt_bsearch")


@torch.inference_mode()
def batch_index_from_offset(
    offsets: Int[Tensor, "B+1"],
) -> Int[Tensor, "N"]:  # type: ignore
    """
    Generates batch indices for a contiguous range of elements defined by offsets.
    `offsets` has B+1 elements, defining B batches.
    Output has N = offsets[B] elements.
    """
    assert len(offsets) > 1, "offsets must have at least two elements. [0, N] for batch size 1"
    count = torch.diff(offsets)
    batch = torch.arange(len(count), device=offsets.device, dtype=torch.long).repeat_interleave(
        count
    )
    return batch


@torch.inference_mode()
def batch_index_from_indices(
    indices: Int[Tensor, "N_indices"],  # type: ignore
    offsets: Int[Tensor, "B_plus_1"],  # type: ignore
    device: Optional[str] = None,
    threads: int = 256,
) -> Int[Tensor, "N_indices"]:  # type: ignore
    """
    Finds batch indices for given `indices` based on `offsets`.
    `offsets` has B+1 elements, defining B batches.
    Output has N_indices elements.
    """
    assert isinstance(indices, torch.Tensor), "indices must be a torch.Tensor"
    assert isinstance(offsets, torch.Tensor), "offsets must be a torch.Tensor"

    _dev = device
    if _dev is None:
        if indices.is_cuda:
            _dev = str(indices.device)
        elif offsets.is_cuda:
            _dev = str(offsets.device)
        else:
            raise ValueError("At least one tensor must be on CUDA if device is not specified.")

    if not indices.is_cuda or str(indices.device) != _dev:
        indices = indices.to(_dev)
    if not offsets.is_cuda or str(offsets.device) != _dev:
        offsets = offsets.to(_dev)

    indices = indices.contiguous().int()
    offsets = offsets.contiguous().int()

    M_len = offsets.shape[0]  # Length of offsets array, M_len = B + 1
    N_indices = indices.shape[0]

    if N_indices == 0:
        return torch.empty(0, dtype=torch.int32, device=_dev)
    if M_len == 0:  # No offsets defined, cannot determine batch
        raise ValueError("Offsets cannot be empty.")
    if M_len == 1:  # Only one offset value, e.g. offsets=[limit]. All indices < limit are batch 0.
        return torch.zeros(N_indices, dtype=torch.int32, device=_dev)

    indices_cp = cp.from_dlpack(indices)
    offsets_cp = cp.from_dlpack(offsets)
    batch_index_buffer_cp = cp.empty(N_indices, dtype=cp.int32)

    blocks = math.ceil(N_indices / threads)
    shared_mem_bytes = M_len * offsets_cp.dtype.itemsize

    # Kernel: find_first_gt_bsearch(const int *srcM, int M, const int *srcN, int N, int *out)
    # srcM: offsets_cp.data.ptr
    # M: M_len (length of offsets array)
    # srcN: indices_cp.data.ptr
    # N: N_indices (number of elements in indices tensor)
    # out: batch_index_buffer_cp.data.ptr
    _bsearch_kernel(
        (blocks,),
        (threads,),
        (
            offsets_cp,
            M_len,
            indices_cp,
            N_indices,
            batch_index_buffer_cp,
        ),
        shared_mem=shared_mem_bytes,
    )

    return torch.from_dlpack(batch_index_buffer_cp).to(_dev)


@torch.inference_mode()
def batch_indexed_coordinates(
    batched_coords: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
) -> Float[Tensor, "N 4"]:  # noqa: F821
    batch_index = batch_index_from_offset(offsets).to(batched_coords)
    batched_coords = torch.cat([batch_index.unsqueeze(1), batched_coords], dim=1)
    return batched_coords


@torch.inference_mode()
def offsets_from_batch_index_consecutive(
    batch_index: Int[Tensor, "N"],  # noqa: F821
) -> Int[Tensor, "B + 1"]:  # noqa: F821
    """
    Given a list of batch indices [0, 0, 1, 1, 2, 2, 2, 3, 3],
    return the offsets [0, 2, 4, 7, 9].
    """
    assert batch_index.ndim == 1, "batch_index must be a 1D tensor"
    assert len(batch_index) > 0, "batch_index must not be empty"
    # Derive offsets (assuming out_indices_batch_indexed[:, 0] is sorted by batch)
    unique_b_idx, counts = torch.unique_consecutive(batch_index, return_counts=True)
    # Basic check if any points returned for all batches up to max batch_idx
    # This logic for offsets needs to be robust for empty batches.
    out_offsets_cpu = [0]
    max_batch_idx_present = unique_b_idx[-1].item()
    temp_counts = torch.zeros(max_batch_idx_present + 1, dtype=counts.dtype)
    temp_counts[unique_b_idx.cpu()] = counts.cpu()
    out_offsets_cpu.extend(torch.cumsum(temp_counts, dim=0).cpu().tolist())
    return torch.IntTensor(out_offsets_cpu)


@torch.inference_mode()
def offsets_from_batch_index(
    batch_index: Int[Tensor, "N"],  # noqa: F821
) -> Int[Tensor, "B + 1"]:  # noqa: F821
    """
    Given a list of batch indices [0, 0, 1, 1, 2, 2, 2, 3, 3],
    return the offsets [0, 2, 4, 7, 9].
    """
    # Get unique elements
    counts = torch.bincount(batch_index)
    counts = counts.cpu()
    # Get the offsets by cumsum
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            counts.cumsum(dim=0),
        ],
        dim=0,
    )
    return offsets


@torch.inference_mode()
def offsets_from_offsets(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    sorted_indices: Int[Tensor, "N"],  # noqa: F821
    device: Optional[str] = None,
) -> Int[Tensor, "B+1"]:  # noqa: F821
    """
    Given a sorted indices, return a new offsets that selects batch indices using the indices.
    """
    B = offsets.shape[0] - 1
    if B == 1:
        new_offsets = torch.IntTensor([0, len(sorted_indices)])
    else:
        batch_index = batch_index_from_offset(offsets)
        if device is not None:
            batch_index = batch_index.to(device)
            sorted_indices = sorted_indices.to(device)
        else:
            # if no device is specified, use the device of sorted_indices
            batch_index = batch_index.to(sorted_indices.device)
        _, batch_counts = torch.unique_consecutive(batch_index[sorted_indices], return_counts=True)
        batch_counts = batch_counts.cpu()
        new_offsets = torch.cat((batch_counts.new_zeros(1), batch_counts.cumsum(dim=0)))
    return new_offsets
