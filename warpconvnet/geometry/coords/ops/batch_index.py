# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.bin
import warp as wp
from jaxtyping import Float, Int
from torch import Tensor

snippet = """
    __shared__ int shared_offsets[256];

    int block_tid = threadIdx.x;

    // Load offsets into shared memory.
    // Make sure that the last block loads the full offsets.
    if (block_tid < offsets_len) {
        shared_offsets[block_tid] = offsets[block_tid];
    }
    __syncthreads();

    // index is the row index of the tid
    if (tid < batch_index_len) {
        // Find bin
        int bin = -1;
        for (int i = 0; i < offsets_len - 1; i++) {
            int start = shared_offsets[i];
            int end = shared_offsets[i + 1];
            if (start <= index && index < end) {
                bin = i;
                break;
            }
        }

        batch_index_wp[tid] = bin;
    }
    """


@wp.func_native(snippet)
def _find_bin_native(
    offsets: wp.array(dtype=Any),
    offsets_len: int,
    tid: int,
    index: int,
    batch_index_wp: wp.array(dtype=Any),
    batch_index_len: int,
): ...


@wp.func
def _find_bin(offsets: wp.array(dtype=Any), tid: int) -> int:
    N = offsets.shape[0] - 1
    bin_id = int(-1)
    for i in range(N):
        start = offsets[i]
        end = offsets[i + 1]
        if start <= tid and tid < end:
            bin_id = i
            break
    return bin_id


@wp.kernel
def _batch_index(
    offsets: wp.array(dtype=wp.int32),
    batch_index_wp: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if offsets.shape[0] > 256:
        batch_index_wp[tid] = _find_bin(offsets, tid)
    else:
        _find_bin_native(
            offsets, offsets.shape[0], tid, tid, batch_index_wp, batch_index_wp.shape[0]
        )


@wp.kernel
def _batch_index_from_indicies(
    indices: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    batch_index_wp: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    index = indices[tid]
    _find_bin_native(
        offsets, offsets.shape[0], tid, index, batch_index_wp, batch_index_wp.shape[0]
    )


def batch_index_from_offset(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    device: Optional[str] = None,
    backend: Literal["auto", "torch", "warp"] = "auto",
    force_cpu_threshold: int = 16384,
) -> Int[Tensor, "N"]:  # noqa: F821
    assert isinstance(offsets, torch.Tensor), "offsets must be a torch.Tensor"
    assert backend in ["auto", "torch", "warp"], "backend must be either torch or warp"

    # cchoy: This function will be inefficient for small offsets[-1].
    # TODO(cchoy): benchmark torch vs warp for small offsets[-1].
    # Force use torch cpu implementation for small offsets[-1].
    if force_cpu_threshold > 0 and offsets[-1].item() < force_cpu_threshold and backend == "auto":
        result = batch_index_from_offset(offsets, device="cpu", backend="torch")
        if device is not None:
            result = result.to(device)
        return result

    if backend == "auto":
        if device is None or "cpu" in device:
            backend = "torch"
        else:
            backend = "warp"

    # force offsets to int
    offsets = offsets.int()

    # ------ Torch Implementation ------
    if backend == "torch":
        batch_index = (
            torch.arange(len(offsets) - 1)
            .to(offsets)
            .repeat_interleave(offsets[1:] - offsets[:-1])
        )
        return batch_index

    # ------ Warp GPU Kernel ------
    # Assert this is not cpu
    if device is not None:
        offsets = offsets.to(device)

    device: str = str(offsets.device)  # warp requires string device
    assert "cpu" not in device, "device must be a cuda device"

    N = offsets[-1].item()
    offsets_wp = wp.from_torch(offsets.int(), dtype=wp.int32).to(device)
    batch_index_wp = wp.zeros(shape=(N,), dtype=wp.int32, device=device)
    wp.launch(
        _batch_index,
        int(np.ceil(N / 256.0)) * 256,
        inputs=[offsets_wp, batch_index_wp],
        device=device,
    )
    return wp.to_torch(batch_index_wp)


def batch_index_from_indicies(
    indices: Int[Tensor, "N"],  # noqa: F821
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    device: Optional[str] = None,
) -> Int[Tensor, "N"]:  # noqa: F821
    assert isinstance(indices, torch.Tensor), "indices must be a torch.Tensor"
    assert isinstance(offsets, torch.Tensor), "offsets must be a torch.Tensor"

    # offset to int
    offsets = offsets.int()

    if device is not None:
        offsets = offsets.to(device)

    if device is None:
        device = str(offsets.device)

    # Assert this is not cpu
    assert "cpu" not in device, "device must be a cuda device"

    N = indices.shape[0]
    indicies_wp = wp.from_torch(indices.int(), dtype=wp.int32).to(device)
    offsets_wp = wp.from_torch(offsets.int(), dtype=wp.int32).to(device)
    batch_index_wp = wp.zeros(shape=(N,), dtype=wp.int32, device=device)
    wp.launch(
        _batch_index_from_indicies,
        int(np.ceil(N / 256.0)) * 256,
        inputs=[indicies_wp, offsets_wp, batch_index_wp],
        device=device,
    )
    return wp.to_torch(batch_index_wp)


def batch_indexed_coordinates(
    batched_coords: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
    backend: Literal["auto", "torch", "warp"] = "auto",
    return_type: Literal["torch", "warp"] = "torch",
) -> Float[Tensor, "N 4"]:  # noqa: F821
    device = str(batched_coords.device)
    batch_index = batch_index_from_offset(offsets, device=device, backend=backend)
    batched_coords = torch.cat([batch_index.unsqueeze(1), batched_coords], dim=1)
    if return_type == "torch":
        return batched_coords
    elif return_type == "warp":
        return wp.from_torch(batched_coords)
    else:
        raise ValueError("return_type must be either torch or warp")


def offsets_from_batch_index(
    batch_index: Int[Tensor, "N"],  # noqa: F821
    backend: Literal["torch", "warp"] = "torch",
) -> Int[Tensor, "B + 1"]:  # noqa: F821
    """
    Given a list of batch indices [0, 0, 1, 1, 2, 2, 2, 3, 3],
    return the offsets [0, 2, 4, 7, 9].
    """
    if backend == "torch":
        # Get unique elements
        _, counts = torch.unique(batch_index, return_counts=True)
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
    elif backend == "warp":
        raise NotImplementedError("warp backend not implemented")
    else:
        raise ValueError("backend must be torch")


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
        batch_index = batch_index_from_offset(offsets, device=device)
        _, batch_counts = torch.unique_consecutive(batch_index[sorted_indices], return_counts=True)
        batch_counts = batch_counts.cpu()
        new_offsets = torch.cat((batch_counts.new_zeros(1), batch_counts.cumsum(dim=0)))
    return new_offsets
