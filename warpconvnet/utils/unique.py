# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import warp as wp
from jaxtyping import Int
from torch import Tensor

from warpconvnet.geometry.coords.search.torch_hashmap import HashMethod, TorchHashTable
from warpconvnet.geometry.coords.ops.serialization import STR2POINT_ORDERING, encode
from warpconvnet.utils.ravel import ravel_multi_index


def unique_segmented(
    x: Int[Tensor, "N"],
    offsets: Int[Tensor, "N+1"],
    return_counts: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Get the sorted unique elements and their counts.
    """
    unique_keys = []
    unique_counts = []
    for i in range(len(offsets) - 1):
        start = offsets[i].item()
        end = offsets[i + 1].item()
        result = torch.unique_consecutive(x[start:end], return_counts=return_counts)
        if return_counts:
            unique_keys.append(result[0])
            unique_counts.append(result[1])
        else:
            unique_keys.append(result)
    return torch.cat(unique_keys), torch.cat(unique_counts)


def unique_inverse(
    x: Tensor,
    dim: int = 0,
) -> Tuple[
    Int[Tensor, "M C"],
    Int[Tensor, "N"],  # noqa: F821
]:
    """
    Get to_unique_indices and to_orig_indices.
    """
    unique, to_orig_indices = torch.unique(x, dim=dim, sorted=True, return_inverse=True)
    to_unique_indices = torch.arange(x.size(dim), dtype=x.dtype, device=x.device)
    to_unique_indices = torch.empty(unique.size(dim), dtype=x.dtype, device=x.device).scatter_(
        dim, to_orig_indices, to_unique_indices
    )
    return to_unique_indices, to_orig_indices


def unique_torch(
    x: Int[Tensor, "N C"],
    dim: int = 0,
    stable: bool = False,
    return_to_unique_indices: bool = False,
) -> Tuple[  # noqa: F821
    Int[Tensor, "M C"],  # noqa: F821
    Int[Tensor, "N"],  # noqa: F821
    Int[Tensor, "N"],  # noqa: F821
    Int[Tensor, "M+1"],  # noqa: F821
    Int[Tensor, "M"],  # noqa: F821
]:
    """
    Get unique elements along a dimension.

    Args:
        x: Tensor
        dim: int
        stable: bool

    Returns:
        unique: M unique coordinates
        to_orig_indices: N indices to original coordinates. unique[to_orig_indices] == x
        all_to_csr_indices: N indices to unique coordinates. x[all_to_csr_indices] == torch.repeat_interleave(unique, counts).
        all_to_csr_offsets: M+1 offsets to unique coordinates. counts = all_to_csr_offsets.diff()
        to_unique_indices: M indices to sample x to unique. x[to_unique_indices] == unique

    from https://github.com/pytorch/pytorch/issues/36748
    """
    unique, to_orig_indices, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    all_to_csr_indices = to_orig_indices.argsort(stable=stable)
    all_to_csr_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))

    if return_to_unique_indices:
        dtype_ind, device = to_orig_indices.dtype, to_orig_indices.device
        to_unique_indices = torch.arange(x.size(dim), dtype=dtype_ind, device=device)
        to_unique_indices = torch.empty(unique.size(dim), dtype=dtype_ind, device=device).scatter_(
            dim, to_orig_indices, to_unique_indices
        )
    else:
        to_unique_indices = None

    return (
        unique,
        to_orig_indices,
        all_to_csr_indices,
        all_to_csr_offsets,
        to_unique_indices,
    )


def unique_ravel(
    x: Int[Tensor, "N C"],
    dim: int = 0,
    sorted: bool = False,
):
    min_coords = x.min(dim=dim).values
    shifted_x = x - min_coords
    shape = shifted_x.max(dim=dim).values + 1
    raveled_x = ravel_multi_index(shifted_x, shape)
    unique_raveled_x, _, _, _, perm = unique_torch(raveled_x, dim=0)
    if sorted:
        perm = perm[unique_raveled_x.argsort()]
    return perm


def unique_hashmap(
    bcoords: Int[Tensor, "N 4"],  # noqa: F821
    hash_method: HashMethod = HashMethod.CITY,
) -> Tuple[Int[Tensor, "M"], TorchHashTable]:  # noqa: F821
    """
    Args:
        bcoords: Batched coordinates.
        hash_method: Hash method.

    Returns:
        unique_indices: bcoords[unique_indices] == unique
        hash_table: Hash table.
    """
    # Append batch index to the coordinates
    assert "cuda" in str(
        bcoords.device
    ), f"Batched coordinates must be on cuda device, got {bcoords.device}"
    table = TorchHashTable(2 * len(bcoords), hash_method)
    table.insert(bcoords)
    return table.unique_index, table  # this is a torch tensor


@dataclass
class UniqueInfo:
    to_orig_indices: Int[Tensor, "N"]  # noqa: F821
    to_csr_indices: Int[Tensor, "N"]  # noqa: F821
    to_csr_offsets: Int[Tensor, "M+1"]  # noqa: F821
    to_unique_indices: Optional[Int[Tensor, "M"]]  # noqa: F821


class ToUnique:
    unique_info: UniqueInfo

    def __init__(
        self,
        unique_method: Literal["torch", "ravel", "morton"] = "torch",
        return_to_unique_indices: bool = False,
    ):
        # Ravel can be used only when the raveled coordinates is less than 2**31
        self.unique_method = unique_method
        self.return_to_unique_indices = return_to_unique_indices or unique_method == "ravel"

    def to_unique(self, x: Int[Tensor, "N C"], dim: int = 0) -> Int[Tensor, "M C"]:
        if self.unique_method == "ravel":
            min_coords = x.min(dim=dim).values
            shifted_x = x - min_coords
            shape = shifted_x.max(dim=dim).values + 1
            unique_input = ravel_multi_index(shifted_x, spatial_shape=shape)
        elif self.unique_method.startswith("morton"):
            assert self.unique_method in STR2POINT_ORDERING, f"Given unique method '{self.unique_method}' - must be one of " + str(STR2POINT_ORDERING.keys())
            unique_input = encode(
                x, order=STR2POINT_ORDERING[self.unique_method], return_perm=False, return_inverse=False
            )
        else:
            unique_input = x

        (
            unique,
            to_orig_indices,
            all_to_unique_indices,
            all_to_unique_offsets,
            to_unique_indices,
        ) = unique_torch(
            unique_input,
            dim=dim,
            stable=True,
            return_to_unique_indices=self.return_to_unique_indices,
        )
        self.unique_info = UniqueInfo(
            to_orig_indices=to_orig_indices,
            to_csr_indices=all_to_unique_indices,
            to_csr_offsets=all_to_unique_offsets,
            to_unique_indices=to_unique_indices,
        )
        if self.unique_method == "ravel":
            return x[self.unique_info.to_unique_indices]
        return unique

    def to_unique_csr(
        self, x: Int[Tensor, "N C"], dim: int = 0
    ) -> Tuple[Int[Tensor, "M C"], Int[Tensor, "N"], Int[Tensor, "M+1"]]:  # noqa: F821
        """
        Convert the the tensor to a unique tensor and return the indices and offsets that can be used for reduction.

        Returns:
            unique: M unique coordinates
            to_csr_indices: N indices to unique coordinates. x[to_csr_indices] == torch.repeat_interleave(unique, counts).
            to_csr_offsets: M+1 offsets to unique coordinates. counts = to_csr_offsets.diff()
        """
        unique = self.to_unique(x, dim=dim)
        return unique, self.unique_info.to_csr_indices, self.unique_info.to_csr_offsets

    def to_original(self, unique: Int[Tensor, "M C"]) -> Int[Tensor, "N C"]:
        return unique[self.unique_info.to_orig_indices]

    @property
    def to_unique_indices(self) -> Int[Tensor, "M"]:  # noqa: F821
        return self.unique_info.to_unique_indices

    @property
    def to_csr_indices(self) -> Int[Tensor, "N"]:  # noqa: F821
        return self.unique_info.to_csr_indices

    @property
    def to_csr_offsets(self) -> Int[Tensor, "M+1"]:  # noqa: F821
        return self.unique_info.to_csr_offsets

    @property
    def to_orig_indices(self) -> Int[Tensor, "N"]:  # noqa: F821
        return self.unique_info.to_orig_indices
