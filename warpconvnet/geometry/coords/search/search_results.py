# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Literal, Optional, Tuple
from jaxtyping import Float, Int

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RealSearchResult:
    """
    Wrapper for the output of a neighbor search operation.
    """

    # N is the total number of neighbors for all M queries
    neighbor_indices: Int[Tensor, "N"]  # noqa: F821
    # M is the number of queries
    neighbor_row_splits: Int[Tensor, "M + 1"]  # noqa: F821
    # optional distance
    neighbor_distances: Optional[Float[Tensor, "N"]] = None  # noqa: F821

    def __init__(self, *args):
        # If there are two args, assume they are neighbor_indices and neighbor_row_splits
        # If there is one arg, assume it is a NeighborSearchReturnType
        if len(args) == 2:
            self.neighbor_indices = args[0].long()
            self.neighbor_row_splits = args[1].long()
        elif len(args) == 1:
            # K-nn search result
            assert isinstance(args[0], torch.Tensor)
            # 2D tensor with shape (M, K)
            assert args[0].ndim == 2
            M, K = args[0].shape
            self.neighbor_indices = args[0].long()
            self.neighbor_row_splits = torch.arange(
                0, M * K + 1, K, device=args[0].device, dtype=torch.long
            )
        else:
            raise ValueError("NeighborSearchReturn must be initialized with 1 or 2 arguments")

    def to(self, device: str | int | torch.device):
        self.neighbor_indices.to(device)
        self.neighbor_row_splits.to(device)
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(neighbor_indices={self.neighbor_indices.shape}, neighbor_row_splits={self.neighbor_row_splits.shape})"


@dataclass
class IntSearchResult:
    """
    Wrapper for the output of a neighbor search operation.
    """

    # The indices of the neighbors
    in_maps: Int[Tensor, "L"]  # noqa: F821
    out_maps: Int[Tensor, "L"]  # noqa: F821
    offsets: Int[Tensor, "K + 1"]  # noqa: F821
    identity_map_index: Optional[int] = None

    def __init__(
        self,
        in_maps: Int[Tensor, "L"],  # noqa: F821
        out_maps: Int[Tensor, "L"],  # noqa: F821
        offsets: Int[Tensor, "K + 1"],  # noqa: F821
        identity_map_index: Optional[int] = None,
    ):
        assert len(in_maps) == len(out_maps) == offsets[-1].item()
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.offsets = offsets.cpu()
        self.identity_map_index = identity_map_index

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[Int[Tensor, "N"], Int[Tensor, "N"]]:  # noqa: F821
        start, end = self.offsets[idx], self.offsets[idx + 1]
        return self.in_maps[start:end], self.out_maps[start:end]

    @torch.no_grad()
    def get_batch(
        self,
        start_idx: int,
        end_idx: int,
        out_format: Literal["list", "tensor"] = "list",
    ) -> Tuple[List[Int[Tensor, "N"]], List[Int[Tensor, "N"]]]:  # noqa: F821
        in_maps = []
        out_maps = []
        for i in range(start_idx, end_idx):
            in_maps.append(self.in_maps[self.offsets[i] : self.offsets[i + 1]])
            out_maps.append(self.out_maps[self.offsets[i] : self.offsets[i + 1]])
        if out_format == "list":
            return in_maps, out_maps
        elif out_format == "tensor":
            max_num_maps = max(len(in_map) for in_map in in_maps)
            in_maps_tensor = -1 * torch.ones(
                len(in_maps),
                max_num_maps,
                device=self.in_maps.device,
                dtype=torch.int64,
            )
            out_maps_tensor = -1 * torch.ones(
                len(out_maps),
                max_num_maps,
                device=self.out_maps.device,
                dtype=torch.int64,
            )
            for i, (in_map, out_map) in enumerate(zip(in_maps, out_maps)):
                in_maps_tensor[i, : len(in_map)] = in_map
                out_maps_tensor[i, : len(out_map)] = out_map
            return in_maps_tensor, out_maps_tensor
        else:
            raise ValueError(f"Invalid output format: {out_format}")

    def __len__(self):
        return len(self.offsets) - 1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"{self.__class__.__name__}(len={len(self)}, iden_map={self.identity_map_index})"

    def numel(self, i: int) -> int:
        """
        Return the number of elements in the i-th map.
        """
        return (self.offsets[i + 1] - self.offsets[i]).item()

    @torch.no_grad()
    def to_csr(
        self,
    ) -> Tuple[Int[Tensor, "L"], Int[Tensor, "K"], Int[Tensor, "K + 1"]]:  # noqa: F821
        """
        Convert the neighbor search result to a CSR format.

        in_maps to row indices
        out_maps to sort and use for columns
        ignore offsets
        """
        in_maps = self.in_maps
        out_maps = self.out_maps

        # Sort the out_maps and get the indices
        out_maps_sorted, out_maps_indices = torch.sort(out_maps)
        # cchoy: Could skip the sorting by implementing a custom warp kernel
        unique_out_maps_sorted, num_unique = torch.unique(
            out_maps_sorted, return_counts=True, sorted=True
        )

        # Get the in_maps from the indices
        in_maps_sorted = in_maps[out_maps_indices]

        # convert to offsets
        offsets = torch.cumsum(num_unique.cpu(), dim=0)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), offsets], dim=0)
        return in_maps_sorted, unique_out_maps_sorted, offsets

    def clone(self):
        return IntSearchResult(
            self.in_maps.clone(),
            self.out_maps.clone(),
            self.offsets.clone(),
            self.identity_map_index,
        )

    @property
    def device(self):
        return self.in_maps.device
