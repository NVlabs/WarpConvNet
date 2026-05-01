# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed hash table for 4D integer coordinates (batch, x, y, z).

Packs coordinates into uint64 keys for single-load probes, single-instruction
comparison, and Splitmix64 hashing. ~2.5-6x faster kernel map generation
compared to the vector-key layout in TorchHashTable.
"""

import enum
from typing import Optional, Union

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search._packed_base import PackedHashTableBase


class SearchMode(enum.IntEnum):
    LINEAR = 0
    DOUBLE_HASH = 1
    WARP_COOP = 2


class PackedHashTable(PackedHashTableBase):
    """Hash table for 4D integer coordinates packed into uint64.

    Bit layout:
      valid:  1 bit (top bit, always 1 for occupied)
      batch:  9 bits unsigned [0, 511]
      x,y,z:  18 bits signed  [-131072, 131071] each

    At 0.01m voxel size, 18-bit spatial range covers +/-1.3 km per axis.
    The top bit is reserved as a validity flag to avoid kEmpty sentinel
    collisions; batch is therefore limited to 9 bits ([0, 511]).
    """

    BATCH_MAX = 511
    COORD_MIN = -131072
    COORD_MAX = 131071

    def __init__(
        self,
        capacity: int,
        device: Union[str, torch.device] = "cuda",
        use_double_hash: bool = False,
    ):
        super().__init__(capacity=capacity, device=device)
        self._use_double_hash = use_double_hash
        self._coords: Optional[Tensor] = None
        self._coarse_cache: dict = {}  # stride -> PackedHashTable

    @property
    def key_dim(self) -> int:
        return 4

    @property
    def vector_keys(self) -> Tensor:
        """Return stored coordinates (backward compat with TorchHashTable)."""
        if self._coords is None:
            raise RuntimeError("No coordinates stored. Call insert() first.")
        return self._coords[: self._num_entries]

    def _prepare_insert_coords(self, coords: Tensor) -> Tensor:
        assert coords.ndim == 2 and coords.shape[1] == 4
        coords = coords.contiguous().to(dtype=torch.int32, device=self._device)
        if coords.shape[0] > 0:
            batch = coords[:, 0]
            spatial = coords[:, 1:]
            if batch.min().item() < 0 or batch.max().item() > self.BATCH_MAX:
                raise ValueError(
                    f"Batch index out of range [0, {self.BATCH_MAX}]: "
                    f"got [{batch.min().item()}, {batch.max().item()}]"
                )
            if spatial.min().item() < self.COORD_MIN or spatial.max().item() > self.COORD_MAX:
                raise ValueError(
                    f"Spatial coord out of range [{self.COORD_MIN}, {self.COORD_MAX}]: "
                    f"got [{spatial.min().item()}, {spatial.max().item()}]"
                )
        return coords

    def _prepare_search_coords(self, queries: Tensor) -> Tensor:
        assert queries.ndim == 2 and queries.shape[1] == 4
        return queries.contiguous().to(dtype=torch.int32, device=self._device)

    def _allocate_storage(self) -> None:
        self._keys = torch.empty(self._capacity, dtype=torch.int64, device=self._device)
        self._values = torch.empty(self._capacity, dtype=torch.int32, device=self._device)

    def _run_prepare(self) -> None:
        _C.cuhash.packed_prepare(self._keys, self._values, self._capacity)

    def _run_insert(self, coords: Tensor, num_keys: int, status_tensor: Tensor) -> None:
        _C.cuhash.packed_insert(
            self._keys,
            self._values,
            coords,
            num_keys,
            self._capacity,
            self._use_double_hash,
            status_tensor,
        )

    def _run_search(
        self,
        queries: Tensor,
        results: Tensor,
        num_search: int,
        mode: SearchMode = SearchMode.LINEAR,
    ) -> None:
        search_mode = int(mode)
        if self._use_double_hash and search_mode == 0:
            search_mode = 1
        _C.cuhash.packed_search(
            self._keys,
            self._values,
            queries,
            results,
            num_search,
            self._capacity,
            search_mode,
        )

    def _post_insert(self, coords: Tensor) -> None:
        self._coords = coords

    @classmethod
    def from_coords(
        cls,
        coords: Tensor,
        device: Union[str, torch.device] = "cuda",
        capacity: Optional[int] = None,
        use_double_hash: bool = False,
    ) -> "PackedHashTable":
        """Create a hash table from 4D coordinates."""
        target = torch.device(device)
        coords = coords.contiguous().to(dtype=torch.int32, device=target)
        n = coords.shape[0]
        cap = capacity if capacity is not None else max(16, n * 2)
        obj = cls(capacity=cap, device=target, use_double_hash=use_double_hash)
        obj.insert(coords)
        return obj

    def _ensure_vector_storage(self, required_capacity: int):
        """Grow the coords storage if needed."""
        if self._coords is None or self._coords.shape[0] < required_capacity:
            new_store = torch.empty((required_capacity, 4), dtype=torch.int32, device=self._device)
            if self._coords is not None:
                n = self._num_entries
                new_store[:n] = self._coords[:n]
            self._coords = new_store

    def expand_with_offsets(
        self,
        base_coords: Tensor,
        offsets: Tensor,
    ):
        """Insert all (base + offset) combinations, deduplicating via the hash table.

        Args:
            base_coords: int32 (N, 4) base coordinates.
            offsets: int32 (K, 4) offsets to add to each base coordinate.
        """
        assert self._keys is not None, "Call insert() first"
        base_coords = base_coords.contiguous().to(dtype=torch.int32, device=self._device)
        offsets = offsets.contiguous().to(dtype=torch.int32, device=self._device)
        assert base_coords.ndim == 2 and base_coords.shape[1] == 4
        assert offsets.ndim == 2 and offsets.shape[1] == 4

        num_base = base_coords.shape[0]
        num_offsets = offsets.shape[0]

        # Ensure we have enough vector storage for the worst case
        max_new = num_base * num_offsets
        required = self._num_entries + max_new
        self._ensure_vector_storage(required)

        # Use GPU-side atomic counter for num_entries
        num_entries_tensor = torch.tensor(
            [self._num_entries], dtype=torch.int32, device=self._device
        )
        status_tensor = torch.zeros(1, dtype=torch.int32, device=self._device)

        _C.cuhash.packed_expand_insert(
            self._keys,
            self._values,
            self._coords,
            base_coords,
            offsets,
            num_base,
            num_offsets,
            self._capacity,
            self._coords.shape[0],  # vector_capacity
            num_entries_tensor,
            status_tensor,
        )

        self._num_entries = num_entries_tensor.item()
        status = status_tensor.item()
        if status == 1:
            raise RuntimeError(
                f"Packed expand: vector storage overflow "
                f"(entries={self._num_entries}, capacity={self._coords.shape[0]})"
            )
        if status == 2:
            raise RuntimeError(
                f"Packed expand: hash table full "
                f"(entries={self._num_entries}, capacity={self._capacity})"
            )

    @property
    def unique_index(self) -> Tensor:
        """Get sorted unique indices from the hash table."""
        assert self._keys is not None, "Call insert() first"
        indices = self.search(self._coords)
        valid_indices = indices[indices != -1]
        return torch.unique(valid_indices)

    def _get_coarse(self, stride: int) -> "PackedHashTable":
        """Get or build a coarse hash table at the given stride.

        The coarse table maps floor-divided spatial coordinates to arbitrary
        (unique) indices. It's cached on the fine table, so repeat calls with
        the same stride reuse the built table.

        Note: Currently unused by the fused C++ hierarchical_kernel_map path,
        which builds the coarse table internally. Kept here so Python callers
        that want explicit control (or a different C++ signature) can benefit.

        Args:
            stride: Coarse table stride (must be a positive power of 2).
        """
        assert (
            stride > 0 and (stride & (stride - 1)) == 0
        ), f"stride must be a positive power of 2, got {stride}"
        if stride in self._coarse_cache:
            return self._coarse_cache[stride]

        assert self._coords is not None, "Call insert() before _get_coarse()"
        fine_coords = self._coords[: self._num_entries]
        coarse_coords = fine_coords.clone()
        coarse_coords[:, 1:] = torch.div(fine_coords[:, 1:], stride, rounding_mode="floor")
        coarse_coords = torch.unique(coarse_coords, dim=0)
        coarse_ht = PackedHashTable.from_coords(coarse_coords, device=self._device)
        self._coarse_cache[stride] = coarse_ht
        return coarse_ht

    def search(
        self,
        query_coords: Tensor,
        mode: SearchMode = SearchMode.LINEAR,
    ) -> Tensor:
        """Search for 4D coordinates in the hash table.

        Returns:
            int32 tensor of shape (M,) with original indices, -1 if not found.
        """
        return super().search(query_coords, mode=mode)
