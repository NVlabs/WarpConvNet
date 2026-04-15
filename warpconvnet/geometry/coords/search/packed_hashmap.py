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


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


class SearchMode(enum.IntEnum):
    LINEAR = 0
    DOUBLE_HASH = 1
    WARP_COOP = 2


class PackedHashTable:
    """Hash table for 4D integer coordinates packed into uint64.

    Bit layout:
      batch:  10 bits unsigned [0, 1023]
      x,y,z:  18 bits signed   [-131072, 131071] each

    At 0.01m voxel size, 18-bit spatial range covers +/-1.3 km per axis.
    """

    BATCH_MAX = 1023
    COORD_MIN = -131072
    COORD_MAX = 131071

    def __init__(
        self,
        capacity: int,
        device: Union[str, torch.device] = "cuda",
        use_double_hash: bool = False,
    ):
        self._capacity = _next_power_of_2(capacity)
        self._device = torch.device(device)
        self._use_double_hash = use_double_hash
        self._keys: Optional[Tensor] = None
        self._values: Optional[Tensor] = None
        self._num_entries = 0
        self._coords: Optional[Tensor] = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def device(self) -> torch.device:
        if self._keys is not None:
            return self._keys.device
        return self._device

    @property
    def num_entries(self) -> int:
        return self._num_entries

    @property
    def key_dim(self) -> int:
        return 4

    @property
    def keys_tensor(self) -> Tensor:
        return self._keys

    @property
    def values_tensor(self) -> Tensor:
        return self._values

    @property
    def vector_keys(self) -> Tensor:
        """Return stored coordinates (backward compat with TorchHashTable)."""
        if self._coords is None:
            raise RuntimeError("No coordinates stored. Call insert() first.")
        return self._coords[: self._num_entries]

    def insert(self, coords: Tensor):
        """Insert 4D integer coordinates into the hash table.

        Args:
            coords: int32 tensor of shape (N, 4) on CUDA device.
        """
        assert coords.ndim == 2 and coords.shape[1] == 4
        assert coords.is_cuda
        coords = coords.contiguous().to(dtype=torch.int32, device=self._device)

        num_keys = coords.shape[0]
        assert (
            num_keys <= self._capacity // 2
        ), f"num_keys={num_keys} exceeds capacity/2={self._capacity // 2}"

        self._keys = torch.empty(self._capacity, dtype=torch.int64, device=self._device)
        self._values = torch.empty(self._capacity, dtype=torch.int32, device=self._device)

        _C.cuhash.packed_prepare(self._keys, self._values, self._capacity)
        _C.cuhash.packed_insert(
            self._keys,
            self._values,
            coords,
            num_keys,
            self._capacity,
            self._use_double_hash,
        )

        self._num_entries = num_keys
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

    def search(
        self,
        query_coords: Tensor,
        mode: SearchMode = SearchMode.LINEAR,
    ) -> Tensor:
        """Search for 4D coordinates in the hash table.

        Returns:
            int32 tensor of shape (M,) with original indices, -1 if not found.
        """
        assert self._keys is not None, "Call insert() first"
        assert query_coords.ndim == 2 and query_coords.shape[1] == 4
        query_coords = query_coords.contiguous().to(dtype=torch.int32, device=self._device)

        num_search = query_coords.shape[0]
        results = torch.empty(num_search, dtype=torch.int32, device=self._device)

        search_mode = int(mode)
        if self._use_double_hash and search_mode == 0:
            search_mode = 1

        _C.cuhash.packed_search(
            self._keys,
            self._values,
            query_coords,
            results,
            num_search,
            self._capacity,
            search_mode,
        )
        return results
