# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared base class for packed-key hash tables.

`PackedHashTable` (4D coords, uint64 keys) and `PackedHashTable128` (D<=7
coords, 128-bit keys) share insert/search skeletons, capacity/device state,
and accessor properties. The base class captures that shared structure;
subclasses override hooks for coord validation, storage allocation, and the
specific C-binding kernel calls.
"""

from typing import Optional, Union

import torch
from torch import Tensor


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


class PackedHashTableBase:
    """Skeleton for packed-key hash tables.

    Concrete subclasses fill in the kernel-binding hooks:
        _prepare_insert_coords(coords) -> Tensor
        _prepare_search_coords(queries) -> Tensor
        _allocate_storage()
        _run_prepare()
        _run_insert(coords, num_keys, status_tensor)
        _run_search(queries, results, num_search, **kwargs)
        _post_insert(coords)              # optional, default no-op

    Shared insert/search call these hooks; subclasses do not override
    insert() unless they need extra behavior.
    """

    def __init__(self, capacity: int, device: Union[str, torch.device] = "cuda"):
        self._capacity = _next_power_of_2(capacity)
        self._device = torch.device(device)
        self._keys: Optional[Tensor] = None
        self._values: Optional[Tensor] = None
        self._num_entries = 0

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
    def keys_tensor(self) -> Tensor:
        return self._keys

    @property
    def values_tensor(self) -> Tensor:
        return self._values

    @property
    def key_dim(self) -> int:
        raise NotImplementedError

    def _prepare_insert_coords(self, coords: Tensor) -> Tensor:
        raise NotImplementedError

    def _prepare_search_coords(self, queries: Tensor) -> Tensor:
        raise NotImplementedError

    def _allocate_storage(self) -> None:
        raise NotImplementedError

    def _run_prepare(self) -> None:
        raise NotImplementedError

    def _run_insert(self, coords: Tensor, num_keys: int, status_tensor: Tensor) -> None:
        raise NotImplementedError

    def _run_search(self, queries: Tensor, results: Tensor, num_search: int, **kwargs) -> None:
        raise NotImplementedError

    def _post_insert(self, coords: Tensor) -> None:
        """Hook for subclass-specific bookkeeping after a successful insert."""

    def insert(self, coords: Tensor) -> None:
        """Insert distinct integer coordinates into the table."""
        assert coords.is_cuda, "coords must be on CUDA"
        coords = self._prepare_insert_coords(coords)
        num_keys = coords.shape[0]
        assert (
            num_keys <= self._capacity // 2
        ), f"num_keys={num_keys} exceeds capacity/2={self._capacity // 2}"

        self._allocate_storage()
        self._run_prepare()
        status = torch.zeros(1, dtype=torch.int32, device=self._device)
        self._run_insert(coords, num_keys, status)
        if int(status.item()) != 0:
            raise RuntimeError(
                f"{type(self).__name__}.insert failed: hash table is full "
                f"(num_keys={num_keys}, capacity={self._capacity}). "
                f"Increase capacity or reduce load factor."
            )
        self._num_entries = num_keys
        self._post_insert(coords)

    def search(self, queries: Tensor, **kwargs) -> Tensor:
        """Search for keys in the table.

        Returns int32 (M,) with original insertion index, or -1 on miss.
        """
        assert self._keys is not None, "Call insert() first"
        queries = self._prepare_search_coords(queries)
        num_search = queries.shape[0]
        results = torch.empty(num_search, dtype=torch.int32, device=self._device)
        self._run_search(queries, results, num_search, **kwargs)
        return results
