# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""128-bit packed hash table for high-dimensional integer coordinate keys.

D up to 8 axes per key with CoordBits per axis (D * CoordBits <= 127).
Skeleton instantiation: D=7, CoordBits=17 (covers permutohedral d<=6 with d+1=7
axes and bilateral_grid d<=6). Lower-D callers pad to 7 with zero trailing
axes; pack/search behave identically.

Contract: callers MUST guarantee distinct keys. Insert path performs no
deduplication. Run torch.unique upstream when input may contain duplicates.

Concurrency: insert and search must run in distinct kernel launches on the
same stream. The kernel-boundary fence makes insert writes visible to search.
Concurrent insert+search produces undefined results.

Range check: gated by env var WARPCONVNET_DEBUG_HASH (truthy) since the check
costs two host syncs.
"""

import os
from typing import Optional, Union

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search._packed_base import PackedHashTableBase


def _debug_hash_enabled() -> bool:
    v = os.environ.get("WARPCONVNET_DEBUG_HASH", "")
    return v not in ("", "0", "false", "False")


class PackedHashTable128(PackedHashTableBase):
    """Hash table for D-dim integer coords packed into a 128-bit key.

    Currently exposes D=7, CoordBits=17. Per-axis range is [-65536, 65535];
    total coord bits = 119, leaving bit 127 for the validity flag and 8
    reserved bits in `hi` for future use (tombstones / type tags).

    Layout:
      keys: int64 [capacity, 2]   reinterpret-cast as PackedKey128 in C++
      values: int32 [capacity]    values[slot] = original insertion index, or -1

    Coord range: coords outside [-65536, 65535] silently truncate via 17-bit
    mask unless WARPCONVNET_DEBUG_HASH=1 is set; truncated keys can collide
    with distinct user inputs. Validate upstream when input range is uncertain.
    """

    DIM = 7
    COORD_BITS = 17
    COORD_MIN = -(1 << (COORD_BITS - 1))  # -65536
    COORD_MAX = (1 << (COORD_BITS - 1)) - 1  # 65535
    MAX_BATCHED_K = 32

    def __init__(
        self,
        capacity: int,
        device: Union[str, torch.device] = "cuda",
        key_dim: int = DIM,
    ):
        if not (1 <= key_dim <= self.DIM):
            raise ValueError(f"key_dim must be in [1, {self.DIM}]; got {key_dim}")
        super().__init__(capacity=capacity, device=device)
        self._key_dim = int(key_dim)

    @property
    def key_dim(self) -> int:
        """Logical key width seen by callers; padded to DIM=7 internally."""
        return self._key_dim

    def _pad_to_dim(self, coords: Tensor) -> Tensor:
        """Zero-pad trailing axes from key_dim to DIM=7 if needed."""
        if self._key_dim == self.DIM:
            return coords.contiguous()
        n = coords.shape[0]
        pad = torch.zeros(
            (n, self.DIM - self._key_dim),
            dtype=coords.dtype,
            device=coords.device,
        )
        return torch.cat([coords, pad], dim=1).contiguous()

    def _prepare_insert_coords(self, coords: Tensor) -> Tensor:
        assert (
            coords.ndim == 2 and coords.shape[1] == self._key_dim
        ), f"Expected (N, {self._key_dim}); got {tuple(coords.shape)}"
        coords = coords.to(dtype=torch.int32, device=self._device)
        if coords.shape[0] > 0 and _debug_hash_enabled():
            mn = coords.min().item()
            mx = coords.max().item()
            if mn < self.COORD_MIN or mx > self.COORD_MAX:
                raise ValueError(
                    f"Coord out of range [{self.COORD_MIN}, {self.COORD_MAX}]: "
                    f"got [{mn}, {mx}]"
                )
        return self._pad_to_dim(coords)

    def _prepare_search_coords(self, queries: Tensor) -> Tensor:
        assert (
            queries.ndim == 2 and queries.shape[1] == self._key_dim
        ), f"queries shape must be (M, {self._key_dim}); got {tuple(queries.shape)}"
        queries = queries.to(dtype=torch.int32, device=self._device)
        return self._pad_to_dim(queries)

    def _allocate_storage(self) -> None:
        self._keys = torch.empty((self._capacity, 2), dtype=torch.int64, device=self._device)
        self._values = torch.empty(self._capacity, dtype=torch.int32, device=self._device)

    def _run_prepare(self) -> None:
        _C.cuhash.packed128_prepare(self._keys, self._values, self._capacity)

    def _run_insert(self, coords: Tensor, num_keys: int, status_tensor: Tensor) -> None:
        _C.cuhash.packed128_insert_d7c17(
            self._keys,
            self._values,
            coords,
            num_keys,
            self._capacity,
            status_tensor,
        )

    def _run_search(self, queries: Tensor, results: Tensor, num_search: int, **kwargs) -> None:
        _C.cuhash.packed128_search_d7c17(
            self._keys,
            self._values,
            queries,
            results,
            num_search,
            self._capacity,
        )

    @classmethod
    def from_keys(
        cls,
        coords: Tensor,
        device: Union[str, torch.device] = "cuda",
        capacity: Optional[int] = None,
        key_dim: Optional[int] = None,
    ) -> "PackedHashTable128":
        """Build a hash table from a coord tensor (caller guarantees distinct rows).

        ``key_dim`` defaults to ``coords.shape[1]``; smaller key dims are
        zero-padded to DIM=7 internally on insert/search/batched_search.
        """
        target = torch.device(device)
        coords = coords.to(dtype=torch.int32, device=target)
        if key_dim is None:
            key_dim = int(coords.shape[1])
        if not (1 <= key_dim <= cls.DIM):
            raise ValueError(f"key_dim must be in [1, {cls.DIM}]; got {key_dim}")
        if coords.shape[1] != key_dim:
            raise ValueError(f"coords width {coords.shape[1]} != key_dim {key_dim}")
        n = coords.shape[0]
        cap = capacity if capacity is not None else max(16, n * 2)
        obj = cls(capacity=cap, device=target, key_dim=key_dim)
        obj.insert(coords)
        return obj

    def batched_search(self, queries: Tensor, offsets: Tensor) -> Tensor:
        """Batched (queries, offsets) -> [K, M] indices in a single kernel.

        For each (query q, offset o) pair, packs (q + o) and searches the
        table. Query coords loaded once per thread, offsets staged through
        shared memory, output written K-major:
            results[k * M + qidx] = values[slot] on hit, -1 on miss.

        Use case: permutohedral lattice / bilateral-grid blur, K = 2*(d+1).

        Args:
            queries: int32 [M, DIM] on CUDA.
            offsets: int32 [K, DIM] on CUDA, K in [1, MAX_BATCHED_K].

        Returns:
            int32 [K, M] on CUDA. -1 indicates miss.
        """
        assert self._keys is not None, "Call insert() first"
        assert (
            queries.ndim == 2 and queries.shape[1] == self._key_dim
        ), f"queries shape must be (M, {self._key_dim}); got {tuple(queries.shape)}"
        assert (
            offsets.ndim == 2 and offsets.shape[1] == self._key_dim
        ), f"offsets shape must be (K, {self._key_dim}); got {tuple(offsets.shape)}"

        queries = self._pad_to_dim(queries.to(dtype=torch.int32, device=self._device))
        offsets = self._pad_to_dim(offsets.to(dtype=torch.int32, device=self._device))

        M = queries.shape[0]
        K = offsets.shape[0]
        assert 1 <= K <= self.MAX_BATCHED_K, f"K={K} out of range [1, {self.MAX_BATCHED_K}]"

        results = torch.empty((K, M), dtype=torch.int32, device=self._device)
        if M == 0:
            return results

        _C.cuhash.packed128_batched_search_d7c17(
            self._keys,
            self._values,
            queries,
            offsets,
            results,
            M,
            K,
            self._capacity,
        )
        return results
