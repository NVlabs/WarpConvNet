# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-table backend for bilateral grid + permutohedral lattice.

Routes (V, key_dim) integer keys with key_dim in [1, 7] and coords in
[-65536, 65535] to PackedHashTable128. Smaller key_dim is zero-padded to D=7
on insert and search. Out-of-range key_dim or coord raises — there is no
fallback hash table in warpconvnet (TorchHashTable was removed).
"""
from __future__ import annotations

import torch
from torch import Tensor

from warpconvnet.geometry.coords.search.packed128_hashmap import PackedHashTable128


_PACKED_DIM = PackedHashTable128.DIM  # 7
_PACKED_MIN = PackedHashTable128.COORD_MIN
_PACKED_MAX = PackedHashTable128.COORD_MAX


class _Packed128Adapter:
    """Surface used by BilateralGrid / PermutohedralLattice.

    Pads (V, key_dim) inputs to (V, 7) with trailing zeros so a single D=7
    instantiation serves all key_dim in [1, 7].
    """

    def __init__(self, packed: PackedHashTable128, key_dim: int):
        self._t = packed
        self._key_dim = key_dim

    def _pad(self, keys: Tensor) -> Tensor:
        if self._key_dim == _PACKED_DIM:
            return keys.contiguous()
        n = keys.shape[0]
        pad = torch.zeros(
            (n, _PACKED_DIM - self._key_dim),
            dtype=keys.dtype,
            device=keys.device,
        )
        return torch.cat([keys, pad], dim=1).contiguous()

    def search(self, keys: Tensor) -> Tensor:
        return self._t.search(self._pad(keys))

    def batched_search(self, queries: Tensor, offsets: Tensor) -> Tensor:
        return self._t.batched_search(self._pad(queries), self._pad(offsets))


def make_hash_table(
    unique_keys: Tensor,
    *,
    device: torch.device,
    capacity: int,
) -> _Packed128Adapter:
    """Build a hash table over (V, key_dim) int32 keys.

    Returns an object exposing
        .search(keys: int32 [N, key_dim]) -> int32 [N]
        .batched_search(queries [M, key_dim], offsets [K, key_dim]) -> int32 [K, M]
    """
    key_dim = int(unique_keys.shape[1])
    if not (1 <= key_dim <= _PACKED_DIM):
        raise ValueError(
            f"PackedHashTable128 supports key_dim in [1, {_PACKED_DIM}]; got {key_dim}"
        )

    if unique_keys.shape[0] > 0:
        mn = int(unique_keys.min().item())
        mx = int(unique_keys.max().item())
        if mn < _PACKED_MIN or mx > _PACKED_MAX:
            raise ValueError(
                f"Key out of range [{_PACKED_MIN}, {_PACKED_MAX}]: got [{mn}, {mx}]. "
                f"Reduce coordinate magnitude or rescale positions before hashing."
            )

    if key_dim < _PACKED_DIM:
        pad = torch.zeros(
            (unique_keys.shape[0], _PACKED_DIM - key_dim),
            dtype=unique_keys.dtype,
            device=unique_keys.device,
        )
        padded = torch.cat([unique_keys, pad], dim=1).contiguous()
    else:
        padded = unique_keys.contiguous()

    packed = PackedHashTable128.from_keys(padded, device=device, capacity=capacity)
    return _Packed128Adapter(packed, key_dim)
