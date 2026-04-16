# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical kernel map search for large kernels (5^3+).

Two-level search: coarse probe builds a uint32 bitmask per query indicating
which coarse cells are occupied, then the fine search skips offsets whose
coarse cell is empty. For 7x7x7 this prunes ~70% of fine probes.

All allocation and kernel launches are fused into a single C++ host call
(_C.cuhash.hierarchical_kernel_map) to eliminate Python round-trip overhead.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.packed_hashmap import PackedHashTable
from warpconvnet.geometry.coords.search.search_results import IntSearchResult


def kernel_map_from_size_hierarchical(
    fine_ht: PackedHashTable,
    query_coords: Tensor,
    kernel_size: Tuple[int, ...],
    stride: int = 4,
    identity_map_index: Optional[int] = None,
) -> IntSearchResult:
    """Hierarchical kernel map: coarse probe then pruned fine search.

    All work (coarse table build, coarse probe, pruned fine search,
    postprocess count+scatter) is done in a single C++ call with zero
    Python round-trips between kernel launches.

    Args:
        fine_ht: PackedHashTable built from input coords.
        query_coords: int32 (M, 4) query coordinates.
        kernel_size: (kx, ky, kz) kernel dimensions.
        stride: Coarse table stride (must be power of 2, default 4).
    """
    query_coords = query_coords.contiguous().to(dtype=torch.int32, device=fine_ht.device)

    # TODO(perf): The fused C++ launcher `_C.cuhash.hierarchical_kernel_map`
    # rebuilds the coarse hash table on every call. `PackedHashTable._get_coarse`
    # already caches coarse tables per stride on the fine table, but the fused
    # launcher does not accept a prebuilt coarse keys/values pair. Add a new
    # C++ signature (e.g. `hierarchical_kernel_map_with_coarse`) that takes
    # `coarse_keys`, `coarse_values`, `coarse_capacity` and skip the internal
    # build when `fine_ht._coarse_cache` already has an entry for `stride`.
    # Until then, correctness is unaffected; we just miss a caching win.
    in_maps, out_maps, offsets, pair_table = _C.cuhash.hierarchical_kernel_map(
        fine_ht.keys_tensor,
        fine_ht.values_tensor,
        fine_ht._coords.contiguous(),
        query_coords,
        list(kernel_size),
        stride,
        fine_ht.capacity,
    )

    result = IntSearchResult(in_maps, out_maps, offsets, identity_map_index=identity_map_index)
    result._pair_table = pair_table
    return result
