# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

import warpconvnet._C as _C

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod


@torch.no_grad()
def _radius_search_cuda(
    points: Float[Tensor, "N 3"],  # noqa: F821
    queries: Float[Tensor, "M 3"],  # noqa: F821
    radius: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Cell-list radius search using CUDA kernels.

    Returns:
        neighbor_index: [Q] (int32)
        neighbor_distance: [Q] (float32)
        neighbor_split: [M + 1] (int32)
    """
    N = points.shape[0]
    M = queries.shape[0]
    device = points.device
    cell_size = radius

    # Quantize points to grid cells
    cell_coords = torch.floor(points / cell_size).int()  # [N, 3]

    # Build cell hash table
    table = TorchHashTable.from_keys(cell_coords, device=device)

    # Get cell IDs for each point
    cell_ids = table.search(cell_coords)  # [N] unique cell ID per point

    # Sort points by cell to create cell-list
    sorted_order = torch.argsort(cell_ids)
    sorted_cell_ids = cell_ids[sorted_order]

    # Compute per-cell start/count using unique_consecutive
    unique_ids, counts = torch.unique_consecutive(sorted_cell_ids, return_counts=True)
    num_cells = table.num_entries

    # Build cell_starts and cell_counts arrays indexed by cell ID
    cell_starts = torch.zeros(num_cells, dtype=torch.int32, device=device)
    cell_counts = torch.zeros(num_cells, dtype=torch.int32, device=device)

    # Fill in starts and counts for cells that have points
    offsets = torch.zeros(len(counts), dtype=torch.int32, device=device)
    torch.cumsum(counts[:-1].int(), dim=0, out=offsets[1:])
    cell_starts[unique_ids.long()] = offsets
    cell_counts[unique_ids.long()] = counts.int()

    # Flatten the table_kvs and vector_keys for kernel access
    table_kvs_flat = table._table_kvs.reshape(-1).contiguous()
    vector_keys_flat = table._vector_keys[: table._num_entries].reshape(-1).contiguous()

    # Ensure contiguous float32 for points/queries
    points_f = points.float().contiguous()
    queries_f = queries.float().contiguous()
    sorted_order_i = sorted_order.int().contiguous()

    # Pass 1: count neighbors per query
    result_count = torch.zeros(M, dtype=torch.int32, device=device)
    _C.coords.radius_search_count(
        points_f,
        queries_f,
        sorted_order_i,
        cell_starts,
        cell_counts,
        table_kvs_flat,
        vector_keys_flat,
        result_count,
        N,
        M,
        num_cells,
        radius,
        cell_size,
        table.capacity,
        table.hash_method.value,
    )
    torch.cuda.synchronize(device)

    # Build offsets from counts
    neighbor_split = torch.zeros(M + 1, dtype=torch.int32, device=device)
    torch.cumsum(result_count, dim=0, out=neighbor_split[1:])
    total = int(neighbor_split[-1].item())

    if total == 0:
        return (
            torch.zeros(0, dtype=torch.int32, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            neighbor_split,
        )

    # Allocate output
    result_indices = torch.zeros(total, dtype=torch.int32, device=device)
    result_distances = torch.zeros(total, dtype=torch.float32, device=device)

    # Pass 2: write results
    _C.coords.radius_search_write(
        points_f,
        queries_f,
        sorted_order_i,
        cell_starts,
        cell_counts,
        table_kvs_flat,
        vector_keys_flat,
        neighbor_split,
        result_indices,
        result_distances,
        N,
        M,
        num_cells,
        radius,
        cell_size,
        table.capacity,
        table.hash_method.value,
    )
    torch.cuda.synchronize(device)

    return result_indices, result_distances, neighbor_split


@torch.no_grad()
def _radius_search_chunk(
    points: Float[Tensor, "N 3"],  # noqa: F821
    queries: Float[Tensor, "M 3"],  # noqa: F821
    radius: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Radius search for a single chunk of queries against all points (CPU fallback)."""
    # Compute pairwise distances: [M, N]
    dists = torch.cdist(queries, points)

    # Find neighbors within radius
    mask = dists <= radius

    # Get indices and distances for each query
    neighbor_indices = []
    neighbor_distances = []
    counts = []
    for i in range(queries.shape[0]):
        valid = mask[i]
        idx = valid.nonzero(as_tuple=True)[0]
        neighbor_indices.append(idx)
        neighbor_distances.append(dists[i, idx])
        counts.append(idx.shape[0])

    counts_tensor = torch.tensor(counts, dtype=torch.int32, device=queries.device)
    if len(neighbor_indices) > 0 and sum(counts) > 0:
        all_indices = torch.cat(neighbor_indices)
        all_distances = torch.cat(neighbor_distances)
    else:
        all_indices = torch.zeros(0, dtype=torch.long, device=queries.device)
        all_distances = torch.zeros(0, dtype=torch.float32, device=queries.device)

    return all_indices, all_distances, counts_tensor


@torch.no_grad()
def radius_search(
    points: Float[Tensor, "N 3"],  # noqa: F821
    queries: Float[Tensor, "M 3"],  # noqa: F821
    radius: float,
    grid_dim: Optional[int | Tuple[int, int, int]] = None,
    chunk_size: int = 4096,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        points: [N, 3]
        queries: [M, 3]
        radius: float
        grid_dim: Unused, kept for API compatibility
        chunk_size: Number of queries to process at a time for memory efficiency (CPU only)

    Returns:
        neighbor_index: [Q] (int32)
        neighbor_distance: [Q] (float32)
        neighbor_split: [M + 1] (int32)
    """
    assert points.is_contiguous(), "points must be contiguous"
    assert queries.is_contiguous(), "queries must be contiguous"

    if queries.device.type == "cuda":
        torch.cuda.set_device(queries.device)

    M = queries.shape[0]
    if M == 0:
        empty_idx = torch.zeros(0, dtype=torch.int32, device=queries.device)
        empty_dist = torch.zeros(0, dtype=torch.float32, device=queries.device)
        empty_split = torch.zeros(1, dtype=torch.int32, device=queries.device)
        return empty_idx, empty_dist, empty_split

    # Use CUDA cell-list kernel for GPU tensors
    if points.device.type == "cuda":
        return _radius_search_cuda(points, queries, radius)

    # CPU fallback: chunked brute-force
    all_indices = []
    all_distances = []
    all_counts = []

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk_queries = queries[start:end]
        idx, dist, counts = _radius_search_chunk(points, chunk_queries, radius)
        all_indices.append(idx)
        all_distances.append(dist)
        all_counts.append(counts)

    if len(all_indices) > 0:
        neighbor_index = torch.cat(all_indices)
        neighbor_distance = torch.cat(all_distances)
        neighbor_counts = torch.cat(all_counts)
    else:
        neighbor_index = torch.zeros(0, dtype=torch.long, device=queries.device)
        neighbor_distance = torch.zeros(0, dtype=torch.float32, device=queries.device)
        neighbor_counts = torch.zeros(0, dtype=torch.int32, device=queries.device)

    # Build split (offsets) from counts
    neighbor_split = torch.zeros(M + 1, dtype=torch.int32, device=queries.device)
    torch.cumsum(neighbor_counts, dim=0, out=neighbor_split[1:])

    return neighbor_index.int(), neighbor_distance, neighbor_split


def batched_radius_search(
    ref_positions: Float[Tensor, "N 3"],  # noqa: F821
    ref_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    query_positions: Float[Tensor, "M 3"],  # noqa: F821
    query_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    radius: float,
    grid_dim: Optional[int | Tuple[int, int, int]] = None,
) -> Tuple[Int[Tensor, "Q"], Float[Tensor, "Q"], Int[Tensor, "M + 1"]]:  # noqa: F821
    """
    Args:
        ref_positions: [N, 3]
        ref_offsets: [B + 1]
        query_positions: [M, 3]
        query_offsets: [B + 1]
        radius: float
        grid_dim: Unused, kept for API compatibility

    Returns:
        neighbor_index: [Q]
        neighbor_distance: [Q]
        neighbor_split: [B + 1]
    """
    assert isinstance(ref_positions, torch.Tensor) and isinstance(
        query_positions, torch.Tensor
    ), "Only torch.Tensor is supported for batched radius search"
    assert (
        ref_positions.device.type == "cuda" and query_positions.device.type == "cuda"
    ), "Only GPU is supported for batched radius search"

    B = len(ref_offsets) - 1
    assert B == len(query_offsets) - 1
    assert (
        ref_offsets[-1] == ref_positions.shape[0]
    ), f"Last offset {ref_offsets[-1]} != {ref_positions.shape[0]}"
    assert (
        query_offsets[-1] == query_positions.shape[0]
    ), f"Last offset {query_offsets[-1]} != {query_positions.shape[0]}"
    neighbor_index_list = []
    neighbor_distance_list = []
    neighbor_split_list = []
    split_offset = 0
    for b in range(B):
        neighbor_index, neighbor_distance, neighbor_split = radius_search(
            points=ref_positions[ref_offsets[b] : ref_offsets[b + 1]],
            queries=query_positions[query_offsets[b] : query_offsets[b + 1]],
            radius=radius,
            grid_dim=grid_dim,
        )
        neighbor_index_list.append(neighbor_index + ref_offsets[b])
        neighbor_distance_list.append(neighbor_distance)
        # if b is last, append all neighbor_split since the last element is the total count
        if b == B - 1:
            neighbor_split_list.append(neighbor_split + split_offset)
        else:
            neighbor_split_list.append(neighbor_split[:-1] + split_offset)

        split_offset += len(neighbor_index)

    # Neighbor index, Neighbor Distance, Neighbor Split
    return (
        torch.cat(neighbor_index_list).long(),
        torch.cat(neighbor_distance_list),
        torch.cat(neighbor_split_list).long(),
    )
