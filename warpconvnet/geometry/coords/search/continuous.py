# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from jaxtyping import Float, Int


from torch import Tensor

from warpconvnet.geometry.coords.search.knn import batched_knn_search
from warpconvnet.geometry.coords.search.radius import batched_radius_search
from warpconvnet.geometry.coords.search.search_configs import (
    RealSearchConfig,
    RealSearchMode,
)
from warpconvnet.geometry.coords.search.search_results import RealSearchResult


def neighbor_search(
    ref_positions: Float[Tensor, "N 3"],  # noqa: F821
    ref_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    query_positions: Float[Tensor, "M 3"],  # noqa: F821
    query_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    search_args: RealSearchConfig,
) -> RealSearchResult:
    """
    Args:
        ref_coords: BatchedCoordinates
        query_coords: BatchedCoordinates
        search_args: NeighborSearchArgs
        grid_dim: Union[int, Tuple[int, int, int]]

    Returns:
        NeighborSearchReturn
    """
    if search_args.mode == RealSearchMode.RADIUS:
        assert (
            search_args.radius is not None
        ), "Radius must be provided for radius search"
        neighbor_index, neighbor_distance, neighbor_split = batched_radius_search(
            ref_positions=ref_positions,
            ref_offsets=ref_offsets,
            query_positions=query_positions,
            query_offsets=query_offsets,
            radius=search_args.radius,
            grid_dim=search_args.grid_dim,
        )
        return RealSearchResult(
            neighbor_index,
            neighbor_split,
        )

    elif search_args.mode == RealSearchMode.KNN:
        assert search_args.knn_k is not None, "knn_k must be provided for knn search"
        # M x K
        neighbor_index = batched_knn_search(
            ref_positions=ref_positions,
            ref_offsets=ref_offsets,
            query_positions=query_positions,
            query_offsets=query_offsets,
            k=search_args.knn_k,
        )
        return RealSearchResult(neighbor_index)

    else:
        raise ValueError(f"search_args.mode {search_args.mode} not supported.")
