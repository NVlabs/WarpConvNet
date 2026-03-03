# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
from warpconvnet.geometry.coords.search.continuous import (
    neighbor_search,
    RealSearchMode,
)
from warpconvnet.geometry.types.points import Points


@pytest.fixture
def setup_points():
    """Setup test points with random coordinates and features."""
    device = torch.device("cuda:0")
    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [10 * torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features, device=device)


def test_radius_search(setup_points):
    """Test radius search method."""
    device = torch.device("cuda:0")
    points: Points = setup_points.to(device)
    radius = 0.1

    search_config = RealSearchConfig(
        mode=RealSearchMode.RADIUS,
        radius=radius,
    )

    # Use same points as both query and reference points
    query_coords = points.coordinate_tensor
    query_offsets = points.offsets
    ref_coords = query_coords
    ref_offsets = query_offsets

    search_result: RealSearchResult = neighbor_search(
        query_coords, query_offsets, ref_coords, ref_offsets, search_config
    )

    # Basic validation
    assert search_result.neighbor_row_splits.shape[0] == len(query_coords) + 1
    assert search_result.neighbor_row_splits[-1] == len(search_result.neighbor_indices)


@pytest.mark.slow
def test_knn_search(setup_points):
    """Test k-nearest neighbors search method."""
    device = torch.device("cuda:0")
    points: Points = setup_points.to(device)
    k = 2

    search_config = RealSearchConfig(
        mode=RealSearchMode.KNN,
        knn_k=k,
    )

    # Use same points as both query and reference points
    query_coords = points.coordinate_tensor
    query_offsets = points.offsets
    ref_coords = query_coords
    ref_offsets = query_offsets

    search_result: RealSearchResult = neighbor_search(
        query_coords, query_offsets, ref_coords, ref_offsets, search_config
    )

    # Basic validation
    assert search_result.neighbor_row_splits.shape[0] == len(query_coords) + 1
    assert search_result.neighbor_indices.numel() == len(query_coords) * k


@pytest.mark.benchmark(group="neighbor_search")
@pytest.mark.parametrize("num_points", [10000, 100000, 1000000])
class TestNeighborSearchPerformance:
    def test_radius_search(self, benchmark: BenchmarkFixture, setup_points, num_points):
        device = torch.device("cuda:0")
        points: Points = setup_points.to(device)
        radius = 0.1
        search_config = RealSearchConfig(mode=RealSearchMode.RADIUS, radius=radius)

        # Use same points as both query and reference points
        query_coords = points.coordinate_tensor
        query_offsets = points.offsets
        ref_coords = query_coords
        ref_offsets = query_offsets

        def run_benchmark():
            return neighbor_search(
                query_coords, query_offsets, ref_coords, ref_offsets, search_config
            )

        benchmark.pedantic(run_benchmark, iterations=4, rounds=3, warmup_rounds=1)

    def test_knn_search(self, benchmark: BenchmarkFixture, setup_points, num_points):
        device = torch.device("cuda:0")
        points: Points = setup_points.to(device)
        k = 16
        search_config = RealSearchConfig(mode=RealSearchMode.KNN, knn_k=k)

        # Use same points as both query and reference points
        query_coords = points.coordinate_tensor
        query_offsets = points.offsets
        ref_coords = query_coords
        ref_offsets = query_offsets

        def run_benchmark():
            return neighbor_search(
                query_coords, query_offsets, ref_coords, ref_offsets, search_config
            )

        benchmark.pedantic(run_benchmark, iterations=4, rounds=3, warmup_rounds=1)
