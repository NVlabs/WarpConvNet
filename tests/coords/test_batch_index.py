# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.coords.ops.batch_index import (
    batch_index_from_indices,
    batch_index_from_offset,
    batch_indexed_coordinates,
    offsets_from_batch_index,
)


@pytest.fixture
def setup_points():
    """Setup test points with random coordinates."""
    torch.manual_seed(0)

    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features)


@pytest.mark.benchmark(group="batch_index")
def test_batch_index_from_offset(setup_points, benchmark):
    """Benchmark batch index generation from offsets."""
    points = setup_points
    offsets = points.offsets

    result = benchmark.pedantic(
        lambda: batch_index_from_offset(offsets),
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="batch_index")
def test_batch_indexed_coordinates(setup_points, benchmark):
    """Benchmark coordinate indexing with batch indices."""
    points = setup_points
    device = "cuda:0"
    points = points.to(device)
    offsets = points.offsets

    result = benchmark.pedantic(
        lambda: batch_indexed_coordinates(points.coordinate_tensor, offsets),
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="batch_index")
def test_offsets_from_batch_index(setup_points, benchmark):
    """Benchmark offset generation from batch indices."""
    points = setup_points
    device = "cuda:0"
    offsets = points.offsets.to(device)
    batch_index = batch_index_from_offset(offsets).to(device)

    result = benchmark.pedantic(
        lambda: offsets_from_batch_index(batch_index),
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )

    # Verify results
    gen_offsets = offsets_from_batch_index(batch_index)
    assert gen_offsets.equal(offsets.cpu())


def test_batch_index_from_indices(setup_points):
    """Test batch index generation from indices."""
    points: Points = setup_points
    device = "cuda:0"
    offsets = points.offsets.to(device)
    tot_N = len(points)

    batch_index = batch_index_from_offset(offsets)
    indices = torch.randint(0, tot_N, (100,))
    sel_batch_index = batch_index[indices]

    pred_batch_index = batch_index_from_indices(indices, offsets, device=device)
    assert torch.allclose(sel_batch_index, pred_batch_index.to(sel_batch_index))
