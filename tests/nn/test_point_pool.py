# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.point_pool import REDUCTIONS, point_pool
from warpconvnet.nn.functional.point_unpool import FEATURE_UNPOOLING_MODE, point_unpool
from warpconvnet.nn.modules.point_pool import PointMaxPool, PointAvgPool, PointSumPool, PointUnpool
from warpconvnet.nn.modules.sparse_pool import PointToSparseWrapper
from warpconvnet.utils.ravel import ravel_multi_index_auto_shape


@pytest.fixture
def setup_points():
    """Setup test points with random coordinates."""
    wp.init()
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features).to(device)


def test_point_max_pool(setup_points):
    """Test max pooling with voxel downsampling."""
    pc: Points = setup_points
    device = pc.device

    pool = PointMaxPool(
        downsample_voxel_size=0.1,
        return_type="point",
    ).to(device)

    # Forward pass
    out = pool(pc)

    # Verify output
    assert isinstance(out, Points)
    assert out.num_channels == pc.num_channels
    assert len(out) < len(pc)  # Should be downsampled
    assert out.voxel_size == 0.1


def test_point_avg_pool(setup_points):
    """Test average pooling with point limit."""
    pc: Points = setup_points
    device = pc.device
    max_points = 1000

    pool = PointAvgPool(
        downsample_max_num_points=max_points,
        return_type="point",
    ).to(device)

    # Forward pass
    out = pool(pc)

    # Verify output
    assert isinstance(out, Points)
    assert out.num_channels == pc.num_channels
    assert len(out) <= max_points * pc.batch_size
    assert out.device == device


def test_point_sum_pool(setup_points):
    """Test sum pooling with neighbor search result."""
    pc: Points = setup_points
    device = pc.device

    pool = PointSumPool(
        downsample_voxel_size=0.1,
        return_type="point",
        return_neighbor_search_result=True,
    ).to(device)

    # Forward pass
    out, search_result = pool(pc)

    # Verify output
    assert isinstance(out, Points)
    assert out.num_channels == pc.num_channels
    assert len(out) < len(pc)  # Should be downsampled
    assert search_result is not None


@pytest.mark.parametrize(
    "unpooling_mode",
    [
        FEATURE_UNPOOLING_MODE.REPEAT,
    ],
)
def test_point_unpool(setup_points, unpooling_mode):
    """Test unpooling with different modes."""
    pc: Points = setup_points
    device = pc.device

    # First create pooled points
    pool = PointMaxPool(downsample_voxel_size=0.1).to(device)
    pooled_pc = pool(pc)

    # Create unpool layer
    unpool = PointUnpool(
        unpooling_mode=unpooling_mode,
        concat_unpooled_pc=False,
    ).to(device)

    # Forward pass
    out = unpool(pooled_pc, pc)

    # Verify output
    assert isinstance(out, Points)
    assert out.num_channels == pooled_pc.num_channels
    assert len(out) == len(pc)  # Should match original size
    assert out.device == device


def test_point_unpool_concat(setup_points):
    """Test unpooling with concatenated features."""
    pc: Points = setup_points
    device = pc.device

    # First create pooled points
    pool = PointMaxPool(downsample_voxel_size=0.1).to(device)
    pooled_pc = pool(pc)

    # Create unpool layer with concatenation
    unpool = PointUnpool(
        unpooling_mode=FEATURE_UNPOOLING_MODE.REPEAT,
        concat_unpooled_pc=True,
    ).to(device)

    # Forward pass
    out = unpool(pooled_pc, pc)

    # Verify output
    assert isinstance(out, Points)
    assert out.num_channels == pooled_pc.num_channels + pc.num_channels
    assert len(out) == len(pc)
    assert out.device == device
