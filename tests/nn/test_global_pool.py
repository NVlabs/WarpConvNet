# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.global_pool import global_pool


@pytest.fixture
def setup_geometries():
    """Setup test points and voxels with random coordinates."""
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    # Generate random point cloud
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    points = Points(coords, features).to(device)

    # Convert to sparse voxels
    voxel_size = 0.01
    voxels = points.to_voxels(voxel_size)

    return points, voxels


@pytest.mark.parametrize("reduce", ["max", "mean", "sum"])
def test_global_pool_points(setup_geometries, reduce):
    """Test global pooling on point clouds with different reduction methods."""
    points: Points = setup_geometries[0]

    # Perform global pooling
    pooled_points = global_pool(points, reduce=reduce)

    # Verify output properties
    assert pooled_points.batch_size == points.batch_size
    assert pooled_points.feature_tensor.shape[0] == points.batch_size
    assert pooled_points.num_channels == points.num_channels

    # Check that coordinates are properly reduced
    assert pooled_points.coordinate_tensor.shape[0] == points.batch_size


@pytest.mark.parametrize("reduce", ["max", "mean", "sum"])
def test_global_pool_voxels(setup_geometries, reduce):
    """Test global pooling on voxels with different reduction methods."""
    voxels: Voxels = setup_geometries[1]

    # Perform global pooling
    pooled_voxels = global_pool(voxels, reduce=reduce)

    # Verify output properties
    assert pooled_voxels.batch_size == voxels.batch_size
    assert pooled_voxels.feature_tensor.shape[0] == voxels.batch_size
    assert pooled_voxels.num_channels == voxels.num_channels

    # Check that coordinates are properly reduced
    assert pooled_voxels.coordinate_tensor.shape[0] == voxels.batch_size


def test_global_pool_consistency(setup_geometries):
    """Test consistency of global pooling between points and voxels."""
    points, voxels = setup_geometries

    # Pool both geometries
    pooled_points = global_pool(points, reduce="max")
    pooled_voxels = global_pool(voxels, reduce="max")

    # Verify consistent output shapes
    assert pooled_points.batch_size == pooled_voxels.batch_size
    assert pooled_points.num_channels == pooled_voxels.num_channels
    assert pooled_points.feature_tensor.shape == pooled_voxels.feature_tensor.shape
