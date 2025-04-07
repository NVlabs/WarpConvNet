# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.types.grid import GridMemoryFormat, Grid
from warpconvnet.geometry.types.conversion.to_grid import (
    points_to_grid,
    voxels_to_grid,
)
from warpconvnet.geometry.coords.grid import GridCoords
from warpconvnet.geometry.features.grid import GridFeatures


@pytest.fixture
def sample_points():
    """Creates a sample Points object for testing."""
    coords = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],  # Point for cell (0,0,0)
            [1.1, 1.1, 1.1],
            [1.9, 1.9, 1.9],  # Point for cell (1,1,1)
            [0.2, 1.2, 1.8],
            [1.8, 0.2, 1.2],  # Points for cells (0,1,1) and (1,0,1)
        ],
        dtype=torch.float32,
    )
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=torch.float32)
    # features = torch.arange(6, dtype=torch.float32).unsqueeze(1) + 1.0
    offsets = torch.tensor([0, 6], dtype=torch.int64)
    return Points(coords, features, offsets)


@pytest.fixture
def sample_voxels():
    """Creates a sample Voxels object for testing."""
    coords = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=torch.int32,
    )
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    offsets = torch.tensor([0, 4], dtype=torch.int64)
    voxel_size = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    return Voxels(coords, features, offsets, voxel_size=voxel_size, origin=origin)


@pytest.mark.parametrize("reduction", ["mean", "max", "sum"])
@pytest.mark.parametrize("search_type", ["radius", "knn"])
@pytest.mark.parametrize(
    "memory_format",
    [
        GridMemoryFormat.b_x_y_z_c,
        GridMemoryFormat.b_c_x_y_z,
        # Add other formats if needed and implemented correctly
    ],
)
def test_points_to_grid(sample_points, reduction, search_type, memory_format):
    """Tests the points_to_grid conversion."""
    grid_shape = (2, 2, 2)
    bounds = (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([2.0, 2.0, 2.0]))
    search_radius = 1.1  # Radius to capture nearest points
    k = 2  # Number of neighbors for knn

    grid: Grid = points_to_grid(
        points=sample_points,
        grid_shape=grid_shape,
        bounds=bounds,
        memory_format=memory_format,
        search_radius=search_radius,
        k=k,
        search_type=search_type,
        reduction=reduction,
    )

    assert isinstance(grid, Grid)
    assert grid.batch_size == sample_points.batch_size
    assert grid.num_channels == sample_points.num_channels
    assert grid.memory_format == memory_format
    assert grid.grid_shape == grid_shape
    assert grid.feature_tensor.device == sample_points.device
    assert grid.coordinate_tensor.device == sample_points.device

    # Check feature tensor shape based on memory format
    expected_shape = {
        GridMemoryFormat.b_x_y_z_c: (1, 2, 2, 2, 1),
        GridMemoryFormat.b_c_x_y_z: (1, 1, 2, 2, 2),
    }
    assert grid.feature_tensor.shape == expected_shape[memory_format]


@pytest.mark.parametrize(
    "memory_format",
    [
        GridMemoryFormat.b_x_y_z_c,
        GridMemoryFormat.b_c_x_y_z,
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "max", "sum"])
def test_voxels_to_grid(sample_voxels, memory_format, reduction):
    """Tests the voxels_to_grid conversion."""
    grid_shape = (2, 2, 2)  # Same shape as implied by voxel coords+origin+size
    grid_bounds = (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([2.0, 2.0, 2.0]))
    grid: Grid = voxels_to_grid(
        voxels=sample_voxels,
        grid_shape=grid_shape,
        grid_bounds=grid_bounds,
        memory_format=memory_format,
        search_radius=None,  # Use default radius search behavior from _points_to_grid_features
        search_type="radius",
        reduction=reduction,
    )

    assert isinstance(grid, Grid)
    assert grid.batch_size == sample_voxels.batch_size
    assert grid.num_channels == sample_voxels.num_channels
    assert grid.memory_format == memory_format
    assert grid.grid_shape == grid_shape
    assert grid.feature_tensor.device == sample_voxels.device
    assert grid.coordinate_tensor.device == sample_voxels.device

    # Check feature tensor shape
    expected_shape = {
        GridMemoryFormat.b_x_y_z_c: (1, 2, 2, 2, 1),
        GridMemoryFormat.b_c_x_y_z: (1, 1, 2, 2, 2),
    }
    assert grid.feature_tensor.shape == expected_shape[memory_format]

    # Convert grid features back to a predictable format for checking
    if memory_format == GridMemoryFormat.b_c_x_y_z:
        check_features = grid.feature_tensor.permute(0, 2, 3, 4, 1)
    else:  # Already b_x_y_z_c
        check_features = grid.feature_tensor

    # Check that other cells are zero
    assert check_features[0, 0, 0, 1] == 0.0
    assert check_features[0, 0, 1, 0] == 0.0
    assert check_features[0, 1, 0, 0] == 0.0
    assert check_features[0, 1, 1, 0] == 0.0
