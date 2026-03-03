# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.features.grid import GridMemoryFormat
from warpconvnet.geometry.coords.grid import GridCoords
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.geometry.types.conversion.to_grid import points_to_grid

NUM_CHANNELS = 7


@pytest.fixture
def setup_device():
    """Setup test device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def setup_point_geometry(setup_device):
    """Setup test point geometry."""
    device = setup_device
    B, min_N, max_N, C = 3, 1000, 10000, NUM_CHANNELS
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features, device=device), Ns


def test_grid_init(setup_device):
    """Test Grid initialization."""
    device = setup_device
    grid_shape = (4, 6, 8)
    batch_size = 2

    # Create from grid shape
    geometry = Grid.from_shape(
        grid_shape=grid_shape,
        num_channels=NUM_CHANNELS,
        memory_format=GridMemoryFormat.b_x_y_z_c,
        batch_size=batch_size,
        device=device,
    )

    # Check properties
    assert geometry.grid_shape == grid_shape
    assert geometry.num_channels == NUM_CHANNELS
    assert geometry.memory_format == GridMemoryFormat.b_x_y_z_c
    assert geometry.batch_size == batch_size
    assert geometry.device == device

    # Check device moving
    if torch.cuda.is_available():
        cpu_geometry = geometry.to(torch.device("cpu"))
        assert cpu_geometry.device == torch.device("cpu")
        assert geometry.device == device  # Original unchanged

    # Test memory format conversion
    for fmt in GridMemoryFormat:
        converted = geometry.to_memory_format(fmt)
        assert converted.memory_format == fmt
        assert converted.grid_shape == grid_shape
        assert converted.num_channels == NUM_CHANNELS


def test_grid_creation_with_tensor(setup_device):
    """Test creating Grid from raw tensors."""
    device = setup_device
    grid_shape = (4, 6, 8)
    num_channels = NUM_CHANNELS
    batch_size = 2
    H, W, D = grid_shape

    # Create offsets
    elements_per_batch = H * W * D
    offsets = torch.tensor([0, elements_per_batch, 2 * elements_per_batch], device=device)

    # Create grid coordinates
    coords = GridCoords.from_shape(
        grid_shape=grid_shape,
        batch_size=batch_size,
        device=device,
    )

    # Test with standard format tensor
    tensor = torch.rand(batch_size, H, W, D, num_channels, device=device)
    geometry = Grid(coords, tensor, memory_format=GridMemoryFormat.b_x_y_z_c)

    assert geometry.grid_shape == grid_shape
    assert geometry.num_channels == num_channels
    assert geometry.memory_format == GridMemoryFormat.b_x_y_z_c
    assert geometry.batch_size == batch_size

    # Test with factorized format tensor
    tensor = torch.rand(batch_size, D * num_channels, H, W, device=device)
    geometry = Grid(
        coords,
        tensor,
        memory_format=GridMemoryFormat.b_zc_x_y,
        grid_shape=grid_shape,
        num_channels=num_channels,
    )

    assert geometry.grid_shape == grid_shape
    assert geometry.num_channels == num_channels
    assert geometry.memory_format == GridMemoryFormat.b_zc_x_y
    assert geometry.batch_size == batch_size


def test_points_to_grid(setup_point_geometry):
    """Test points_to_grid function."""
    points, Ns = setup_point_geometry
    grid_shape = (4, 6, 8)

    # Test with different memory formats
    for memory_format in [GridMemoryFormat.b_x_y_z_c, GridMemoryFormat.b_c_x_y_z]:
        grid = points_to_grid(
            points,
            grid_shape,
            memory_format=memory_format,
            search_radius=0.1,
            search_type="radius",
            reduction="mean",
        )

        assert isinstance(grid, Grid)
        assert grid.memory_format == memory_format
        assert grid.grid_shape == grid_shape
        assert grid.num_channels == NUM_CHANNELS
        assert grid.batch_size == points.batch_size


def test_extra_attributes():
    # Test for extra attribute in the Geometry base class
    # Critical for X.replace() to work
    device = "cuda:0"
    grid_shape = (4, 6, 8)
    bounds = (torch.zeros(3), torch.ones(3))
    grid = Grid.from_shape(
        grid_shape, num_channels=NUM_CHANNELS, bounds=bounds, device=device, test_attribute="test"
    )
    # Add extra attribute
    replaced = grid.replace(batched_features=grid.features + 1)
    # Check that the extra attribute is present
    assert replaced.extra_attributes["test_attribute"] == "test"
    assert replaced.bounds == bounds
    # Check if it is_lazy
    assert not replaced.grid_coords.is_initialized
