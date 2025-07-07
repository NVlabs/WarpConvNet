# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.features.grid import GridMemoryFormat
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.geometry.types.factor_grid import FactorGrid, points_to_factor_grid

NUM_CHANNELS = 7


@pytest.fixture
def setup_device():
    """Setup test device."""
    wp.init()
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


def test_factorized_grid_init(setup_device):
    """Test FactorizedGrid initialization."""
    device = setup_device
    # Create a list of grid shapes
    grid_shapes = [(2, 32, 64), (16, 2, 64), (16, 32, 2)]
    batch_size = 2

    # Create from grid shape
    factorized = FactorGrid.create_from_grid_shape(
        grid_shapes,
        NUM_CHANNELS,
        memory_formats=[
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ],
        batch_size=batch_size,
        device=device,
    )

    # Check properties
    assert factorized.num_channels == NUM_CHANNELS
    assert factorized.batch_size == batch_size
    assert len(factorized) == 3

    # Check each geometry
    for i, fmt in enumerate(
        [GridMemoryFormat.b_zc_x_y, GridMemoryFormat.b_xc_y_z, GridMemoryFormat.b_yc_x_z]
    ):
        assert factorized[i].memory_format == fmt
        assert factorized[i].grid_shape == grid_shapes[i]
        assert factorized[i].num_channels == NUM_CHANNELS

    # Test get_by_format
    z_geometry = factorized.get_by_format(GridMemoryFormat.b_zc_x_y)
    assert z_geometry is not None
    assert z_geometry.memory_format == GridMemoryFormat.b_zc_x_y

    # Test with non-existent format
    standard_geometry = factorized.get_by_format(GridMemoryFormat.b_x_y_z_c)
    assert standard_geometry is None


def test_factorized_grid_validation(setup_device):
    """Test FactorizedGrid validation."""
    device = setup_device
    grid_shape = (4, 6, 8)
    num_channels = 16
    batch_size = 2

    # Create standard geometry
    standard_geo = Grid.from_shape(
        grid_shape,
        num_channels,
        memory_format=GridMemoryFormat.b_x_y_z_c,
        batch_size=batch_size,
        device=device,
    )

    # Create geometries with different formats
    geometries = [
        standard_geo.to_memory_format(GridMemoryFormat.b_zc_x_y),
        standard_geo.to_memory_format(GridMemoryFormat.b_xc_y_z),
        standard_geo.to_memory_format(GridMemoryFormat.b_yc_x_z),
    ]

    # Test valid creation
    factorized = FactorGrid(geometries)
    assert len(factorized) == 3

    # Test creating with duplicate formats - should raise an error
    geometries_with_duplicate = [
        standard_geo.to_memory_format(GridMemoryFormat.b_zc_x_y),
        standard_geo.to_memory_format(GridMemoryFormat.b_zc_x_y),  # Duplicate format
        standard_geo.to_memory_format(GridMemoryFormat.b_yc_x_z),
    ]

    with pytest.raises(AssertionError):
        FactorGrid(geometries_with_duplicate)

    # Test creating with non-factorized format - should raise an error
    geometries_with_standard = [
        standard_geo,  # Standard format (not factorized)
        standard_geo.to_memory_format(GridMemoryFormat.b_xc_y_z),
        standard_geo.to_memory_format(GridMemoryFormat.b_yc_x_z),
    ]

    with pytest.raises(AssertionError):
        FactorGrid(geometries_with_standard)


def test_factorized_grid_shapes(setup_device):
    """Test the shapes property of FactorizedGrid."""
    device = setup_device
    grid_shapes = [(2, 32, 64), (16, 2, 64), (16, 32, 2)]
    num_channels = NUM_CHANNELS
    batch_size = 2

    # Create factorized grid
    factorized = FactorGrid.create_from_grid_shape(
        grid_shapes,
        num_channels,
        memory_formats=[
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ],
        batch_size=batch_size,
        device=device,
    )

    # Check shapes property
    shapes = factorized.shapes
    assert len(shapes) == 3

    for shape, grid_shape in zip(shapes, grid_shapes):
        assert shape["grid_shape"] == grid_shape
        assert shape["batch_size"] == batch_size
        assert shape["num_channels"] == num_channels
        assert (
            shape["total_elements"] == grid_shape[0] * grid_shape[1] * grid_shape[2] * batch_size
        )


@pytest.mark.parametrize("search_type", ["radius", "knn", "voxel"])
@pytest.mark.parametrize("reduction", ["mean", "max", "sum"])
def test_points_to_factor_grid(setup_point_geometry, search_type, reduction):
    """Test points_to_factor_grid with different search types and reductions."""
    points, Ns = setup_point_geometry
    grid_shapes = [(4, 8, 16), (8, 4, 16), (8, 16, 4)]
    memory_formats = [
        GridMemoryFormat.b_zc_x_y,
        GridMemoryFormat.b_xc_y_z,
        GridMemoryFormat.b_yc_x_z,
    ]

    # Test with explicit bounds
    min_coords = points.coordinate_tensor.min(dim=0)[0]
    max_coords = points.coordinate_tensor.max(dim=0)[0]
    padding = (max_coords - min_coords) * 0.1
    bounds = (min_coords - padding, max_coords + padding)

    factorized = points_to_factor_grid(
        points,
        grid_shapes,
        memory_formats,
        bounds=bounds,
        search_radius=0.2,
        k=8,
        search_type=search_type,
        reduction=reduction,
    )

    # Basic structure checks
    assert isinstance(factorized, FactorGrid)
    assert len(factorized) == 3
    assert factorized.batch_size == points.batch_size
    assert factorized.num_channels == points.num_channels
    assert factorized.device == points.device

    # Check each grid
    for i, fmt in enumerate(memory_formats):
        grid = factorized[i]
        assert grid.memory_format == fmt
        assert grid.grid_shape == grid_shapes[i]
        assert grid.num_channels == points.num_channels
        assert grid.batch_size == points.batch_size

        # Check that features are not all zeros (should have some values from points)
        features = grid.grid_features.batched_tensor
        assert features.numel() > 0
        # For most reduction methods, we should have some non-zero values
        if reduction in ["mean", "sum"] and search_type != "voxel":
            # Note: voxel search might result in all zeros if points don't align well
            assert (
                features.abs().sum() > 0
            ), f"All features are zero for {fmt} with {search_type}-{reduction}"


def test_points_to_factor_grid_memory_format_strings(setup_point_geometry):
    """Test points_to_factor_grid with string memory format specifications."""
    points, Ns = setup_point_geometry
    grid_shapes = [(4, 8, 16), (8, 4, 16)]
    memory_formats_str = ["b_zc_x_y", "b_xc_y_z"]  # String versions
    memory_formats_enum = [GridMemoryFormat.b_zc_x_y, GridMemoryFormat.b_xc_y_z]  # Enum versions

    factorized_str = points_to_factor_grid(
        points,
        grid_shapes,
        memory_formats_str,
        search_radius=0.1,
        search_type="radius",
        reduction="mean",
    )

    factorized_enum = points_to_factor_grid(
        points,
        grid_shapes,
        memory_formats_enum,
        search_radius=0.1,
        search_type="radius",
        reduction="mean",
    )

    # Should produce identical results
    assert len(factorized_str) == len(factorized_enum)
    for i in range(len(factorized_str)):
        assert factorized_str[i].memory_format == factorized_enum[i].memory_format
        assert factorized_str[i].grid_shape == factorized_enum[i].grid_shape


def test_points_to_factor_grid_bounds_handling(setup_point_geometry):
    """Test points_to_factor_grid with different bounds settings."""
    points, Ns = setup_point_geometry
    grid_shapes = [(4, 8, 16)]
    memory_formats = [GridMemoryFormat.b_zc_x_y]

    # Test with None bounds (should auto-compute)
    factorized_auto = points_to_factor_grid(
        points,
        grid_shapes,
        memory_formats,
        bounds=None,
        search_radius=0.1,
        search_type="radius",
        reduction="mean",
    )

    # Test with explicit bounds
    min_coords = torch.tensor([0.0, 0.0, 0.0], device=points.device)
    max_coords = torch.tensor([1.0, 1.0, 1.0], device=points.device)
    bounds = (min_coords, max_coords)

    factorized_explicit = points_to_factor_grid(
        points,
        grid_shapes,
        memory_formats,
        bounds=bounds,
        search_radius=0.1,
        search_type="radius",
        reduction="mean",
    )

    # Both should work and produce valid FactorGrid objects
    assert isinstance(factorized_auto, FactorGrid)
    assert isinstance(factorized_explicit, FactorGrid)
    assert factorized_auto[0].grid_shape == factorized_explicit[0].grid_shape
    assert factorized_auto[0].memory_format == factorized_explicit[0].memory_format


def test_points_to_factor_grid_error_cases(setup_point_geometry):
    """Test points_to_factor_grid error handling."""
    points, Ns = setup_point_geometry

    # Test mismatched lengths
    grid_shapes = [(4, 8, 16), (8, 4, 16)]
    memory_formats = [GridMemoryFormat.b_zc_x_y]  # Different length

    with pytest.raises(AssertionError, match="must have the same length"):
        points_to_factor_grid(
            points,
            grid_shapes,
            memory_formats,
            search_radius=0.1,
            search_type="radius",
            reduction="mean",
        )


def test_strided_vertices(setup_device):
    """Test strided_vertices method for different resolutions."""
    device = setup_device
    original_grid_shape = (16, 32, 64)
    num_channels = NUM_CHANNELS
    batch_size = 2

    # Create a grid with a specific shape
    grid = Grid.from_shape(
        original_grid_shape,
        num_channels,
        memory_format=GridMemoryFormat.b_x_y_z_c,
        batch_size=batch_size,
        device=device,
    )

    # Test with exact divisible resolution (striding case)
    divisible_resolution = (8, 16, 32)  # Half of the original in each dimension
    # TODO(cchoy): Not supported yet
    return

    strided_vertices = grid.strided_vertices(divisible_resolution)

    # Check shape
    assert strided_vertices.shape[1:4] == divisible_resolution

    # Compare with manual striding
    original_vertices = grid.grid_coords.batched_tensor
    B, H, W, D = batch_size, *original_grid_shape
    original_reshaped = original_vertices.reshape(B, H, W, D, 3)
    manually_strided = original_reshaped[:, ::2, ::2, ::2, :]
    assert torch.allclose(strided_vertices, manually_strided)

    # Test with non-divisible resolution (interpolation case)
    non_divisible_resolution = (7, 15, 30)
    interpolated_vertices = grid.strided_vertices(non_divisible_resolution)

    # Check shape
    assert interpolated_vertices.shape[1:4] == non_divisible_resolution

    # Verify the interpolated vertices are within the bounds of the original
    # Get min and max of original vertices
    min_coords = original_vertices.reshape(-1, 3).min(dim=0)[0]
    max_coords = original_vertices.reshape(-1, 3).max(dim=0)[0]

    # Check min and max of interpolated vertices
    interp_min = interpolated_vertices.reshape(-1, 3).min(dim=0)[0]
    interp_max = interpolated_vertices.reshape(-1, 3).max(dim=0)[0]

    # Interpolated vertices should be within the bounds of the original vertices
    assert torch.all(interp_min >= min_coords)
    assert torch.all(interp_max <= max_coords)


def test_channel_size(setup_device):
    """Test channel_size method for different memory formats."""
    device = setup_device
    grid_shape = (4, 8, 16)
    num_channels = NUM_CHANNELS
    batch_size = 2

    # Create grid with standard memory format
    grid = Grid.from_shape(
        grid_shape,
        num_channels,
        memory_format=GridMemoryFormat.b_x_y_z_c,
        batch_size=batch_size,
        device=device,
    )

    # Test channel size for different memory formats
    # Standard formats should return the original channel count
    assert grid.grid_features.channel_size(GridMemoryFormat.b_x_y_z_c) == num_channels
    assert grid.grid_features.channel_size(GridMemoryFormat.b_c_x_y_z) == num_channels

    # Factorized formats should return channels * corresponding dimension
    assert (
        grid.grid_features.channel_size(GridMemoryFormat.b_xc_y_z) == num_channels * grid_shape[0]
    )
    assert (
        grid.grid_features.channel_size(GridMemoryFormat.b_yc_x_z) == num_channels * grid_shape[1]
    )
    assert (
        grid.grid_features.channel_size(GridMemoryFormat.b_zc_x_y) == num_channels * grid_shape[2]
    )

    # Convert to a factorized format and test again
    z_factorized = grid.to_memory_format(GridMemoryFormat.b_zc_x_y)
    assert z_factorized.grid_features.channel_size() == num_channels * grid_shape[2]
    assert z_factorized.grid_features.channel_size(GridMemoryFormat.b_x_y_z_c) == num_channels
