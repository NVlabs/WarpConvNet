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
    points_to_closest_voxel_mapping,
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
    return Points(coords, features, offsets, device="cuda")


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
    return Voxels(coords, features, offsets, voxel_size=voxel_size, origin=origin, device="cuda")


@pytest.mark.parametrize("reduction", ["mean", "max", "sum"])
@pytest.mark.parametrize("search_type", ["radius", "knn", "voxel"])
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


def test_points_to_closest_voxel_mapping():
    """Tests the CUDA accelerated points_to_closest_voxel_mapping function."""

    import warpconvnet._C as _C

    device = "cuda"

    # Create test data
    batch_sizes = [100, 150]
    coords_list = []
    features_list = []
    for batch_size in batch_sizes:
        # Generate points in [0, 2] range
        coords = torch.rand(batch_size, 3, device=device) * 2.0
        features = torch.rand(batch_size, 4, device=device)
        coords_list.append(coords)
        features_list.append(features)

    # Create Points object
    all_coords = torch.cat(coords_list, dim=0)
    all_features = torch.cat(features_list, dim=0)
    offsets = torch.tensor([0, batch_sizes[0], sum(batch_sizes)], dtype=torch.int64, device=device)
    points_obj = Points(all_coords, all_features, offsets, device=device)

    grid_shape = (4, 4, 4)
    bounds = (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([2.0, 2.0, 2.0]))

    if device == "cuda":
        # Test CUDA implementation directly with raw tensors
        offsets_int32 = offsets.to(torch.int32)
        grid_shape_tensor = torch.tensor(grid_shape, dtype=torch.int32)
        voxel_indices = _C.utils.points_to_closest_voxel_mapping(
            all_coords, offsets_int32, grid_shape_tensor, bounds[0], bounds[1]
        )

        # Verify results
        assert voxel_indices.shape[0] == all_coords.shape[0]
        assert voxel_indices.dtype == torch.int32
        assert voxel_indices.min() >= 0
        assert voxel_indices.max() < len(batch_sizes) * grid_shape_tensor.prod()

    # Test Python wrapper with Points object
    result = points_to_closest_voxel_mapping(points_obj, grid_shape, bounds)
    assert result.shape[0] == all_coords.shape[0]  # Number of points


def test_voxel_mapping_edge_cases():
    """Tests edge cases for voxel mapping."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import warpconvnet._C as _C

    # Test 1: Single point at origin
    points = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    grid_shape = torch.tensor([2, 2, 2], dtype=torch.int32)
    bounds_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    bounds_max = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )
    assert result[0] == 0  # Should map to first voxel

    # Test 2: Point at grid boundary
    points = torch.tensor([[2.0, 2.0, 2.0]], device="cuda", dtype=torch.float32)
    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )
    expected = 1 * 2 * 2 + 1 * 2 + 1  # Last voxel in 2x2x2 grid
    assert result[0] == expected

    # Test 3: Points outside bounds (should clamp)
    points = torch.tensor(
        [
            [-1.0, -1.0, -1.0],  # Below minimum
            [3.0, 3.0, 3.0],  # Above maximum
        ],
        device="cuda",
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2], device="cuda", dtype=torch.int32)

    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )
    assert result[0] == 0  # Clamped to first voxel
    assert result[1] == expected  # Clamped to last voxel


def test_voxel_mapping_multi_batch():
    """Tests voxel mapping with multiple batches."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import warpconvnet._C as _C

    # Create points for 3 batches
    batch1 = torch.rand(100, 3, device="cuda") * 2.0
    batch2 = torch.rand(150, 3, device="cuda") * 2.0
    batch3 = torch.rand(200, 3, device="cuda") * 2.0

    points = torch.cat([batch1, batch2, batch3], dim=0)
    offsets = torch.tensor([0, 100, 250, 450], device="cuda", dtype=torch.int32)

    grid_shape = torch.tensor([8, 8, 8], dtype=torch.int32)
    bounds_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    bounds_max = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )

    # Check that voxel indices are within expected ranges for each batch
    batch_voxel_size = 8 * 8 * 8

    # Batch 0 indices should be in [0, 512)
    batch0_indices = result[:100]
    assert batch0_indices.min() >= 0
    assert batch0_indices.max() < batch_voxel_size

    # Batch 1 indices should be in [512, 1024)
    batch1_indices = result[100:250]
    assert batch1_indices.min() >= batch_voxel_size
    assert batch1_indices.max() < 2 * batch_voxel_size

    # Batch 2 indices should be in [1024, 1536)
    batch2_indices = result[250:450]
    assert batch2_indices.min() >= 2 * batch_voxel_size
    assert batch2_indices.max() < 3 * batch_voxel_size


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_voxel_mapping_dtypes(dtype):
    """Tests voxel mapping with different data types."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import warpconvnet._C as _C

    points = torch.rand(100, 3, device="cuda", dtype=dtype) * 2.0
    offsets = torch.tensor([0, 100], device="cuda", dtype=torch.int32)
    grid_shape = torch.tensor([4, 4, 4], dtype=torch.int32)
    bounds_min = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    bounds_max = torch.tensor([2.0, 2.0, 2.0], dtype=dtype)

    # Should work with both float32 and float64
    result = _C.utils.points_to_closest_voxel_mapping(
        points, offsets, grid_shape, bounds_min, bounds_max
    )

    assert result.shape[0] == 100
    assert result.dtype == torch.int32
    assert result.min() >= 0
    assert result.max() < 64  # 4*4*4
