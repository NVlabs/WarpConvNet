# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels


def _create_points_data(B: int, min_N: int, max_N: int, C: int, device: str = "cuda:0"):
    """Helper function to create points data with given parameters."""
    torch.manual_seed(0)
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.randint(0, 1000, (int(N.item()), 3)) for N in Ns]
    features = [torch.rand((int(N.item()), C)).requires_grad_() for N in Ns]
    points = Points(coords, features, device=device)
    return points


def _create_voxels_data(
    B: int, min_N: int, max_N: int, C: int, device: str = "cuda", voxel_size: float = 0.01
):
    """Helper function to create voxels data with given parameters."""
    torch.manual_seed(0)
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [(torch.rand((int(N.item()), 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((int(N.item()), C)) for N in Ns]
    return Voxels(coords, features, device=device).unique()


# Small fixtures for fast tests
@pytest.fixture
def setup_small_points():
    """Setup small test points for fast testing."""
    return _create_points_data(B=2, min_N=100, max_N=500, C=4, device="cuda:0")


@pytest.fixture
def setup_small_voxels():
    """Setup small test voxels for fast testing."""
    return _create_voxels_data(B=2, min_N=100, max_N=500, C=4, device="cuda", voxel_size=0.1)


# Medium fixtures for standard tests
@pytest.fixture
def setup_points():
    """Setup medium test points with random coordinates and features."""
    return _create_points_data(B=3, min_N=1000, max_N=10000, C=7, device="cuda:0")


@pytest.fixture
def setup_voxels():
    """Setup medium test voxels with random coordinates and features."""
    return _create_voxels_data(B=3, min_N=100000, max_N=1000000, C=7, device="cuda")


# Large fixtures for benchmarking
@pytest.fixture(scope="module")
def setup_large_voxels():
    """Setup large test voxels for benchmarking."""
    return _create_voxels_data(B=3, min_N=500000, max_N=1000000, C=64, device="cuda")


# Specialized fixtures
@pytest.fixture
def setup_geometries():
    """Setup test points and voxels with random coordinates."""
    torch.manual_seed(0)
    device = "cuda:0"

    # Generate random point cloud
    points = _create_points_data(B=3, min_N=1000, max_N=10000, C=7, device=device)

    # Convert to sparse voxels
    voxel_size = 0.01
    voxels = points.to_sparse(voxel_size)  # type: ignore[attr-defined]

    return points, voxels


@pytest.fixture
def setup_voxels_for_spconv(setup_test_data):
    """Setup voxels specifically for spconv testing."""
    raw_coords_list, features_list, Ns, C, _ = setup_test_data
    return Voxels(raw_coords_list, features_list)


@pytest.fixture
def setup_voxel_data():
    """Setup fixed coordinate and feature data for benchmarking."""
    torch.manual_seed(0)
    device = "cuda:0"

    # Fixed configuration for data generation
    B, min_N, max_N = 3, 100000, 1000000
    base_channels = 32  # Use maximum channel size for feature generation

    # Generate fixed coordinates and features
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    # default dtype is float32
    coords = [(torch.rand((int(N.item()), 3), device=device) / voxel_size).int() for N in Ns]
    features = [torch.randn((int(N.item()), base_channels), device=device) for N in Ns]

    return coords, features


# Sample fixtures for specific test cases
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
