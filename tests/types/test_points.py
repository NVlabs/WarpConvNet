# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import pytest
import torch

from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig, RealSearchMode
from warpconvnet.geometry.types.points import Points


@pytest.fixture
def setup_points():
    """Setup test points with random coordinates and features."""
    device = "cuda:0"
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features, device=device), Ns


def test_point_indexing(setup_points):
    """Test point collection indexing."""
    points, Ns = setup_points

    for i in range(len(Ns)):
        assert points[i].batched_coordinates.batch_size == 1
        assert points[i].batched_coordinates.batched_tensor.shape[0] == Ns[i]
        assert points[i].batched_features.batch_size == 1
        assert points[i].batched_features.batched_tensor.shape[0] == Ns[i]


def test_point_construction(setup_points):
    """Test point collection construction."""
    points, Ns = setup_points
    Ns_cumsum = Ns.cumsum(dim=0).tolist()

    # Test basic properties
    assert points.batched_coordinates.batch_size == len(Ns)
    assert torch.equal(points.batched_coordinates.offsets, torch.IntTensor([0] + Ns_cumsum))
    assert points.batched_coordinates.batched_tensor.shape == (Ns_cumsum[-1], 3)
    assert points.batched_features.batch_size == len(Ns)
    assert points.batched_features.batched_tensor.shape == (Ns_cumsum[-1], 7)

    # Test device movement
    device = torch.device("cuda:0")
    points_gpu = points.to(device)
    assert points_gpu.batched_coordinates.batched_tensor.device == device

    # Test construction from concatenated tensors
    coords = torch.cat([torch.rand((N, 3)) for N in Ns], dim=0)
    features = torch.cat([torch.rand((N, 7)) for N in Ns], dim=0)
    offsets = torch.IntTensor([0] + Ns_cumsum)
    points_cat = Points(coords, features, offsets=offsets)
    assert points_cat.batched_coordinates.batch_size == len(Ns)


def test_point_dataclass_serialization(setup_points):
    """Test dataclass serialization."""
    points, _ = setup_points
    points_downsampled = points.voxel_downsample(0.1)

    # Test dictionary conversion
    points_dict = dataclasses.asdict(points_downsampled)
    assert "_extra_attributes" in points_dict
    assert points_dict["_extra_attributes"]["voxel_size"] == 0.1

    # Test replacement
    points_replaced = dataclasses.replace(points_downsampled)
    assert "voxel_size" in points_replaced.extra_attributes


def test_radius_search(setup_points):
    """Test radius search functionality."""
    points, Ns = setup_points
    radius = 0.1

    search_config = RealSearchConfig(
        mode=RealSearchMode.RADIUS,
        radius=radius,
    )
    search_result = points.batched_coordinates.neighbors(search_config)

    assert isinstance(search_result, RealSearchResult)
    assert sum(Ns) == search_result.neighbor_row_splits.shape[0] - 1
    assert search_result.neighbor_row_splits[-1] == search_result.neighbor_indices.numel()


def test_knn_search(setup_points):
    """Test k-nearest neighbors search."""
    points, Ns = setup_points
    knn_k = 10

    search_config = RealSearchConfig(
        mode=RealSearchMode.KNN,
        knn_k=knn_k,
    )
    search_result = points.batched_coordinates.neighbors(search_config)

    assert isinstance(search_result, RealSearchResult)
    assert sum(Ns) == search_result.neighbor_row_splits.shape[0] - 1
    assert sum(Ns) * knn_k == search_result.neighbor_indices.numel()
    assert search_result.neighbor_row_splits[-1] == search_result.neighbor_indices.numel()


def test_voxel_downsample(setup_points):
    """Test voxel downsampling."""
    points, _ = setup_points
    voxel_size = 0.1

    downsampled = points.voxel_downsample(voxel_size)
    assert downsampled.batched_coordinates.batched_tensor.shape[1] == 3
    assert (
        downsampled.batched_features.batched_tensor.shape[0]
        == downsampled.batched_coordinates.batched_tensor.shape[0]
    )


def test_binary_operations(setup_points):
    """Test binary operations on points."""
    points, _ = setup_points

    # Test scalar operations
    points_add = points + 1
    assert torch.allclose(points.feature_tensor + 1, points_add.feature_tensor)

    points_mul = points * 2
    assert torch.allclose(points.feature_tensor * 2, points_mul.feature_tensor)

    points_pow = points**2
    assert torch.allclose(points.feature_tensor**2, points_pow.feature_tensor)

    # Test point-wise operations
    points_add_points = points + points_pow
    assert (
        points_add_points.batched_coordinates.batched_tensor.shape[0]
        == points.batched_coordinates.batched_tensor.shape[0]
    )
    assert (
        points_add_points.batched_features.batched_tensor.shape[0]
        == points.batched_features.batched_tensor.shape[0]
    )

    points_mul_points = points * points_pow
    assert (
        points_mul_points.batched_coordinates.batched_tensor.shape[0]
        == points.batched_coordinates.batched_tensor.shape[0]
    )
    assert (
        points_mul_points.batched_features.batched_tensor.shape[0]
        == points.batched_features.batched_tensor.shape[0]
    )


def test_from_coordinates():
    """Test point creation from coordinates."""
    B, N, D = 7, 100000, 3
    coords = [torch.rand(N, D) for _ in range(B)]

    points = Points.from_list_of_coordinates(coords, encoding_channels=10, encoding_range=1)
    assert points.batched_coordinates.batched_tensor.shape == (B * N, D)
    assert points.batched_features.batched_tensor.shape == (B * N, 10 * D)


def test_contiguous(setup_points):
    """Test making point coordinates and features contiguous."""
    points, _ = setup_points

    # Create a non-contiguous tensor by transposing then back
    non_contiguous_coords = points.coordinate_tensor.transpose(0, 1).transpose(0, 1)
    non_contiguous_features = points.feature_tensor.transpose(0, 1).transpose(0, 1)

    # Verify tensors are not contiguous after this operation
    if non_contiguous_coords.is_contiguous() and non_contiguous_features.is_contiguous():
        # If still contiguous, use another method to make non-contiguous
        non_contiguous_coords = non_contiguous_coords.expand_as(non_contiguous_coords)
        non_contiguous_features = non_contiguous_features.expand_as(non_contiguous_features)

    # Create a new points object with non-contiguous tensors
    from warpconvnet.geometry.coords.real import RealCoords
    from warpconvnet.geometry.features.cat import CatFeatures

    non_contiguous_points = Points(
        batched_coordinates=RealCoords(
            batched_tensor=non_contiguous_coords,
            offsets=points.offsets.clone(),
        ),
        batched_features=CatFeatures(
            batched_tensor=non_contiguous_features,
            offsets=points.offsets.clone(),
        ),
    )

    # Make contiguous
    contiguous_points = non_contiguous_points.contiguous()

    # Verify tensors are now contiguous
    assert contiguous_points.coordinate_tensor.is_contiguous()
    assert contiguous_points.feature_tensor.is_contiguous()

    # Verify data is the same
    assert torch.allclose(
        non_contiguous_points.coordinate_tensor, contiguous_points.coordinate_tensor
    )
    assert torch.allclose(non_contiguous_points.feature_tensor, contiguous_points.feature_tensor)


def test_extra_attributes():
    # Test for extra attribute in the Geometry base class
    # Critical for X.replace() to work
    device = "cuda:0"
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    points = Points(coords, features, device=device, test_attribute="test")
    # Add extra attribute
    replaced = points.replace(batched_features=points.batched_features + 1)
    # Check that the extra attribute is present
    assert replaced.extra_attributes["test_attribute"] == "test"
