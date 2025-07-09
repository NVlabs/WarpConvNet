# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FactorGrid functional operations.

This module tests all functional operations in warpconvnet.nn.functional.factor_grid
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from warpconvnet.geometry.features.grid import GridMemoryFormat, GridFeatures
from warpconvnet.geometry.types.factor_grid import FactorGrid, points_to_factor_grid
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.coords.real import RealCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.nn.functional.factor_grid import (
    factor_grid_transform,
    factor_grid_cat,
    factor_grid_pool,
    factor_grid_intra_communication,
)


@pytest.fixture
def setup_factor_grid_geometry():
    """Create a sample FactorGrid for testing."""
    # Create sample points
    batch_size = 2
    num_points_per_batch = [100, 150]
    total_points = sum(num_points_per_batch)

    # Generate random coordinates
    coords = torch.randn(total_points, 3) * 2.0  # Scale to [-4, 4] range
    features = torch.randn(total_points, 8)  # 8 feature channels

    # Create offsets for batching
    offsets = torch.tensor(
        [0] + [sum(num_points_per_batch[: i + 1]) for i in range(batch_size)]
    )

    # Create Points geometry
    points = Points(
        batched_coordinates=RealCoords(coords, offsets),
        batched_features=CatFeatures(features, offsets),
    )

    # Convert to FactorGrid
    grid_shapes = [(4, 8, 16), (8, 4, 16), (8, 16, 4)]
    memory_formats = [
        GridMemoryFormat.b_zc_x_y,
        GridMemoryFormat.b_xc_y_z,
        GridMemoryFormat.b_yc_x_z,
    ]

    factor_grid = points_to_factor_grid(
        points,
        grid_shapes,
        memory_formats,
        search_radius=0.2,
        search_type="radius",
        reduction="mean",
    )

    return factor_grid, points


@pytest.fixture
def simple_factor_grid():
    """Create a simple FactorGrid for basic testing."""
    batch_size = 2
    num_channels = 3
    # Create simple grids with known shapes
    grids = []
    bounds = (torch.tensor([-2.0, -1.0, -1.0]), torch.tensor([2.0, 1.0, 1.0]))

    for i, (shape, fmt) in enumerate(
        [
            ((2, 4, 4), GridMemoryFormat.b_zc_x_y),
            ((4, 2, 4), GridMemoryFormat.b_xc_y_z),
            ((4, 4, 2), GridMemoryFormat.b_yc_x_z),
        ]
    ):
        grid = Grid.from_shape(
            grid_shape=shape,
            num_channels=num_channels,
            memory_format=fmt,
            bounds=bounds,
            batch_size=batch_size,
        )

        grids.append(grid)

    return FactorGrid(grids)


class TestFactorGridTransform:
    """Test factor_grid_transform functional operation."""

    def test_transform_basic(self, simple_factor_grid):
        """Test basic transform operation."""
        # Create a simple transform (ReLU)
        transform_fn = lambda x: F.relu(x)  # noqa: E731

        # Apply transform
        result = factor_grid_transform(simple_factor_grid, transform_fn, in_place=False)

        # Check that result is a FactorGrid
        assert isinstance(result, FactorGrid)
        assert len(result) == len(simple_factor_grid)

        # Check that negative values are zeroed out
        for i, grid in enumerate(result):
            features = grid.grid_features.batched_tensor
            assert torch.all(features >= 0), f"Grid {i} has negative values after ReLU"

    def test_transform_in_place(self, simple_factor_grid):
        """Test in-place transform operation."""
        original_grids = [
            grid.grid_features.batched_tensor.clone() for grid in simple_factor_grid
        ]

        # Apply transform in-place
        transform_fn = lambda x: x * 2.0  # noqa: E731
        result = factor_grid_transform(simple_factor_grid, transform_fn, in_place=True)

        # Check that features were modified
        for i, (original, result_grid) in enumerate(zip(original_grids, result)):
            result_features = result_grid.grid_features.batched_tensor
            expected = original * 2.0
            assert torch.allclose(
                result_features, expected, atol=1e-6
            ), f"Grid {i} transform failed"

    def test_transform_not_in_place(self, simple_factor_grid):
        """Test not-in-place transform operation."""
        original_grids = [
            grid.grid_features.batched_tensor.clone() for grid in simple_factor_grid
        ]

        # Apply transform not in-place
        transform_fn = lambda x: x * 3.0  # noqa: E731
        result = factor_grid_transform(simple_factor_grid, transform_fn, in_place=False)

        # Check that original wasn't modified and result is correct
        for i, (original, original_grid, result_grid) in enumerate(
            zip(original_grids, simple_factor_grid, result)
        ):
            # Original should be unchanged
            current_original = original_grid.grid_features.batched_tensor
            assert torch.allclose(
                current_original, original, atol=1e-6
            ), f"Grid {i} was modified in-place"

            # Result should be transformed
            result_features = result_grid.grid_features.batched_tensor
            expected = original * 3.0
            assert torch.allclose(
                result_features, expected, atol=1e-6
            ), f"Grid {i} transform failed"


class TestFactorGridCat:
    """Test factor_grid_cat functional operation."""

    def test_cat_basic(self, simple_factor_grid):
        """Test basic concatenation operation."""
        # Create a second FactorGrid with same structure but different features
        factor_grid2 = FactorGrid(
            [
                grid.replace(
                    batched_features=torch.randn_like(grid.grid_features.batched_tensor)
                )
                for grid in simple_factor_grid
            ]
        )

        # Concatenate
        result = factor_grid_cat(simple_factor_grid, factor_grid2)

        # Check result structure
        assert isinstance(result, FactorGrid)
        assert len(result) == len(simple_factor_grid)

        # Check that channels were doubled
        for original_grid, result_grid in zip(simple_factor_grid, result):
            original_channels = original_grid.num_channels
            result_channels = result_grid.num_channels
            assert (
                result_channels == original_channels * 2
            ), "Channels not properly concatenated"

    def test_cat_different_lengths(self, simple_factor_grid):
        """Test concatenation with different lengths should fail."""
        # Create FactorGrid with different number of grids
        shorter_grid = FactorGrid(simple_factor_grid.grids[:2])

        with pytest.raises(AssertionError, match="FactorGrid lengths must match"):
            factor_grid_cat(simple_factor_grid, shorter_grid)

    def test_cat_memory_formats(self, simple_factor_grid):
        """Test concatenation preserves memory formats."""
        factor_grid2 = FactorGrid(
            [
                grid.replace(
                    batched_features=torch.randn_like(grid.grid_features.batched_tensor)
                )
                for grid in simple_factor_grid
            ]
        )

        result = factor_grid_cat(simple_factor_grid, factor_grid2)

        # Check memory formats are preserved
        for original_grid, result_grid in zip(simple_factor_grid, result):
            assert result_grid.memory_format == original_grid.memory_format


class TestFactorGridPool:
    """Test factor_grid_pool functional operation."""

    @pytest.mark.parametrize("pooling_type", ["max", "mean"])
    def test_pool_basic(self, simple_factor_grid, pooling_type):
        """Test basic pooling operation."""
        result = factor_grid_pool(
            simple_factor_grid,
            pooling_type=pooling_type,
        )

        # Check output shape
        batch_size = simple_factor_grid[0].batch_size
        # Total channels = sum of effective channels from all grids
        total_channels = 0
        for grid in simple_factor_grid:
            if grid.memory_format == GridMemoryFormat.b_zc_x_y:
                total_channels += grid.grid_features.batched_tensor.shape[1]  # Z*C
            elif grid.memory_format == GridMemoryFormat.b_xc_y_z:
                total_channels += grid.grid_features.batched_tensor.shape[1]  # X*C
            elif grid.memory_format == GridMemoryFormat.b_yc_x_z:
                total_channels += grid.grid_features.batched_tensor.shape[1]  # Y*C

        assert isinstance(result, torch.Tensor)
        assert result.shape == (batch_size, total_channels)

    def test_pool_attention(self, simple_factor_grid):
        """Test attention pooling."""
        # Create attention layer
        # Need to determine feature dimensions first
        sample_features = simple_factor_grid[0].grid_features.batched_tensor
        if simple_factor_grid[0].memory_format == GridMemoryFormat.b_zc_x_y:
            feature_dim = sample_features.shape[1]  # Z*C
        elif simple_factor_grid[0].memory_format == GridMemoryFormat.b_xc_y_z:
            feature_dim = sample_features.shape[1]  # X*C
        elif simple_factor_grid[0].memory_format == GridMemoryFormat.b_yc_x_z:
            feature_dim = sample_features.shape[1]  # Y*C

        attention_layer = nn.MultiheadAttention(
            feature_dim, num_heads=2, batch_first=True
        )

        result = factor_grid_pool(
            simple_factor_grid,
            pooling_type="attention",
            attention_layer=attention_layer,
        )

        # Check output
        batch_size = simple_factor_grid[0].batch_size
        total_channels = sum(
            grid.grid_features.batched_tensor.shape[1] for grid in simple_factor_grid
        )
        assert result.shape == (batch_size, total_channels)


class TestFactorGridIntraCommunication:
    """Test factor_grid_intra_communication functional operation."""

    @pytest.mark.parametrize("communication_type", ["sum", "mul"])
    def test_intra_communication_basic(self, simple_factor_grid, communication_type):
        """Test basic intra-communication."""

        factor_grid = simple_factor_grid

        # Update the grid values to random values
        for grid in factor_grid:
            grid.grid_features.batched_tensor.random_()

        # Apply intra-communication
        result = factor_grid_intra_communication(factor_grid, communication_type)

        # Check result structure
        assert isinstance(result, FactorGrid)
        assert len(result) == len(factor_grid)

        # Check that features were modified (communication occurred)
        for i, (original_grid, result_grid) in enumerate(zip(factor_grid, result)):
            original_features = original_grid.grid_features.batched_tensor
            result_features = result_grid.grid_features.batched_tensor

            # Features should be different after communication
            assert not torch.allclose(
                original_features, result_features
            ), f"Grid {i} features unchanged after communication"

    def test_intra_communication_single_grid(self, simple_factor_grid):
        """Test intra-communication with single grid should return unchanged."""
        single_grid = FactorGrid([simple_factor_grid.grids[0]])

        result = factor_grid_intra_communication(single_grid, "sum")

        # Should return the same FactorGrid
        assert len(result) == 1
        original_features = single_grid[0].grid_features.batched_tensor
        result_features = result[0].grid_features.batched_tensor
        assert torch.allclose(original_features, result_features)

    def test_intra_communication_invalid_type(self, simple_factor_grid):
        """Test invalid communication type should raise error."""
        with pytest.raises(ValueError, match="Unknown communication type"):
            factor_grid_intra_communication(simple_factor_grid, "invalid")


class TestFactorGridIntraCommunications:
    """Test factor_grid_intra_communications functional operation."""

    def test_multiple_communications_single(self, simple_factor_grid):
        """Test multiple communications with single type."""
        result = factor_grid_intra_communication(simple_factor_grid, ["sum"])

        assert isinstance(result, FactorGrid)
        assert len(result) == len(simple_factor_grid)

    def test_multiple_communications_two_types(self, simple_factor_grid):
        """Test multiple communications with two types."""
        result = factor_grid_intra_communication(simple_factor_grid, ["sum", "mul"])

        assert isinstance(result, FactorGrid)
        assert len(result) == len(simple_factor_grid)

        # Channels should be doubled due to concatenation
        for original_grid, result_grid in zip(simple_factor_grid, result):
            original_channels = original_grid.num_channels
            result_channels = result_grid.num_channels
            assert result_channels == original_channels * 2

    def test_multiple_communications_too_many(self, simple_factor_grid):
        """Test multiple communications with more than 2 types should fail."""
        with pytest.raises(
            NotImplementedError, match="More than 2 communication types"
        ):
            factor_grid_intra_communication(simple_factor_grid, ["sum", "mul", "div"])

    def test_multiple_communications_custom_cat(self, simple_factor_grid):
        """Test multiple communications with custom concatenation function."""
        custom_cat_called = []

        def custom_cat_fn(grid1, grid2):
            custom_cat_called.append(True)
            return factor_grid_cat(grid1, grid2)

        result = factor_grid_intra_communication(
            simple_factor_grid, ["sum", "mul"], cat_fn=custom_cat_fn
        )

        assert len(custom_cat_called) == 1, "Custom cat function should be called"
        assert isinstance(result, FactorGrid)


class TestFactorGridFunctionalIntegration:
    """Integration tests combining multiple functional operations."""

    def test_transform_then_cat(self, simple_factor_grid):
        """Test combining transform and concatenation."""
        # Transform first FactorGrid
        transformed1 = factor_grid_transform(
            simple_factor_grid, lambda x: F.relu(x), in_place=False
        )

        # Transform second FactorGrid differently
        transformed2 = factor_grid_transform(
            simple_factor_grid, lambda x: torch.sigmoid(x), in_place=False
        )

        # Concatenate results
        result = factor_grid_cat(transformed1, transformed2)

        assert isinstance(result, FactorGrid)
        assert len(result) == len(simple_factor_grid)

        # Check that channels doubled
        for original_grid, result_grid in zip(simple_factor_grid, result):
            assert result_grid.num_channels == original_grid.num_channels * 2

    def test_communication_then_pool(self, simple_factor_grid):
        """Test combining communication and pooling."""
        # Apply intra-communication
        communicated = factor_grid_intra_communication(simple_factor_grid, "sum")

        # Pool the results
        pooled = factor_grid_pool(communicated, pooling_type="mean")

        assert isinstance(pooled, torch.Tensor)
        batch_size = simple_factor_grid[0].batch_size
        # Calculate expected channels based on actual feature dimensions
        total_channels = sum(
            grid.grid_features.batched_tensor.shape[1] for grid in communicated
        )
        assert pooled.shape == (batch_size, total_channels)


class TestFactorGridFunctionalErrorCases:
    """Test error handling in functional operations."""

    def test_empty_factor_grid(self):
        """Test operations with empty FactorGrid."""
        empty_grid = FactorGrid([])

        # Transform should work with empty grid
        result = factor_grid_transform(empty_grid, lambda x: x * 2, in_place=False)
        assert len(result) == 0

        # Pool should work with empty grid
        pooled = factor_grid_pool(empty_grid, "mean")
        assert pooled.numel() == 0  # Should return empty tensor

    def test_mismatched_bounds(self):
        """Test communication with mismatched grid bounds."""
        # This test would check if grids with different bounds handle communication properly
        # Implementation depends on how bounds are handled in grid sampling
        pass

    def test_invalid_memory_format(self):
        """Test operations with unsupported memory formats."""
        # This would test error handling for unsupported memory formats
        # Implementation depends on the complete set of supported formats
        pass
