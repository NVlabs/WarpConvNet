# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Operations for converting between point and grid features.
Primarily used for the Factorized Implicit Grid (FIG) approach.
"""

from typing import Union

import torch
import torch.nn.functional as F

from warpconvnet.geometry.features.grid import GridMemoryFormat
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.geometry.types.factor_grid import FactorGrid
from warpconvnet.geometry.types.points import Points


def grid_to_points(
    grid: Union[Grid, FactorGrid],
    points: Points,
    mode: str = "bilinear",
) -> Points:
    """Convert grid features to point features.

    Args:
        grid: Input grid (either single or factorized)
        points: Target point geometry
        mode: Interpolation mode ('bilinear', 'nearest')

    Returns:
        Points: Point features with interpolated values
    """
    # If we have a factorized grid, use the first one for sampling
    # (in the future, could average results from all grids)
    if isinstance(grid, FactorGrid):
        grid = grid[0]

    batch_size = grid.batch_size
    device = grid.device
    H, W, D = grid.grid_shape

    # Get normalized coordinates in the range [-1, 1]
    point_coords = points.batched_coordinates.batched_tensor
    point_offsets = points.batched_coordinates.offsets
    grid_min = grid.grid_coords.min_bound
    grid_max = grid.grid_coords.max_bound

    # Convert grid features to channels-first format for grid_sample
    if grid.memory_format == GridMemoryFormat.b_x_y_z_c:
        # Need to convert to channels-first format for grid_sample
        grid_features = grid.grid_features.batched_tensor.permute(0, 4, 1, 2, 3)
    elif grid.memory_format == GridMemoryFormat.b_c_x_y_z:
        grid_features = grid.grid_features.batched_tensor
    else:
        # For factorized formats, first convert to standard format
        grid_features = grid.grid_features.to_standard_format()
        grid_features = grid_features.permute(0, 4, 1, 2, 3)

    # Initialize output point features
    out_point_features = torch.zeros_like(points.batched_features.batched_tensor)

    # Process each batch separately since the number of points per batch may vary
    for b in range(batch_size):
        start_idx = point_offsets[b].item()
        end_idx = point_offsets[b + 1].item()

        # Skip empty batches
        if start_idx == end_idx:
            continue

        # Get point coordinates for this batch
        batch_coords = point_coords[start_idx:end_idx]

        # Normalize to [-1, 1] for grid_sample
        batch_norm_coords = (
            2.0 * (batch_coords - grid_min) / (grid_max - grid_min) - 1.0
        )

        # Reshape for grid_sample
        batch_norm_coords = batch_norm_coords.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 3]

        # Use grid_sample to interpolate features
        sampled_features = F.grid_sample(
            grid_features[b : b + 1],  # [1, C, H, W, D]
            batch_norm_coords.unsqueeze(0),  # [1, N, 1, 1, 3]
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )  # [1, C, N, 1, 1]
        # Assert the dimensions are correct
        assert sampled_features.shape == (
            1,
            grid.num_channels,
            end_idx - start_idx,
            1,
            1,
        )

        # Reshape to [N, C]
        batch_point_features = (
            sampled_features.squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)
        )

        # Set output features
        out_point_features[start_idx:end_idx] = batch_point_features

    # Create new Points geometry with interpolated features
    return points.replace(batched_features=out_point_features)
