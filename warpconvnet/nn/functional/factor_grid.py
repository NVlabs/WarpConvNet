# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Literal, Optional, Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.features.grid import GridMemoryFormat
from warpconvnet.geometry.types.factor_grid import FactorGrid
from warpconvnet.geometry.types.grid import Grid

__all__ = [
    "factor_grid_transform",
    "factor_grid_cat",
    "factor_grid_pool",
    "factor_grid_intra_communication",
]


def factor_grid_transform(
    factor_grid: FactorGrid,
    transform_fn: Callable[[Tensor], Tensor],
) -> FactorGrid:
    """Apply a transform function to all grids in a FactorGrid.

    Args:
        factor_grid: Input FactorGrid
        transform_fn: Function to apply to each grid's features

    Returns:
        FactorGrid with transformed features
    """
    # Apply transform to each grid's features
    transformed_grids = []
    for grid in factor_grid:
        # Apply transform to features and create new grid
        transformed_features = transform_fn(grid.grid_features.batched_tensor)
        transformed_grid = grid.replace(batched_features=transformed_features)
        transformed_grids.append(transformed_grid)

    return FactorGrid(transformed_grids)


def factor_grid_cat(factor_grid1: FactorGrid, factor_grid2: FactorGrid) -> FactorGrid:
    """Concatenate features from two FactorGrid objects.

    Args:
        factor_grid1: First FactorGrid
        factor_grid2: Second FactorGrid

    Returns:
        FactorGrid with concatenated features
    """
    assert len(factor_grid1) == len(
        factor_grid2
    ), f"FactorGrid lengths must match: {len(factor_grid1)} != {len(factor_grid2)}"

    concatenated_grids = []
    for grid1, grid2 in zip(factor_grid1, factor_grid2):
        # Get features from both grids
        features1 = grid1.grid_features.batched_tensor
        features2 = grid2.grid_features.batched_tensor

        # Concatenate along channel dimension based on memory format
        if grid1.memory_format == GridMemoryFormat.b_x_y_z_c:
            # Channel is last dimension
            concatenated_features = torch.cat([features1, features2], dim=-1)
        elif grid1.memory_format == GridMemoryFormat.b_c_x_y_z:
            # Channel is second dimension
            concatenated_features = torch.cat([features1, features2], dim=1)
        elif grid1.memory_format in [
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ]:
            # For factorized formats, channel is combined with spatial dimension
            concatenated_features = torch.cat([features1, features2], dim=1)
        else:
            raise ValueError(f"Unsupported memory format: {grid1.memory_format}")

        # Create new grid with concatenated features
        concatenated_grid = grid1.replace(batched_features=concatenated_features)
        concatenated_grids.append(concatenated_grid)

    return FactorGrid(concatenated_grids)


def factor_grid_pool(
    factor_grid: FactorGrid,
    pooling_type: Literal["max", "mean", "attention"] = "max",
    pool_op: Optional[Callable] = None,
    attention_layer: Optional[Callable] = None,
) -> Tensor:
    """Pool features from FactorGrid to a single tensor.

    Args:
        factor_grid: Input FactorGrid
        pooling_type: Type of pooling ("max", "mean", "attention")
        pool_op: Pooling operation function
        attention_layer: Attention layer for attention pooling

    Returns:
        Pooled tensor of shape [B, total_channels]
    """
    pooled_features = []

    for grid in factor_grid:
        # Get the features tensor
        features = grid.grid_features.batched_tensor  # Shape depends on memory format
        fmt = grid.memory_format

        # Convert to appropriate format for pooling
        if fmt == GridMemoryFormat.b_zc_x_y:
            # Shape: B, Z*C, X, Y -> flatten spatial -> B, Z*C, X*Y
            B, ZC, X, Y = features.shape
            features_flat = features.view(B, ZC, -1)
        elif fmt == GridMemoryFormat.b_xc_y_z:
            # Shape: B, X*C, Y, Z -> flatten spatial -> B, X*C, Y*Z
            B, XC, Y, Z = features.shape
            features_flat = features.view(B, XC, -1)
        elif fmt == GridMemoryFormat.b_yc_x_z:
            # Shape: B, Y*C, X, Z -> flatten spatial -> B, Y*C, X*Z
            B, YC, X, Z = features.shape
            features_flat = features.view(B, YC, -1)
        else:
            raise ValueError(f"Unsupported memory format for pooling: {fmt}")

        # Apply pooling directly to flattened features
        if pooling_type in ["max", "mean"]:
            if pool_op is not None:
                pooled = pool_op(features_flat).squeeze(-1)  # B, channels
            elif pooling_type == "max":
                pooled = F.adaptive_max_pool1d(features_flat, 1).squeeze(-1)
            else:  # mean
                pooled = F.adaptive_avg_pool1d(features_flat, 1).squeeze(-1)
        elif pooling_type == "attention":
            if attention_layer is not None:
                # Convert to B, N, C for attention
                features_t = features_flat.transpose(1, 2)  # B, N, C
                attended, _ = attention_layer(features_t, features_t, features_t)
                pooled = attended.mean(dim=1)  # B, C
            else:
                # Fallback: simple mean pooling
                pooled = F.adaptive_avg_pool1d(features_flat, 1).squeeze(-1)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        pooled_features.append(pooled)

    # Concatenate pooled features from all grids
    return torch.cat(pooled_features, dim=-1)


def _factor_grid_intra_communication(
    factor_grid: FactorGrid,
    communication_type: Literal["sum", "mul"] = "sum",
) -> FactorGrid:
    """Perform intra-communication between grids in a FactorGrid.

    Args:
        factor_grid: Input FactorGrid
        communication_type: Type of communication ("sum" or "mul")

    Returns:
        FactorGrid with inter-grid communication applied
    """
    if len(factor_grid) == 1:
        # No communication needed for single grid
        return factor_grid

    # Convert all grids to standard format temporarily for grid sampling
    orig_grids = []
    standard_grids = []

    for grid in factor_grid:
        orig_grids.append(grid)
        # Convert to standard b_c_x_y_z format for grid_sample
        if grid.memory_format != GridMemoryFormat.b_c_x_y_z:
            standard_grid = grid.to_memory_format(GridMemoryFormat.b_c_x_y_z)
        else:
            standard_grid = grid
        standard_grids.append(standard_grid)

    # Assert all grids have the same number of channels
    num_channels = standard_grids[0].num_channels
    for grid in standard_grids:
        assert (
            grid.num_channels == num_channels
        ), f"All grids must have same channels: {grid.num_channels} != {num_channels}"

    # Get normalized coordinates for grid sampling
    normalized_coords_list = []
    with torch.no_grad():
        for grid in standard_grids:
            # Get the grid coordinates in physical space
            coords = grid.grid_coords.batched_tensor  # Shape: B*H*W*D, 3
            B, H, W, D = grid.batch_size, *grid.grid_shape

            # Reshape to grid format: B, H, W, D, 3
            coords_grid = coords.view(B, H, W, D, 3)

            # Normalize coordinates to [-1, 1] for grid_sample
            bounds_min, bounds_max = grid.bounds
            normalized_coords = 2.0 * (coords_grid - bounds_min) / (bounds_max - bounds_min) - 1.0
            normalized_coords_list.append(normalized_coords)

    # Perform feature communication
    updated_grids = []
    for i, target_grid in enumerate(standard_grids):
        target_features = target_grid.grid_features.batched_tensor.clone()  # B, C, H, W, D

        # Sample features from all other grids and accumulate
        for j, source_grid in enumerate(standard_grids):
            if i == j:
                continue  # Skip self

            source_features = source_grid.grid_features.batched_tensor  # B, C, H, W, D
            target_coords = normalized_coords_list[i]  # B, H, W, D, 3

            # Use grid_sample to interpolate source features at target coordinates
            sampled_features = F.grid_sample(
                source_features,  # B, C, H_src, W_src, D_src
                target_coords,  # B, H_tgt, W_tgt, D_tgt, 3
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # B, C, H_tgt, W_tgt, D_tgt

            # Apply communication operation
            if communication_type == "sum":
                target_features += sampled_features
            elif communication_type == "mul":
                target_features *= sampled_features
            else:
                raise ValueError(f"Unknown communication type: {communication_type}")

        # Create updated grid with new features
        updated_grid = target_grid.replace(batched_features=target_features)
        updated_grids.append(updated_grid)

    # Convert back to original memory formats
    final_grids = []
    for orig_grid, updated_grid in zip(orig_grids, updated_grids):
        if orig_grid.memory_format != GridMemoryFormat.b_c_x_y_z:
            final_grid = updated_grid.to_memory_format(orig_grid.memory_format)
        else:
            final_grid = updated_grid
        final_grids.append(final_grid)

    return FactorGrid(final_grids)


def factor_grid_intra_communication(
    factor_grid: FactorGrid,
    communication_types: List[Literal["sum", "mul"]] = ["sum"],
    cat_fn: Optional[Callable] = None,
) -> FactorGrid:
    """Perform intra-communication between grids in a FactorGrid with multiple communication types.

    Args:
        factor_grid: Input FactorGrid
        communication_types: List of communication types to apply
        cat_fn: Function to concatenate results from multiple communication types

    Returns:
        FactorGrid with inter-grid communication applied
    """
    if len(communication_types) == 1:
        return _factor_grid_intra_communication(factor_grid, communication_types[0])
    elif len(communication_types) == 2:
        # Apply both communication types and concatenate
        result1 = _factor_grid_intra_communication(factor_grid, communication_types[0])
        result2 = _factor_grid_intra_communication(factor_grid, communication_types[1])

        if cat_fn is not None:
            return cat_fn(result1, result2)
        else:
            return factor_grid_cat(result1, result2)
    else:
        raise ValueError(f"Unsupported number of communication types: {len(communication_types)}")
