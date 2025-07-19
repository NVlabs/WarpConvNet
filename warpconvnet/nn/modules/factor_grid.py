# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Neural network modules for FactorGrid operations.

This module provides neural network layers and operations specifically designed
for working with FactorGrid geometries in the FIGConvNet architecture.
"""

from typing import List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.features.grid import (
    GridMemoryFormat,
    NON_COMPRESSED_FORMATS,
    COMPRESSED_FORMATS,
)
from warpconvnet.geometry.types.factor_grid import FactorGrid
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.factor_grid import (
    factor_grid_transform,
    factor_grid_cat,
    factor_grid_pool,
    factor_grid_intra_communication,
)

__all__ = [
    "FactorGridTransform",
    "FactorGridCat",
    "FactorGridPool",
    "FactorGridGlobalConv",
]


class FactorGridTransform(BaseSpatialModule):
    """Apply a transform operation to all grids in a FactorGrid.

    This is equivalent to GridFeatureGroupTransform but works with FactorGrid objects.
    """

    def __init__(self, transform: nn.Module, in_place: bool = True) -> None:
        super().__init__()
        self.transform = transform
        self.in_place = in_place

    def forward(self, factor_grid: FactorGrid) -> FactorGrid:
        """Apply transform to all grids in the FactorGrid."""
        return factor_grid_transform(factor_grid, self.transform, self.in_place)


class FactorGridCat(BaseSpatialModule):
    """Concatenate features of two FactorGrid objects.

    This is equivalent to GridFeatureGroupCat but works with FactorGrid objects.
    """

    def __init__(self):
        super().__init__()

    def forward(self, factor_grid1: FactorGrid, factor_grid2: FactorGrid) -> FactorGrid:
        """Concatenate features from two FactorGrid objects."""
        return factor_grid_cat(factor_grid1, factor_grid2)


class FactorGridPool(BaseSpatialModule):
    """Pooling operation for FactorGrid.

    This is equivalent to GridFeatureGroupPool but works with FactorGrid objects.
    """

    def __init__(
        self,
        pooling_type: Literal["max", "mean", "attention"] = "max",
    ):
        super().__init__()
        self.pooling_type = pooling_type

        # Pooling operation
        if pooling_type == "max":
            self.pool_op = nn.AdaptiveMaxPool1d(1)
        elif pooling_type == "mean":
            self.pool_op = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == "attention":
            # For now, use simple attention mechanism
            # Note: attention layer dimensions will depend on actual feature dimensions
            self.attention = None  # Will be set based on input if needed
            self.pool_op = None
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

    def forward(self, factor_grid: FactorGrid) -> Tensor:
        """Pool features from FactorGrid to a single tensor."""
        return factor_grid_pool(
            factor_grid,
            self.pooling_type,
            pool_op=self.pool_op,
            attention_layer=getattr(self, "attention", None),
        )


class FactorGridIntraCommunication(BaseSpatialModule):
    """Intra-communication between grids in a FactorGrid.

    This is equivalent to GridFeaturesGroupIntraCommunication but works with FactorGrid objects.
    """

    def __init__(self, communication_types: List[Literal["sum", "mul"]] = ["sum"]) -> None:
        super().__init__()
        assert len(communication_types) > 0, "At least one communication type must be provided"
        assert len(communication_types) <= 2, "At most two communication types can be provided"
        self.communication_types = communication_types

    def forward(self, factor_grid: FactorGrid) -> FactorGrid:
        """Perform intra-communication between grids in the FactorGrid."""
        return factor_grid_intra_communication(factor_grid, self.communication_types)


class _FactorGridConvNormAct(BaseSpatialModule):
    """2D Convolution with normalization and activation for FactorGrid.

    This applies 2D convolution followed by normalization and activation to each grid in the FactorGrid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dim: int,
        compressed_memory_format: GridMemoryFormat,
        stride: int = 1,
        up_stride: Optional[int] = None,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        assert (
            compressed_memory_format in COMPRESSED_FORMATS
        ), "Compressed memory format must be a compressed format, got {compressed_memory_format}"

        # Create convolution + normalization layers for each compressed spatial dimension
        self.conv_norm_layers = nn.Sequential(
            (
                nn.Conv2d(
                    in_channels * compressed_spatial_dim,
                    out_channels * compressed_spatial_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    bias=True,
                )
                if up_stride is None
                else nn.ConvTranspose2d(
                    in_channels * compressed_spatial_dim,
                    out_channels * compressed_spatial_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    bias=True,
                )
            ),
            norm(out_channels),
            activation(),
        )

    def forward(self, grid: Grid) -> Grid:
        """Apply convolution + normalization to a grid."""
        # Convert grid to compressed format if it is not already
        grid_features = grid.grid_features.to_memory_format(self.compressed_memory_format)
        if grid_features.memory_format != self.compressed_memory_format:
            grid_features = grid_features.to_memory_format(self.compressed_memory_format)

        # Apply convolution + normalization
        processed_features = self.conv_norm_layers(grid_features.batched_tensor)
        return grid.replace(batched_features=processed_features)


class FactorGridProjection(BaseSpatialModule):
    """Projection operation for FactorGrid."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dims: Tuple[int, ...],
        compressed_memory_formats: Tuple[GridMemoryFormat, ...],
        stride: int = 1,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for compressed_spatial_dim, compressed_memory_format in zip(
            compressed_spatial_dims, compressed_memory_formats
        ):
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels * compressed_spatial_dim,
                    out_channels * compressed_spatial_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    bias=True,
                ),
                norm(out_channels * compressed_spatial_dim),
                activation(),
            )
            self.convs.append(block)

    def forward(self, grid: FactorGrid) -> FactorGrid:
        projected_grids = []
        for grid, conv in zip(grid, self.convs):
            projected_grid = conv(grid)
            projected_grids.append(projected_grid)
        return FactorGrid(projected_grids)


class FactorGridGlobalConv(BaseSpatialModule):
    """Global convolution with intra-communication for FactorGrid.

    This is equivalent to GridFeatureConv2DBlocksAndIntraCommunication but works with FactorGrid objects.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dims: Tuple[int, ...],
        compressed_memory_formats: Tuple[GridMemoryFormat, ...],
        stride: int = 1,
        up_stride: Optional[int] = None,
        communication_types: List[Literal["sum", "mul"]] = ["sum"],
        norm: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        assert len(compressed_spatial_dims) == len(
            compressed_memory_formats
        ), "Number of compressed spatial dimensions and compressed memory formats must match"

        # Create convolution blocks for each compressed spatial dimension
        self.conv_blocks = nn.ModuleList()
        for compressed_spatial_dim, compressed_memory_format in zip(
            compressed_spatial_dims, compressed_memory_formats
        ):
            self.conv_blocks.append(
                _FactorGridConvNormAct(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    compressed_spatial_dim=compressed_spatial_dim,
                    compressed_memory_format=compressed_memory_format,
                    stride=stride,
                    up_stride=up_stride,
                    norm=norm,
                    activation=activation,
                )
            )

        # Intra-communication module
        self.intra_communications = FactorGridIntraCommunication(
            communication_types=communication_types
        )

        # Projection layer if multiple communication types
        if len(communication_types) > 1:
            self.proj = FactorGridProjection(
                in_channels=out_channels * len(communication_types),
                out_channels=out_channels,
                kernel_size=1,
                compressed_spatial_dims=compressed_spatial_dims,
                compressed_memory_formats=compressed_memory_formats,
                stride=1,
            )
        else:
            self.proj = nn.Identity()

    def forward(self, factor_grid: FactorGrid) -> FactorGrid:
        """Forward pass through the global convolution module."""
        assert len(factor_grid) == len(
            self.conv_blocks
        ), f"Expected {len(self.conv_blocks)} grids, got {len(factor_grid)}"

        # Apply convolution blocks to each grid
        convolved_grids = []
        for grid, conv_block in zip(factor_grid, self.conv_blocks):
            convolved = conv_block(grid)
            convolved_grids.append(convolved)

        # Apply intra-communication
        factor_grid = self.intra_communications(FactorGrid(convolved_grids))

        # Apply projection if needed
        factor_grid = self.proj(factor_grid)

        return factor_grid
