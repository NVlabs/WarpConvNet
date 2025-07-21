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
    FORMAT_TO_AXIS,
    COMPRESSED_FORMATS,
    GridFeatures,
)
from warpconvnet.geometry.types.factor_grid import FactorGrid
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.geometry.coords.grid import GridCoords
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.functional.encodings import sinusoidal_encoding, get_freqs
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
    "FactorGridPadToMatch",
    "FactorGridToPoint",
    "PointToFactorGrid",
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


class FactorGridPadToMatch(BaseSpatialModule):
    """Pad FactorGrid features to match spatial dimensions for UNet skip connections."""

    def __init__(self):
        super().__init__()

    def forward(self, up_factor_grid: FactorGrid, down_factor_grid: FactorGrid) -> FactorGrid:
        """Pad up_factor_grid to match down_factor_grid spatial dimensions."""
        assert len(up_factor_grid) == len(
            down_factor_grid
        ), "FactorGrids must have same number of grids"

        padded_grids = []
        for up_grid, down_grid in zip(up_factor_grid, down_factor_grid):
            # Get features and shapes
            up_features = up_grid.grid_features.batched_tensor
            down_features = down_grid.grid_features.batched_tensor

            # Get spatial dimensions (excluding batch and channel)
            up_shape = up_features.shape[2:]  # Spatial dimensions
            down_shape = down_features.shape[2:]  # Spatial dimensions

            if up_shape == down_shape:
                # No padding needed
                padded_grids.append(up_grid)
            else:
                # Calculate padding needed
                pad_h = max(0, down_shape[0] - up_shape[0])
                pad_w = max(0, down_shape[1] - up_shape[1])
                pad_d = max(0, down_shape[2] - up_shape[2]) if len(down_shape) > 2 else 0

                # Apply padding
                if len(down_shape) == 2:  # 2D case
                    padded_features = F.pad(up_features, (0, pad_w, 0, pad_h), mode="replicate")
                else:  # 3D case
                    padded_features = F.pad(
                        up_features, (0, pad_d, 0, pad_w, 0, pad_h), mode="replicate"
                    )

                # Create new grid with padded features
                padded_grid = up_grid.replace(batched_features=padded_features)
                padded_grids.append(padded_grid)

        return FactorGrid(padded_grids)


class FactorGridToPoint(BaseSpatialModule):
    """Convert FactorGrid features back to point features using trilinear interpolation."""

    def __init__(
        self,
        grid_in_channels: int,
        point_in_channels: int,
        num_grids: int,
        out_channels: int,
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        pos_embed_dim: int = 32,
        sample_method: Literal["graphconv", "interp"] = "interp",
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
        reductions: List[str] = ["mean"],
    ):
        super().__init__()
        self.grid_in_channels = grid_in_channels
        self.point_in_channels = point_in_channels
        self.out_channels = out_channels
        self.use_rel_pos = use_rel_pos
        self.use_rel_pos_embed = use_rel_pos_embed
        self.pos_embed_dim = pos_embed_dim
        self.sample_method = sample_method
        self.neighbor_search_type = neighbor_search_type
        self.knn_k = knn_k
        self.reductions = reductions
        self.freqs = get_freqs(pos_embed_dim)

        # Calculate combined channels for MLP
        combined_channels = grid_in_channels * num_grids + point_in_channels
        if use_rel_pos_embed:
            combined_channels += pos_embed_dim * 3
        elif use_rel_pos:
            combined_channels += 3

        self.combine_mlp = nn.Sequential(
            nn.Linear(combined_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def _normalize_coordinates(self, coords: Tensor, grid: Grid) -> Tensor:
        """Normalize coordinates to [-1, 1] range for grid_sample using grid bounds."""
        # Get bounds from the grid
        bounds_min, bounds_max = grid.bounds

        # Normalize to [0, 1] first
        normalized = (coords - bounds_min) / (bounds_max - bounds_min)

        # Convert to [-1, 1] range for grid_sample
        normalized = 2.0 * normalized - 1.0

        # Ensure coordinates are within bounds
        normalized = torch.clamp(normalized, -1.0, 1.0)

        return normalized

    def _sample_from_grid(self, grid: Grid, point_coords: Tensor) -> Tensor:
        """Sample features from grid using trilinear interpolation."""
        grid_features_tensor = grid.grid_features.batched_tensor
        batch_size = grid_features_tensor.shape[0]

        # Normalize point coordinates using grid bounds
        normalized_coords = self._normalize_coordinates(point_coords, grid)

        # Reshape coordinates for grid_sample: (B, N, 3) -> (B, N, 1, 1, 3)
        # grid_sample expects (B, H, W, D, 3) for 3D grids
        normalized_coords = normalized_coords.view(batch_size, -1, 1, 1, 3)

        # Convert grid to standard format for grid_sample
        grid_reshaped = grid.grid_features.to_memory_format(GridMemoryFormat.b_c_x_y_z)

        # Use grid_sample for trilinear interpolation
        sampled_features = F.grid_sample(
            grid_reshaped.batched_tensor,
            normalized_coords,
            mode="bilinear",  # For 3D, this becomes trilinear
            padding_mode="border",
            align_corners=True,
        )  # Shape: B, C, N, 1, 1

        # Reshape to (B*N, C)
        sampled_features = sampled_features.squeeze(-1).squeeze(-1).transpose(1, 2)  # B, N, C
        sampled_features = sampled_features.flatten(start_dim=0, end_dim=1)  # B*N, C
        return sampled_features

    def forward(self, factor_grid: FactorGrid, point_features: Points) -> Points:
        """Convert FactorGrid features to point features using trilinear interpolation."""
        # Get point coordinates and features
        point_coords = point_features.coordinate_tensor
        point_feats = point_features.feature_tensor
        batch_size = point_features.batch_size
        num_points = point_coords.shape[0]

        # Sample features from all grids and concatenate
        all_grid_features = []

        for grid in factor_grid:
            # Sample features from this grid
            sampled_features = self._sample_from_grid(grid, point_coords)
            all_grid_features.append(sampled_features)

        # Concatenate features from all grids
        if len(all_grid_features) > 1:
            grid_feat_per_point = torch.cat(all_grid_features, dim=-1)
        else:
            grid_feat_per_point = all_grid_features[0]

        # Add relative position features if requested
        if self.use_rel_pos_embed:
            # Use bounds from the first grid for relative position calculation
            bounds_min, bounds_max = factor_grid[0].bounds
            rel_pos = point_coords - ((bounds_max + bounds_min) / 2.0)
            pos_encoding = sinusoidal_encoding(rel_pos, self.pos_embed_dim, data_range=2)
            combined_features = torch.cat([point_feats, grid_feat_per_point, pos_encoding], dim=-1)
        elif self.use_rel_pos:
            # Use raw relative positions
            # Use bounds from the first grid for relative position calculation
            bounds_min, bounds_max = factor_grid[0].bounds
            rel_pos = point_coords - ((bounds_max + bounds_min) / 2.0)
            combined_features = torch.cat([point_feats, grid_feat_per_point, rel_pos], dim=-1)
        else:
            # Just concatenate point and grid features
            combined_features = torch.cat([point_feats, grid_feat_per_point], dim=-1)

        # Apply MLP
        output_features = self.combine_mlp(combined_features)

        # Create new Points object
        return Points(
            batched_coordinates=point_features.batched_coordinates,
            batched_features=output_features,
        )


class PointToFactorGrid(BaseSpatialModule):
    """Convert point features to FactorGrid representation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_shapes: List[Tuple[int, int, int]],
        memory_formats: List[GridMemoryFormat],
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_encode_dim: int = 32,
        search_radius: Optional[float] = None,
        k: int = 8,
        search_type: Literal["radius", "knn", "voxel"] = "radius",
        reduction: str = "mean",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_shapes = grid_shapes
        self.memory_formats = memory_formats
        self.aabb_max = aabb_max
        self.aabb_min = aabb_min
        self.use_rel_pos = use_rel_pos
        self.use_rel_pos_embed = use_rel_pos_embed
        self.pos_encode_dim = pos_encode_dim
        self.search_radius = search_radius
        self.k = k
        self.search_type = search_type
        self.reduction = reduction

        # Calculate compressed spatial dimensions
        self.compressed_spatial_dims = []
        for grid_shape, memory_format in zip(grid_shapes, memory_formats):
            # Determine compressed spatial dimension
            if memory_format == GridMemoryFormat.b_zc_x_y:
                compressed_dim = grid_shape[0]  # Z dimension
            elif memory_format == GridMemoryFormat.b_xc_y_z:
                compressed_dim = grid_shape[0]  # X dimension
            elif memory_format == GridMemoryFormat.b_yc_x_z:
                compressed_dim = grid_shape[1]  # Y dimension
            else:
                raise ValueError(f"Unsupported memory format: {memory_format}")

            self.compressed_spatial_dims.append(compressed_dim)

        # Create projection layers for each grid
        self.projections = nn.ModuleList()
        for compressed_dim in self.compressed_spatial_dims:
            # Create projection layer
            proj = nn.Linear(in_channels, out_channels * compressed_dim)
            self.projections.append(proj)

    def forward(self, points: Points) -> FactorGrid:
        """Convert point features to FactorGrid."""
        from warpconvnet.geometry.types.conversion.to_factor_grid import points_to_factor_grid

        # Convert points to FactorGrid using the existing function
        factor_grid = points_to_factor_grid(
            points=points,
            grid_shapes=self.grid_shapes,
            memory_formats=self.memory_formats,
            bounds=(torch.tensor(self.aabb_min), torch.tensor(self.aabb_max)),
            search_radius=self.search_radius,
            k=self.k,
            search_type=self.search_type,
            reduction=self.reduction,
        )

        # For now, return the factor grid as is without projections
        # The projections can be applied later in the pipeline if needed
        return factor_grid


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

        self.stride = stride
        self.up_stride = up_stride
        self.compressed_spatial_dim = compressed_spatial_dim
        self.compressed_memory_format = compressed_memory_format

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
                    stride=up_stride,
                    padding=(kernel_size - 1) // 2,
                    bias=True,
                )
            ),
            norm(out_channels * compressed_spatial_dim) if norm is not None else nn.Identity(),
            activation() if activation is not None else nn.Identity(),
        )

    def forward(self, grid: Grid) -> Grid:
        """Apply convolution + normalization to a grid."""
        # Get grid features
        grid_features = grid.grid_features.batched_tensor

        # Apply convolution + normalization
        processed_features = self.conv_norm_layers(grid_features)

        # For the upsampled, and downsampled grids, the grid shape changes properly when downsampling/upsampling
        if self.up_stride is not None or self.stride != 1:
            # Up or down stride applied only on non compressed dimensions
            axis = FORMAT_TO_AXIS[grid.memory_format]
            if self.up_stride is not None:
                grid_shape = [
                    grid.grid_shape[0] * self.up_stride,
                    grid.grid_shape[1] * self.up_stride,
                    grid.grid_shape[2] * self.up_stride,
                ]
            else:
                grid_shape = [
                    grid.grid_shape[0] // self.stride,
                    grid.grid_shape[1] // self.stride,
                    grid.grid_shape[2] // self.stride,
                ]
            grid_shape[axis] = self.compressed_spatial_dim
            grid_shape = tuple(grid_shape)

            # For upsampled grids, the grid shape and the feature shape might not match
            if self.up_stride is not None:
                non_compressed_axes = [i for i in range(3) if i != axis]
                # Pad the features to match the grid shape
                pad_h = max(0, grid_shape[non_compressed_axes[0]] - processed_features.shape[2])
                pad_w = max(0, grid_shape[non_compressed_axes[1]] - processed_features.shape[3])
                processed_features = F.pad(
                    processed_features, (0, pad_w, 0, pad_h), mode="replicate"
                )

            return Grid(
                batched_coordinates=GridCoords.from_shape(
                    batch_size=grid.batched_coordinates.batch_size,
                    grid_shape=grid_shape,
                    bounds=grid.bounds,
                ),
                batched_features=processed_features,
                grid_shape=grid_shape,
                memory_format=grid.memory_format,
            )
        else:
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
