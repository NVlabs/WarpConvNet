# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FIGConvNet: Factorized Grid Convolutional Network

This module implements the FIGConvNet architecture, which is the warpconvnet equivalent
of PointFeatureToGridGroupUNet. It uses factorized grid representations for efficient
3D point cloud processing.
"""

from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.features.grid import GridMemoryFormat
from warpconvnet.geometry.types.factor_grid import FactorGrid
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.base_module import BaseSpatialModel
from warpconvnet.nn.modules.factor_grid import (
    FactorGridGlobalConv,
    FactorGridPool,
    FactorGridToPoint,
    PointToFactorGrid,
)
from warpconvnet.nn.modules.mlp import FeatureMLPBlock


class FIGConvNet(BaseSpatialModel):
    """FIGConvNet: Factorized Grid Convolutional Network.

    This is the warpconvnet equivalent of PointFeatureToGridGroupUNet, using
    factorized grid representations for efficient 3D point cloud processing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        mlp_channels: List[int] = [2048, 2048],
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[Tuple[GridMemoryFormat, Tuple[int, int, int]]] = [
            (GridMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "radius",
        knn_k: int = 16,
        reductions: List[str] = ["mean"],
        pooling_type: Literal["attention", "max", "mean"] = "max",
        pooling_layers: List[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.aabb_max = aabb_max
        self.aabb_min = aabb_min

        # Extract grid shapes and memory formats
        grid_shapes = [res for _, res in resolution_memory_format_pairs]
        memory_formats = [fmt for fmt, _ in resolution_memory_format_pairs]
        self.grid_feature_group_size = len(resolution_memory_format_pairs)

        # Calculate compressed spatial dimensions
        compressed_spatial_dims = []
        for mem_fmt, res in resolution_memory_format_pairs:
            if mem_fmt == GridMemoryFormat.b_xc_y_z:
                compressed_spatial_dims.append(res[0])  # X dimension
            elif mem_fmt == GridMemoryFormat.b_yc_x_z:
                compressed_spatial_dims.append(res[1])  # Y dimension
            elif mem_fmt == GridMemoryFormat.b_zc_x_y:
                compressed_spatial_dims.append(res[2])  # Z dimension
            else:
                raise ValueError(f"Unsupported memory format: {mem_fmt}")

        self.compressed_spatial_dims = compressed_spatial_dims

        # Point to FactorGrid conversion
        self.point_to_factor_grids = PointToFactorGrid(
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            grid_shapes=grid_shapes,
            memory_formats=memory_formats,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_encode_dim=pos_encode_dim,
            search_radius=voxel_size * 2 if voxel_size else None,
            k=knn_k,
            search_type=neighbor_search_type,
            reduction=reductions[0] if reductions else "mean",
        )

        # UNet down and up blocks
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Handle num_down_blocks and num_up_blocks
        if isinstance(num_down_blocks, int):
            num_down_blocks = [num_down_blocks] * num_levels
        if isinstance(num_up_blocks, int):
            num_up_blocks = [num_up_blocks] * num_levels

        # Create down blocks
        for level in range(num_levels):
            down_block = []
            for _ in range(num_down_blocks[level]):
                down_block.append(
                    FactorGridGlobalConv(
                        in_channels=hidden_channels[level],
                        out_channels=hidden_channels[level + 1],
                        kernel_size=kernel_size,
                        compressed_spatial_dims=tuple(compressed_spatial_dims),
                        compressed_memory_formats=tuple(memory_formats),
                        stride=2 if _ == 0 else 1,  # First block downsamples
                        communication_types=communication_types,
                    )
                )
            self.down_blocks.append(nn.Sequential(*down_block))

        # Create up blocks
        for level in range(num_levels):
            up_block = []
            for _ in range(num_up_blocks[level]):
                up_block.append(
                    FactorGridGlobalConv(
                        in_channels=hidden_channels[level + 1],
                        out_channels=hidden_channels[level],
                        kernel_size=kernel_size,
                        compressed_spatial_dims=tuple(compressed_spatial_dims),
                        compressed_memory_formats=tuple(memory_formats),
                        up_stride=2 if _ == 0 else None,  # First block upsamples
                        communication_types=communication_types,
                    )
                )
            self.up_blocks.append(nn.Sequential(*up_block))

        # Global pooling for drag prediction
        if pooling_layers is None:
            pooling_layers = [num_levels]
        else:
            assert isinstance(
                pooling_layers, list
            ), f"pooling_layers must be a list, got {type(pooling_layers)}."
            for layer in pooling_layers:
                assert (
                    layer <= num_levels
                ), f"pooling_layer {layer} is greater than num_levels {num_levels}."

        self.pooling_layers = pooling_layers
        self.grid_pools = nn.ModuleList(
            [FactorGridPool(pooling_type=pooling_type) for _ in pooling_layers]
        )

        # MLP for global prediction
        total_pooled_channels = sum(
            sum(
                hidden_channels[layer] * compressed_spatial_dims[i]
                for i in range(len(compressed_spatial_dims))
            )
            for layer in pooling_layers
        )

        self.mlp = FeatureMLPBlock(
            in_channels=total_pooled_channels,
            out_channels=mlp_channels[-1],
            hidden_channels=mlp_channels[:-1],
        )
        self.mlp_projection = nn.Linear(mlp_channels[-1], 1)

        # FactorGrid to Point conversion
        self.to_point = FactorGridToPoint(
            grid_in_channels=hidden_channels[0],
            point_in_channels=in_channels,
            num_grids=len(resolution_memory_format_pairs),
            out_channels=hidden_channels[0] * 2,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_embed_dim=pos_encode_dim,
            sample_method=to_point_sample_method,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
        )

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[0] * 2, hidden_channels[0] * 2),
            nn.LayerNorm(hidden_channels[0] * 2),
            nn.GELU(),
            nn.Linear(hidden_channels[0] * 2, out_channels),
        )

    def _grid_forward(self, point_features: Points) -> Tuple[FactorGrid, Tensor]:
        """Forward pass through the grid-based UNet."""
        # Convert points to FactorGrid
        factor_grid = self.point_to_factor_grids(point_features)

        # Downsampling path
        down_factor_grids = [factor_grid]
        for down_block in self.down_blocks:
            out_features = down_block(down_factor_grids[-1])
            down_factor_grids.append(out_features)

        # Global pooling for drag prediction
        pooled_feats = []
        for grid_pool, layer in zip(self.grid_pools, self.pooling_layers):
            pooled_feat = grid_pool(down_factor_grids[layer])
            pooled_feats.append(pooled_feat)

        if len(pooled_feats) > 1:
            pooled_feats = torch.cat(pooled_feats, dim=-1)
        else:
            pooled_feats = pooled_feats[0]

        drag_pred = self.mlp_projection(self.mlp(pooled_feats))

        # Upsampling path with skip connections
        for level in reversed(range(self.num_levels)):
            up_factor_grid = self.up_blocks[level](down_factor_grids[level + 1])
            up_factor_grid = up_factor_grid + down_factor_grids[level]
            down_factor_grids[level] = up_factor_grid

        return down_factor_grids[0], drag_pred

    def forward(self, point_features: Points) -> Tuple[Points, Tensor]:
        """Forward pass through the FIGConvNet.

        Args:
            point_features: Input point features

        Returns:
            Tuple of (output_point_features, drag_prediction)
        """
        # Grid-based processing
        grid_features, drag_pred = self._grid_forward(point_features)

        # Convert back to point features
        out_point_features = self.to_point(grid_features, point_features)
        out_point_features = self.projection(out_point_features.features)

        # Create new Points object with output features
        output_points = Points(
            batched_coordinates=point_features.batched_coordinates,
            batched_features=out_point_features,
        )

        return output_points, drag_pred


class FIGConvNetDrivAer(FIGConvNet):
    """FIGConvNet specialized for DrivAer dataset."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        aabb_max: Tuple[float, float, float] = (2.5, 1.5, 1.0),
        aabb_min: Tuple[float, float, float] = (-2.5, -1.5, -1.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[Tuple[GridMemoryFormat, Tuple[int, int, int]]] = [
            (GridMemoryFormat.b_xc_y_z, (4, 120, 80)),
            (GridMemoryFormat.b_yc_x_z, (200, 3, 80)),
            (GridMemoryFormat.b_zc_x_y, (200, 120, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "knn",
        knn_k: int = 16,
        reductions: List[str] = ["mean"],
        pooling_type: Literal["attention", "max", "mean"] = "max",
        pooling_layers: List[int] = None,
        drag_loss_weight: Optional[float] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            voxel_size=voxel_size,
            resolution_memory_format_pairs=resolution_memory_format_pairs,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_encode_dim=pos_encode_dim,
            communication_types=communication_types,
            to_point_sample_method=to_point_sample_method,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
            pooling_type=pooling_type,
            pooling_layers=pooling_layers,
        )

        if drag_loss_weight is not None:
            self.drag_loss_weight = drag_loss_weight
