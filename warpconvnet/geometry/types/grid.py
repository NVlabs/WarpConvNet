# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Grid geometry implementation that combines grid coordinates and features.
"""

from typing import Dict, Literal, Optional, Tuple, Union

from jaxtyping import Float, Int
import torch
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.grid import GridCoords
from warpconvnet.geometry.features.grid import GridFeatures, GridMemoryFormat


class Grid(Geometry):
    """Grid geometry representation that combines coordinates and features.

    This class provides a unified interface for grid-based geometries with any
    memory format, combining grid coordinates with grid features.

    Args:
        batched_coordinates (GridCoords): Coordinate system for the grid
        batched_features (Union[GridFeatures, Tensor]): Grid features
        memory_format (GridMemoryFormat): Memory format for the features
        grid_shape (Tuple[int, int, int], optional): Grid resolution (H, W, D)
        num_channels (int, optional): Number of feature channels
        **kwargs: Additional parameters
    """

    def __init__(
        self,
        batched_coordinates: GridCoords,
        batched_features: Union[GridFeatures, Tensor],
        memory_format: Optional[GridMemoryFormat] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        num_channels: Optional[int] = None,
        **kwargs,
    ):
        if isinstance(batched_features, Tensor):
            assert (
                memory_format is not None
            ), "Memory format must be provided if features are a tensor"
            if grid_shape is None:
                grid_shape = batched_coordinates.grid_shape

            # If num_channels not provided, infer it from tensor shape and memory format
            if num_channels is None:
                if memory_format == GridMemoryFormat.b_x_y_z_c:
                    num_channels = batched_features.shape[-1]
                elif memory_format == GridMemoryFormat.b_c_x_y_z:
                    num_channels = batched_features.shape[1]
                elif memory_format == GridMemoryFormat.b_c_z_x_y:
                    num_channels = batched_features.shape[1]
                elif memory_format == GridMemoryFormat.b_zc_x_y:
                    zc = batched_features.shape[1]
                    num_channels = zc // grid_shape[2]
                elif memory_format == GridMemoryFormat.b_xc_y_z:
                    xc = batched_features.shape[1]
                    num_channels = xc // grid_shape[0]
                elif memory_format == GridMemoryFormat.b_yc_x_z:
                    yc = batched_features.shape[1]
                    num_channels = yc // grid_shape[1]
                else:
                    raise ValueError(f"Unsupported memory format: {memory_format}")

            # Create GridFeatures with same offsets as coordinates
            batched_features = GridFeatures(
                batched_features,
                batched_coordinates.offsets.clone(),
                memory_format,
                grid_shape,
                num_channels,
            )
        else:
            assert (
                memory_format is None or memory_format == batched_features.memory_format
            ), f"Memory format must be None or match the GridFeatures memory format: {batched_features.memory_format}. Provided: {memory_format}"

        # Check that the grid is valid
        self.check(batched_coordinates, batched_features)

        # Ensure offsets match if coordinates are not lazy
        assert (
            batched_coordinates.offsets == batched_features.offsets
        ).all(), "Coordinate and feature offsets must match"

        # Initialize base class
        super().__init__(batched_coordinates, batched_features, **kwargs)

    def check(self, coords: GridCoords, features: GridFeatures):
        """
        Check if the grid dimensions are consistent
        """
        assert coords.shape[-1] == 3

        num_coords = coords.numel() // 3
        num_features = features.numel() // features.num_channels
        assert (
            num_coords == num_features
        ), f"Number of coordinates ({num_coords}) must match number of features ({num_features})"
        assert (
            coords.grid_shape == features.grid_shape
        ), f"Grid shape ({coords.grid_shape}) must match feature grid shape ({features.grid_shape})"

    @classmethod
    def from_shape(
        cls,
        grid_shape: Tuple[int, int, int],
        num_channels: int,
        memory_format: GridMemoryFormat = GridMemoryFormat.b_x_y_z_c,
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "Grid":
        """
        Create a new Grid geometry from a grid shape. The coordinates will be lazily initialized and the features will be created as an empty tensor.

        Args:
            grid_shape: Grid resolution (H, W, D)
            num_channels: Number of feature channels
            memory_format: Memory format for features
            bounds: Min and max bounds for the grid
            batch_size: Number of batches
            device: Device to create tensors on
            dtype: Data type for feature tensors
            **kwargs: Additional parameters

        Returns:
            Initialized grid geometry
        """
        # Create coordinates. By default, data will be lazily initialized and coordinates will be flattened.
        coords = GridCoords.from_shape(
            grid_shape=grid_shape,
            bounds=bounds,
            batch_size=batch_size,
            device=device,
            flatten=True,
        )

        # Create empty features with same offsets
        features = GridFeatures.create_empty(
            grid_shape=grid_shape,
            num_channels=num_channels,
            batch_size=batch_size,
            memory_format=memory_format,
            device=device,
            dtype=dtype,
        )

        # Make sure offsets match
        assert (
            coords.offsets == features.offsets
        ).all(), "Coordinate and feature offsets must match"

        return cls(coords, features, memory_format, **kwargs)

    @property
    def grid_features(self) -> GridFeatures:
        """Get the grid features."""
        return self.batched_features

    @property
    def grid_coords(self) -> GridCoords:
        """Get the grid coordinates."""
        return self.batched_coordinates

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Get the grid shape (H, W, D)."""
        return self.grid_coords.grid_shape

    @property
    def bounds(self) -> Tuple[Tensor, Tensor]:
        """Get the bounds of the grid."""
        return self.grid_coords.bounds

    @property
    def num_channels(self) -> int:
        """Get the number of feature channels."""
        return self.grid_features.num_channels

    @property
    def memory_format(self) -> GridMemoryFormat:
        """Get the memory format."""
        return self.grid_features.memory_format

    def channel_size(self, memory_format: Optional[GridMemoryFormat] = None):
        if memory_format is None:
            memory_format = self.memory_format
        if memory_format == GridMemoryFormat.b_x_y_z_c:
            return self.num_channels
        elif memory_format == GridMemoryFormat.b_c_x_y_z:
            return self.num_channels
        elif memory_format == GridMemoryFormat.b_xc_y_z:
            return self.num_channels * self.grid_shape[0]
        elif memory_format == GridMemoryFormat.b_yc_x_z:
            return self.num_channels * self.grid_shape[1]
        elif memory_format == GridMemoryFormat.b_zc_x_y:
            return self.num_channels * self.grid_shape[2]
        else:
            raise ValueError(f"Unsupported memory format: {memory_format}")

    def to_memory_format(self, memory_format: GridMemoryFormat) -> "Grid":
        """Convert to a different memory format."""
        if memory_format != self.memory_format:
            return self.replace(
                batched_features=self.grid_features.to_memory_format(memory_format),
                memory_format=memory_format,
            )
        return self

    @property
    def shape(self) -> Dict[str, Union[int, Tuple[int, ...]]]:
        """Get the shape information."""
        H, W, D = self.grid_shape
        return {
            "grid_shape": self.grid_shape,
            "batch_size": self.batch_size,
            "num_channels": self.num_channels,
            "total_elements": H * W * D * self.batch_size,
        }

    def to(self, device: torch.device) -> "Grid":
        """Move the geometry to the specified device."""
        return Grid(
            self.grid_coords.to(device),
            self.grid_features.to(device),
            self.memory_format,
        )

    def replace(
        self,
        batched_coordinates: Optional[GridCoords] = None,
        batched_features: Optional[Union[GridFeatures, Tensor]] = None,
        **kwargs,
    ) -> "Grid":
        """Create a new instance with replaced coordinates and/or features."""
        # Convert the batched_features to a GridFeatures if it is a tensor
        if isinstance(batched_features, Tensor) and batched_features.ndim == 5:
            # Based on the memory format, we have to check the shape of the tensor
            if self.memory_format == GridMemoryFormat.b_x_y_z_c:
                in_H, in_W, in_D, in_C = batched_features.shape[1:5]
                assert in_H == self.grid_shape[0]
                assert in_W == self.grid_shape[1]
                assert in_D == self.grid_shape[2]
                assert in_C == self.num_channels
            elif self.memory_format == GridMemoryFormat.b_c_x_y_z:
                in_C, in_H, in_W, in_D = batched_features.shape[1:5]
                assert in_C == self.num_channels
                assert in_H == self.grid_shape[0]
                assert in_W == self.grid_shape[1]
                assert in_D == self.grid_shape[2]
            elif self.memory_format == GridMemoryFormat.b_c_z_x_y:
                in_C, in_D, in_H, in_W = batched_features.shape[1:5]
                assert in_C == self.num_channels
                assert in_D == self.grid_shape[2]
                assert in_H == self.grid_shape[0]
                assert in_W == self.grid_shape[1]
            else:
                raise ValueError(f"Unsupported memory format: {self.memory_format}")

            batched_features = GridFeatures(
                batched_tensor=batched_features,
                offsets=self.grid_features.offsets,
                memory_format=self.memory_format,
                grid_shape=self.grid_shape,
                num_channels=in_C,
            )
        elif isinstance(batched_features, Tensor) and batched_features.ndim == 4:
            # This is the compressed format
            assert self.memory_format in [
                GridMemoryFormat.b_zc_x_y,
                GridMemoryFormat.b_xc_y_z,
                GridMemoryFormat.b_yc_x_z,
            ], f"Unsupported memory format: {self.memory_format} for feature tensor of shape {batched_features.shape}"
            # Assert that the grid shape is consistent with the feature tensor shape
            # Only the channel dim can change when using .replace()
            # e.g. in_H, in_W, in_D == self.grid_shape[0], self.grid_shape[1], self.grid_shape[2]
            compressed_dim = batched_features.shape[1]  # this is the compressed_dim * channels
            new_channel_dim = None
            if self.memory_format == GridMemoryFormat.b_zc_x_y:
                assert batched_features.shape[2] == self.grid_shape[0]
                assert batched_features.shape[3] == self.grid_shape[1]
                new_channel_dim = compressed_dim // self.grid_shape[2]
            elif self.memory_format == GridMemoryFormat.b_xc_y_z:
                assert batched_features.shape[2] == self.grid_shape[1]
                assert batched_features.shape[3] == self.grid_shape[2]
                new_channel_dim = compressed_dim // self.grid_shape[0]
            elif self.memory_format == GridMemoryFormat.b_yc_x_z:
                assert batched_features.shape[2] == self.grid_shape[0]
                assert batched_features.shape[3] == self.grid_shape[2]
                new_channel_dim = compressed_dim // self.grid_shape[1]
            else:
                raise ValueError(f"Unsupported memory format: {self.memory_format}")

            batched_features = GridFeatures(
                batched_tensor=batched_features,
                offsets=self.grid_features.offsets,
                memory_format=self.memory_format,
                grid_shape=self.grid_shape,
                num_channels=new_channel_dim,
            )
        elif isinstance(batched_features, GridFeatures):
            # no action needed
            pass
        else:
            raise ValueError(f"Unsupported feature tensor shape: {batched_features.shape}")

        return super().replace(batched_coordinates, batched_features, **kwargs)
