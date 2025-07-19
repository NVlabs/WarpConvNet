# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Feature representations for grid-based geometries.
This module defines the GridFeatures class which supports all memory formats
including standard and factorized grid representations.
"""

from typing import Optional, Tuple, List, Union

from enum import Enum, auto

import torch
from torch import Tensor

from warpconvnet.geometry.base.features import Features
from warpconvnet.geometry.features.ops.convert_grid import (
    convert_to_standard_format,
    convert_from_standard_format,
)


class GridMemoryFormat(Enum):
    """Memory format used for grid-based feature representations.

    The memory format defines how the grid features are stored in memory:

    b_x_y_z_c: Batch, X, Y, Z, Channels (3D Grid)
    b_c_x_y_z: Batch, Channels, X, Y, Z (3D Grid)
    b_zc_x_y: Batch, Z * Channels, X, Y (2D Grid, Z compressed)
    b_xc_y_z: Batch, X * Channels, Y, Z (2D Grid, X compressed)
    b_yc_x_z: Batch, Y * Channels, X, Z (2D Grid, Y compressed)
    """

    b_x_y_z_c = auto()
    b_c_x_y_z = auto()
    b_c_z_x_y = auto()  # Channels, Z, X, Y which is consistent with the pytorch conv3d format

    # Factorized 3D formats
    b_zc_x_y = auto()
    b_xc_y_z = auto()
    b_yc_x_z = auto()


# Dictionary mapping memory formats to string representations
FORMAT_TO_STR = {
    GridMemoryFormat.b_x_y_z_c: "b_x_y_z_c",
    GridMemoryFormat.b_c_x_y_z: "b_c_x_y_z",
    GridMemoryFormat.b_c_z_x_y: "b_c_z_x_y",
    GridMemoryFormat.b_zc_x_y: "b_zc_x_y",
    GridMemoryFormat.b_xc_y_z: "b_xc_y_z",
    GridMemoryFormat.b_yc_x_z: "b_yc_x_z",
}


# Dictionary mapping memory formats to axis indices for compression
FORMAT_TO_AXIS = {
    GridMemoryFormat.b_x_y_z_c: -1,  # No compression
    GridMemoryFormat.b_c_x_y_z: -1,  # No compression
    GridMemoryFormat.b_c_z_x_y: -1,  # No compression
    GridMemoryFormat.b_zc_x_y: 2,  # Z-axis compressed
    GridMemoryFormat.b_xc_y_z: 0,  # X-axis compressed
    GridMemoryFormat.b_yc_x_z: 1,  # Y-axis compressed
}

NON_COMPRESSED_FORMATS = [
    GridMemoryFormat.b_x_y_z_c,
    GridMemoryFormat.b_c_x_y_z,
    GridMemoryFormat.b_c_z_x_y,
]

COMPRESSED_FORMATS = [
    GridMemoryFormat.b_zc_x_y,
    GridMemoryFormat.b_xc_y_z,
    GridMemoryFormat.b_yc_x_z,
]


class GridFeatures(Features):
    """Grid feature representation for 3D data with support for all memory formats.

    This unified class supports both standard and factorized memory formats
    in a single implementation, combining the functionality of the previous
    GridFeatures and FactorizedGridFeatures classes.

    Args:
        tensor (Tensor): Feature tensor in the specified memory format
        offsets (Tensor): Offsets for batched data
        memory_format (GridMemoryFormat): The memory layout of the tensor
        grid_shape (Tuple[int, int, int], optional): The 3D resolution of the grid (H, W, D)
            Required for factorized formats, inferred for standard formats
        num_channels (int, optional): Number of feature channels
            Required for factorized formats, inferred for standard formats
    """

    def __init__(
        self,
        batched_tensor: Tensor,
        offsets: Tensor,
        memory_format: GridMemoryFormat = GridMemoryFormat.b_x_y_z_c,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        num_channels: Optional[int] = None,
    ):
        """

        Args:
            tensor: Feature tensor in the specified memory format
            offsets: Offsets for batched data
            memory_format: The memory layout of the tensor
            grid_shape: The 3D resolution of the grid (H, W, D). Even if the memory format is factorized (e.g. b_zc_x_y), permuted (e.g. b_c_z_x_y), or standard (e.g. b_x_y_z_c), the grid_shape is always the original shape of the grid in the order of (H, W, D) == (X, Y, Z)
            num_channels: Number of feature channels
        """

        # Determine grid shape of the feature tensor
        B, H, W, D, C = None, None, None, None, None
        if memory_format == GridMemoryFormat.b_x_y_z_c:
            assert (
                batched_tensor.ndim == 5
            ), f"Expected 5D tensor for b_x_y_z_c format, got {batched_tensor.ndim}D"
            B, H, W, D, C = batched_tensor.shape
        elif memory_format == GridMemoryFormat.b_c_x_y_z:
            assert (
                batched_tensor.ndim == 5
            ), f"Expected 5D tensor for b_c_x_y_z format, got {batched_tensor.ndim}D"
            B, C, H, W, D = batched_tensor.shape
        elif memory_format == GridMemoryFormat.b_c_z_x_y:
            assert (
                batched_tensor.ndim == 5
            ), f"Expected 5D tensor for b_c_z_x_y format, got {batched_tensor.ndim}D"
            B, C, D, H, W = batched_tensor.shape
        elif memory_format == GridMemoryFormat.b_zc_x_y:
            assert (
                batched_tensor.ndim == 4
            ), f"Expected 4D tensor for b_zc_x_y format, got {batched_tensor.ndim}D"
            assert num_channels is not None, "num_channels must be provided for b_zc_x_y format"
            B, ZC, H, W = batched_tensor.shape
            D, _ = divmod(ZC, num_channels)
            C = num_channels
        elif memory_format == GridMemoryFormat.b_xc_y_z:
            assert (
                batched_tensor.ndim == 4
            ), f"Expected 4D tensor for b_xc_y_z format, got {batched_tensor.ndim}D"
            assert num_channels is not None, "num_channels must be provided for b_xc_y_z format"
            B, XC, W, D = batched_tensor.shape
            H, _ = divmod(XC, num_channels)
            C = num_channels
        elif memory_format == GridMemoryFormat.b_yc_x_z:
            assert (
                batched_tensor.ndim == 4
            ), f"Expected 4D tensor for b_yc_x_z format, got {batched_tensor.ndim}D"
            assert num_channels is not None, "num_channels must be provided for b_yc_x_z format"
            B, YC, H, D = batched_tensor.shape
            W, _ = divmod(YC, num_channels)
            C = num_channels
        else:
            raise ValueError(f"Unsupported memory format: {memory_format}")

        # Check the input grid_shape matches the inferred grid_shape
        if grid_shape is not None:
            assert len(grid_shape) == 3, "grid_shape must be a tuple of 3 integers"
            assert (
                H == grid_shape[0] and W == grid_shape[1] and D == grid_shape[2]
            ), f"Input grid_shape ({grid_shape}) does not match inferred grid_shape ({H}, {W}, {D})"

        # Batch size check
        assert B == offsets.shape[0] - 1, f"Batch size mismatch: {B} != {offsets.shape[0] - 1}"

        # Initialize the parent class and save the attributes
        super().__init__(batched_tensor, offsets)
        self.memory_format = memory_format
        self._grid_shape = (H, W, D)
        self._num_channels = C

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Return the 3D grid shape (height, width, depth)."""
        return self._grid_shape

    @property
    def resolution(self) -> Tuple[int, int, int]:
        """Return the 3D grid resolution (alias for grid_shape)."""
        return self._grid_shape

    @property
    def num_channels(self) -> int:
        """Return the number of feature channels."""
        # Since the parent class num_channels is the last dimension of the tensor, we need to return the correct one based on the memory format
        return self._num_channels

    def channel_size(self, memory_format: Optional[GridMemoryFormat] = None) -> int:
        """Get the channel size for a specific memory format.

        Args:
            memory_format: Target memory format (use current format if None)

        Returns:
            Number of channels in the specified format
        """
        if memory_format is None:
            memory_format = self.memory_format

        if memory_format in [
            GridMemoryFormat.b_x_y_z_c,
            GridMemoryFormat.b_c_x_y_z,
            GridMemoryFormat.b_c_z_x_y,
        ]:
            return self._num_channels
        elif memory_format == GridMemoryFormat.b_xc_y_z:
            return self._num_channels * self._grid_shape[0]
        elif memory_format == GridMemoryFormat.b_yc_x_z:
            return self._num_channels * self._grid_shape[1]
        elif memory_format == GridMemoryFormat.b_zc_x_y:
            return self._num_channels * self._grid_shape[2]
        else:
            raise ValueError(f"Unsupported memory format: {memory_format}")

    def to_standard_format(self) -> Tensor:
        """Convert features to standard b_x_y_z_c format regardless of current format."""
        if self.memory_format == GridMemoryFormat.b_x_y_z_c:
            return self.batched_tensor

        return convert_to_standard_format(
            self.batched_tensor, self.memory_format, self._num_channels, self._grid_shape
        )

    def to_memory_format(self, target_format: GridMemoryFormat) -> "GridFeatures":
        """Convert to a different memory format.

        Args:
            target_format: The memory format to convert to

        Returns:
            A new GridFeatures instance with the requested memory format
        """
        if self.memory_format == target_format:
            return self

        # Special case for non compressed formats
        if (
            self.memory_format in NON_COMPRESSED_FORMATS
            and target_format in NON_COMPRESSED_FORMATS
        ):
            # Only perform permutations
            axes = FORMAT_TO_STR[self.memory_format].split("_")
            target_axes = FORMAT_TO_STR[target_format].split("_")
            perm = [axes.index(axis) for axis in target_axes]
            return GridFeatures(
                batched_tensor=self.batched_tensor.permute(perm),
                offsets=self.offsets,
                memory_format=target_format,
                grid_shape=self._grid_shape,
                num_channels=self._num_channels,
            )

        # First convert to standard format if not already
        standard = self.to_standard_format()

        # Then convert to target format
        tensor = convert_from_standard_format(standard, target_format, self._grid_shape)

        return GridFeatures(
            batched_tensor=tensor,
            offsets=self.offsets,
            memory_format=target_format,
            grid_shape=self._grid_shape,
            num_channels=self._num_channels,
        )

    def equal_shape(self, other: "GridFeatures") -> bool:
        """Check if two grid features have the same shape."""
        if not isinstance(other, GridFeatures):
            return False

        return (
            self._grid_shape == other._grid_shape
            and self._num_channels == other._num_channels
            and len(self.offsets) == len(other.offsets)
            and (self.offsets == other.offsets).all()
        )

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "GridFeatures":
        """Move the features to the specified device and optionally convert dtype."""
        tensor = self.batched_tensor.to(device=device, dtype=dtype if dtype is not None else None)
        offsets = self.offsets.to(device)
        return GridFeatures(
            tensor, offsets, self.memory_format, self._grid_shape, self._num_channels
        )

    @classmethod
    def create_empty(
        cls,
        grid_shape: Tuple[int, int, int],
        num_channels: int,
        batch_size: int = 1,
        memory_format: GridMemoryFormat = GridMemoryFormat.b_x_y_z_c,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "GridFeatures":
        """Create empty grid features with the specified format.

        Args:
            grid_shape: 3D grid resolution (H, W, D)
            num_channels: Number of feature channels
            batch_size: Number of batches to create
            memory_format: Memory format to use
            device: Device to create tensor on
            dtype: Data type for the tensor

        Returns:
            Empty grid features with the requested configuration
        """
        tensor = init_grid_feature(
            grid_shape, batch_size, num_channels, memory_format, device, dtype
        )

        # Create batch offsets for grid - these should match the pattern of coordinates
        # For grid coordinates, each batch has grid_shape[0] * grid_shape[1] * grid_shape[2] elements
        elements_per_batch = grid_shape[0] * grid_shape[1] * grid_shape[2]
        offsets = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
        for i in range(1, batch_size + 1):
            offsets[i] = offsets[i - 1] + elements_per_batch

        return cls(
            batched_tensor=tensor,
            offsets=offsets,
            memory_format=memory_format,
            grid_shape=grid_shape,
            num_channels=num_channels,
        )

    @staticmethod
    def from_conv_output(
        conv_output: Tensor,
        offsets: Tensor,
        memory_format: GridMemoryFormat,
        grid_shape: Tuple[int, int, int],
        num_channels: int,
    ) -> "GridFeatures":
        """Initialize GridFeatures from the output of a convolutional layer.

        Args:
            conv_output: Output tensor from a convolutional layer
            offsets: Batch offsets tensor
            memory_format: The memory format of the conv_output tensor
            grid_shape: Tuple of spatial dimensions (H, W, D)
            num_channels: The number of output channels from the convolution

        Returns:
            A new GridFeatures object with the given memory format
        """
        # Infer spatial dimensions based on the memory format
        rem = 0
        if memory_format == GridMemoryFormat.b_zc_x_y:
            B, DC, H, W = conv_output.shape
            D, rem = divmod(DC, num_channels)
            assert (
                D == grid_shape[2]
            ), f"Spatial dimension D does not match: {D} != {grid_shape[2]}"
        elif memory_format == GridMemoryFormat.b_xc_y_z:
            B, HC, W, D = conv_output.shape
            H, rem = divmod(HC, num_channels)
            assert (
                H == grid_shape[0]
            ), f"Spatial dimension H does not match: {H} != {grid_shape[0]}"
        elif memory_format == GridMemoryFormat.b_yc_x_z:
            B, WC, H, D = conv_output.shape
            W, rem = divmod(WC, num_channels)
            assert (
                W == grid_shape[1]
            ), f"Spatial dimension W does not match: {W} != {grid_shape[1]}"
        elif memory_format == GridMemoryFormat.b_c_x_y_z:
            B, C, H, W, D = conv_output.shape
            assert C == num_channels, f"Number of channels does not match: {C} != {num_channels}"
        else:
            raise ValueError(f"Unsupported memory format: {memory_format}")
        assert rem == 0, "Number of channels does not divide evenly"

        return GridFeatures(
            batched_tensor=conv_output,
            offsets=offsets,
            memory_format=memory_format,
            grid_shape=grid_shape,
            num_channels=num_channels,
        )

    @staticmethod
    def create_factorized_formats(
        features: "GridFeatures",
        formats: List[GridMemoryFormat] = [
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ],
    ) -> List["GridFeatures"]:
        """Create a list of factorized grid features from a single grid feature.

        This is a convenient method to create factorized representations from
        a single grid feature, typically used for creating a factorized grid
        geometry.

        Args:
            features: Source grid features to convert
            formats: List of factorized formats to convert to

        Returns:
            List of grid features with the requested factorized formats
        """
        return [features.to_memory_format(fmt) for fmt in formats]


def init_grid_feature(
    grid_shape: Tuple[int, int, int],
    batch_size: int,
    num_channels: int,
    memory_format: GridMemoryFormat,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Create a tensor for grid features in the specified memory format.

    Args:
        grid_shape: 3D grid resolution (H, W, D)
        batch_size: Number of batches
        num_channels: Number of feature channels
        memory_format: Memory format for the tensor
        device: Device to create tensor on
        dtype: Data type for the tensor

    Returns:
        Empty tensor in the requested format
    """

    H, W, D = grid_shape

    if memory_format == GridMemoryFormat.b_x_y_z_c:
        return torch.zeros((batch_size, H, W, D, num_channels), device=device, dtype=dtype)

    if memory_format == GridMemoryFormat.b_c_x_y_z:
        return torch.zeros((batch_size, num_channels, H, W, D), device=device, dtype=dtype)

    if memory_format == GridMemoryFormat.b_c_z_x_y:
        return torch.zeros((batch_size, num_channels, D, H, W), device=device, dtype=dtype)

    if memory_format == GridMemoryFormat.b_zc_x_y:
        return torch.zeros((batch_size, D * num_channels, H, W), device=device, dtype=dtype)

    if memory_format == GridMemoryFormat.b_xc_y_z:
        return torch.zeros((batch_size, H * num_channels, W, D), device=device, dtype=dtype)

    if memory_format == GridMemoryFormat.b_yc_x_z:
        return torch.zeros((batch_size, W * num_channels, H, D), device=device, dtype=dtype)

    raise ValueError(f"Unsupported memory format: {memory_format}")
