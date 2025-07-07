# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Factorized grid geometry implementation for FIGConvNet.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import torch
from torch import Tensor

from warpconvnet.geometry.features.grid import GridMemoryFormat
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.geometry.types.conversion.to_grid import points_to_grid
from warpconvnet.geometry.types.points import Points
from typing import Literal

# Import the REDUCTION_TYPES_STR type
try:
    from warpconvnet.ops.reductions import REDUCTION_TYPES_STR
except ImportError:
    REDUCTION_TYPES_STR = Literal["mean", "max", "sum", "mul"]


@dataclass
class FactorGrid:
    """A group of grid geometries with different factorized memory formats.

    This class implements the core concept of FIGConvNet where the 3D space
    is represented as multiple factorized 2D grids with different memory formats.

    Args:
        geometries: List of GridGeometry objects with different factorized formats
    """

    grids: List[Grid]
    _extra_attributes: Dict[str, Any] = field(default_factory=dict, init=True)  # Store extra args

    def __init__(self, grids: List[Grid], **kwargs):
        self.grids = grids

        # Validate we have at least one geometry
        assert len(grids) > 0, "At least one geometry must be provided"

        batch_size = grids[0].batch_size
        num_channels = grids[0].num_channels

        # Verify all geometries have the same batch size, channels, and grid shape
        for geo in grids:
            assert geo.batch_size == batch_size, "All geometries must have the same batch size"
            assert (
                geo.num_channels == num_channels
            ), "All geometries must have the same number of channels"

            # Ensure each geometry uses a factorized format
            assert geo.memory_format in [
                GridMemoryFormat.b_zc_x_y,
                GridMemoryFormat.b_xc_y_z,
                GridMemoryFormat.b_yc_x_z,
            ], f"Expected factorized format, got {geo.memory_format}"

        # Check for memory format duplicates
        memory_formats = [geo.memory_format for geo in grids]
        assert len(memory_formats) == len(
            set(memory_formats)
        ), "Each geometry must have a unique memory format"

        # Extra arguments for subclasses
        # First check _extra_attributes in kwargs. This happens when we use dataclasses.replace
        if "_extra_attributes" in kwargs:
            attr = kwargs.pop("_extra_attributes")
            assert isinstance(attr, dict), f"_extra_attributes must be a dictionary, got {attr}"
            # Update kwargs
            for k, v in attr.items():
                kwargs[k] = v
        self._extra_attributes = kwargs

    @classmethod
    def create_from_grid_shape(
        cls,
        grid_shapes: List[Tuple[int, int, int]],
        num_channels: int,
        memory_formats: List[Union[GridMemoryFormat, str]] = [
            "b_zc_x_y",
            "b_xc_y_z",
            "b_yc_x_z",
        ],
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FactorGrid":
        """Create a new factorized grid geometry with initialized geometries.

        Args:
            grid_shapes: List of grid resolutions (H, W, D)
            num_channels: Number of feature channels
            memory_formats: List of factorized formats to use
            bounds: Min and max bounds for the grid
            batch_size: Number of batches
            device: Device to create tensors on
            dtype: Data type for feature tensors

        Returns:
            Initialized factorized grid geometry
        """
        assert len(grid_shapes) == len(
            memory_formats
        ), "grid_shapes and memory_formats must have the same length"
        for grid_shape in grid_shapes:
            assert (
                isinstance(grid_shape, tuple) and len(grid_shape) == 3
            ), f"grid_shape: {grid_shape} must be a tuple of 3 integers."
        # First create a standard grid geometry
        geometries = []
        for grid_shape, memory_format in zip(grid_shapes, memory_formats):
            if isinstance(memory_format, str):
                memory_format = GridMemoryFormat(memory_format)
            geometry = Grid.from_shape(
                grid_shape,
                num_channels,
                memory_format=memory_format,
                bounds=bounds,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            geometries.append(geometry)

        # Then convert to each factorized format
        return cls(geometries)

    @property
    def batch_size(self) -> int:
        """Return the batch size of the geometries."""
        return self.grids[0].batch_size

    @property
    def num_channels(self) -> int:
        """Return the number of channels in the geometries."""
        return self.grids[0].num_channels

    @property
    def device(self) -> torch.device:
        """Return the device of the geometries."""
        return self.grids[0].device

    def to(self, device: torch.device) -> "FactorGrid":
        """Move all geometries to the specified device."""
        return FactorGrid([geo.to(device) for geo in self.grids])

    def __getitem__(self, idx: int) -> Grid:
        """Get a specific geometry from the group."""
        return self.grids[idx]

    def __len__(self) -> int:
        """Get the number of geometries in the group."""
        return len(self.grids)

    def __iter__(self):
        """Iterate over the grids."""
        return iter(self.grids)

    def __repr__(self) -> str:
        """String representation of the FactorGrid."""
        out_str = "FactorGrid("
        for grid in self.grids:
            out_str += f"\n\t{grid}"
        out_str += "\n)"
        return out_str

    def __add__(self, other: "FactorGrid") -> "FactorGrid":
        """Add two FactorGrid objects together element-wise."""
        assert len(self) == len(
            other
        ), f"FactorGrid lengths must match: {len(self)} != {len(other)}"
        new_grids = []
        for grid_a, grid_b in zip(self.grids, other.grids):
            # Add features together using Grid.replace()
            new_features = (
                grid_a.grid_features.batched_tensor + grid_b.grid_features.batched_tensor
            )
            new_grid = grid_a.replace(batched_features=new_features)
            new_grids.append(new_grid)
        return FactorGrid(new_grids)

    def get_by_format(self, memory_format: GridMemoryFormat) -> Optional[Grid]:
        """Get a geometry with the specified memory format.

        Args:
            memory_format: The memory format to look for

        Returns:
            The geometry with the requested format, or None if not found
        """
        for geo in self.grids:
            if geo.memory_format == memory_format:
                return geo
        return None

    @property
    def shapes(self) -> List[Dict[str, Union[int, Tuple[int, ...]]]]:
        """Get shape information for all geometries."""
        return [geo.shape for geo in self.grids]


def points_to_factor_grid(
    points: Points,
    grid_shapes: List[Tuple[int, int, int]],
    memory_formats: List[Union[GridMemoryFormat, str]] = [
        GridMemoryFormat.b_zc_x_y,
        GridMemoryFormat.b_xc_y_z,
        GridMemoryFormat.b_yc_x_z,
    ],
    bounds: Optional[Tuple[Tensor, Tensor]] = None,
    search_radius: Optional[float] = None,
    k: int = 8,
    search_type: Literal["radius", "knn", "voxel"] = "radius",
    reduction: REDUCTION_TYPES_STR = "mean",
) -> FactorGrid:
    """Convert points to a factorized grid representation.

    Args:
        points: Input point geometry
        grid_shapes: List of grid shapes for each factorized representation
        memory_formats: List of memory formats for each grid
        bounds: Min and max bounds for the grids
        search_radius: Search radius for radius search
        k: Number of neighbors for kNN search
        search_type: Search type ('radius', 'knn', 'voxel')
        reduction: Reduction method ('mean', 'max', 'sum', 'mul')

    Returns:
        FactorGrid: Factorized grid representation
    """
    assert len(grid_shapes) == len(
        memory_formats
    ), f"grid_shapes and memory_formats must have the same length: {len(grid_shapes)} != {len(memory_formats)}"

    # Convert points to individual grids with different shapes and formats
    grids = []
    for grid_shape, memory_format in zip(grid_shapes, memory_formats):
        if isinstance(memory_format, str):
            memory_format = GridMemoryFormat[memory_format]

        grid = points_to_grid(
            points=points,
            grid_shape=grid_shape,
            memory_format=memory_format,
            bounds=bounds,
            search_radius=search_radius,
            k=k,
            search_type=search_type,
            reduction=reduction,
        )
        grids.append(grid)

    return FactorGrid(grids)
