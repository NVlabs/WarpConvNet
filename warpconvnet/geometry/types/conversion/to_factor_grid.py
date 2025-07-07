# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Literal, Optional, Tuple, Union


def points_to_factor_grid(
    points: "Points",
    grid_shapes: List[Tuple[int, int, int]],
    memory_formats: List[Union["GridMemoryFormat", str]] = [
        "b_zc_x_y",
        "b_xc_y_z",
        "b_yc_x_z",
    ],
    bounds: Optional[Tuple[Tensor, Tensor]] = None,
    search_radius: Optional[float] = None,
    k: int = 8,
    search_type: Literal["radius", "knn", "voxel"] = "radius",
    reduction: str = "mean",
) -> "FactorGrid":
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
    # To prevent circular import, import here
    from warpconvnet.geometry.types.conversion.to_grid import points_to_grid
    from warpconvnet.geometry.types.factor_grid import FactorGrid
    from warpconvnet.geometry.types.grid import GridMemoryFormat

    assert len(grid_shapes) == len(
        memory_formats
    ), f"grid_shapes and memory_formats must have the same length: {len(grid_shapes)} != {len(memory_formats)}"

    # Convert points to individual grids with different shapes and formats
    grids = []
    for grid_shape, memory_format in zip(grid_shapes, memory_formats):
        if isinstance(memory_format, str):
            memory_format = GridMemoryFormat(memory_format)

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
