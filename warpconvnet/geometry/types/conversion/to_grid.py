# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Tuple, TYPE_CHECKING
from jaxtyping import Float, Int

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.utils.ravel import ravel_multi_index
from warpconvnet.ops.reductions import REDUCTION_TYPES_STR, REDUCTIONS

# Import types for type checking only to avoid circular imports
if TYPE_CHECKING:
    from warpconvnet.geometry.coords.search.search_results import RealSearchResult
    from warpconvnet.geometry.coords.grid import GridCoords
    from warpconvnet.geometry.features.grid import GridFeatures
    from warpconvnet.geometry.types.grid import GridMemoryFormat, Grid
    from warpconvnet.geometry.types.points import Points
    from warpconvnet.geometry.types.voxels import Voxels


def points_to_closest_voxel_mapping(points, grid_shape, bounds, memory_format=None):
    """Find the closest voxel center for each point using CUDA acceleration.

    Args:
        points: Points object with coordinates and offsets
        grid_shape: Tuple of (H, W, D) for the grid shape
        bounds: Tuple of (min_bounds, max_bounds) tensors
        memory_format: Grid memory format (unused but kept for compatibility)

    Returns:
        Tensor of voxel indices for each point
    """
    assert points.device.type == "cuda", "points_to_closest_voxel_mapping only supports CUDA"
    # Extract coordinates and offsets
    coords = points.coordinate_tensor
    offsets = points.offsets

    # Convert grid shape to tensor
    grid_shape_tensor = torch.tensor(grid_shape, dtype=torch.int32)

    # Get bounds
    if bounds is None:
        # Compute bounds from points if not provided
        min_coords = coords.min(dim=0)[0]
        max_coords = coords.max(dim=0)[0]
        # Add some padding
        padding = (max_coords - min_coords) * 0.01
        bounds_min = min_coords - padding
        bounds_max = max_coords + padding
    else:
        bounds_min = bounds[0]
        bounds_max = bounds[1]

    # Ensure bounds are on the correct device
    bounds_min = bounds_min.to(device=points.device, dtype=coords.dtype)
    bounds_max = bounds_max.to(device=points.device, dtype=coords.dtype)

    # Call CUDA kernel
    voxel_indices = _C.utils.points_to_closest_voxel_mapping(
        coords,
        offsets.to(device=points.device, dtype=torch.int32),
        grid_shape_tensor,
        bounds_min,
        bounds_max,
    )
    # Convert to int64 as expected by PyTorch operations
    return voxel_indices.to(torch.int64)


def _point_to_grid_mapping(
    point_coords: Float[Tensor, "N 3"],  # noqa: F821
    point_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    grid_coords: Float[Tensor, "M 3"],  # noqa: F821
    grid_offsets: Int[Tensor, "B + 1"],  # noqa: F821
    search_radius: Optional[float] = None,
    k: Optional[int] = None,
    search_type: Literal["radius", "knn", "voxel"] = "radius",
    search_grid_dim: Optional[int] = None,
) -> "RealSearchResult":
    """Compute the mapping from points to grid voxels.

    Args:
        point_coords: Point coordinates (N, 3)
        point_offsets: Point offsets (B+1)
        grid_coords: Grid coordinates (M, 3)
        grid_offsets: Grid offsets (B+1)
        search_radius: Search radius for radius search
        k: Number of neighbors for kNN search
        search_type: Search type ('radius' or 'knn')
        search_grid_dim: Grid dimension for radius search

    Returns:
        RealSearchResult: Search results class
    """
    from warpconvnet.geometry.coords.search.continuous import neighbor_search
    from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig, RealSearchMode
    from warpconvnet.geometry.coords.search.search_results import RealSearchResult

    # Create search config
    if search_type == "radius":
        assert search_radius is not None, "Search radius must be provided for radius search"
        search_mode = RealSearchMode.RADIUS
        search_config = RealSearchConfig(
            mode=search_mode, radius=search_radius, grid_dim=search_grid_dim
        )
    elif search_type == "knn":
        search_mode = RealSearchMode.KNN
        search_config = RealSearchConfig(mode=search_mode, knn_k=k)
    elif search_type == "voxel":
        # Use the closest voxel center to the point
        raise ValueError(
            "Voxel search should not be used for _point_to_grid_mapping as the grid_shape and bounds should be used to efficiently find the closest voxel center"
        )
    else:
        raise ValueError(f"Unsupported search type: {search_type}")

    # Note: We're reversing the query and reference compared to the original function
    # as we want to find the points near each grid voxel
    results: RealSearchResult = neighbor_search(
        ref_positions=point_coords,
        ref_offsets=point_offsets,
        query_positions=grid_coords,
        query_offsets=grid_offsets,
        search_args=search_config,
    )

    return results


def _process_radius_search_results(
    search_results: "RealSearchResult",
    neighbor_indices: Int[Tensor, "Q"],  # noqa: F821
    point_features: Float[Tensor, "N C"],  # noqa: F821
    grid_features: Float[Tensor, "B H W D C"],  # noqa: F821
    grid_coords: "GridCoords",
    W: int,
    D: int,
    reduction: REDUCTION_TYPES_STR,
) -> Float[Tensor, "B H W D C"]:  # noqa: F821
    """Process radius search results and update grid features.

    Args:
        search_results: Search results containing neighbor information
        neighbor_indices: Indices of neighboring points
        point_features: Features of all points
        grid_features: Output grid features tensor to be modified
        grid_coords: Grid coordinates
        W: Grid width
        D: Grid depth
        reduction: Reduction method to apply ('mean', 'max', 'sum', 'mul')
    """
    # TODO(cchoy) 2025-04-08: accelerate with a warp kernel
    # Process each grid cell
    from warpconvnet.geometry.coords.search.search_results import RealSearchResult

    assert isinstance(
        search_results, RealSearchResult
    ), f"Expected RealSearchResult, got {type(search_results)}"
    reduction_type = REDUCTIONS(reduction)

    for grid_idx, start_idx in enumerate(search_results.row_offsets[:-1]):
        end_idx = search_results.row_offsets[grid_idx + 1]

        # Skip if no neighbors found
        if start_idx == end_idx:
            continue

        # Get indices of neighboring points
        point_indices = neighbor_indices[start_idx:end_idx]

        # Get the corresponding point features
        neighbor_features = point_features[point_indices]

        # Calculate batch and cell indices
        batch_idx = torch.searchsorted(grid_coords.offsets, grid_idx, right=True) - 1
        cell_idx = grid_idx - grid_coords.offsets[batch_idx]

        # Compute grid indices
        h_idx = cell_idx // (W * D)
        w_idx = (cell_idx % (W * D)) // D
        d_idx = cell_idx % D

        # Apply reduction
        if reduction_type == REDUCTIONS.MEAN:
            reduced_features = torch.mean(neighbor_features, dim=0)
        elif reduction_type == REDUCTIONS.MAX:
            reduced_features = torch.max(neighbor_features, dim=0)[0]
        elif reduction_type == REDUCTIONS.SUM:
            reduced_features = torch.sum(neighbor_features, dim=0)
        elif reduction_type == REDUCTIONS.MUL:
            reduced_features = torch.prod(neighbor_features, dim=0)
        else:
            raise ValueError(f"Unsupported reduction: {reduction} for radius search")

        # Set grid features
        grid_features[batch_idx, h_idx, w_idx, d_idx] = reduced_features

    return grid_features


def _process_knn_search_results(
    neighbor_indices: Tensor,
    point_features: Tensor,
    grid_features: Tensor,
    grid_coords: "GridCoords",
    H: int,
    W: int,
    D: int,
    reduction: REDUCTION_TYPES_STR,
) -> None:
    """Process KNN search results and update grid features.

    Args:
        neighbor_indices: Indices of neighboring points
        point_features: Features of all points
        grid_features: Output grid features tensor to be modified
        grid_coords: Grid coordinates
        H: Grid height
        W: Grid width
        D: Grid depth
        reduction: Reduction method to apply ('mean', 'max', 'sum')
    """
    # TODO(cchoy) 2025-04-08: accelerate with a warp kernel
    reduction_type = REDUCTIONS(reduction)
    total_grid_cells = H * W * D

    # Calculate batch and spatial indices for all grid cells
    # Make sure arange is on the same device as offsets
    batch_indices = (
        torch.searchsorted(
            grid_coords.offsets,
            torch.arange(total_grid_cells, device=grid_coords.offsets.device),
            right=True,
        )
        - 1
    )
    cell_indices = (
        torch.arange(total_grid_cells, device=grid_coords.offsets.device)
        - grid_coords.offsets[batch_indices]
    )

    # Compute 3D grid indices
    h_indices = cell_indices // (W * D)
    w_indices = (cell_indices % (W * D)) // D
    d_indices = cell_indices % D

    # Process each grid cell
    for i in range(total_grid_cells):
        # Get neighboring point indices
        point_indices = neighbor_indices[i]

        # Skip invalid indices
        valid_mask = point_indices >= 0
        if not torch.any(valid_mask):
            continue

        point_indices = point_indices[valid_mask]

        # Get point features
        neighbor_features = point_features[point_indices]

        # Apply reduction
        if reduction_type == REDUCTIONS.MEAN:
            reduced_features = torch.mean(neighbor_features, dim=0)
        elif reduction_type == REDUCTIONS.MAX:
            reduced_features = torch.max(neighbor_features, dim=0)[0]
        elif reduction_type == REDUCTIONS.SUM:
            reduced_features = torch.sum(neighbor_features, dim=0)
        elif reduction_type == REDUCTIONS.MUL:
            reduced_features = torch.prod(neighbor_features, dim=0)
        else:
            raise ValueError(f"Unsupported reduction: {reduction} for knn search")

        # Set grid features
        batch_idx = batch_indices[i]
        h_idx = h_indices[i]
        w_idx = w_indices[i]
        d_idx = d_indices[i]

        grid_features[batch_idx, h_idx, w_idx, d_idx] = reduced_features

    return grid_features


def _points_to_grid_features(
    points: "Points",
    grid_coords: "GridCoords",
    memory_format: Optional["GridMemoryFormat"] = None,
    search_radius: Optional[float] = None,
    k: int = 8,
    search_type: Literal["radius", "knn"] = "radius",
    reduction: REDUCTION_TYPES_STR = "mean",
) -> "GridFeatures":
    """Convert point features to grid features.

    Args:
        points: Input point geometry
        grid_coords: Grid coordinates
        memory_format: Memory format for grid features
        search_radius: Search radius for radius search
        k: Number of neighbors for kNN search
        search_type: Search type ('radius', 'knn', 'voxel')
        reduction: Reduction method ('mean', 'max', 'sum', 'mul')

    Returns:
        GridFeatures: Grid features with the specified memory format
    """
    from warpconvnet.geometry.types.grid import GridMemoryFormat
    from warpconvnet.geometry.types.points import Points
    from warpconvnet.geometry.coords.grid import GridCoords
    from warpconvnet.geometry.features.grid import GridFeatures

    assert isinstance(points, Points), f"Expected Points, got {type(points)}"
    assert isinstance(grid_coords, GridCoords), f"Expected GridCoords, got {type(grid_coords)}"

    if memory_format is None:
        memory_format = GridMemoryFormat.b_x_y_z_c

    batch_size = points.batch_size
    device = points.device
    num_channels = points.num_channels

    # Map points to grid
    point_coords_tensor = points.coordinate_tensor
    point_offsets = points.offsets

    # Flatten grid coords
    assert (
        grid_coords.batched_tensor.ndim == 2
    ), f"Grid coords must be 2D, got {grid_coords.batched_tensor.ndim}D"
    grid_coords_tensor = grid_coords.batched_tensor
    grid_offsets = grid_coords.offsets

    search_results = _point_to_grid_mapping(
        point_coords_tensor,
        point_offsets,
        grid_coords_tensor,
        grid_offsets,
        search_radius=search_radius,
        k=k,
        search_type=search_type,
        search_grid_dim=None,  # For hash grid
    )

    # Get neighbor indices
    neighbor_indices: Int[Tensor, "Q"] = search_results.neighbor_indices  # noqa: F821

    # Gather point features
    point_features = points.batched_features.batched_tensor

    # Initialize grid features
    H, W, D = grid_coords.grid_shape

    if memory_format == GridMemoryFormat.b_x_y_z_c:
        grid_tensor = torch.zeros(
            (batch_size, H, W, D, num_channels),
            device=device,
            dtype=point_features.dtype,
        )
    else:
        # For other memory formats, we'll first create in standard format
        # and then convert
        grid_tensor = torch.zeros(
            (batch_size, H, W, D, num_channels),
            device=device,
            dtype=point_features.dtype,
        )

    # Process search results differently based on search type
    if search_type == "radius":
        grid_tensor = _process_radius_search_results(
            search_results,
            neighbor_indices,
            point_features,
            grid_tensor,
            grid_coords,
            W,
            D,
            reduction,
        )
    else:  # knn search
        grid_tensor = _process_knn_search_results(
            neighbor_indices,
            point_features,
            grid_tensor,
            grid_coords,
            H,
            W,
            D,
            reduction,
        )

    # Convert to target memory format if needed
    if memory_format != GridMemoryFormat.b_x_y_z_c:
        B = batch_size
        C = num_channels

        if memory_format == GridMemoryFormat.b_c_x_y_z:
            grid_tensor = grid_tensor.permute(0, 4, 1, 2, 3)
        elif memory_format == GridMemoryFormat.b_zc_x_y:
            grid_tensor = grid_tensor.permute(0, 3, 4, 1, 2).reshape(B, D * C, H, W)
        elif memory_format == GridMemoryFormat.b_xc_y_z:
            grid_tensor = grid_tensor.permute(0, 1, 4, 2, 3).reshape(B, H * C, W, D)
        elif memory_format == GridMemoryFormat.b_yc_x_z:
            grid_tensor = grid_tensor.permute(0, 2, 4, 1, 3).reshape(B, W * C, H, D)
        else:
            raise ValueError(f"Unsupported memory format: {memory_format}")

    # Create GridFeatures
    return GridFeatures(
        grid_tensor, grid_coords.offsets, memory_format, grid_coords.grid_shape, num_channels
    )


def points_to_grid(
    points: "Points",
    grid_shape: Tuple[int, int, int],
    memory_format: Optional["GridMemoryFormat"] = None,
    bounds: Optional[Tuple[Tensor, Tensor]] = None,
    search_radius: Optional[float] = None,
    k: int = 8,
    search_type: Literal["radius", "knn", "voxel"] = "radius",
    reduction: REDUCTION_TYPES_STR = "mean",
) -> "Grid":
    """Convert point features to a grid.

    Args:
        points: Input point geometry
        grid_shape: Grid shape (H, W, D)
        memory_format: Memory format for grid features
        bounds: Min and max bounds for the grid
        search_radius: Search radius for radius search
        k: Number of neighbors for kNN search
        search_type: Search type ('radius' or 'knn')
        reduction: Reduction method ('mean', 'max', 'sum', 'mul')

    Returns:
        Grid: Grid with the specified memory format
    """
    from warpconvnet.geometry.coords.grid import GridCoords
    from warpconvnet.geometry.types.grid import Grid, GridMemoryFormat

    if memory_format is None:
        memory_format = GridMemoryFormat.b_x_y_z_c

    # Create grid coordinates
    batch_size = points.batch_size
    device = points.device

    if search_type == "voxel":
        to_voxel_indices = points_to_closest_voxel_mapping(
            points, grid_shape, bounds, memory_format
        )

        # Create grid structure
        from warpconvnet.geometry.features.grid import GridFeatures

        H, W, D = grid_shape
        num_channels = points.num_channels
        total_voxels = batch_size * H * W * D

        # Initialize grid tensor with zeros
        grid_tensor = torch.zeros(
            (batch_size, H, W, D, num_channels),
            device=device,
            dtype=points.batched_features.batched_tensor.dtype,
        )

        # Flatten grid for indexing
        flat_grid = grid_tensor.view(total_voxels, num_channels)

        # Apply reduction by voxel
        if reduction == "mean":
            # Count points per voxel
            counts = torch.zeros(total_voxels, device=device)
            counts.scatter_add_(
                0, to_voxel_indices, torch.ones_like(to_voxel_indices, dtype=torch.float32)
            )
            # Sum features per voxel
            flat_grid.scatter_add_(
                0,
                to_voxel_indices.unsqueeze(1).expand(-1, num_channels),
                points.batched_features.batched_tensor,
            )
            # Average where count > 0
            mask = counts > 0
            flat_grid[mask] /= counts[mask].unsqueeze(1)
        elif reduction == "max":
            # Use scatter_reduce with reduce='amax'
            flat_grid.scatter_reduce_(
                0,
                to_voxel_indices.unsqueeze(1).expand(-1, num_channels),
                points.batched_features.batched_tensor,
                reduce="amax",
            )
        elif reduction == "sum":
            flat_grid.scatter_add_(
                0,
                to_voxel_indices.unsqueeze(1).expand(-1, num_channels),
                points.batched_features.batched_tensor,
            )
        else:
            raise ValueError(f"Unsupported reduction: {reduction} for voxel search")

        # Convert to target memory format if needed
        if memory_format != GridMemoryFormat.b_x_y_z_c:
            B = batch_size
            C = num_channels
            if memory_format == GridMemoryFormat.b_c_x_y_z:
                grid_tensor = grid_tensor.permute(0, 4, 1, 2, 3)
            elif memory_format == GridMemoryFormat.b_zc_x_y:
                grid_tensor = grid_tensor.permute(0, 3, 4, 1, 2).reshape(B, D * C, H, W)
            elif memory_format == GridMemoryFormat.b_xc_y_z:
                grid_tensor = grid_tensor.permute(0, 1, 4, 2, 3).reshape(B, H * C, W, D)
            elif memory_format == GridMemoryFormat.b_yc_x_z:
                grid_tensor = grid_tensor.permute(0, 2, 4, 1, 3).reshape(B, W * C, H, D)
            else:
                raise ValueError(f"Unsupported memory format: {memory_format}")

        # Create grid coordinates
        grid_coords = GridCoords.from_shape(
            grid_shape=grid_shape,
            bounds=bounds,
            batch_size=batch_size,
            device=device,
            flatten=False,
        )

        # Create grid features
        grid_features = GridFeatures(
            grid_tensor, grid_coords.offsets, memory_format, grid_shape, num_channels
        )

        # Return Grid object
        return Grid(grid_coords, grid_features, memory_format)

    grid_coords = GridCoords.from_shape(
        grid_shape=grid_shape,
        bounds=bounds,
        batch_size=batch_size,
        device=device,
        flatten=True,
    )

    # Convert point features to grid features
    grid_features = _points_to_grid_features(
        points,
        grid_coords,
        memory_format=memory_format,
        search_radius=search_radius,
        k=k,
        search_type=search_type,
        reduction=reduction,
    )

    # Create and return grid geometry
    return Grid(grid_coords, grid_features, memory_format)


def voxels_to_grid(
    voxels: "Voxels",
    grid_shape: Tuple[int, int, int],
    grid_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    memory_format: Optional["GridMemoryFormat"] = None,
    search_type: Literal["voxel"] = "voxel",
    reduction: REDUCTION_TYPES_STR = "mean",
) -> "Grid":
    """Convert voxel features to a grid.

    Args:
        voxels: Input voxel geometry
        grid_shape: Grid shape (H, W, D)
        memory_format: Memory format for grid features
        bounds: Min and max bounds for the grid
        search_radius: Search radius for radius search
        k: Number of neighbors for kNN search
        search_type: Search type ('cell')
        reduction: Reduction method defined in REDUCTION_TYPES_STR

    Returns:
        Grid: Grid with the specified memory format
    """
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.geometry.types.grid import Grid, GridMemoryFormat, GridFeatures

    assert isinstance(voxels, Voxels), f"Expected Voxels, got {type(voxels)}"

    if memory_format is None:
        memory_format = GridMemoryFormat.b_x_y_z_c

    # Create grid coordinates
    batch_size = voxels.batch_size
    device = voxels.device

    grid = Grid.from_shape(
        grid_shape=grid_shape,
        num_channels=voxels.num_channels,
        bounds=grid_bounds,
        batch_size=batch_size,
        device=device,
        flatten=True,
    )
    grid_feats = grid.features

    if search_type == "voxel":
        from warpconvnet.geometry.coords.ops.batch_index import batch_index_from_offset

        # Find cell indices that voxels belong to
        voxel_coords: Int[Tensor, "N 3"] = voxels.coordinate_tensor  # noqa: F821
        voxel_offsets: Int[Tensor, "B + 1"] = voxels.offsets  # noqa: F821
        device: torch.device = voxels.device
        # Use the coordinate to find the cell index
        grid_dimension = grid_bounds[1] - grid_bounds[0]
        grid_shape_pt = torch.tensor(grid_shape)
        cell_size = grid_dimension / grid_shape_pt
        # Find the cell index for each voxel
        cell_indices: Int[Tensor, "N 3"] = (  # noqa: F821
            voxel_coords - grid_bounds[0].to(device)
        ) / cell_size.to(device)
        # Out of bound indices should be clamped to the nearest bound
        cell_indices = torch.floor(cell_indices).long()
        # Clamp to Nx3 to 3-vector applied on the last dimension separately
        cell_indices = torch.minimum(cell_indices, (grid_shape_pt - 1).to(device))
        # Convert to integer indices
        batch_indices = batch_index_from_offset(voxel_offsets)

        # multi index ravel
        cell_flat_indices = ravel_multi_index(cell_indices, grid_shape_pt)
        batched_flat_indices = (
            batch_indices * grid_shape_pt[0] * grid_shape_pt[1] * grid_shape_pt[2]
        )
        batched_cell_flat_indices = cell_flat_indices + batched_flat_indices.to(device)

        # fill in the grid features using the cell indices
        # BxHxWxDxC -> (BxHxWxD)xC
        flat_grid_feats_view = grid_feats.view(-1, grid_feats.shape[4])
        # This will randomly select one of the voxel features for each grid cell
        # TODO(cchoy) 2025-04-10: reduction should be applied here
        flat_grid_feats_view[batched_cell_flat_indices] = voxels.batched_features.batched_tensor

        # Restore the
        # Replace the grid features with the reduced features
        return grid.replace(
            batched_features=GridFeatures(
                grid_feats,
                grid.offsets,
                memory_format=GridMemoryFormat.b_x_y_z_c,
                grid_shape=grid_shape,
                num_channels=voxels.num_channels,
            ).to_memory_format(memory_format)
        )
    else:
        raise ValueError(f"Unsupported search type: {search_type}")
