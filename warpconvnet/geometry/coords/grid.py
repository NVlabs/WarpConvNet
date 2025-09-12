# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from jaxtyping import Int

from dataclasses import dataclass
import numpy as np
from copy import deepcopy

import torch
from torch import Tensor

from warpconvnet.geometry.base.coords import Coords
from warpconvnet.geometry.base.batched import BatchedTensor
from warpconvnet.geometry.coords.ops.grid import create_grid_coordinates


@dataclass
class GridCoordsLazyInit:
    batch_size: int = 1
    flatten: bool = True
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def replace(self, **kwargs):
        return GridCoordsLazyInit(**{**self.__dict__, **kwargs})


class GridCoords(Coords):
    """Grid coordinates representation with lazy tensor creation.

    This implementation only creates the full coordinate tensor when it's actually
    needed, improving memory efficiency for large grids.
    """

    bounds: Tuple[Tensor, Tensor] = (torch.zeros(3), torch.ones(3))
    grid_shape: Tuple[int, int, int] = (1, 1, 1)

    def __init__(
        self,
        batched_tensor: Tensor,
        offsets: Tensor,
        grid_shape: Tuple[int, int, int],
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        lazy_init: Optional[GridCoordsLazyInit] = None,
    ):
        """Internal initialization method.

        Note: Users should prefer factory methods like from_shape(), from_tensor(),
        or create_regular_grid() instead of using this constructor directly.

        Args:
            batched_tensor: Coordinate tensor (real or dummy)
            offsets: Offset tensor (real or dummy)
            grid_shape: Shape of the grid (H, W, D) or (X, Y, Z)
            bounds: Min and max bounds (default: unit cube)
            lazy_init: Parameters for lazy initialization
        """
        # Set initialization flag first to avoid triggering lazy init
        # during parent class initialization
        self._is_initialized = lazy_init is None
        assert len(grid_shape) == 3, "Grid shape must be (H, W, D)"
        self.grid_shape = grid_shape

        # Set bounds
        if bounds is None:
            # Default to unit cube
            min_bound = torch.zeros(3)
            max_bound = torch.ones(3)
        else:
            assert len(bounds) == 2, "Bounds must be a tuple of (min, max)"
            assert (
                bounds[0].shape == bounds[1].shape
            ), "Min and max bounds must have the same shape"
            assert bounds[0].shape == (3,), "Bounds must be a tuple of (min, max)"
            min_bound = bounds[0].to("cpu")
            max_bound = bounds[1].to("cpu")

        self.bounds = (min_bound, max_bound)

        # Store lazy initialization parameters if needed
        if lazy_init is not None:
            self._lazy_params = lazy_init
        else:
            assert (
                batched_tensor.ndim == 5 or batched_tensor.ndim == 2
            ), f"Tensor must have shape (B,H,W,D,3) or (N,3). Got {batched_tensor.shape}"
            assert (
                batched_tensor.shape[-1] == 3
            ), f"Last dimension must be 3. Got {batched_tensor.shape}"
            if batched_tensor.ndim == 5:
                assert batched_tensor.shape[1:4] == grid_shape

        # Call parent's __init__ directly to avoid attribute lookup problems
        BatchedTensor.__init__(
            self,
            batched_tensor=batched_tensor,
            offsets=offsets,
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        offsets: Tensor,
        grid_shape: Tuple[int, int, int],
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> "GridCoords":
        """Create grid coordinates from an existing tensor.

        This is a convenience method for creating GridCoords from a pre-created
        coordinate tensor. Since the coordinate tensor is created, it is eagerly
        initialized.

        Args:
            tensor: Pre-created coordinate tensor of shape (B,H,W,D,3) or (N,3)
            offsets: Offset tensor for batched coordinates
            grid_shape: Shape of the grid (H, W, D)
            bounds: Optional min/max bounds

        Returns:
            GridCoords: Grid coordinates with eager initialization
        """
        return cls(
            batched_tensor=tensor,
            offsets=offsets,
            grid_shape=grid_shape,
            bounds=bounds,
        )

    @classmethod
    def from_shape(
        cls,
        grid_shape: Tuple[int, int, int],
        bounds: Optional[Tuple[Tensor, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        flatten: bool = True,
    ) -> "GridCoords":
        """Create grid coordinates lazily from a shape.

        Coordinates will only be created when actually needed.

        Args:
            grid_shape: Grid resolution (H, W, D)
            bounds: Min and max bounds (default: unit cube)
            batch_size: Number of batches
            device: Device to create tensors on
            flatten: Whether to flatten the coordinates

        Returns:
            GridCoords: Lazily initialized grid coordinates
        """
        # Create minimal dummy tensors
        dummy_tensor = torch.zeros((1, 3))
        # Create the offset using the batch size and grid shape
        num_elements = np.prod(grid_shape)
        offsets = torch.tensor(
            [i * num_elements for i in range(batch_size + 1)], dtype=torch.long, device="cpu"
        )

        lazy_init = GridCoordsLazyInit(
            batch_size=batch_size,
            flatten=flatten,
            device=device,
        )

        return cls(
            batched_tensor=dummy_tensor,
            offsets=offsets,
            grid_shape=grid_shape,
            bounds=bounds,
            lazy_init=lazy_init,
        )

    def _ensure_initialized(self):
        """Ensure coordinates are initialized."""
        if not self._is_initialized:
            # Create the actual coordinate tensor
            coords, offsets = create_grid_coordinates(
                self.grid_shape,
                self.bounds,
                self.batch_size,
                device=self._lazy_params.device,
                flatten=self._lazy_params.flatten,
                dtype=self._lazy_params.dtype,
            )

            # Replace dummy tensors
            # Need to use direct attribute access on parent class
            # to avoid triggering our custom __setattr__
            object.__setattr__(self, "batched_tensor", coords)
            object.__setattr__(self, "offsets", offsets)
            self._is_initialized = True

    @property
    def is_initialized(self):
        return hasattr(self, "_is_initialized") and self._is_initialized

    # Override __getattribute__ to intercept attribute access
    def __getattribute__(self, name):
        # First get the _is_initialized flag (if it exists)
        is_initialized = object.__getattribute__(self, "_is_initialized")

        # If accessing tensor attributes and not initialized, ensure initialization
        if not is_initialized and name in ("batched_tensor"):
            # Call _ensure_initialized through object.__getattribute__
            # to avoid recursion
            object.__getattribute__(self, "_ensure_initialized")()

        # Use default attribute lookup
        return super().__getattribute__(name)

    # Override other key methods
    def __getitem__(self, idx):
        # __getitem__ uses offsets, which will trigger lazy init via __getattribute__
        return super().__getitem__(idx)

    def to(self, device=None, dtype=None):
        """Handle device transfers while preserving lazy status."""
        if not self.is_initialized:
            # Create a new lazy GridCoords with updated device
            lazy_init = self._lazy_params.replace(
                device=device if device is not None else self._lazy_params.device,
                dtype=dtype if dtype is not None else self._lazy_params.dtype,
            )
            return self.__class__(
                batched_tensor=self.batched_tensor,
                offsets=self.offsets,
                grid_shape=self.grid_shape,
                bounds=self.bounds,
                lazy_init=lazy_init,
            )

        # Otherwise use standard implementation
        return super().to(device=device, dtype=dtype)

    def check(self):
        """Override check to allow lazy initialization."""
        if not self.is_initialized:
            # Skip check for uninitialized tensors
            return
        super().check()

    # Override properties defined in BatchedTensor to avoid initialization
    @property
    def device(self) -> torch.device:
        if not self.is_initialized:
            return self._lazy_params.device
        return super().device

    @property
    def shape(self):
        """Override shape to avoid initialization."""
        if not self.is_initialized:
            # Return expected shape without initializing tensor
            H, W, D = self.grid_shape
            if self._lazy_params.flatten:
                # If flattened, shape is (N, 3) where N is batch_size * H * W * D
                return (self._lazy_params.batch_size * H * W * D, 3)
            else:
                # If not flattened, shape would depend on batch size
                return (self._lazy_params.batch_size, H, W, D, 3)
        return super().shape

    @property
    def dtype(self):
        if not self.is_initialized:
            return self._lazy_params.dtype
        return super().dtype

    def half(self):
        if not self.is_initialized:
            return GridCoords.from_tensor(
                self.batched_tensor.half(), self.offsets, self.grid_shape, self.bounds
            )
        return super().half()

    def float(self):
        if not self.is_initialized:
            return GridCoords.from_tensor(
                self.batched_tensor.float(), self.offsets, self.grid_shape, self.bounds
            )
        return super().float()

    def double(self):
        if not self.is_initialized:
            return GridCoords.from_tensor(
                self.batched_tensor.double(), self.offsets, self.grid_shape, self.bounds
            )
        return super().double()

    def numel(self):
        """Override numel to avoid initialization."""
        if not self.is_initialized:
            # Calculate expected number of elements
            H, W, D = self.grid_shape
            return self._lazy_params.batch_size * H * W * D * 3
        return super().numel()

    def __len__(self):
        """Override len to avoid initialization."""
        if not self.is_initialized:
            # Calculate expected length
            H, W, D = self.grid_shape
            return self.batch_size * H * W * D
        return super().__len__()

    # Methods specific to GridCoords that need to be preserved
    def get_spatial_indices(
        self, flat_indices: Int[Tensor, "M"]  # noqa: F821
    ) -> Tuple[Int[Tensor, "M"], Int[Tensor, "M"], Int[Tensor, "M"]]:  # noqa: F821
        """Convert flattened indices to 3D spatial indices."""
        H, W, D = self.grid_shape

        # Calculate indices for each dimension
        h_indices = flat_indices // (W * D)
        w_indices = (flat_indices % (W * D)) // D
        d_indices = flat_indices % D

        return h_indices, w_indices, d_indices

    def get_flattened_indices(
        self,
        h_indices: Int[Tensor, "M"],  # noqa: F821
        w_indices: Int[Tensor, "M"],  # noqa: F821
        d_indices: Int[Tensor, "M"],  # noqa: F821
    ) -> Int[Tensor, "M"]:  # noqa: F821
        """Convert 3D spatial indices to flattened indices."""
        H, W, D = self.grid_shape

        return h_indices * (W * D) + w_indices * D + d_indices

    # Prevent unwanted initialization
    def __repr__(self):
        if not self.is_initialized:
            return f"{self.__class__.__name__}(grid_shape={self.grid_shape}, lazy=True, device={self.device})"
        return super().__repr__()

    def __str__(self):
        if not self.is_initialized:
            return f"{self.__class__.__name__}(grid_shape={self.grid_shape}, lazy=True)"
        return super().__str__()
