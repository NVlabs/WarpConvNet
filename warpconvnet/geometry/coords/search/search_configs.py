# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

from jaxtyping import Int
from torch import Tensor


class RealSearchMode(Enum):
    RADIUS = "radius"
    KNN = "knn"
    VOXEL = "voxel"


class IntSearchMode(Enum):
    MANHATTAN = "manhattan_distance"
    CUSTOM_OFFSETS = "custom_offsets"


@dataclass
class RealSearchConfig:
    """
    Wrapper for the input of a neighbor search operation.
    """

    # The mode of the neighbor search
    mode: RealSearchMode
    # The radius for radius search
    radius: Optional[float]
    # The number of neighbors for knn search
    knn_k: Optional[int]
    # Grid dim
    grid_dim: Optional[int | Tuple[int, int, int]]

    def __init__(
        self,
        mode: RealSearchMode,
        radius: Optional[float] = None,
        knn_k: Optional[int] = None,
        grid_dim: Optional[int | Tuple[int, int, int]] = None,
    ):
        if isinstance(mode, str):
            mode = RealSearchMode(mode)

        self.mode = mode
        self.radius = radius
        self.knn_k = knn_k
        self.grid_dim = grid_dim

    def __repr__(self):
        if self.mode == RealSearchMode.RADIUS:
            out_str = f"{self.mode.name}({self.radius})"
        elif self.mode == RealSearchMode.KNN:
            out_str = f"{self.mode._name}({self.knn_k})"
        return out_str

    def replace(
        self,
        mode: Optional[RealSearchMode] = None,
        radius: Optional[float] = None,
        k: Optional[int] = None,
        grid_dim: Optional[int | Tuple[int, int, int]] = None,
    ):
        return RealSearchConfig(
            mode=mode if mode is not None else self.mode,
            radius=radius if radius is not None else self.radius,
            knn_k=k if k is not None else self.knn_k,
            grid_dim=grid_dim if grid_dim is not None else self.grid_dim,
        )

    def __hash__(self):
        return int(hash(self.mode) ^ hash(self.radius) ^ hash(self.knn_k))

    def __eq__(self, other):
        if not isinstance(other, RealSearchConfig):
            return False
        return (
            self.mode == other.mode
            and self.radius == other.radius
            and self.knn_k == other.knn_k
        )


@dataclass
class IntSearchConfig:

    mode: IntSearchMode
    kernel_sizes: Optional[int | Tuple[int, ...]]
    offsets: Optional[Int[Tensor, "K 3"]]

    def __init__(
        self,
        mode: IntSearchMode,
        kernel_sizes: Optional[int | Tuple[int, ...]] = None,
        offsets: Optional[Int[Tensor, "K 3"]] = None,
    ):
        self.mode = mode
        if mode == IntSearchMode.MANHATTAN_DISTANCE:
            assert (
                kernel_sizes is not None
            ), "Distance threshold must be provided for manhattan distance search"
            self.kernel_sizes = kernel_sizes
        elif mode == IntSearchMode.CUSTOM_OFFSETS:
            assert (
                offsets is not None
            ), "Offsets must be provided for custom offsets search"
            self.offsets = offsets
        else:
            raise ValueError(f"Invalid neighbor search mode: {mode}")
