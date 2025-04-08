# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from jaxtyping import Float, Int

import torch

from warpconvnet.geometry.base.coords import Coords
from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
from warpconvnet.geometry.coords.sample import random_sample_per_batch
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING, morton_code
from warpconvnet.geometry.coords.search.continuous import (
    neighbor_search,
)
from warpconvnet.geometry.coords.ops.voxel import voxel_downsample_random_indices


class RealCoords(Coords):
    def check(self):
        Coords.check(self)
        assert self.batched_tensor.shape[-1] == 3, "Coordinates must have 3 dimensions"

    def voxel_downsample(self, voxel_size: float):
        """
        Voxel downsample the coordinates
        """
        assert self.device.type != "cpu", "Voxel downsample is only supported on GPU"
        perm, down_offsets = voxel_downsample_random_indices(
            coords=self.batched_tensor,
            voxel_size=voxel_size,
            offsets=self.offsets,
        )
        return self.__class__(tensors=self.batched_tensor[perm], offsets=down_offsets)

    def downsample(self, sample_points: int):
        """
        Downsample the coordinates to the specified number of points
        """
        sampled_indices, sample_offsets = random_sample_per_batch(self.offsets, sample_points)
        return self.__class__(
            batched_tensor=self.batched_tensor[sampled_indices], offsets=sample_offsets
        )

    def neighbors(
        self,
        search_args: RealSearchConfig,
        query_coords: Optional["Coords"] = None,
    ) -> RealSearchResult:
        """
        Returns CSR format neighbor indices
        """
        if query_coords is None:
            query_coords = self

        assert isinstance(query_coords, Coords), "query_coords must be BatchedCoordinates"

        return neighbor_search(
            self.batched_tensor,
            self.offsets,
            query_coords.batched_tensor,
            query_coords.offsets,
            search_args,
        )

    def sort(
        self,
        ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
        voxel_size: Optional[float] = None,
    ):
        """
        Sort the points according to the ordering provided.
        The voxel size defines the smallest descritization and points in the same voxel will have random order.
        """
        assert ordering == POINT_ORDERING.Z_ORDER, "Only Z-ordering is supported atm"
        # Warp uses int32 so only 10 bits per coordinate supported. Thus max 1024.
        assert self.device.type != "cpu", "Sorting is only supported on GPU"
        code, perm = morton_code(
            torch.floor(self.batched_tensor / voxel_size).int(), self.offsets, ordering
        )
        return self.__class__(
            batched_tensor=self.batched_tensor[perm],
            offsets=self.offsets,
        )
