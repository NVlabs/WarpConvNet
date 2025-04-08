# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Literal, Optional, Tuple, Union
from jaxtyping import Float, Int

import torch
from torch import Tensor

from warpconvnet.geometry.base.coords import Coords
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.real import RealCoords
from warpconvnet.geometry.coords.search.cache import RealSearchCache
from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig
from warpconvnet.geometry.coords.sample import random_sample_per_batch
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING, morton_code
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.pad import PadFeatures
from warpconvnet.geometry.coords.search.continuous import (
    neighbor_search,
)
from warpconvnet.geometry.coords.ops.voxel import (
    voxel_downsample_csr_mapping,
    voxel_downsample_random_indices,
)
from warpconvnet.geometry.features.ops.convert import to_batched_features
from warpconvnet.nn.functional.encodings import sinusoidal_encoding
from warpconvnet.ops.reductions import REDUCTIONS, REDUCTION_TYPES_STR, row_reduction
from warpconvnet.geometry.types.conversion.to_voxels import points_to_voxels


class Points(Geometry):
    """
    Interface class for collections of points

    A point collection is a set of points in a geometric space
    (dim=1 (line), 2 (plane), 3 (space), 4 (space-time)).
    """

    def __init__(
        self,
        batched_coordinates: (
            List[Float[Tensor, "N 3"]] | Float[Tensor, "N 3"] | RealCoords
        ),  # noqa: F722,F821
        batched_features: (
            List[Float[Tensor, "N C"]]
            | Float[Tensor, "N C"]
            | Float[Tensor, "B M C"]
            | CatFeatures
            | PadFeatures
        ),  # noqa: F722,F821
        offsets: Optional[Int[Tensor, "B + 1"]] = None,  # noqa: F722,F821
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize a point collection with coordinates and features.
        """
        if isinstance(batched_coordinates, list):
            assert isinstance(
                batched_features, list
            ), "If coords is a list, features must be a list too."
            assert len(batched_coordinates) == len(batched_features)
            # Assert all elements in coords and features have same length
            assert all(
                len(c) == len(f) for c, f in zip(batched_coordinates, batched_features)
            ), "All elements in coords and features must have same length"
            batched_coordinates = RealCoords(batched_coordinates, device=device)
        elif isinstance(batched_coordinates, Tensor):
            assert (
                isinstance(batched_features, Tensor) and offsets is not None
            ), "If coordinate is a tensor, features must be a tensor and offsets must be provided."
            batched_coordinates = RealCoords(batched_coordinates, offsets=offsets, device=device)

        if isinstance(batched_features, list):
            batched_features = CatFeatures(batched_features, device=device)
        elif isinstance(batched_features, Tensor):
            batched_features = to_batched_features(
                batched_features, batched_coordinates.offsets, device=device
            )

        Geometry.__init__(
            self,
            batched_coordinates,
            batched_features,
            **kwargs,
        )

    def sort(
        self,
        voxel_size: float,
        ordering: POINT_ORDERING = POINT_ORDERING.Z_ORDER,
    ):
        """
        Sort the points according to the ordering provided.
        The voxel size defines the smallest discretization and points in the same voxel will have random order.
        """
        assert ordering == POINT_ORDERING.Z_ORDER, "Only Z-ordering is supported atm"
        # Warp uses int32 so only 10 bits per coordinate supported. Thus max 1024.
        assert self.device.type != "cpu", "Sorting is only supported on GPU"
        code, perm = morton_code(
            torch.floor(self.coordinate_tensor / voxel_size).int(),
            self.offsets,
            ordering,
        )
        kwargs = self.extra_attributes.copy()
        kwargs["ordering"] = ordering
        kwargs["code"] = code[perm]
        return self.__class__(
            batched_coordinates=RealCoords(
                batched_tensor=self.coordinate_tensor[perm],
                offsets=self.offsets,
            ),
            batched_features=CatFeatures(
                batched_tensor=self.feature_tensor[perm],
                offsets=self.offsets,
            ),
            **kwargs,
        )

    def voxel_downsample(
        self,
        voxel_size: float,
        reduction: Union[REDUCTIONS | REDUCTION_TYPES_STR] = REDUCTIONS.RANDOM,
    ) -> "Points":
        """
        Voxel downsample the coordinates
        """
        assert self.device.type != "cpu", "Voxel downsample is only supported on GPU"
        extra_args = self.extra_attributes
        extra_args["voxel_size"] = voxel_size
        assert isinstance(
            self.batched_features, CatFeatures
        ), "Voxel downsample is only supported for CatBatchedFeatures"
        if reduction == REDUCTIONS.RANDOM:
            to_unique_indicies, unique_offsets = voxel_downsample_random_indices(
                batched_points=self.coordinate_tensor,
                offsets=self.offsets,
                voxel_size=voxel_size,
            )
            return self.__class__(
                batched_coordinates=RealCoords(
                    batched_tensor=self.coordinate_tensor[to_unique_indicies],
                    offsets=unique_offsets,
                ),
                batched_features=CatFeatures(
                    batched_tensor=self.feature_tensor[to_unique_indicies],
                    offsets=unique_offsets,
                ),
                **extra_args,
            )

        # perm, down_offsets, vox_inices, vox_offsets = voxel_downsample_csr_mapping(
        #     batched_points=self.coordinate_tensor,
        #     offsets=self.offsets,
        #     voxel_size=voxel_size,
        # )
        (
            batch_indexed_down_coords,
            unique_offsets,
            to_csr_indices,
            to_csr_offsets,
            to_unique,
        ) = voxel_downsample_csr_mapping(
            batched_points=self.coordinate_tensor,
            offsets=self.offsets,
            voxel_size=voxel_size,
        )

        neighbors = RealSearchResult(to_csr_indices, to_csr_offsets)
        down_features = row_reduction(
            self.feature_tensor,
            neighbors.neighbor_row_splits,
            reduction,
        )

        return self.__class__(
            batched_coordinates=RealCoords(
                batched_tensor=self.coordinate_tensor[to_unique.to_unique_indices],
                offsets=unique_offsets,
            ),
            batched_features=CatFeatures(batched_tensor=down_features, offsets=unique_offsets),
            **extra_args,
        )

    def random_downsample(self, num_sample_points: int) -> "Points":
        """
        Downsample the coordinates to the specified number of points.

        If the batch size is N, the total number of output points is N * num_sample_points.
        """
        sampled_indices, sample_offsets = random_sample_per_batch(self.offsets, num_sample_points)
        return self.__class__(
            batched_coordinates=RealCoords(
                batched_tensor=self.coordinate_tensor[sampled_indices],
                offsets=sample_offsets,
            ),
            batched_features=CatFeatures(
                batched_tensor=self.feature_tensor[sampled_indices],
                offsets=sample_offsets,
            ),
            **self.extra_attributes,
        )

    def contiguous(self) -> "Points":
        """Ensure coordinates and features are contiguous in memory.

        This is important for memory access patterns and can improve
        performance for operations that require contiguous memory.

        Returns:
            Points: A new Points instance with contiguous tensors
        """
        if self.coordinate_tensor.is_contiguous() and self.feature_tensor.is_contiguous():
            return self

        return self.__class__(
            batched_coordinates=RealCoords(
                batched_tensor=self.coordinate_tensor.contiguous(),
                offsets=self.offsets,
            ),
            batched_features=CatFeatures(
                batched_tensor=self.feature_tensor.contiguous(),
                offsets=self.offsets,
            ),
            **self.extra_attributes,
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
            query_coords = self.batched_coordinates

        assert isinstance(query_coords, Coords), "query_coords must be BatchedCoordinates"

        # cache the neighbor search result
        if self.cache is not None:
            neighbor_search_result = self.cache.get(
                search_args, self.offsets, query_coords.offsets
            )
            if neighbor_search_result is not None:
                return neighbor_search_result

        neighbor_search_result = neighbor_search(
            self.coordinate_tensor,
            self.offsets,
            query_coords.batched_tensor,
            query_coords.offsets,
            search_args,
        )
        if self.cache is None:
            self._extra_attributes["_cache"] = RealSearchCache()
        self.cache.put(search_args, self.offsets, query_coords.offsets, neighbor_search_result)
        return neighbor_search_result

    @property
    def voxel_size(self):
        return self._extra_attributes.get("voxel_size", None)

    @property
    def ordering(self):
        return self._extra_attributes.get("ordering", None)

    @classmethod
    def from_list_of_coordinates(
        cls,
        coordinates: List[Float[Tensor, "N 3"]],
        features: Optional[List[Float[Tensor, "N C"]]] = None,
        encoding_channels: Optional[int] = None,
        encoding_range: Optional[Tuple[float, float]] = None,
        encoding_dim: Optional[int] = -1,
    ):
        """
        Create a point collection from a list of coordinates.
        """
        # if the input is a tensor, expand it to a list of tensors
        if isinstance(coordinates, Tensor):
            coordinates = list(coordinates)  # this expands the tensor to a list of tensors

        if features is None:
            assert (
                encoding_range is not None
            ), "Encoding range must be provided if encoding channels are provided"
            features = [
                sinusoidal_encoding(coordinates, encoding_channels, encoding_range, encoding_dim)
                for coordinates in coordinates
            ]

        # Create BatchedContinuousCoordinates
        batched_coordinates = RealCoords(coordinates)
        # Create CatBatchedFeatures
        batched_features = CatFeatures(features)

        return cls(batched_coordinates, batched_features)

    def to_voxels(self, voxel_size: float) -> "Voxels":
        """
        Convert the point collection to a spatially sparse tensor.
        """
        return points_to_voxels(self, voxel_size)
