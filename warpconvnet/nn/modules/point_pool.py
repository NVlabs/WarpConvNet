# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Tuple, Union

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.search.search_results import RealSearchResult
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.point_pool import point_pool
from warpconvnet.nn.functional.point_unpool import FEATURE_UNPOOLING_MODE, point_unpool
from warpconvnet.ops.reductions import REDUCTIONS

__all__ = [
    "PointPoolBase",
    "PointMaxPool",
    "PointAvgPool",
    "PointSumPool",
    "PointUnpool",
]


class PointPoolBase(BaseSpatialModule):
    """Base module for pooling points or voxels.

    Parameters
    ----------
    reduction : str or :class:`REDUCTIONS`, optional
        Reduction method used when merging features. Defaults to ``REDUCTIONS.MAX``.
    downsample_max_num_points : int, optional
        Maximum number of points to keep when downsampling.
    downsample_voxel_size : float, optional
        Size of voxels used for downsampling.
    return_type : {"point", "sparse"}, optional
        Output geometry type. Defaults to ``"point"``.
    unique_method : {"torch", "ravel", "morton"}, optional
        Method used to find unique voxel indices. Defaults to ``"torch"``.
    avereage_pooled_coordinates : bool, optional
        If ``True`` average coordinates of points within each voxel. Defaults to ``False``.
    return_neighbor_search_result : bool, optional
        If ``True`` also return the neighbor search result. Defaults to ``False``.
    """

    def __init__(
        self,
        reduction: Union[str, REDUCTIONS] = REDUCTIONS.MAX,
        downsample_max_num_points: Optional[int] = None,
        downsample_voxel_size: Optional[float] = None,
        return_type: Literal["point", "sparse"] = "point",
        unique_method: Literal["torch", "ravel", "morton"] = "torch",
        avereage_pooled_coordinates: bool = False,
        return_neighbor_search_result: bool = False,
    ):
        super().__init__()
        if isinstance(reduction, str):
            reduction = REDUCTIONS(reduction)
        self.reduction = reduction
        self.downsample_max_num_points = downsample_max_num_points
        self.downsample_voxel_size = downsample_voxel_size
        self.return_type = return_type
        self.return_neighbor_search_result = return_neighbor_search_result
        self.unique_method = unique_method
        self.avereage_pooled_coordinates = avereage_pooled_coordinates

    def forward(self, pc: Points) -> Union[Geometry, Tuple[Geometry, RealSearchResult]]:
        return point_pool(
            pc=pc,
            reduction=self.reduction,
            downsample_max_num_points=self.downsample_max_num_points,
            downsample_voxel_size=self.downsample_voxel_size,
            return_type=self.return_type,
            return_neighbor_search_result=self.return_neighbor_search_result,
            unique_method=self.unique_method,
            avereage_pooled_coordinates=self.avereage_pooled_coordinates,
        )


class PointMaxPool(PointPoolBase):
    """Point pooling using ``max`` reduction.

    Parameters
    ----------
    downsample_max_num_points : int, optional
        Maximum number of points to keep when downsampling.
    downsample_voxel_size : float, optional
        Size of voxels used for downsampling.
    return_type : {"point", "sparse"}, optional
        Output geometry type. Defaults to ``"point"``.
    return_neighbor_search_result : bool, optional
        If ``True`` also return the neighbor search result. Defaults to ``False``.
    """

    def __init__(
        self,
        downsample_max_num_points: Optional[int] = None,
        downsample_voxel_size: Optional[float] = None,
        return_type: Literal["point", "sparse"] = "point",
        return_neighbor_search_result: bool = False,
    ):
        super().__init__(
            reduction=REDUCTIONS.MAX,
            downsample_max_num_points=downsample_max_num_points,
            downsample_voxel_size=downsample_voxel_size,
            return_type=return_type,
            return_neighbor_search_result=return_neighbor_search_result,
        )


class PointAvgPool(PointPoolBase):
    """Point pooling using ``mean`` reduction.

    Parameters
    ----------
    downsample_max_num_points : int, optional
        Maximum number of points to keep when downsampling.
    downsample_voxel_size : float, optional
        Size of voxels used for downsampling.
    return_type : {"point", "sparse"}, optional
        Output geometry type. Defaults to ``"point"``.
    return_neighbor_search_result : bool, optional
        If ``True`` also return the neighbor search result. Defaults to ``False``.
    """

    def __init__(
        self,
        downsample_max_num_points: Optional[int] = None,
        downsample_voxel_size: Optional[float] = None,
        return_type: Literal["point", "sparse"] = "point",
        return_neighbor_search_result: bool = False,
    ):
        super().__init__(
            reduction=REDUCTIONS.MEAN,
            downsample_max_num_points=downsample_max_num_points,
            downsample_voxel_size=downsample_voxel_size,
            return_type=return_type,
            return_neighbor_search_result=return_neighbor_search_result,
        )


class PointSumPool(PointPoolBase):
    """Point pooling using ``sum`` reduction.

    Parameters
    ----------
    downsample_max_num_points : int, optional
        Maximum number of points to keep when downsampling.
    downsample_voxel_size : float, optional
        Size of voxels used for downsampling.
    return_type : {"point", "sparse"}, optional
        Output geometry type. Defaults to ``"point"``.
    return_neighbor_search_result : bool, optional
        If ``True`` also return the neighbor search result. Defaults to ``False``.
    """

    def __init__(
        self,
        downsample_max_num_points: Optional[int] = None,
        downsample_voxel_size: Optional[float] = None,
        return_type: Literal["point", "sparse"] = "point",
        return_neighbor_search_result: bool = False,
    ):
        super().__init__(
            reduction=REDUCTIONS.SUM,
            downsample_max_num_points=downsample_max_num_points,
            downsample_voxel_size=downsample_voxel_size,
            return_type=return_type,
            return_neighbor_search_result=return_neighbor_search_result,
        )


class PointUnpool(BaseSpatialModule):
    """Undo point pooling by scattering pooled features back to the input cloud.

    Parameters
    ----------
    unpooling_mode : {"repeat", "interpolate"} or :class:`FEATURE_UNPOOLING_MODE`, optional
        Strategy used when unpooling features. Defaults to ``FEATURE_UNPOOLING_MODE.REPEAT``.
    concat_unpooled_pc : bool, optional
        If ``True`` concatenate the unpooled point cloud with the input. Defaults to ``False``.
    """

    def __init__(
        self,
        unpooling_mode: Union[
            str, FEATURE_UNPOOLING_MODE
        ] = FEATURE_UNPOOLING_MODE.REPEAT,
        concat_unpooled_pc: bool = False,
    ):
        super().__init__()
        if isinstance(unpooling_mode, str):
            unpooling_mode = FEATURE_UNPOOLING_MODE(unpooling_mode)
        self.unpooling_mode = unpooling_mode
        self.concat_unpooled_pc = concat_unpooled_pc

    def forward(self, pooled_pc: Points, unpooled_pc: Points):
        return point_unpool(
            pooled_pc=pooled_pc,
            unpooled_pc=unpooled_pc,
            unpooling_mode=self.unpooling_mode,
            concat_unpooled_pc=self.concat_unpooled_pc,
        )
