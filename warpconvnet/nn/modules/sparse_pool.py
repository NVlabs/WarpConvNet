# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Tuple, Union

import torch
import torch.nn as nn

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.global_pool import global_pool
from warpconvnet.nn.functional.point_pool import point_pool
from warpconvnet.nn.functional.point_unpool import point_unpool
from warpconvnet.nn.functional.sparse_pool import sparse_reduce, sparse_unpool
from warpconvnet.ops.reductions import REDUCTION_TYPES_STR, REDUCTIONS


class SparsePool(BaseSpatialModule):
    """Reduce features of a ``Voxels`` object using a strided kernel.

    Parameters
    ----------
    kernel_size : int
        Size of the pooling kernel.
    stride : int
        Stride between pooling windows.
    reduce : {"max", "min", "mean", "sum", "random"}, optional
        Reduction to apply within each window. Defaults to ``"max"``.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        reduce: Literal["max", "min", "mean", "sum", "random"] = "max",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduce = reduce

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, reduce={self.reduce})"

    def forward(self, st: Voxels):
        return sparse_reduce(
            st,
            self.kernel_size,
            self.stride,
            self.reduce,
        )


class SparseMaxPool(SparsePool):
    """Max pooling for sparse tensors.

    Parameters
    ----------
    kernel_size : int
        Size of the pooling kernel.
    stride : int
        Stride between pooling windows.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int,
    ):
        super().__init__(kernel_size, stride, "max")


class SparseMinPool(SparsePool):
    """Min pooling for sparse tensors.

    Parameters
    ----------
    kernel_size : int
        Size of the pooling kernel.
    stride : int
        Stride between pooling windows.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int,
    ):
        super().__init__(kernel_size, stride, "min")


class GlobalPool(BaseSpatialModule):
    """Pool features across the entire geometry.

    Parameters
    ----------
    reduce : {"min", "max", "mean", "sum"}, optional
        Reduction to apply over all features. Defaults to ``"max"``.
    """

    def __init__(self, reduce: Literal["min", "max", "mean", "sum"] = "max"):
        super().__init__()
        self.reduce = reduce

    def forward(self, x: Geometry):
        return global_pool(x, self.reduce)


class SparseUnpool(BaseSpatialModule):
    """Unpool a sparse tensor back to a higher resolution.

    Parameters
    ----------
    kernel_size : int
        Size of the unpooling kernel.
    stride : int
        Stride between unpooling windows.
    concat_unpooled_st : bool, optional
        If ``True`` concatenate the unpooled tensor with the input. Defaults to ``True``.
    """

    def __init__(self, kernel_size: int, stride: int, concat_unpooled_st: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.concat_unpooled_st = concat_unpooled_st

    def forward(self, st: Voxels, unpooled_st: Voxels):
        return sparse_unpool(
            st,
            unpooled_st,
            self.kernel_size,
            self.stride,
            self.concat_unpooled_st,
        )


class PointToSparseWrapper(BaseSpatialModule):
    """Pool points into a sparse tensor, apply an inner module and unpool back to points.

    Parameters
    ----------
    inner_module : :class:`BaseSpatialModule`
        Module applied on the pooled sparse tensor.
    voxel_size : float
        Voxel size used to pool the input points.
    reduction : :class:`REDUCTIONS` or str, optional
        Reduction used when pooling points. Defaults to ``REDUCTIONS.MEAN``.
    unique_method : {"morton", "ravel", "torch"}, optional
        Method used for hashing voxel indices. Defaults to ``"morton"``.
    concat_unpooled_pc : bool, optional
        If ``True`` concatenate the unpooled result with the original input. Defaults to ``True``.
    """

    def __init__(
        self,
        inner_module: BaseSpatialModule,
        voxel_size: float,
        reduction: Union[REDUCTIONS, REDUCTION_TYPES_STR] = REDUCTIONS.MEAN,
        unique_method: Literal["morton", "ravel", "torch"] = "morton",
        concat_unpooled_pc: bool = True,
    ):
        super().__init__()
        self.inner_module = inner_module
        self.voxel_size = voxel_size
        self.reduction = reduction
        self.concat_unpooled_pc = concat_unpooled_pc
        self.unique_method = unique_method

    def forward(self, pc: Points) -> Points:
        st, to_unique = point_pool(
            pc,
            reduction=self.reduction,
            downsample_voxel_size=self.voxel_size,
            return_type="voxel",
            return_to_unique=True,
            unique_method=self.unique_method,
        )
        out_st = self.inner_module(st)
        assert isinstance(out_st, Voxels), "Output of inner module must be a Voxels"
        unpooled_pc = point_unpool(
            out_st.to_point(self.voxel_size),
            pc,
            concat_unpooled_pc=self.concat_unpooled_pc,
            to_unique=to_unique,
        )
        return unpooled_pc
