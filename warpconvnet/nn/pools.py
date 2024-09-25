from typing import Literal

import torch
import torch.nn as nn

from warpconvnet.geometry.base_geometry import BatchedSpatialFeatures
from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor
from warpconvnet.nn.base_module import BaseSpatialModule
from warpconvnet.nn.functional.global_pool import global_pool
from warpconvnet.nn.functional.point_pool import point_pool
from warpconvnet.nn.functional.sparse_pool import sparse_reduce


class SparsePool(BaseSpatialModule):
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

    def forward(self, st: SpatiallySparseTensor):
        return sparse_reduce(st, self.kernel_size, self.stride, self.reduce)


class SparseMaxPool(SparsePool):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__(kernel_size, stride, "max")


class SparseMinPool(SparsePool):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__(kernel_size, stride, "min")


class GlobalPool(BaseSpatialModule):
    def __init__(self, reduce: Literal["min", "max", "mean", "sum"] = "max"):
        super().__init__()
        self.reduce = reduce

    def forward(self, x: BatchedSpatialFeatures):
        return global_pool(x, self.reduce)
