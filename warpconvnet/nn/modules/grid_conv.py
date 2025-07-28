# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union, Literal
import warnings

import torch
import torch.nn as nn
from torch.nn import init

from warpconvnet.geometry.types.grid import Grid, GridMemoryFormat
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.grid_conv import grid_conv
from warpconvnet.utils.ntuple import ntuple


class GridConv(BaseSpatialModule):
    """Convolutional layer for `warpconvnet.geometry.types.grid.Grid` data.

    Parameters mirror those of `torch.nn.Conv3d` but operate on a
    ``Grid`` object instead of plain tensors.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    stride : int or tuple of int, optional
        Stride of the convolution. Defaults to ``1``.
    padding : int or tuple of int, optional
        Zero-padding added to all three sides of the input. Defaults to ``0``.
    dilation : int or tuple of int, optional
        Spacing between kernel elements. Defaults to ``1``.
    bias : bool, optional
        If ``True``, adds a learnable bias to the output. Defaults to ``True``.
    num_spatial_dims : int, optional
        Number of spatial dimensions. Defaults to ``3``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: bool = True,
        num_spatial_dims: Optional[int] = 3,
    ):
        super().__init__()
        kernel_size = ntuple(kernel_size, ndim=num_spatial_dims)
        stride = ntuple(stride, ndim=num_spatial_dims)
        padding = ntuple(padding, ndim=num_spatial_dims)
        dilation = ntuple(dilation, ndim=num_spatial_dims)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.num_spatial_dims = num_spatial_dims

        # For 3D convolution, shape is (out_channels, in_channels, depth, height, width)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"dilation={self.dilation}, "
            f"bias={self.bias is not None}"
            f")"
        )

    def reset_parameters(self):
        # Standard initialization for convolutional layers
        init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_grid: Grid) -> Grid:
        return grid_conv(
            grid=input_grid,
            weight=self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.bias,
        )
