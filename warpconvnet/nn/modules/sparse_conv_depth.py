# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.init import calculate_gain

from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.sparse_conv_depth import (
    SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE,
    SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE,
    spatially_sparse_depthwise_conv,
)
from warpconvnet.nn.functional.sparse_conv import (
    STRIDED_CONV_MODE,
    generate_output_coords_and_kernel_map,
)
from warpconvnet.utils.ntuple import ntuple
from warpconvnet.constants import (
    WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE,
    WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE,
)


class SpatiallySparseDepthwiseConv(BaseSpatialModule):
    """
    Spatially sparse depthwise convolution module.

    In depthwise convolution, each input channel is convolved with its own kernel,
    so the number of input channels must equal the number of output channels.
    The weight shape is (K, C) where K is the kernel volume and C is the number of channels.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: bool = True,
        transposed: bool = False,
        generative: bool = False,
        num_spatial_dims: int = 3,
        fwd_algo: Optional[Union[SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE, str]] = None,
        bwd_algo: Optional[Union[SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE, str]] = None,
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
        stride_reduce: str = "max",
        order: POINT_ORDERING = POINT_ORDERING.RANDOM,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_spatial_dims = num_spatial_dims
        self.channels = channels
        self.in_channels = channels  # For compatibility with PyTorch naming
        self.out_channels = channels  # For depthwise, in_channels == out_channels

        # Ensure kernel_size, stride, dilation are tuples for consistent use
        self.kernel_size = ntuple(kernel_size, ndim=self.num_spatial_dims)
        self.stride = ntuple(stride, ndim=self.num_spatial_dims)
        self.dilation = ntuple(dilation, ndim=self.num_spatial_dims)

        self.transposed = transposed
        self.generative = generative
        self.stride_reduce = stride_reduce

        # Use environment variable values if not explicitly provided
        if fwd_algo is None:
            fwd_algo = WARPCONVNET_DEPTHWISE_CONV_FWD_ALGO_MODE
        if bwd_algo is None:
            bwd_algo = WARPCONVNET_DEPTHWISE_CONV_BWD_ALGO_MODE

        # Map string algo names to depthwise-specific enums if needed
        if isinstance(fwd_algo, str):
            # Map generic algorithm names to depthwise-specific ones
            if fwd_algo.lower() in ["explicit", "explicit_gemm"]:
                self.fwd_algo = SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT
            elif fwd_algo.lower() in ["implicit", "implicit_gemm"]:
                self.fwd_algo = SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT
            elif fwd_algo.lower() == "auto":
                self.fwd_algo = SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.AUTO
            else:
                self.fwd_algo = SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE(fwd_algo)
        else:
            self.fwd_algo = fwd_algo

        if isinstance(bwd_algo, str):
            # Map generic algorithm names to depthwise-specific ones
            if bwd_algo.lower() in ["explicit", "explicit_gemm"]:
                self.bwd_algo = SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT
            elif bwd_algo.lower() in ["implicit", "implicit_gemm"]:
                self.bwd_algo = SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT
            elif bwd_algo.lower() == "auto":
                self.bwd_algo = SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.AUTO
            else:
                self.bwd_algo = SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE(bwd_algo)
        else:
            self.bwd_algo = bwd_algo

        self.stride_mode = stride_mode
        self.order = order
        self.compute_dtype = compute_dtype

        # Depthwise convolution weight shape: (K, C) where K is kernel volume
        kernel_volume = int(np.prod(self.kernel_size))
        self.weight = nn.Parameter(torch.randn(kernel_volume, channels))

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.randn(channels))
        else:
            self.bias = None

        self.reset_parameters()

    def __repr__(self):
        out_str = (
            f"{self.__class__.__name__}(channels={self.channels}, "
            f"kernel_size={self.kernel_size}"
        )
        if self.stride != (1,) * self.num_spatial_dims:
            out_str += f", stride={self.stride}"
        if self.dilation != (1,) * self.num_spatial_dims:
            out_str += f", dilation={self.dilation}"
        if self.transposed:
            out_str += f", transposed={self.transposed}"
        if self.generative:
            out_str += f", generative={self.generative}"
        if self.order != POINT_ORDERING.RANDOM:
            out_str += f", order={self.order}"
        if self.bias is None:
            out_str += ", bias=False"
        out_str += ")"
        return out_str

    def _calculate_fan_in_and_fan_out(self):
        """Calculate fan_in and fan_out for depthwise convolution."""
        receptive_field_size = np.prod(self.kernel_size)
        # For depthwise convolution, each channel has its own kernel
        fan_in = receptive_field_size  # One kernel per channel
        fan_out = receptive_field_size  # One output per channel
        return fan_in, fan_out

    def _calculate_correct_fan(self, mode: str):
        """Calculate correct fan for initialization."""
        mode = mode.lower()
        assert mode in ["fan_in", "fan_out"]

        fan_in, fan_out = self._calculate_fan_in_and_fan_out()
        return fan_in if mode == "fan_in" else fan_out

    def _custom_kaiming_uniform_(self, tensor, a=0.0, mode="fan_in", nonlinearity="leaky_relu"):
        """Custom Kaiming uniform initialization for depthwise convolution."""
        fan = self._calculate_correct_fan(mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(self.num_spatial_dims) * std
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    @torch.no_grad()
    def reset_parameters(self):
        """Reset module parameters using appropriate initialization."""
        self._custom_kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
            mode="fan_out" if self.transposed else "fan_in",
        )

        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out()
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        input_sparse_tensor: Voxels,
        output_spatially_sparse_tensor: Optional[Voxels] = None,
    ) -> Voxels:
        """
        Forward pass for spatially sparse depthwise convolution.

        Args:
            input_sparse_tensor: Input sparse tensor
            output_spatially_sparse_tensor: Optional output sparse tensor for transposed conv

        Returns:
            Output sparse tensor
        """
        # Generate output coordinates and kernel map
        batch_indexed_out_coords, out_offsets, kernel_map = generate_output_coords_and_kernel_map(
            input_sparse_tensor=input_sparse_tensor,
            kernel_size=self.kernel_size,
            kernel_dilation=self.dilation,
            stride=self.stride,
            generative=self.generative,
            transposed=self.transposed,
            output_spatially_sparse_tensor=output_spatially_sparse_tensor,
            stride_mode=self.stride_mode,
            order=self.order,
        )

        num_out_coords = batch_indexed_out_coords.shape[0]

        # Apply depthwise convolution
        output_features = spatially_sparse_depthwise_conv(
            input_sparse_tensor.feature_tensor,
            self.weight,
            kernel_map,
            num_out_coords,
            fwd_algo=self.fwd_algo,
            bwd_algo=self.bwd_algo,
            compute_dtype=self.compute_dtype,
        )

        # Add bias if present
        if self.bias is not None:
            output_features = output_features + self.bias

        # Determine output tensor stride
        in_tensor_stride = input_sparse_tensor.tensor_stride
        if in_tensor_stride is None:
            in_tensor_stride = (1,) * self.num_spatial_dims

        if not self.transposed:
            out_tensor_stride = tuple(o * s for o, s in zip(self.stride, in_tensor_stride))
        else:
            if (
                output_spatially_sparse_tensor is not None
                and output_spatially_sparse_tensor.tensor_stride is not None
            ):
                out_tensor_stride = output_spatially_sparse_tensor.tensor_stride
            else:
                out_tensor_stride = (1,) * self.num_spatial_dims

        # Create output voxels
        out_offsets_cpu = out_offsets.cpu().int()
        out_coords = IntCoords(
            batch_indexed_out_coords[:, 1:],
            offsets=out_offsets_cpu,
        )
        return input_sparse_tensor.replace(
            batched_coordinates=out_coords,
            batched_features=output_features,
            tensor_stride=out_tensor_stride,
        )


class SparseDepthwiseConv2d(SpatiallySparseDepthwiseConv):
    """2D spatially sparse depthwise convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        transposed: bool = False,
        generative: bool = False,
        fwd_algo: Optional[Union[SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE, str]] = None,
        bwd_algo: Optional[Union[SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE, str]] = None,
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
        stride_reduce: str = "max",
        order: POINT_ORDERING = POINT_ORDERING.RANDOM,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            transposed=transposed,
            generative=generative,
            num_spatial_dims=2,
            fwd_algo=fwd_algo,
            bwd_algo=bwd_algo,
            stride_mode=stride_mode,
            stride_reduce=stride_reduce,
            order=order,
            compute_dtype=compute_dtype,
        )


class SparseDepthwiseConv3d(SpatiallySparseDepthwiseConv):
    """3D spatially sparse depthwise convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        transposed: bool = False,
        generative: bool = False,
        fwd_algo: Optional[Union[SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE, str]] = None,
        bwd_algo: Optional[Union[SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE, str]] = None,
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
        stride_reduce: str = "max",
        order: POINT_ORDERING = POINT_ORDERING.RANDOM,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            transposed=transposed,
            generative=generative,
            num_spatial_dims=3,
            fwd_algo=fwd_algo,
            bwd_algo=bwd_algo,
            stride_mode=stride_mode,
            stride_reduce=stride_reduce,
            order=order,
            compute_dtype=compute_dtype,
        )


# Alias for convenience
SparseDepthwiseConv = SparseDepthwiseConv3d
