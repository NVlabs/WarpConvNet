# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.init import calculate_gain

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.sparse_conv import (
    SPARSE_CONV_FWD_ALGO_MODE,
    SPARSE_CONV_BWD_ALGO_MODE,
    STRIDED_CONV_MODE,
    spatially_sparse_conv,
)
from warpconvnet.utils.ntuple import ntuple
from warpconvnet.constants import (
    WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP,
    WARPCONVNET_FWD_ALGO_MODE,
    WARPCONVNET_BWD_ALGO_MODE,
)


class SpatiallySparseConv(BaseSpatialModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: bool = True,
        transposed: bool = False,
        generative: bool = False,
        kernel_matmul_batch_size: int = 2,
        num_spatial_dims: Optional[int] = 3,
        fwd_algo: Optional[Union[SPARSE_CONV_FWD_ALGO_MODE, str]] = None,
        bwd_algo: Optional[Union[SPARSE_CONV_BWD_ALGO_MODE, str]] = None,
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
        out_code_backend: Literal["hashmap", "unique", "ravel", "morton"] = "hashmap",
        compute_dtype: Optional[torch.dtype] = None,
        implicit_matmul_fwd_block_size: Optional[int] = None,
        implicit_matmul_bwd_block_size: Optional[int] = None,
        skip_symmetric_kernel_map: Optional[bool] = None,
    ):
        super().__init__()
        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure kernel_size, stride, dilation are tuples for consistent use
        _kernel_size = ntuple(kernel_size, ndim=self.num_spatial_dims)
        _stride = ntuple(stride, ndim=self.num_spatial_dims)
        _dilation = ntuple(dilation, ndim=self.num_spatial_dims)

        self.kernel_size = _kernel_size
        self.stride = _stride
        self.dilation = _dilation

        self.transposed = transposed
        self.generative = generative
        self.kernel_matmul_batch_size = kernel_matmul_batch_size

        # Use environment variable values if not explicitly provided
        if fwd_algo is None:
            fwd_algo = WARPCONVNET_FWD_ALGO_MODE
        if bwd_algo is None:
            bwd_algo = WARPCONVNET_BWD_ALGO_MODE

        # If string, update to enum
        self.fwd_algo = (
            SPARSE_CONV_FWD_ALGO_MODE(fwd_algo) if isinstance(fwd_algo, str) else fwd_algo
        )
        self.bwd_algo = (
            SPARSE_CONV_BWD_ALGO_MODE(bwd_algo) if isinstance(bwd_algo, str) else bwd_algo
        )
        self.stride_mode = stride_mode
        self.out_code_backend = out_code_backend
        self.compute_dtype = compute_dtype
        self.implicit_matmul_fwd_block_size = implicit_matmul_fwd_block_size
        self.implicit_matmul_bwd_block_size = implicit_matmul_bwd_block_size
        # Skip symmetric kernel map
        if skip_symmetric_kernel_map is None:
            skip_symmetric_kernel_map = WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP
        is_odd_kernel = all(k % 2 == 1 for k in _kernel_size)
        no_stride = all(s == 1 for s in _stride)
        is_same_in_out_coords = no_stride and not generative
        self.skip_symmetric_kernel_map = (
            skip_symmetric_kernel_map and is_same_in_out_coords and is_odd_kernel
        )

        self.bias: Optional[nn.Parameter] = None

        self.weight = nn.Parameter(torch.randn(np.prod(_kernel_size), in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None  # Explicitly set to None if bias is False
        self.reset_parameters()  # Call after parameters are defined for the chosen backend

    def __repr__(self):
        # return class name and parameters that are not default
        out_str = f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}"
        if self.stride != 1:
            out_str += f", stride={self.stride}"
        if self.dilation != 1:
            out_str += f", dilation={self.dilation}"
        if self.transposed:
            out_str += f", transposed={self.transposed}"
        if self.generative:
            out_str += f", generative={self.generative}"
        if self.skip_symmetric_kernel_map:
            out_str += f", skip_symmetric_kernel={self.skip_symmetric_kernel_map}"
        out_str += ")"
        return out_str

    def _calculate_fan_in_and_fan_out(self):
        receptive_field_size = np.prod(self.kernel_size)
        fan_in = self.in_channels * receptive_field_size
        fan_out = self.out_channels * receptive_field_size
        return fan_in, fan_out

    def _calculate_correct_fan(self, mode: Literal["fan_in", "fan_out"]):
        mode = mode.lower()
        assert mode in ["fan_in", "fan_out"]

        fan_in, fan_out = self._calculate_fan_in_and_fan_out()
        return fan_in if mode == "fan_in" else fan_out

    def _custom_kaiming_uniform_(self, tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
        fan = self._calculate_correct_fan(mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(self.num_spatial_dims) * std
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    @torch.no_grad()
    def reset_parameters(self):
        self._custom_kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
            mode="fan_out" if self.transposed else "fan_in",
        )

        if self.skip_symmetric_kernel_map:
            identity_map_idx = np.prod(self.kernel_size).item() // 2
            # Add the symmetric kernel parts to the first half of the weight
            for i in range(identity_map_idx):
                self.weight[i] = self.weight[i] + self.weight[-1 - i]

            # Remove the symmetric kernel parts from the second half of the weight
            self.weight[identity_map_idx + 1 :] = 0

        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out()
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        input_sparse_tensor: Voxels,
        output_spatially_sparse_tensor: Optional[Voxels] = None,
    ):
        return spatially_sparse_conv(
            input_sparse_tensor=input_sparse_tensor,
            weight=self.weight,
            kernel_size=self.kernel_size,
            stride=self.stride,
            kernel_dilation=self.dilation,
            bias=self.bias,
            kernel_matmul_batch_size=self.kernel_matmul_batch_size,
            output_spatially_sparse_tensor=output_spatially_sparse_tensor,
            transposed=self.transposed,
            generative=self.generative,
            fwd_algo=self.fwd_algo,
            bwd_algo=self.bwd_algo,
            stride_mode=self.stride_mode,
            out_code_backend=self.out_code_backend,
            compute_dtype=self.compute_dtype,
            implicit_matmul_fwd_block_size=self.implicit_matmul_fwd_block_size,
            implicit_matmul_bwd_block_size=self.implicit_matmul_bwd_block_size,
            skip_symmetric_kernel_map=self.skip_symmetric_kernel_map,
        )


class SparseConv2d(SpatiallySparseConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        transposed=False,
        generative: bool = False,
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
        fwd_algo: Optional[Union[SPARSE_CONV_FWD_ALGO_MODE, str]] = None,
        bwd_algo: Optional[Union[SPARSE_CONV_BWD_ALGO_MODE, str]] = None,
        kernel_matmul_batch_size: int = 2,
        out_code_backend: Literal["hashmap", "unique", "ravel", "morton"] = "hashmap",
        compute_dtype: Optional[torch.dtype] = None,
        implicit_matmul_fwd_block_size: Optional[int] = None,
        implicit_matmul_bwd_block_size: Optional[int] = None,
        skip_symmetric_kernel_map: Optional[bool] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            transposed=transposed,
            generative=generative,
            num_spatial_dims=2,
            stride_mode=stride_mode,
            fwd_algo=fwd_algo,
            bwd_algo=bwd_algo,
            kernel_matmul_batch_size=kernel_matmul_batch_size,
            out_code_backend=out_code_backend,
            compute_dtype=compute_dtype,
            implicit_matmul_fwd_block_size=implicit_matmul_fwd_block_size,
            implicit_matmul_bwd_block_size=implicit_matmul_bwd_block_size,
            skip_symmetric_kernel_map=skip_symmetric_kernel_map,
        )


class SparseConv3d(SpatiallySparseConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        transposed=False,
        generative: bool = False,
        stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
        fwd_algo: Optional[Union[SPARSE_CONV_FWD_ALGO_MODE, str]] = None,
        bwd_algo: Optional[Union[SPARSE_CONV_BWD_ALGO_MODE, str]] = None,
        kernel_matmul_batch_size: int = 2,
        out_code_backend: Literal["hashmap", "unique", "ravel", "morton"] = "hashmap",
        compute_dtype: Optional[torch.dtype] = None,
        implicit_matmul_fwd_block_size: Optional[int] = None,
        implicit_matmul_bwd_block_size: Optional[int] = None,
        skip_symmetric_kernel_map: Optional[bool] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            transposed=transposed,
            generative=generative,
            num_spatial_dims=3,
            stride_mode=stride_mode,
            fwd_algo=fwd_algo,
            bwd_algo=bwd_algo,
            kernel_matmul_batch_size=kernel_matmul_batch_size,
            out_code_backend=out_code_backend,
            compute_dtype=compute_dtype,
            implicit_matmul_fwd_block_size=implicit_matmul_fwd_block_size,
            implicit_matmul_bwd_block_size=implicit_matmul_bwd_block_size,
            skip_symmetric_kernel_map=skip_symmetric_kernel_map,
        )
