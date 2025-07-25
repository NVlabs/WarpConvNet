# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule

__all__ = ["Linear", "BatchedLinear", "LinearNormActivation"]


class Linear(BaseSpatialModule):
    """Apply a linear layer to ``Geometry`` features.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, optional
        If ``True`` adds a bias term to the layer. Defaults to ``True``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.block = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Geometry):
        return x.replace(batched_features=self.block(x.feature_tensor))


class LinearNormActivation(BaseSpatialModule):
    """Linear layer followed by ``LayerNorm`` and ``ReLU``.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, optional
        Whether to include a bias term. Defaults to ``True``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            nn.LayerNorm(out_features),
            nn.ReLU(),
        )

    def forward(self, x: Geometry):
        return x.replace(batched_features=self.block(x.feature_tensor))


class BatchedLinear(nn.Module):
    """
    A linear layer with batched weights for Muon-friendly optimization.

    Instead of a single weight matrix [in_features, out_features * num_matrices],
    this uses separate weight matrices stacked as [num_matrices, in_features, out_features].
    This structure is more suitable for Muon optimization as it can orthogonalize
    each [in_features, out_features] matrix independently.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension per matrix
        num_matrices: Number of separate matrices (e.g., 3 for Q, K, V)
        bias: Whether to use bias parameters
    """

    def __init__(
        self, in_features: int, out_features: int, num_matrices: int = 3, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_matrices = num_matrices

        # Create batched weight: [num_matrices, in_features, out_features]
        self.weight = nn.Parameter(torch.empty(num_matrices, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            # Use flat bias for Muon - 1D parameter gets Adam optimization
            self.bias = nn.Parameter(torch.zeros(num_matrices * out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass with batched matrix multiplication.

        Args:
            input: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., num_matrices, out_features]
        """
        # input: [..., in_features], weight: [num_matrices, in_features, out_features]
        # output: [..., num_matrices, out_features]
        output = torch.einsum("...i,kio->...ko", input, self.weight)

        if self.bias is not None:
            output = output + self.bias.view(self.num_matrices, self.out_features)

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, num_matrices={self.num_matrices}, bias={self.bias is not None}"
