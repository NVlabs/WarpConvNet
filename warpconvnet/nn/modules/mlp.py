# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule

__all__ = ["FeatureResidualMLPBlock", "Linear"]


class FeatureMLPBlock(nn.Module):
    """Simple fully connected block with activation and normalization.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    hidden_channels : int, optional
        Size of the hidden layer. Defaults to ``None`` which uses ``in_channels``.
    activation : ``nn.Module``, optional
        Activation module to apply. Defaults to :class:`torch.nn.ReLU`.
    bias : bool, optional
        If ``True`` adds bias terms to the linear layers. Defaults to ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        bias: bool = True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.LayerNorm(out_channels),
            activation(),
        )

    def forward(self, x: Float[Tensor, "B C"]):
        return self.block(x)


class FeatureResidualMLPBlock(nn.Module):
    """MLP block with a residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int, optional
        Number of output features. Defaults to ``in_channels``.
    hidden_channels : int, optional
        Hidden layer size. Defaults to ``in_channels``.
    activation : ``nn.Module``, optional
        Activation module to apply. Defaults to :class:`torch.nn.ReLU`.
    bias : bool, optional
        If ``True`` adds bias terms to the linear layers. Defaults to ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        activation=nn.ReLU,
        bias: bool = True,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = (
            nn.Linear(in_channels, out_channels, bias=bias)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = activation()

    def forward(self, x: Float[Tensor, "B C"]):
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        # add skip connection
        out = self.activation(out + self.shortcut(x))
        return out


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


class ResidualMLPBlock(FeatureResidualMLPBlock):
    """Residual MLP block operating on ``Geometry`` features.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int, optional
        Number of output features. Defaults to ``in_features``.
    hidden_features : int, optional
        Size of the hidden layer. Defaults to ``in_features``.
    """

    def __init__(
        self, in_features: int, out_features: int = None, hidden_features: int = None
    ):
        super().__init__(in_features, out_features, hidden_features)

    def forward(self, x: Geometry):
        return x.replace(batched_features=super().forward(x.feature_tensor))
