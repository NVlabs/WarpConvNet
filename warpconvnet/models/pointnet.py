# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.functional.bmm import bmm
from warpconvnet.nn.modules.activations import ReLU
from warpconvnet.nn.modules.mlp import Linear
from warpconvnet.nn.modules.normalizations import BatchNorm
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_pool import GlobalPool


class SpatialLinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Geometry):
        return self.block(x)


class MLPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, activation=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            activation(),
        )

    def forward(self, x: Float[Tensor, "B C"]):
        return self.block(x)


class STNkd(nn.Module):
    """
    Line to Line translation of https://github.com/guochengqian/openpoints/blob/master/models/backbone/pointnet.py
    """

    def __init__(self, channel: int):
        super().__init__()
        self.net = nn.Sequential(
            SpatialLinearBlock(channel, 64),
            SpatialLinearBlock(64, 128),
            SpatialLinearBlock(128, 1024),
            GlobalPool(reduce="max"),
            SpatialLinearBlock(1024, 512),
            SpatialLinearBlock(512, 256),
            Linear(256, channel**2),
        )
        self.register_buffer("iden", torch.eye(channel).view(1, -1))
        self.add_iden = lambda x: x.replace(batched_features=x.features + self.iden)

    def forward(self, x: Points) -> Points:
        x = self.net(x)
        x = self.add_iden(x)
        return x


class PointNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int] = [1024, 1024],
        input_transform: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stn = STNkd(in_channels) if input_transform else None
        self.net1 = nn.Sequential(
            Linear(in_channels, 64),
            BatchNorm(64),
            ReLU(),
            Linear(64, 64),
            BatchNorm(64),
            ReLU(),
        )

        self.net2 = nn.Sequential(
            Linear(64, 64),
            BatchNorm(64),
            ReLU(),
            Linear(64, 128),
            BatchNorm(128),
            ReLU(),
            Linear(128, 1024),
            BatchNorm(1024),
        )

        self.global_pool = GlobalPool(reduce="max")

        self.fstn_channels = 64 if feature_transform else None
        self.fstn = STNkd(self.fstn_channels) if feature_transform else None

        self.mlp = nn.Sequential(
            MLPBlock(1024, hidden_channels[0]),
            *[
                MLPBlock(hidden_channels[i], hidden_channels[i + 1])
                for i in range(len(hidden_channels) - 1)
            ],
            nn.Linear(hidden_channels[-1], out_channels),
        )

    def forward(self, pc: Points) -> Float[torch.Tensor, "B C"]:
        assert isinstance(pc, Points)
        if self.stn is not None:
            trans = self.stn(pc).features
            pc = bmm(pc, trans.view(pc.batch_size, self.in_channels, self.in_channels))

        x = self.net1(pc)
        if self.fstn is not None:
            trans_f = self.fstn(x).features
            x = bmm(x, trans_f.view(pc.batch_size, self.fstn_channels, self.fstn_channels))

        x = self.net2(x)
        x = self.global_pool(x)
        x = x.features
        x = self.mlp(x)
        return x
