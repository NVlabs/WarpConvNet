# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import torch
import torch.nn as nn
from warpconvnet.geometry.coords.search.search_configs import (
    RealSearchConfig,
    RealSearchMode,
)
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.functional.global_pool import global_pool
from warpconvnet.nn.functional.transforms import cat
from warpconvnet.nn.modules.mlp import Linear
from warpconvnet.nn.modules.point_conv import PointConv


# One DGCNN convolution consists of
# 1. K features per point that is concatenation of f and x
# 2. Linear, BN, LeakyReLU
# 3. Max pooling K features to get one feature per point
# All of which are done in the PointConv layer
class PointConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        knn_k: int = 20,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.conv = PointConv(
            in_channels,
            out_channels,
            RealSearchConfig(mode=RealSearchMode.KNN, knn_k=knn_k),
            edge_transform_mlp=nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(negative_slope=negative_slope),
            ),
            out_transform_mlp=nn.Identity(),
            reductions=["max"],
            out_point_type="same",
        )

    def forward(self, x: Points):
        return self.conv(x)


class DGCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        knn_ks: Union[List[int], int],
        emb_dims: int,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        if isinstance(knn_ks, int):
            knn_ks = [knn_ks] * 4
        self.conv1 = PointConvBlock(
            in_channels, 64, knn_k=knn_ks[0], negative_slope=negative_slope
        )
        self.conv2 = PointConvBlock(64, 64, knn_k=knn_ks[1], negative_slope=negative_slope)
        self.conv3 = PointConvBlock(64, 128, knn_k=knn_ks[2], negative_slope=negative_slope)
        self.conv4 = PointConvBlock(128, 256, knn_k=knn_ks[3], negative_slope=negative_slope)
        self.conv5 = Linear(256 + 128 + 64 + 64, emb_dims)

    def forward(self, x: Points):
        assert isinstance(x, Points)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = cat(x1, x2, x3, x4)
        x = self.conv5(x)
        return x


class DGCNN(nn.Module):
    """
    https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        knn_ks: Union[List[int], int] = 20,
        emb_dims: int = 1024,
        negative_slope: float = 0.2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = DGCNNEncoder(in_channels, knn_ks, emb_dims, negative_slope)

        self.head = nn.Sequential(
            nn.Linear(2 * emb_dims, emb_dims),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            nn.Linear(emb_dims, emb_dims),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            nn.Linear(emb_dims, out_channels),
        )

    def forward(self, x: Points):
        x = self.encoder(x)
        x_max = global_pool(x, reduce="max").feature_tensor
        x_mean = global_pool(x, reduce="mean").feature_tensor
        x = torch.cat([x_max, x_mean], dim=1)
        x = self.head(x)
        return x
