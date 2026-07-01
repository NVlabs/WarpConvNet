# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WarpConvNet-native port of FCGF (Fully Convolutional Geometric Features).

Reference: Choy, Park, Koltun, "Fully Convolutional Geometric Features", ICCV 2019.
Original (MinkowskiEngine) implementation: https://github.com/chrischoy/FCGF

This reimplements the ``ResUNetBN2C`` sparse-convolutional U-Net that produces
dense, per-point geometric descriptors. The architecture is a faithful port of
``model/resunet.py`` from the original repository, built on WarpConvNet
``Voxels`` + ``SparseConv3d`` primitives rather than MinkowskiEngine.

Key design notes matching the original:
- Input feature is 1-D occupancy (all ones); output is ``out_channels`` (32).
- Encoder downsamples with stride-2 k3 convs; decoder upsamples with stride-2
  k3 transposed convs whose output sparsity is pinned to the matching-resolution
  encoder skip tensor (so coordinates line up for the concatenation).
- ``BasicBlock`` (from ``warpconvnet.models.mink_unet``) equals FCGF's
  ``BasicBlockBN`` at stride 1 / in==out (conv3-BN-relu-conv3-BN + residual, relu).
- When ``normalize_feature`` is set the final descriptors are L2-normalized.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.transforms import cat
from warpconvnet.nn.modules.activations import ReLU
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.models.mink_unet import BasicBlock, ConvBlock, ConvTrBlock


class ResUNet2(nn.Module):
    """FCGF ResUNet backbone. Subclasses set ``CHANNELS`` / ``TR_CHANNELS``.

    Parameters
    ----------
    in_channels : int
        Input feature dimension (1 for occupancy on 3DMatch).
    out_channels : int
        Output descriptor dimension (32 in the paper).
    bn_momentum : float
        BatchNorm momentum (0.05 in FCGF; nn default is 0.1).
    normalize_feature : bool
        If True, L2-normalize the output descriptors along the channel dim.
    conv1_kernel_size : int
        Kernel size of the stem convolution (5 in the training config; the
        released 3DMatch checkpoint used 7).
    """

    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        bn_momentum: float = 0.05,
        normalize_feature: bool = True,
        conv1_kernel_size: int = 5,
    ):
        super().__init__()
        self.normalize_feature = normalize_feature
        C = self.CHANNELS
        TR = self.TR_CHANNELS

        # --- Encoder ---------------------------------------------------------
        # conv + BN, no activation (FCGF applies relu implicitly via the block).
        self.conv1 = ConvBlock(
            in_channels, C[1], kernel_size=conv1_kernel_size, stride=1, activation=None
        )
        self.block1 = BasicBlock(C[1], C[1])

        self.conv2 = ConvBlock(C[1], C[2], kernel_size=3, stride=2, activation=None)
        self.block2 = BasicBlock(C[2], C[2])

        self.conv3 = ConvBlock(C[2], C[3], kernel_size=3, stride=2, activation=None)
        self.block3 = BasicBlock(C[3], C[3])

        self.conv4 = ConvBlock(C[3], C[4], kernel_size=3, stride=2, activation=None)
        self.block4 = BasicBlock(C[4], C[4])

        # --- Decoder (transposed convs, coords pinned to encoder skips) ------
        self.conv4_tr = ConvTrBlock(C[4], TR[4], kernel_size=3, stride=2, activation=None)
        self.block4_tr = BasicBlock(TR[4], TR[4])

        self.conv3_tr = ConvTrBlock(C[3] + TR[4], TR[3], kernel_size=3, stride=2, activation=None)
        self.block3_tr = BasicBlock(TR[3], TR[3])

        self.conv2_tr = ConvTrBlock(C[2] + TR[3], TR[2], kernel_size=3, stride=2, activation=None)
        self.block2_tr = BasicBlock(TR[2], TR[2])

        # conv1_tr is a regular 1x1 conv (not transposed), NO norm, followed by relu
        # (matches FCGF: conv1_tr -> relu -> final, with no BatchNorm on conv1_tr).
        self.conv1_tr = SparseConv3d(C[1] + TR[2], TR[1], kernel_size=1, stride=1, bias=False)
        self.relu = ReLU(inplace=True)

        self.final = SparseConv3d(TR[1], out_channels, kernel_size=1, stride=1, bias=True)

        self._set_bn_momentum(bn_momentum)

    def _set_bn_momentum(self, momentum: float) -> None:
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = momentum

    def forward(self, x: Voxels) -> Voxels:
        # Encoder. Block outputs are post-relu, so they double as the pre-relu
        # feed to the next stage (relu(block_out) == block_out).
        out_s1 = self.block1(self.conv1(x))
        out_s2 = self.block2(self.conv2(out_s1))
        out_s4 = self.block3(self.conv3(out_s2))
        out_s8 = self.block4(self.conv4(out_s4))

        # Decoder.
        out = self.block4_tr(self.conv4_tr(out_s8, out_s4))
        out = cat(out, out_s4)

        out = self.block3_tr(self.conv3_tr(out, out_s2))
        out = cat(out, out_s2)

        out = self.block2_tr(self.conv2_tr(out, out_s1))
        out = cat(out, out_s1)

        out = self.relu(self.conv1_tr(out))
        out = self.final(out)

        if self.normalize_feature:
            out = out.replace(batched_features=F.normalize(out.feature_tensor, p=2, dim=1))
        return out


class ResUNetBN2C(ResUNet2):
    """Default FCGF descriptor network (out=32). Matches the paper's headline model."""

    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2B(ResUNet2):
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2D(ResUNet2):
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
    CHANNELS = [None, 128, 128, 128, 256]
    TR_CHANNELS = [None, 64, 128, 128, 128]
