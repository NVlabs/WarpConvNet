# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvNeXt block on `Voxels` (sparse 3D analogue).

Standard ConvNeXt layout:
    SparseConv3d (k=3, submanifold)  → conv
    LayerNorm (affine, fp32)         → norm
    Linear → SiLU → Linear (zero-init) on .feats   → mlp
    + skip (residual)

Uses warpconvnet's native `SparseConv3d` (weight layout
``(K^3, in_channels, out_channels)``). Models that need to load
checkpoints stored in a different sparse-conv weight layout (TRELLIS
uses ``(Cout, Kd, Kh, Kw, Cin)`` in some checkpoints, for instance) ship
their own block with a layout-compatible conv wrapper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.normalizations import LayerNorm32
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.utils import zero_module


__all__ = ["SparseConvNeXtBlock3d"]


class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        kernel_size: int = 3,
        use_checkpoint: bool = False,
        conv_cls: type[nn.Module] = SparseConv3d,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint

        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = conv_cls(channels, channels, kernel_size=kernel_size)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            zero_module(nn.Linear(int(channels * mlp_ratio), channels)),
        )

    def _forward(self, x: Voxels) -> Voxels:
        h = self.conv(x)
        h = h.replace_features(self.norm(h.feats))
        h = h.replace_features(self.mlp(h.feats))
        return h + x

    def forward(self, x: Voxels) -> Voxels:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
