# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable dense 3D convolutional blocks."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.nn.functional.pixel_shuffle import pixel_shuffle_3d
from warpconvnet.nn.modules.normalizations import ChannelLayerNorm32, GroupNorm32
from warpconvnet.nn.utils import zero_module


__all__ = [
    "DownsampleBlock3d",
    "ResBlock3d",
    "UpsampleBlock3d",
    "norm_layer_3d",
]


def norm_layer_3d(norm_type: Literal["group", "layer"], channels: int) -> nn.Module:
    """Build a fp32-internal normalization layer for ``(B, C, D, H, W)`` tensors."""
    if norm_type == "group":
        return GroupNorm32(32, channels)
    if norm_type == "layer":
        return ChannelLayerNorm32(channels)
    raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    """Pre-norm 3D residual block: norm -> SiLU -> conv, twice, plus skip."""

    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = norm_layer_3d(norm_type, channels)
        self.norm2 = norm_layer_3d(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip_connection(x)


class DownsampleBlock3d(nn.Module):
    """2x dense 3D downsampling by strided conv or average pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        super().__init__()
        assert mode in ("conv", "avgpool")
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        else:
            assert in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    """2x dense 3D upsampling by conv + pixel shuffle or nearest interpolation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        super().__init__()
        assert mode in ("conv", "nearest")
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)
        else:
            assert in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return pixel_shuffle_3d(self.conv(x), 2)
        return F.interpolate(x, scale_factor=2, mode="nearest")
