# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 Sparse-Structure VAE (dense Conv3d encoder/decoder).

Encoder: 64³ binary occupancy → 16³ × latent_channels Gaussian latent.
Decoder: 16³ × latent_channels latent → 64³ logits (thresholded for occupancy).

Despite the name "sparse structure", these are *dense* 3D convolutions on
small cubes (e.g. 64³). The "sparse" here refers to what the latent represents
(an occupied subset of voxels), not the operator backend.

Mirrors `trellis2.models.sparse_structure_vae` exactly so the upstream
state_dict loads 1:1 by attribute name.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from warpconvnet.nn.functional.pixel_shuffle import pixel_shuffle_3d
from warpconvnet.nn.modules.conv3d_blocks import (
    DownsampleBlock3d,
    ResBlock3d,
    UpsampleBlock3d,
    norm_layer_3d,
)
from warpconvnet.nn.utils import DEFAULT_MIXED_PRECISION_MODULES, convert_module_parameters_to


def _norm_layer(norm_type: str, channels: int) -> nn.Module:
    return norm_layer_3d(norm_type, channels)


def _zero_module(module: nn.Module) -> nn.Module:
    from warpconvnet.nn.utils import zero_module

    zero_module(module)
    return module


_MIX_PRECISION_MODULES = DEFAULT_MIXED_PRECISION_MODULES


def _convert_module_to(module: nn.Module, dtype: torch.dtype) -> None:
    convert_module_parameters_to(module, dtype, module_types=_MIX_PRECISION_MODULES)


def _convert_module_to_f16(m: nn.Module) -> None:
    _convert_module_to(m, torch.float16)


def _convert_module_to_f32(m: nn.Module) -> None:
    _convert_module_to(m, torch.float32)


# -----------------------------------------------------------------------------
# VAE encoder / decoder
# -----------------------------------------------------------------------------
class SparseStructureEncoder(nn.Module):
    """Dense 3D Conv encoder producing a Gaussian VAE latent.

    Inference path returns ``z = mean`` (no posterior sampling).
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: list[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(DownsampleBlock3d(ch, channels[i + 1]))

        self.middle_block = nn.Sequential(
            *[ResBlock3d(channels[-1], channels[-1]) for _ in range(num_res_blocks_middle)]
        )

        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1),
        )

        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(_convert_module_to_f16)
        self.middle_block.apply(_convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(_convert_module_to_f32)
        self.middle_block.apply(_convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = False,
        return_raw: bool = False,
    ) -> torch.Tensor:
        h = self.input_layer(x).type(self.dtype)
        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h).type(x.dtype)
        h = self.out_layer(h)
        mean, logvar = h.chunk(2, dim=1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        if return_raw:
            return z, mean, logvar
        return z


class SparseStructureDecoder(nn.Module):
    """Dense 3D Conv decoder: latent → 64³ occupancy logits."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: list[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(
            *[ResBlock3d(channels[0], channels[0]) for _ in range(num_res_blocks_middle)]
        )

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))

        self.out_layer = nn.Sequential(
            _norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(_convert_module_to_f16)
        self.middle_block.apply(_convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(_convert_module_to_f32)
        self.middle_block.apply(_convert_module_to_f32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x).type(self.dtype)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = h.type(x.dtype)
        return self.out_layer(h)


__all__ = [
    "DownsampleBlock3d",
    "ResBlock3d",
    "SparseStructureDecoder",
    "SparseStructureEncoder",
    "UpsampleBlock3d",
]
