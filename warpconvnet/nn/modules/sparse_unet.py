# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable sparse U-Net blocks and stage assembly on Voxels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.normalizations import LayerNorm32
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.sparse_resample import SparseChannel2Spatial
from warpconvnet.nn.utils import zero_module


__all__ = ["SparseChannelToSpatialResBlock3d", "SparseUNetDecoderStages"]


class SparseUNetDecoderStages(nn.ModuleList):
    """Resolution-stage assembly for sparse U-Net decoders.

    The class subclasses ``torch.nn.ModuleList`` so assigning it as
    ``self.blocks`` preserves familiar state-dict names such as
    ``blocks.0.0.weight``. Each stage contains ``num_blocks[i]`` residual
    blocks followed by an optional upsample block between resolutions.
    """

    def __init__(
        self,
        model_channels: list[int],
        num_blocks: list[int],
        block_type: list[str],
        up_block_type: list[str],
        block_args: list[dict[str, Any]],
        block_registry: Mapping[str, type[nn.Module]],
        up_block_kwargs: dict[str, Any] | None = None,
    ):
        if not (len(model_channels) == len(num_blocks) == len(block_type) == len(block_args)):
            raise ValueError("model_channels, num_blocks, block_type, and block_args must align")
        if len(up_block_type) != max(0, len(num_blocks) - 1):
            raise ValueError("up_block_type must have one entry between each resolution stage")

        stages: list[nn.ModuleList] = []
        up_block_kwargs = up_block_kwargs or {}
        for i, n_blocks in enumerate(num_blocks):
            stage = nn.ModuleList([])
            for _ in range(n_blocks):
                stage.append(block_registry[block_type[i]](model_channels[i], **block_args[i]))
            if i < len(num_blocks) - 1:
                kwargs = dict(block_args[i])
                kwargs.update(up_block_kwargs)
                stage.append(
                    block_registry[up_block_type[i]](
                        model_channels[i],
                        model_channels[i + 1],
                        **kwargs,
                    )
                )
            stages.append(stage)
        super().__init__(stages)

        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.up_block_type = up_block_type

    def run(
        self,
        x: Voxels,
        guide_subs: list[Voxels] | None = None,
        return_subs: bool = False,
        stop_before_stage: int | None = None,
    ):
        """Run decoder stages.

        ``guide_subs`` supplies explicit subdivision masks for upsample blocks.
        ``return_subs`` collects subdivision predictions from blocks that return
        ``(x, subdiv)``. ``stop_before_stage`` returns early before a stage is
        executed, useful for cascade coordinate upsampling.
        """
        if guide_subs is not None and return_subs:
            raise ValueError("guide_subs and return_subs are mutually exclusive")

        subs: list[Voxels] = []
        for i, stage in enumerate(self):
            if stop_before_stage is not None and i == stop_before_stage:
                return (x, subs) if return_subs else x
            for j, block in enumerate(stage):
                is_last_in_stage = j == len(stage) - 1
                is_upsample = i < len(self) - 1 and is_last_in_stage
                if is_upsample and guide_subs is not None:
                    x = block(x, subdiv=guide_subs[i])
                    continue

                out = block(x)
                if isinstance(out, tuple):
                    x, sub = out
                    if return_subs:
                        subs.append(sub)
                else:
                    x = out

        return (x, subs) if return_subs else x


class SparseChannelToSpatialResBlock3d(nn.Module):
    """Residual block that upsamples sparse voxels via channel-to-spatial unpacking.

    The block projects ``channels`` to ``out_channels * factor**3`` channels,
    unpacks neighbouring child voxels with ``SparseChannel2Spatial``, then
    applies a zero-initialized sparse conv and residual skip. An optional
    subdivision head can predict which child slots to materialize.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        factor: int = 2,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
        conv_cls: type[nn.Module] = SparseConv3d,
        norm_cls: type[nn.Module] = LayerNorm32,
        kernel_size: int | tuple[int, int, int] = 3,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.factor = factor
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        self.num_children = factor**3

        if channels % self.num_children != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by factor**3 ({self.num_children})"
            )
        if self.out_channels % (channels // self.num_children) != 0:
            raise ValueError(
                "out_channels must be divisible by channels // factor**3 "
                f"({channels // self.num_children})"
            )

        self.norm1 = norm_cls(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = norm_cls(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = conv_cls(channels, self.out_channels * self.num_children, kernel_size)
        self.conv2 = zero_module(conv_cls(self.out_channels, self.out_channels, kernel_size))
        self._repeat = self.out_channels // (channels // self.num_children)
        if pred_subdiv:
            self.to_subdiv = nn.Linear(channels, self.num_children)
        self.updown = SparseChannel2Spatial(factor)

    def _skip(self, x: Voxels) -> Voxels:
        return x.replace_features(x.feats.repeat_interleave(self._repeat, dim=1))

    def _forward(
        self,
        x: Voxels,
        subdiv: Voxels | None = None,
    ):
        if self.pred_subdiv:
            subdiv = x.replace_features(self.to_subdiv(x.feats))
        h = x.replace_features(self.norm1(x.feats))
        h = h.replace_features(F.silu(h.feats))
        h = self.conv1(h)
        sub_bin = subdiv.replace_features(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, sub_bin)
        x = self.updown(x, sub_bin)
        h = h.replace_features(self.norm2(h.feats))
        h = h.replace_features(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self._skip(x)
        if self.pred_subdiv:
            return h, subdiv
        return h

    def forward(
        self,
        x: Voxels,
        subdiv: Voxels | None = None,
    ):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        return self._forward(x, subdiv)
