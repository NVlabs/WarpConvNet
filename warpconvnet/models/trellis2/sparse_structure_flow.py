# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 Sparse-Structure flow-matching model (dense DiT).

Operates on a (B, in_channels, R, R, R) cube (R=16 for the 4B checkpoint).
Tokens are flattened to (B, R³, model_channels) and processed through a stack
of `ModulatedTransformerCrossBlock`s with image cross-attention.

Mirrors `trellis2.models.sparse_structure_flow.SparseStructureFlowModel` so the
upstream state_dict loads 1:1 by attribute name.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warpconvnet.nn.utils import convert_module_parameters_to

from .blocks_dense import (
    AbsolutePositionEmbedder,
    ModulatedTransformerCrossBlock,
    RotaryPositionEmbedder,
    TimestepEmbedder,
    manual_cast,
    str_to_dtype,
)


def _convert_module_to(m: nn.Module, dtype: torch.dtype) -> None:
    convert_module_parameters_to(m, dtype)


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: int | None = None,
        num_head_channels: int | None = 64,
        mlp_ratio: float = 4.0,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = "vanilla",
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        coords = torch.meshgrid(*[torch.arange(resolution) for _ in range(3)], indexing="ij")
        coords = torch.stack(coords, dim=-1).reshape(-1, 3).float()
        if pe_mode == "ape":
            ape = AbsolutePositionEmbedder(model_channels, 3)
            self.register_buffer("pos_emb", ape(coords), persistent=False)
            self.rope_phases = None
        elif pe_mode == "rope":
            rope = RotaryPositionEmbedder(model_channels // self.num_heads, 3)
            # complex64 ⇒ safetensors-incompatible; recompute on load.
            self.register_buffer("rope_phases", rope(coords), persistent=False)
            self.pos_emb = None
        else:
            raise ValueError(f"Invalid pe_mode {pe_mode}")

        self.input_layer = nn.Linear(in_channels, model_channels)

        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    rope_freq=rope_freq,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_layer = nn.Linear(model_channels, out_channels)

        self.convert_to(self.dtype)

    def convert_to(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.blocks.apply(partial(_convert_module_to, dtype=dtype))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert list(x.shape) == [
            x.shape[0],
            self.in_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        ], f"Input shape mismatch: {x.shape}"
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        h = manual_cast(h, self.dtype)
        cond = manual_cast(cond, self.dtype)
        phases = self.rope_phases if self.pe_mode == "rope" else None
        for block in self.blocks:
            h = block(h, t_emb, cond, phases)
        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()
        return h


__all__ = ["SparseStructureFlowModel"]
