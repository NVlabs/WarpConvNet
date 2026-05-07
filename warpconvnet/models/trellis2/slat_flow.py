# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 Sparse Latent (SLAT) flow-matching model.

Operates on a `Voxels` token sequence (sparse 32^3 latent at the configured
resolution) and predicts velocity for rectified-flow sampling.

Mirrors `trellis2.models.structured_latent_flow.SLatFlowModel` exactly so the
upstream state_dict (`microsoft/TRELLIS.2-4B/ckpts/slat_flow_*.safetensors`)
loads 1:1 by attribute name. Inference-only: the elastic-mixin variant
(`ElasticSLatFlowModel`) is not ported because it only matters for training.
"""
from __future__ import annotations

from functools import partial
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.embeddings import (
    SinusoidalPositionEmbedder,
    TimestepEmbedder,
)
from warpconvnet.nn.utils import convert_module_parameters_to

from .blocks_dense import manual_cast, str_to_dtype
from .blocks_sparse import ModulatedSparseTransformerCrossBlock


__all__ = ["SLatFlowModel"]


def _convert_module_to(m: nn.Module, dtype: torch.dtype) -> None:
    convert_module_parameters_to(m, dtype)


class SLatFlowModel(nn.Module):
    """Image- (or shape-) conditioned sparse DiT for SLAT diffusion.

    Architecture:
        input_layer (nn.Linear, in_channels -> model_channels)
        + t_embedder (sinusoidal -> MLP)
        + optional shared adaLN_modulation (when ``share_mod=True``)
        + N x ModulatedSparseTransformerCrossBlock
        + LayerNorm (no affine) + out_layer (nn.Linear)

    Cross-attention context can be a dense ``(B, L, cond_channels)`` tensor
    (image features from DinoV3) or any tensor matching the cross-attn API.
    Variable-length sparse contexts (List[Tensor]) — used for shape-to-tex
    flow — are not in this Phase-8 cut.
    """

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
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = SinusoidalPositionEmbedder(model_channels)

        self.input_layer = nn.Linear(in_channels, model_channels)

        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """Cast the transformer torso (Linears) to ``dtype``; norms stay fp32."""
        self.dtype = dtype
        self.blocks.apply(partial(_convert_module_to, dtype=dtype))

    def forward(
        self,
        x: Voxels,
        t: torch.Tensor,
        cond: torch.Tensor | list[torch.Tensor],
        **kwargs,
    ) -> Voxels:
        if isinstance(cond, list):
            raise NotImplementedError(
                "List[Tensor] cond (variable-length sparse context) not yet ported"
            )

        # Input projection (per-voxel Linear).
        h = x.replace_features(self.input_layer(x.feats))
        h = h.replace_features(manual_cast(h.feats, self.dtype))

        # Timestep + (optional) shared modulation.
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        # Optional absolute position embedding.
        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:].float())
            pe_per_voxel = manual_cast(pe, self.dtype)
            h = h.replace_features(h.feats + pe_per_voxel)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        # Output projection (LayerNorm w/o affine + per-voxel Linear).
        h = h.replace_features(manual_cast(h.feats, x.dtype))
        h = h.replace_features(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = h.replace_features(self.out_layer(h.feats))
        return h
