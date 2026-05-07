# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse DiT blocks (adaLN-modulated transformer over `Voxels`).

Generic sparse counterpart to `warpconvnet.nn.modules.dit`. ``x`` is a
`Voxels` token sequence; ``mod`` is a per-batch ``(B, C)`` (or ``(B, 6*C)``
when ``share_mod=True``) conditioning tensor; ``context`` is either a dense
``(B, L, ctx_channels)`` tensor (image features) or a `Voxels` (sparse
cross-attention).
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.normalizations import LayerNorm32
from warpconvnet.nn.modules.sparse_dit_attention import SparseMultiHeadAttention


__all__ = [
    "ModulatedSparseTransformerBlock",
    "ModulatedSparseTransformerCrossBlock",
    "SparseFeedForwardNet",
]


# -----------------------------------------------------------------------------
# Per-voxel modulation helper
# -----------------------------------------------------------------------------
def _per_voxel(x: Voxels, t: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-batch ``(B, C)`` tensor to per-voxel ``(N_total, C)``."""
    return t[x.coords[:, 0].long()]


# -----------------------------------------------------------------------------
# Sparse FFN
# -----------------------------------------------------------------------------
class SparseFeedForwardNet(nn.Module):
    """Linear-GELU(tanh)-Linear MLP applied per-voxel.

    Keeps the upstream attribute layout (``self.mlp[0]``/``self.mlp[2]``) so
    state_dict keys remain ``mlp.mlp.0.weight`` etc.
    """

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: Voxels) -> Voxels:
        return x.replace_features(self.mlp(x.feats))


# -----------------------------------------------------------------------------
# Modulated self-only block
# -----------------------------------------------------------------------------
class ModulatedSparseTransformerBlock(nn.Module):
    """Sparse DiT block: ``norm1 → adaLN(MSA) → norm2 → adaLN(FFN)``."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full"] = "full",
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)

    def _split_mod(self, mod: torch.Tensor):
        if self.share_mod:
            return (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        return self.adaLN_modulation(mod).chunk(6, dim=1)

    def _forward(self, x: Voxels, mod: torch.Tensor) -> Voxels:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_mod(mod)
        # MSA branch ---------------------------------------------------------
        h_feats = self.norm1(x.feats)
        h_feats = h_feats * (1 + _per_voxel(x, scale_msa)) + _per_voxel(x, shift_msa)
        h = x.replace_features(h_feats)
        h = self.attn(h)
        h = h.replace_features(h.feats * _per_voxel(x, gate_msa))
        x = x + h
        # FFN branch ---------------------------------------------------------
        h_feats = self.norm2(x.feats)
        h_feats = h_feats * (1 + _per_voxel(x, scale_mlp)) + _per_voxel(x, shift_mlp)
        h = x.replace_features(h_feats)
        h = self.mlp(h)
        h = h.replace_features(h.feats * _per_voxel(x, gate_mlp))
        return x + h

    def forward(self, x: Voxels, mod: torch.Tensor) -> Voxels:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        return self._forward(x, mod)


# -----------------------------------------------------------------------------
# Modulated cross block (self + cross + FFN)
# -----------------------------------------------------------------------------
class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Sparse DiT cross block: ``MSA → MCA → FFN`` with adaLN modulation.

    The cross-attn norm (``norm2``) is the only norm with affine params,
    matching upstream `trellis2.modules.sparse.transformer.modulated`.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full"] = "full",
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)

    def _split_mod(self, mod: torch.Tensor):
        if self.share_mod:
            return (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        return self.adaLN_modulation(mod).chunk(6, dim=1)

    def _forward(
        self,
        x: Voxels,
        mod: torch.Tensor,
        context: torch.Tensor | Voxels,
    ) -> Voxels:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_mod(mod)
        # Self-attn ----------------------------------------------------------
        h_feats = self.norm1(x.feats)
        h_feats = h_feats * (1 + _per_voxel(x, scale_msa)) + _per_voxel(x, shift_msa)
        h = x.replace_features(h_feats)
        h = self.self_attn(h)
        h = h.replace_features(h.feats * _per_voxel(x, gate_msa))
        x = x + h
        # Cross-attn (no scale/shift/gate per upstream) ----------------------
        h = x.replace_features(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        # FFN ----------------------------------------------------------------
        h_feats = self.norm3(x.feats)
        h_feats = h_feats * (1 + _per_voxel(x, scale_mlp)) + _per_voxel(x, shift_mlp)
        h = x.replace_features(h_feats)
        h = self.mlp(h)
        h = h.replace_features(h.feats * _per_voxel(x, gate_mlp))
        return x + h

    def forward(
        self,
        x: Voxels,
        mod: torch.Tensor,
        context: torch.Tensor | Voxels,
    ) -> Voxels:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, use_reentrant=False
            )
        return self._forward(x, mod, context)
