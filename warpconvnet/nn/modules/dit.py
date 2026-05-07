# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiT-style transformer building blocks for diffusion / flow models.

Operates on dense `(B, L, C)` token sequences. Provides:

- `MultiHeadAttention`         — self/cross attention with optional RoPE + qk-norm
- `FeedForwardNet`             — Linear-GELU(tanh)-Linear MLP
- `ModulatedTransformerBlock`  — DiT block (self-attn + FFN with adaLN modulation)
- `ModulatedTransformerCrossBlock` — DiT cross block (self + cross + FFN, adaLN)

These mirror the building blocks introduced by DiT / SD3 / Flux / TRELLIS and
are usable for any diffusion-transformer port. For sparse-voxel attention see
`warpconvnet.nn.modules.attention` and `warpconvnet.nn.modules.space_attention`.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.nn.modules.embeddings import RotaryPositionEmbedder
from warpconvnet.nn.modules.normalizations import LayerNorm32, MultiHeadRMSNorm


__all__ = [
    "FeedForwardNet",
    "ModulatedTransformerBlock",
    "ModulatedTransformerCrossBlock",
    "MultiHeadAttention",
]


import os as _os

try:
    import flash_attn as _flash_attn  # noqa: F401

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


def _sdpa_4d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """`(N, L, H, C)` attention → `(N, L, H, C)`.

    Prefers `flash_attn.flash_attn_func` for bf16/fp16 inputs (matches the
    backend used by the published TRELLIS.2 / Stable-Diffusion-3 / Flux
    checkpoints — different reduction order than torch SDPA leads to
    ~10% rtol drift on long DiT stacks if mixed). Falls back to torch
    SDPA when flash-attn is unavailable, input is fp32 (flash-attn requires
    fp16/bf16), or ``ATTN_BACKEND=sdpa`` is set in the environment (used by
    upstream TRELLIS2 + sibling tests to force the slower-but-CPU-capable
    backend).
    """
    use_flash = (
        _HAS_FLASH_ATTN
        and q.dtype in (torch.float16, torch.bfloat16)
        and _os.environ.get("ATTN_BACKEND", "flash_attn") != "sdpa"
    )
    if use_flash:
        from flash_attn import flash_attn_func

        return flash_attn_func(q, k, v)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3).contiguous()


class MultiHeadAttention(nn.Module):
    """Dense multi-head attention with optional RoPE + qk-RMSNorm.

    Supports ``type='self'`` (Q=K=V from `x`) or ``type='cross'``
    (Q from `x`, K/V from `context`). Cross attention is always full.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: int | None = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full"] = "full",
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ("self", "cross")
        assert attn_mode == "full"
        assert type == "self" or attn_mode == "full"

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1)
            q, k, v = qkv.unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            if self.use_rope:
                assert phases is not None, "phases required when use_rope=True"
                q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
            h = _sdpa_4d(q, k, v)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x).reshape(B, L, self.num_heads, -1)
            kv = self.to_kv(context).reshape(B, Lkv, 2, self.num_heads, -1)
            k, v = kv.unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            h = _sdpa_4d(q, k, v)
        return self.to_out(h.reshape(B, L, -1))


class FeedForwardNet(nn.Module):
    """Linear-GELU(tanh)-Linear feed-forward block (mlp_ratio scaled hidden)."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ModulatedTransformerBlock(nn.Module):
    """DiT block: norm → adaLN-modulated self-attn → norm → adaLN-modulated FFN."""

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
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)

    def _forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                mod
            ).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, phases=phases)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        return x + h

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, phases, use_reentrant=False
            )
        return self._forward(x, mod, phases)


class ModulatedTransformerCrossBlock(nn.Module):
    """DiT cross block: self-attn → cross-attn → FFN with adaLN modulation."""

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
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)

    def _forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                mod
            ).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, phases=phases)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        return x + h

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, phases, use_reentrant=False
            )
        return self._forward(x, mod, context, phases)
