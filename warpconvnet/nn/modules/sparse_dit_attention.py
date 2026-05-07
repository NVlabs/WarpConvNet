# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse multi-head attention on `Voxels` (varlen flash-attn backend).

Generic sparse-voxel attention surface used by sparse DiT-style diffusion
models (e.g. TRELLIS.2 SLAT flow):

- Self-attention on voxel tokens (full attention via varlen-packed qkv).
- Cross-attention with sparse Q (voxels) and dense KV (image features).
- Optional RoPE phases derived from voxel coordinates, with a per-instance
  cache so multi-block stacks compute phases once.
- Optional qk-RMSNorm for attention-logit stability under bf16/fp16.

For dense `(B, L, C)` token attention see `warpconvnet.nn.modules.dit`.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.flash_attn_utils import flash_attn_varlen_qkvpacked
from warpconvnet.nn.modules.normalizations import MultiHeadRMSNorm


__all__ = [
    "SparseMultiHeadAttention",
    "SparseRotaryPositionEmbedder",
    "sparse_scaled_dot_product_attention",
]


# -----------------------------------------------------------------------------
# Cu-seqlens helpers
# -----------------------------------------------------------------------------
def _voxels_cu_seqlens(voxels: Voxels) -> tuple[torch.Tensor, int]:
    """Return ``(cu_seqlens, max_seqlen)`` for flash-attn varlen kernels.

    `voxels.offsets` is a CPU LongTensor; flash-attn wants an int32 device
    tensor of cumulative lengths.
    """
    off = voxels.offsets.to(device=voxels.device, dtype=torch.int32)
    max_seqlen = int((off[1:] - off[:-1]).max().item()) if off.numel() > 1 else 0
    return off.contiguous(), max_seqlen


def _dense_cu_seqlens(B: int, L: int, device: torch.device) -> torch.Tensor:
    """Cu-seqlens for a fixed-length dense (B, L, ...) sequence."""
    return (torch.arange(B + 1, device=device, dtype=torch.int32) * L).contiguous()


# -----------------------------------------------------------------------------
# RoPE on voxel coordinates
# -----------------------------------------------------------------------------
class SparseRotaryPositionEmbedder(nn.Module):
    """Per-voxel RoPE phases. Phases are cached on the input ``Voxels`` so
    multi-block stacks pay the polar-conversion cost once."""

    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        return torch.polar(torch.ones_like(phases), phases)

    @staticmethod
    def _rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rot = x_complex * phases.unsqueeze(-2)
        return torch.view_as_real(x_rot).reshape(*x_rot.shape[:-1], -1).to(x.dtype)

    def _phases_for(self, voxels: Voxels) -> torch.Tensor:
        cache_key = (
            f"rope_phase_{self.dim}d"
            f"_freq{self.rope_freq[0]}-{self.rope_freq[1]}_hd{self.head_dim}"
        )
        cache = voxels.spatial_cache
        phases = cache.get(cache_key)
        if phases is not None:
            return phases
        coords = voxels.coords[..., 1:]  # drop batch col
        phases = self._get_phases(coords.reshape(-1)).reshape(*coords.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
            pad_n = self.head_dim // 2 - phases.shape[-1]
            ones = torch.ones(*phases.shape[:-1], pad_n, device=phases.device)
            zeros = torch.zeros(*phases.shape[:-1], pad_n, device=phases.device)
            phases = torch.cat([phases, torch.polar(ones, zeros)], dim=-1)
        cache[cache_key] = phases
        return phases

    def forward(
        self,
        voxels: Voxels,
        q_feats: torch.Tensor,
        k_feats: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rotate per-token Q/K features inplace-style.

        ``q_feats`` / ``k_feats`` are the raw ``(T, H, D)`` tensors taken from
        the voxels' feature column post-Linear. Coords come from ``voxels``.
        """
        phases = self._phases_for(voxels)
        q_out = self._rotary_embedding(q_feats, phases)
        if k_feats is None:
            return q_out
        return q_out, self._rotary_embedding(k_feats, phases)


# -----------------------------------------------------------------------------
# Sparse SDPA dispatcher (varlen flash-attn)
# -----------------------------------------------------------------------------
def sparse_scaled_dot_product_attention(
    *args,
) -> torch.Tensor:
    """Dispatch full sparse attention.

    Three call shapes supported (matching the subset of upstream
    ``trellis2.modules.sparse.attention.full_attn`` that the SLAT flow model
    uses):

    - ``(qkv_feats, voxels)``: self-attn. ``qkv_feats`` is ``(T, 3, H, D)``,
      sequences delimited by ``voxels.offsets``. Returns ``(T, H, D)``.
    - ``(q_feats, voxels, kv_dense)``: cross-attn with dense KV. ``q_feats``
      is ``(T_q, H, D)``; ``kv_dense`` is ``(B, L, 2, H, D)``. Returns
      ``(T_q, H, D)``.
    - ``(q_feats, voxels, k_dense, v_dense)``: same as above but K and V
      passed separately as ``(B, L, H, D)``.
    """
    import flash_attn  # noqa: WPS433 — lazy: CUDA-only

    if len(args) == 2:
        qkv, voxels = args
        cu_q, max_q = _voxels_cu_seqlens(voxels)
        return flash_attn_varlen_qkvpacked(qkv, cu_q, max_q)

    if len(args) == 3:
        q, voxels, kv_dense = args
        assert kv_dense.ndim == 5, "kv_dense must be (B, L, 2, H, D)"
        B, L, _, H, D = kv_dense.shape
        cu_q, max_q = _voxels_cu_seqlens(voxels)
        cu_kv = _dense_cu_seqlens(B, L, q.device)
        return flash_attn.flash_attn_varlen_kvpacked_func(
            q,
            kv_dense.reshape(B * L, 2, H, D),
            cu_q,
            cu_kv,
            max_seqlen_q=max_q,
            max_seqlen_k=L,
        )

    if len(args) == 4:
        q, voxels, k_dense, v_dense = args
        assert k_dense.ndim == 4 and v_dense.ndim == 4
        B, L, H, _ = k_dense.shape
        cu_q, max_q = _voxels_cu_seqlens(voxels)
        cu_kv = _dense_cu_seqlens(B, L, q.device)
        return flash_attn.flash_attn_varlen_func(
            q,
            k_dense.reshape(B * L, *k_dense.shape[2:]),
            v_dense.reshape(B * L, *v_dense.shape[2:]),
            cu_q,
            cu_kv,
            max_seqlen_q=max_q,
            max_seqlen_k=L,
        )

    raise ValueError(f"sparse_scaled_dot_product_attention: bad arity {len(args)}")


# -----------------------------------------------------------------------------
# SparseMultiHeadAttention (self + cross, full attn only)
# -----------------------------------------------------------------------------
class SparseMultiHeadAttention(nn.Module):
    """Multi-head attention on a `Voxels` token sequence.

    Self-attention modes share Q/K/V via a single ``to_qkv`` Linear; cross
    modes split into ``to_q`` and ``to_kv`` (with separate ``ctx_channels``).
    Only ``attn_mode='full'`` is implemented in this Phase-6 cut; windowed
    attention is a Phase-8/9 extension if the downstream models need it.

    The ``context`` for cross attention may be either a ``Voxels`` (sparse KV)
    or a dense ``(B, L, ctx_channels)`` Tensor (image features).
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
        if attn_mode != "full":
            raise NotImplementedError(
                "SparseMultiHeadAttention currently supports only attn_mode='full'"
            )
        if type == "cross" and use_rope:
            raise ValueError("Rotary position embeddings only supported for self-attn")

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

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)

    # -- self attention --------------------------------------------------------
    def _forward_self(self, x: Voxels) -> Voxels:
        T = x.feats.shape[0]
        qkv = self.to_qkv(x.feats).reshape(T, 3, self.num_heads, self.head_dim)
        if self.qk_rms_norm or self.use_rope:
            q, k, v = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            if self.use_rope:
                q, k = self.rope(x, q, k)
            qkv = torch.stack([q, k, v], dim=1)
        h = sparse_scaled_dot_product_attention(qkv, x)  # (T, H, D)
        h = h.reshape(T, -1)
        return x.replace_features(self.to_out(h))

    # -- cross attention -------------------------------------------------------
    def _forward_cross(
        self,
        x: Voxels,
        context: torch.Tensor | Voxels,
    ) -> Voxels:
        T = x.feats.shape[0]
        q = self.to_q(x.feats).reshape(T, self.num_heads, self.head_dim)
        if isinstance(context, Voxels):
            T_kv = context.feats.shape[0]
            kv = self.to_kv(context.feats).reshape(T_kv, 2, self.num_heads, self.head_dim)
            cu_q, max_q = _voxels_cu_seqlens(x)
            cu_kv, max_kv = _voxels_cu_seqlens(context)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                import flash_attn

                h = flash_attn.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_q,
                    cu_kv,
                    max_seqlen_q=max_q,
                    max_seqlen_k=max_kv,
                )
            else:
                import flash_attn

                h = flash_attn.flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_q,
                    cu_kv,
                    max_seqlen_q=max_q,
                    max_seqlen_k=max_kv,
                )
        else:
            assert context.ndim == 3, "dense context must be (B, L, ctx_channels)"
            B, L, _ = context.shape
            kv = self.to_kv(context).reshape(B, L, 2, self.num_heads, self.head_dim)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = sparse_scaled_dot_product_attention(q, x, k, v)
            else:
                h = sparse_scaled_dot_product_attention(q, x, kv)
        h = h.reshape(T, -1)
        return x.replace_features(self.to_out(h))

    def forward(
        self,
        x: Voxels,
        context: torch.Tensor | Voxels | None = None,
    ) -> Voxels:
        if self._type == "self":
            return self._forward_self(x)
        assert context is not None, "cross-attention requires `context`"
        return self._forward_cross(x, context)
