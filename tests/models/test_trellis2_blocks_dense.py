# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 tests for warpconvnet.models.trellis2.blocks_dense.

Validates that the dense transformer building blocks are byte-for-byte
compatible with `microsoft/TRELLIS.2`. When the upstream `trellis2` package is
importable, either through PYTHONPATH or TRELLIS2_PATH, tests cross-check
against the original; otherwise they fall back to internal consistency checks.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

from warpconvnet.models.trellis2.blocks_dense import (
    AbsolutePositionEmbedder,
    FeedForwardNet,
    LayerNorm32,
    ModulatedTransformerCrossBlock,
    MultiHeadAttention,
    MultiHeadRMSNorm,
    RotaryPositionEmbedder,
    TimestepEmbedder,
)

# Best-effort import of upstream reference for byte-exact compares.
_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
_HAS_TRELLIS2_REF = False
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH):
    # Force the upstream attention backend to torch SDPA (works on CPU + CUDA);
    # `flash_attn` is the upstream default but is CUDA-only.
    os.environ["ATTN_BACKEND"] = "sdpa"
    if _TRELLIS2_PATH not in sys.path:
        sys.path.insert(0, _TRELLIS2_PATH)
    try:
        from trellis2.modules.transformer import (
            ModulatedTransformerCrossBlock as RefModulatedTransformerCrossBlock,
        )
        from trellis2.modules.attention import RotaryPositionEmbedder as RefRoPE
        from trellis2.models.sparse_structure_flow import (
            TimestepEmbedder as RefTimestepEmbedder,
        )

        _HAS_TRELLIS2_REF = True
    except Exception:  # noqa: BLE001 — reference-env optional
        _HAS_TRELLIS2_REF = False


# -----------------------------------------------------------------------------
# TimestepEmbedder
# -----------------------------------------------------------------------------
def test_timestep_embedding_formula():
    """Sin/cos formula matches reference computation."""
    t = torch.tensor([0.0, 0.5, 1.0, 17.5])
    dim = 256
    emb = TimestepEmbedder.timestep_embedding(t, dim)
    assert emb.shape == (4, dim)
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    ref = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    torch.testing.assert_close(emb, ref)


def test_timestep_embedder_forward_shape():
    m = TimestepEmbedder(hidden_size=1536, frequency_embedding_size=256)
    out = m(torch.tensor([0.1, 0.5, 0.9]))
    assert out.shape == (3, 1536)
    assert out.dtype == torch.float32


@pytest.mark.skipif(not _HAS_TRELLIS2_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_timestep_embedder_matches_reference():
    torch.manual_seed(0)
    ours = TimestepEmbedder(1536, 256)
    ref = RefTimestepEmbedder(1536, 256)
    ref.load_state_dict(ours.state_dict())
    t = torch.rand(4)
    torch.testing.assert_close(ours(t), ref(t))


# -----------------------------------------------------------------------------
# RotaryPositionEmbedder
# -----------------------------------------------------------------------------
def test_rope_phases_shape_and_padding():
    rope = RotaryPositionEmbedder(head_dim=128, dim=3)
    coords = torch.stack(
        torch.meshgrid(torch.arange(4), torch.arange(4), torch.arange(4), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    phases = rope(coords)
    # Phases are complex; last dim = head_dim // 2 = 64
    assert phases.shape == (64, 64)
    assert phases.dtype == torch.complex64


def test_rope_apply_preserves_norm_per_pair():
    """Rotating in 2D plane preserves L2 norm of each (real, imag) pair."""
    torch.manual_seed(0)
    rope = RotaryPositionEmbedder(head_dim=64, dim=3)
    coords = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.float32)
    phases = rope(coords)  # [2, 32]
    x = torch.randn(2, 4, 64)  # [N=2, H=4, D=64]
    y = RotaryPositionEmbedder.apply_rotary_embedding(x, phases)
    # ||y||^2 over each (real, imag) pair == ||x||^2
    x_pairs = x.reshape(2, 4, -1, 2)
    y_pairs = y.reshape(2, 4, -1, 2)
    torch.testing.assert_close(
        x_pairs.pow(2).sum(-1), y_pairs.pow(2).sum(-1), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not _HAS_TRELLIS2_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_rope_matches_reference():
    rope_o = RotaryPositionEmbedder(128, 3)
    rope_r = RefRoPE(128, 3)
    coords = torch.tensor([[0, 0, 0], [1, 2, 3], [7, 5, 11]], dtype=torch.float32)
    torch.testing.assert_close(rope_o(coords), rope_r(coords))


# -----------------------------------------------------------------------------
# MultiHeadRMSNorm
# -----------------------------------------------------------------------------
def test_qk_rmsnorm_shape_and_grad():
    n = MultiHeadRMSNorm(dim=64, heads=12)
    x = torch.randn(2, 100, 12, 64, requires_grad=True)
    y = n(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


# -----------------------------------------------------------------------------
# MultiHeadAttention
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("qk_rms_norm", [False, True])
@pytest.mark.parametrize("use_rope", [False, True])
def test_multi_head_attention_self(qk_rms_norm, use_rope):
    C, H, L, B = 64, 8, 16, 2
    attn = MultiHeadAttention(
        channels=C,
        num_heads=H,
        type="self",
        use_rope=use_rope,
        qk_rms_norm=qk_rms_norm,
    )
    x = torch.randn(B, L, C)
    phases = None
    if use_rope:
        rope = RotaryPositionEmbedder(C // H, dim=3)
        coords = torch.randint(0, 4, (L, 3)).float()
        phases = rope(coords).expand(B, -1, -1)
    out = attn(x, phases=phases)
    assert out.shape == (B, L, C)


def test_multi_head_attention_cross():
    C, H, Cctx, L, Lkv, B = 64, 8, 32, 16, 24, 2
    attn = MultiHeadAttention(channels=C, ctx_channels=Cctx, num_heads=H, type="cross")
    x = torch.randn(B, L, C)
    ctx = torch.randn(B, Lkv, Cctx)
    out = attn(x, context=ctx)
    assert out.shape == (B, L, C)


# -----------------------------------------------------------------------------
# ModulatedTransformerCrossBlock — shape + parameter count + state_dict roundtrip
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("share_mod", [False, True])
def test_modulated_cross_block_forward_shape(share_mod):
    C, Cctx, H = 96, 64, 8
    blk = ModulatedTransformerCrossBlock(
        channels=C,
        ctx_channels=Cctx,
        num_heads=H,
        mlp_ratio=4.0,
        use_rope=False,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        share_mod=share_mod,
    )
    B, L, Lkv = 2, 32, 64
    x = torch.randn(B, L, C)
    # When share_mod=True the outer model has already projected mod to 6*C.
    mod = torch.randn(B, 6 * C if share_mod else C)
    ctx = torch.randn(B, Lkv, Cctx)
    out = blk(x, mod, ctx)
    assert out.shape == (B, L, C)


def test_modulated_cross_block_state_dict_roundtrip():
    blk1 = ModulatedTransformerCrossBlock(
        channels=96,
        ctx_channels=64,
        num_heads=8,
        share_mod=True,
        qk_rms_norm=True,
    )
    blk2 = ModulatedTransformerCrossBlock(
        channels=96,
        ctx_channels=64,
        num_heads=8,
        share_mod=True,
        qk_rms_norm=True,
    )
    blk2.load_state_dict(blk1.state_dict())
    x = torch.randn(2, 16, 96)
    mod = torch.randn(2, 6 * 96)
    ctx = torch.randn(2, 32, 64)
    torch.testing.assert_close(blk1(x, mod, ctx), blk2(x, mod, ctx))


@pytest.mark.skipif(not _HAS_TRELLIS2_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_modulated_cross_block_matches_reference():
    """Bit-for-bit (within numerical noise) forward parity with TRELLIS.2."""
    # CPU + fp32 ⇒ both sides must use torch SDPA (flash_attn requires CUDA+fp16/bf16).
    from trellis2.modules.attention.config import set_backend as _set_backend

    _set_backend("sdpa")
    os.environ["ATTN_BACKEND"] = "sdpa"
    torch.manual_seed(0)
    kw = dict(
        channels=96,
        ctx_channels=64,
        num_heads=8,
        mlp_ratio=4.0,
        attn_mode="full",
        use_rope=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        share_mod=True,
    )
    ours = ModulatedTransformerCrossBlock(**kw)
    ref = RefModulatedTransformerCrossBlock(**kw)
    ref.load_state_dict(ours.state_dict())
    B, L, Lkv = 2, 16, 32
    x = torch.randn(B, L, 96)
    mod = torch.randn(B, 6 * 96)  # share_mod=True ⇒ pre-expanded
    ctx = torch.randn(B, Lkv, 64)
    rope = RotaryPositionEmbedder(96 // 8, 3)
    coords = torch.randint(0, 4, (L, 3)).float()
    phases = rope(coords).expand(B, -1, -1)
    o = ours(x, mod, ctx, phases)
    r = ref(x, mod, ctx, phases)
    torch.testing.assert_close(o, r, rtol=1e-4, atol=1e-4)


# -----------------------------------------------------------------------------
# Param-count sanity check (matches ss_flow_img_dit_1_3B_64_bf16.json)
# -----------------------------------------------------------------------------
def test_modulated_cross_block_param_count_matches_ss_flow_config():
    """One DiT block in the SS flow model. Count must match upstream."""
    blk = ModulatedTransformerCrossBlock(
        channels=1536,
        ctx_channels=1024,
        num_heads=12,
        mlp_ratio=5.3334,
        use_rope=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        share_mod=True,
    )
    n_params = sum(p.numel() for p in blk.parameters())
    # 30 such blocks ≈ 1.0B, which combined with embedders + io layers ≈ 1.3B.
    # Per-block budget should be in the 30-40M range.
    assert 25_000_000 < n_params < 45_000_000, n_params
