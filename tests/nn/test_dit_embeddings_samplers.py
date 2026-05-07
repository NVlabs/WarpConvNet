# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for promoted diffusion building blocks (Tier A).

Covers the public API in:
- ``warpconvnet.nn.modules.embeddings``
- ``warpconvnet.nn.modules.dit``
- ``warpconvnet.nn.modules.normalizations`` (new fp32-internal + qk variants)
- ``warpconvnet.nn.functional.pixel_shuffle``
- ``warpconvnet.nn.samplers.flow_euler``
"""
import math

import torch

from warpconvnet.nn.functional.pixel_shuffle import (
    pixel_shuffle_3d,
    pixel_unshuffle_3d,
)
from warpconvnet.nn.modules.dit import (
    FeedForwardNet,
    ModulatedTransformerBlock,
    ModulatedTransformerCrossBlock,
    MultiHeadAttention,
)
from warpconvnet.nn.modules.embeddings import (
    RotaryPositionEmbedder,
    SinusoidalPositionEmbedder,
    TimestepEmbedder,
)
from warpconvnet.nn.modules.normalizations import (
    ChannelLayerNorm32,
    GroupNorm32,
    LayerNorm32,
    MultiHeadRMSNorm,
)
from warpconvnet.nn.samplers.flow_euler import (
    FlowEulerCfgSampler,
    FlowEulerSampler,
)


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------
def test_timestep_embedder_shape_and_formula():
    m = TimestepEmbedder(hidden_size=128, frequency_embedding_size=64)
    out = m(torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (3, 128)
    raw = TimestepEmbedder.timestep_embedding(torch.tensor([0.5]), 64)
    half = 32
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = torch.tensor([0.5])[:, None] * freqs[None]
    ref = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    torch.testing.assert_close(raw, ref)


def test_sinusoidal_position_embedder_shape():
    e = SinusoidalPositionEmbedder(channels=96, in_channels=3)
    coords = torch.randint(0, 8, (16, 3)).float()
    out = e(coords)
    assert out.shape == (16, 96)


def test_rope_phases_norm_preserving():
    rope = RotaryPositionEmbedder(head_dim=64, dim=3)
    coords = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.float32)
    phases = rope(coords)
    x = torch.randn(2, 4, 64)
    y = RotaryPositionEmbedder.apply_rotary_embedding(x, phases)
    x_pairs = x.reshape(2, 4, -1, 2)
    y_pairs = y.reshape(2, 4, -1, 2)
    torch.testing.assert_close(
        x_pairs.pow(2).sum(-1), y_pairs.pow(2).sum(-1), rtol=1e-5, atol=1e-5
    )


# -----------------------------------------------------------------------------
# Norms
# -----------------------------------------------------------------------------
def test_layernorm32_preserves_dtype():
    ln = LayerNorm32(64)
    x = torch.randn(4, 16, 64, dtype=torch.float16)
    assert ln(x).dtype == torch.float16


def test_channel_layer_norm32_normalizes_along_channel():
    ln = ChannelLayerNorm32(32)
    x = torch.randn(2, 32, 4, 4, 4)
    y = ln(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.mean(dim=1), torch.zeros_like(y.mean(dim=1)), atol=1e-5, rtol=0)


def test_groupnorm32_runs_in_fp32():
    gn = GroupNorm32(8, 32)
    x = torch.randn(2, 32, 4, 4, dtype=torch.float16)
    assert gn(x).dtype == torch.float16


def test_multi_head_rmsnorm_shape_and_grad():
    n = MultiHeadRMSNorm(dim=64, heads=12)
    x = torch.randn(2, 100, 12, 64, requires_grad=True)
    y = n(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


# -----------------------------------------------------------------------------
# Pixel shuffle / unshuffle
# -----------------------------------------------------------------------------
def test_pixel_shuffle_3d_inverse():
    x = torch.randn(2, 32, 4, 4, 4)
    y = pixel_unshuffle_3d(pixel_shuffle_3d(x, 2), 2)
    torch.testing.assert_close(x, y)


def test_pixel_shuffle_3d_shapes():
    x = torch.randn(1, 16, 4, 4, 4)
    y = pixel_shuffle_3d(x, 2)
    assert y.shape == (1, 2, 8, 8, 8)
    z = pixel_unshuffle_3d(y, 2)
    assert z.shape == (1, 16, 4, 4, 4)


# -----------------------------------------------------------------------------
# DiT blocks
# -----------------------------------------------------------------------------
def test_multi_head_attention_self():
    m = MultiHeadAttention(channels=64, num_heads=8, type="self", qk_rms_norm=True)
    x = torch.randn(2, 16, 64)
    assert m(x).shape == (2, 16, 64)


def test_multi_head_attention_cross():
    m = MultiHeadAttention(
        channels=64, ctx_channels=32, num_heads=8, type="cross", qk_rms_norm=True
    )
    x = torch.randn(2, 16, 64)
    ctx = torch.randn(2, 24, 32)
    assert m(x, context=ctx).shape == (2, 16, 64)


def test_feed_forward_net_shape():
    m = FeedForwardNet(channels=96, mlp_ratio=4.0)
    assert m(torch.randn(2, 8, 96)).shape == (2, 8, 96)


def test_modulated_transformer_block_shape():
    m = ModulatedTransformerBlock(channels=96, num_heads=8, share_mod=True)
    x = torch.randn(2, 16, 96)
    mod = torch.randn(2, 6 * 96)
    assert m(x, mod).shape == (2, 16, 96)


def test_modulated_transformer_cross_block_shape():
    m = ModulatedTransformerCrossBlock(
        channels=96, ctx_channels=64, num_heads=8, share_mod=True, qk_rms_norm=True
    )
    x = torch.randn(2, 16, 96)
    mod = torch.randn(2, 6 * 96)
    ctx = torch.randn(2, 32, 64)
    assert m(x, mod, ctx).shape == (2, 16, 96)


# -----------------------------------------------------------------------------
# Flow-matching sampler
# -----------------------------------------------------------------------------
class _ZeroVelocity(torch.nn.Module):
    def forward(self, x_t, t, cond=None, **kw):
        return torch.zeros_like(x_t)


def test_flow_euler_zero_velocity_preserves_noise():
    s = FlowEulerSampler(sigma_min=1e-5)
    noise = torch.randn(1, 4, 4, 4)
    out = s.sample(_ZeroVelocity(), noise, cond=None, steps=4, verbose=False)
    torch.testing.assert_close(out["samples"], noise)


def test_flow_euler_cfg_strength_one_equals_uncfg():
    cfg = FlowEulerCfgSampler(sigma_min=1e-5)
    plain = FlowEulerSampler(sigma_min=1e-5)
    noise = torch.randn(1, 4, 4, 4)
    cond = torch.zeros(1, 4)
    out_cfg = cfg.sample(
        _ZeroVelocity(),
        noise,
        cond=cond,
        neg_cond=cond,
        steps=4,
        guidance_strength=1.0,
        verbose=False,
    )
    out_plain = plain.sample(_ZeroVelocity(), noise, cond=cond, steps=4, verbose=False)
    torch.testing.assert_close(out_cfg["samples"], out_plain["samples"])
