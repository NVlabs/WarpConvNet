# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 tests: dense Sparse-Structure VAE (encoder + decoder).

Validates 1:1 forward parity with `trellis2.models.sparse_structure_vae` when
the upstream package is on PYTHONPATH or via TRELLIS2_PATH.
"""
import os
import sys

import pytest
import torch

from warpconvnet.models.trellis2.sparse_structure_vae import (
    ChannelLayerNorm32,
    DownsampleBlock3d,
    GroupNorm32,
    ResBlock3d,
    SparseStructureDecoder,
    SparseStructureEncoder,
    UpsampleBlock3d,
    pixel_shuffle_3d,
)

_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
_HAS_REF = False
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH):
    if _TRELLIS2_PATH not in sys.path:
        sys.path.insert(0, _TRELLIS2_PATH)
    try:
        from trellis2.models.sparse_structure_vae import (
            ResBlock3d as RefResBlock3d,
            DownsampleBlock3d as RefDownsampleBlock3d,
            UpsampleBlock3d as RefUpsampleBlock3d,
            SparseStructureEncoder as RefEncoder,
            SparseStructureDecoder as RefDecoder,
        )
        from trellis2.modules.spatial import pixel_shuffle_3d as ref_pixel_shuffle_3d

        _HAS_REF = True
    except Exception:  # noqa: BLE001
        _HAS_REF = False


# -----------------------------------------------------------------------------
# pixel_shuffle_3d
# -----------------------------------------------------------------------------
def test_pixel_shuffle_3d_shape():
    x = torch.randn(1, 16, 4, 4, 4)
    y = pixel_shuffle_3d(x, 2)
    assert y.shape == (1, 2, 8, 8, 8)


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_pixel_shuffle_3d_matches_reference():
    x = torch.randn(2, 32, 4, 5, 6)
    torch.testing.assert_close(pixel_shuffle_3d(x, 2), ref_pixel_shuffle_3d(x, 2))


# -----------------------------------------------------------------------------
# Norms
# -----------------------------------------------------------------------------
def test_groupnorm32_runs_in_fp32_and_casts_back():
    n = GroupNorm32(8, 32)
    x = torch.randn(2, 32, 4, 4, 4, dtype=torch.float16)
    y = n(x)
    assert y.dtype == torch.float16


def test_channel_layer_norm32_normalizes_along_channels():
    n = ChannelLayerNorm32(32)
    x = torch.randn(2, 32, 4, 4, 4)
    y = n(x)
    assert y.shape == x.shape
    # Per-position mean ≈ 0, var ≈ 1 across channels
    mean = y.mean(dim=1)
    var = y.var(dim=1, unbiased=False)
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=0)
    torch.testing.assert_close(var, torch.ones_like(var), atol=5e-2, rtol=5e-2)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("norm_type", ["group", "layer"])
def test_resblock3d_forward_shape(norm_type):
    blk = ResBlock3d(channels=64, out_channels=128, norm_type=norm_type)
    x = torch.randn(2, 64, 8, 8, 8)
    out = blk(x)
    assert out.shape == (2, 128, 8, 8, 8)


def test_resblock3d_zero_init_conv2_is_identity_at_start():
    """conv2 is zero-init ⇒ ResBlock(x) == skip_connection(x) before training."""
    blk = ResBlock3d(channels=32, out_channels=32)
    x = torch.randn(1, 32, 4, 4, 4)
    out = blk(x)
    torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)


def test_downsample_block_halves_spatial():
    d = DownsampleBlock3d(in_channels=16, out_channels=32, mode="conv")
    x = torch.randn(1, 16, 16, 16, 16)
    assert d(x).shape == (1, 32, 8, 8, 8)


def test_upsample_block_doubles_spatial():
    u = UpsampleBlock3d(in_channels=32, out_channels=16, mode="conv")
    x = torch.randn(1, 32, 8, 8, 8)
    assert u(x).shape == (1, 16, 16, 16, 16)


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_resblock3d_matches_reference():
    ours = ResBlock3d(64, 128, norm_type="layer")
    ref = RefResBlock3d(64, 128, norm_type="layer")
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(2, 64, 8, 8, 8)
    torch.testing.assert_close(ours(x), ref(x), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_downsample_upsample_match_reference():
    d_ours = DownsampleBlock3d(16, 32)
    d_ref = RefDownsampleBlock3d(16, 32)
    d_ref.load_state_dict(d_ours.state_dict())
    x = torch.randn(1, 16, 16, 16, 16)
    torch.testing.assert_close(d_ours(x), d_ref(x), rtol=1e-5, atol=1e-5)

    u_ours = UpsampleBlock3d(32, 16)
    u_ref = RefUpsampleBlock3d(32, 16)
    u_ref.load_state_dict(u_ours.state_dict())
    x = torch.randn(1, 32, 8, 8, 8)
    torch.testing.assert_close(u_ours(x), u_ref(x), rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Encoder + Decoder end-to-end
# -----------------------------------------------------------------------------
@pytest.fixture
def vae_kwargs():
    """Small VAE config (3-stage, 64³→16³ encoder)."""
    return dict(
        latent_channels=8,
        num_res_blocks=2,
        channels=[32, 64, 128],
        num_res_blocks_middle=2,
        norm_type="layer",
        use_fp16=False,
    )


def test_encoder_decoder_shape_roundtrip(vae_kwargs):
    enc = SparseStructureEncoder(in_channels=1, **vae_kwargs)
    dec = SparseStructureDecoder(out_channels=1, **vae_kwargs)
    x = torch.randn(2, 1, 64, 64, 64)
    z = enc(x)
    # 3-stage encoder ⇒ 2 downsamples ⇒ 64 → 16
    assert z.shape == (2, 8, 16, 16, 16)
    out = dec(z)
    assert out.shape == (2, 1, 64, 64, 64)


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_encoder_matches_reference(vae_kwargs):
    ours = SparseStructureEncoder(in_channels=1, **vae_kwargs)
    ref = RefEncoder(in_channels=1, **vae_kwargs)
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(2, 1, 32, 32, 32)
    torch.testing.assert_close(ours(x), ref(x), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_decoder_matches_reference(vae_kwargs):
    ours = SparseStructureDecoder(out_channels=1, **vae_kwargs)
    ref = RefDecoder(out_channels=1, **vae_kwargs)
    ref.load_state_dict(ours.state_dict())
    z = torch.randn(2, 8, 8, 8, 8)
    torch.testing.assert_close(ours(z), ref(z), rtol=1e-5, atol=1e-5)


def test_decoder_state_dict_roundtrip(vae_kwargs):
    a = SparseStructureDecoder(out_channels=1, **vae_kwargs)
    b = SparseStructureDecoder(out_channels=1, **vae_kwargs)
    b.load_state_dict(a.state_dict())
    z = torch.randn(1, 8, 8, 8, 8)
    torch.testing.assert_close(a(z), b(z), rtol=1e-5, atol=1e-5)
