# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 tests: SparseMultiHeadAttention + SparseRotaryPositionEmbedder.

Sparse attention requires flash-attn varlen kernels which are CUDA-only;
non-CUDA hosts skip the forward tests but still cover state_dict /
shape-derivation paths.
"""
import importlib
import os
import sys

import pytest
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.trellis2.sparse_attention import (
    SparseMultiHeadAttention,
    SparseRotaryPositionEmbedder,
)
from warpconvnet.models.trellis2.sparse_ops import from_feats_coords

_HAS_CUDA = torch.cuda.is_available()
_HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
_skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for flash-attn")
_skip_no_flash = pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")

_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
_HAS_REF = False
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH):
    os.environ["ATTN_BACKEND"] = "flash_attn"
    os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    if _TRELLIS2_PATH not in sys.path:
        sys.path.insert(0, _TRELLIS2_PATH)
    try:
        from trellis2.modules.sparse.attention.modules import (
            SparseMultiHeadAttention as RefSparseMultiHeadAttention,
        )
        from trellis2.modules.sparse.basic import SparseTensor as RefSparseTensor

        _HAS_REF = True
    except Exception:  # noqa: BLE001
        _HAS_REF = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _make_voxels(B: int = 2, N_per: int = 32, C: int = 96, R: int = 8, seed: int = 0) -> Voxels:
    g = torch.Generator().manual_seed(seed)
    coords_list, feats_list = [], []
    for b in range(B):
        flat = torch.randperm(R**3, generator=g)[:N_per]
        x = flat // (R * R)
        y = (flat // R) % R
        z = flat % R
        coords_list.append(torch.stack([torch.full_like(x, b), x, y, z], dim=-1).int())
        feats_list.append(torch.randn(N_per, C, generator=g))
    return from_feats_coords(torch.cat(feats_list, dim=0), torch.cat(coords_list, dim=0))


# -----------------------------------------------------------------------------
# RoPE phase computation (CPU OK)
# -----------------------------------------------------------------------------
def test_rope_phases_shape_and_caching():
    rope = SparseRotaryPositionEmbedder(head_dim=64, dim=3)
    v = _make_voxels(B=1, N_per=16)
    p1 = rope._phases_for(v)
    assert p1.shape == (16, 32)  # head_dim // 2
    assert p1.dtype == torch.complex64
    # Second call should hit the spatial cache (same object identity).
    p2 = rope._phases_for(v)
    assert p1 is p2


# -----------------------------------------------------------------------------
# SparseMultiHeadAttention construction + state_dict
# -----------------------------------------------------------------------------
def test_self_attn_construction():
    m = SparseMultiHeadAttention(
        channels=96, num_heads=8, type="self", qk_rms_norm=True, use_rope=True
    )
    sd = m.state_dict()
    assert "to_qkv.weight" in sd
    assert "to_out.weight" in sd
    assert "q_rms_norm.gamma" in sd and sd["q_rms_norm.gamma"].shape == (8, 12)


def test_cross_attn_construction():
    m = SparseMultiHeadAttention(
        channels=96,
        ctx_channels=64,
        num_heads=8,
        type="cross",
        qk_rms_norm=True,
    )
    sd = m.state_dict()
    assert "to_q.weight" in sd
    assert "to_kv.weight" in sd
    assert sd["to_kv.weight"].shape == (2 * 96, 64)


def test_cross_attn_disallows_rope():
    with pytest.raises(ValueError, match="self-attn"):
        SparseMultiHeadAttention(channels=96, num_heads=8, type="cross", use_rope=True)


# -----------------------------------------------------------------------------
# CUDA forward (flash-attn)
# -----------------------------------------------------------------------------
@_skip_no_cuda
@_skip_no_flash
@pytest.mark.parametrize("qk_rms_norm", [False, True])
@pytest.mark.parametrize("use_rope", [False, True])
def test_self_attn_forward(qk_rms_norm, use_rope):
    torch.manual_seed(0)
    m = (
        SparseMultiHeadAttention(
            channels=96,
            num_heads=8,
            type="self",
            qk_rms_norm=qk_rms_norm,
            use_rope=use_rope,
        )
        .cuda()
        .to(torch.float16)
    )
    v = _make_voxels(B=2, N_per=32, C=96).to("cuda").half()
    out = m(v)
    assert isinstance(out, Voxels)
    assert out.feats.shape == v.feats.shape


@_skip_no_cuda
@_skip_no_flash
@pytest.mark.parametrize("qk_rms_norm", [False, True])
def test_cross_attn_dense_kv_forward(qk_rms_norm):
    torch.manual_seed(0)
    m = (
        SparseMultiHeadAttention(
            channels=96,
            ctx_channels=64,
            num_heads=8,
            type="cross",
            qk_rms_norm=qk_rms_norm,
        )
        .cuda()
        .to(torch.float16)
    )
    v = _make_voxels(B=2, N_per=32, C=96).to("cuda").half()
    cond = torch.randn(2, 24, 64, device="cuda", dtype=torch.float16)
    out = m(v, context=cond)
    assert out.feats.shape == v.feats.shape


@_skip_no_cuda
@_skip_no_flash
@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
@pytest.mark.parametrize("qk_rms_norm", [False, True])
@pytest.mark.parametrize("use_rope", [False, True])
def test_self_attn_matches_reference(qk_rms_norm, use_rope):
    torch.manual_seed(0)
    kw = dict(
        channels=96,
        num_heads=8,
        type="self",
        qk_rms_norm=qk_rms_norm,
        use_rope=use_rope,
    )
    ours = SparseMultiHeadAttention(**kw).cuda().to(torch.float16)
    ref = RefSparseMultiHeadAttention(**kw).cuda().to(torch.float16)
    ref.load_state_dict(ours.state_dict())

    v_ours = _make_voxels(B=2, N_per=32, C=96).to("cuda").half()
    v_ref = RefSparseTensor(v_ours.feats.clone(), v_ours.coords.clone())

    o = ours(v_ours).feats
    r = ref(v_ref).feats
    torch.testing.assert_close(o, r, rtol=5e-3, atol=5e-3)


@_skip_no_cuda
@_skip_no_flash
@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
@pytest.mark.parametrize("qk_rms_norm", [False, True])
def test_cross_attn_dense_kv_matches_reference(qk_rms_norm):
    torch.manual_seed(0)
    kw = dict(
        channels=96,
        ctx_channels=64,
        num_heads=8,
        type="cross",
        qk_rms_norm=qk_rms_norm,
    )
    ours = SparseMultiHeadAttention(**kw).cuda().to(torch.float16)
    ref = RefSparseMultiHeadAttention(**kw).cuda().to(torch.float16)
    ref.load_state_dict(ours.state_dict())

    v_ours = _make_voxels(B=2, N_per=32, C=96).to("cuda").half()
    v_ref = RefSparseTensor(v_ours.feats.clone(), v_ours.coords.clone())
    cond = torch.randn(2, 24, 64, device="cuda", dtype=torch.float16)

    o = ours(v_ours, context=cond).feats
    r = ref(v_ref, context=cond).feats
    torch.testing.assert_close(o, r, rtol=5e-3, atol=5e-3)
