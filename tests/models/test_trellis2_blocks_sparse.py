# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 7 tests: ModulatedSparseTransformerBlock + ModulatedSparseTransformerCrossBlock.

Forward parity tests rely on flash-attn (CUDA-only). State_dict + shape
sanity checks run on CPU.
"""
import importlib
import os
import sys

import pytest
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.trellis2.blocks_sparse import (
    ModulatedSparseTransformerBlock,
    ModulatedSparseTransformerCrossBlock,
    SparseFeedForwardNet,
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
        from trellis2.modules.sparse.basic import SparseTensor as RefSparseTensor
        from trellis2.modules.sparse.transformer.modulated import (
            ModulatedSparseTransformerBlock as RefModulatedSparseTransformerBlock,
            ModulatedSparseTransformerCrossBlock as RefModulatedSparseTransformerCrossBlock,
        )

        _HAS_REF = True
    except Exception:  # noqa: BLE001
        _HAS_REF = False


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
# Construction / state_dict
# -----------------------------------------------------------------------------
def test_ffn_state_dict_keys_match_upstream():
    m = SparseFeedForwardNet(channels=96, mlp_ratio=4.0)
    keys = set(m.state_dict().keys())
    assert {"mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"} <= keys


def test_modulated_self_block_state_dict():
    blk = ModulatedSparseTransformerBlock(
        channels=96, num_heads=8, share_mod=True, qk_rms_norm=True, use_rope=True
    )
    sd = blk.state_dict()
    assert "modulation" in sd
    assert "attn.to_qkv.weight" in sd
    assert "mlp.mlp.0.weight" in sd


def test_modulated_cross_block_state_dict():
    blk = ModulatedSparseTransformerCrossBlock(
        channels=96,
        ctx_channels=64,
        num_heads=8,
        share_mod=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        use_rope=True,
    )
    sd = blk.state_dict()
    assert "modulation" in sd
    assert "self_attn.to_qkv.weight" in sd
    assert "cross_attn.to_kv.weight" in sd
    assert "norm2.weight" in sd  # affine on norm2 only


# -----------------------------------------------------------------------------
# CUDA forward
# -----------------------------------------------------------------------------
def _half_linears(module: torch.nn.Module) -> torch.nn.Module:
    """Cast `nn.Linear` params to fp16 in-place; leaves LayerNorm/RMSNorm fp32.

    Mirrors `trellis2.modules.utils.convert_module_to_f16` (mix-precision
    inference: feature path runs at fp16 while normalisation stays fp32).
    """
    for sub in module.modules():
        if isinstance(sub, torch.nn.Linear):
            sub.weight.data = sub.weight.data.half()
            if sub.bias is not None:
                sub.bias.data = sub.bias.data.half()
    return module


@_skip_no_cuda
@_skip_no_flash
@pytest.mark.parametrize("share_mod", [False, True])
def test_modulated_self_block_forward(share_mod):
    torch.manual_seed(0)
    blk = ModulatedSparseTransformerBlock(
        channels=96, num_heads=8, share_mod=share_mod, qk_rms_norm=True
    ).cuda()
    _half_linears(blk)
    v = _make_voxels(B=2, N_per=32, C=96).to("cuda").half()
    mod = torch.randn(2, (6 if share_mod else 1) * 96, device="cuda", dtype=torch.float16)
    out = blk(v, mod)
    assert out.feats.shape == v.feats.shape


@_skip_no_cuda
@_skip_no_flash
@pytest.mark.parametrize("share_mod", [False, True])
def test_modulated_cross_block_forward(share_mod):
    torch.manual_seed(0)
    blk = ModulatedSparseTransformerCrossBlock(
        channels=96,
        ctx_channels=64,
        num_heads=8,
        share_mod=share_mod,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
    ).cuda()
    _half_linears(blk)
    v = _make_voxels(B=2, N_per=32, C=96).to("cuda").half()
    mod = torch.randn(2, (6 if share_mod else 1) * 96, device="cuda", dtype=torch.float16)
    cond = torch.randn(2, 24, 64, device="cuda", dtype=torch.float16)
    out = blk(v, mod, cond)
    assert out.feats.shape == v.feats.shape


# -----------------------------------------------------------------------------
# Reference parity (upstream trellis2)
# -----------------------------------------------------------------------------
@_skip_no_cuda
@_skip_no_flash
@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
@pytest.mark.parametrize("share_mod", [False, True])
def test_modulated_self_block_matches_reference(share_mod):
    torch.manual_seed(0)
    kw = dict(
        channels=96,
        num_heads=8,
        share_mod=share_mod,
        qk_rms_norm=True,
        use_rope=True,
    )
    ours = ModulatedSparseTransformerBlock(**kw).cuda()
    _half_linears(ours)
    ref = RefModulatedSparseTransformerBlock(**kw).cuda()
    _half_linears(ref)
    ref.load_state_dict(ours.state_dict())

    v_ours = _make_voxels(B=2, N_per=32, C=96).to("cuda").half()
    v_ref = RefSparseTensor(v_ours.feats.clone(), v_ours.coords.clone())
    mod = torch.randn(2, (6 if share_mod else 1) * 96, device="cuda", dtype=torch.float16)

    o = ours(v_ours, mod).feats
    r = ref(v_ref, mod).feats
    torch.testing.assert_close(o, r, rtol=5e-3, atol=5e-3)


@_skip_no_cuda
@_skip_no_flash
@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
@pytest.mark.parametrize("share_mod", [False, True])
def test_modulated_cross_block_matches_reference(share_mod):
    torch.manual_seed(0)
    kw = dict(
        channels=96,
        ctx_channels=64,
        num_heads=8,
        share_mod=share_mod,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        use_rope=True,
    )
    ours = ModulatedSparseTransformerCrossBlock(**kw).cuda()
    _half_linears(ours)
    ref = RefModulatedSparseTransformerCrossBlock(**kw).cuda()
    _half_linears(ref)
    ref.load_state_dict(ours.state_dict())

    v_ours = _make_voxels(B=2, N_per=32, C=96).to("cuda").half()
    v_ref = RefSparseTensor(v_ours.feats.clone(), v_ours.coords.clone())
    mod = torch.randn(2, (6 if share_mod else 1) * 96, device="cuda", dtype=torch.float16)
    cond = torch.randn(2, 24, 64, device="cuda", dtype=torch.float16)

    o = ours(v_ours, mod, cond).feats
    r = ref(v_ref, mod, cond).feats
    torch.testing.assert_close(o, r, rtol=5e-3, atol=5e-3)
