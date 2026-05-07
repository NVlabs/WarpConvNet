# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 8 tests: SLatFlowModel.

Construction, state_dict shape, real-weight bit-parity vs upstream.
"""
import importlib
import json
import os
import sys

import pytest
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.trellis2.slat_flow import SLatFlowModel
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
        from trellis2.models.structured_latent_flow import (
            SLatFlowModel as RefSLatFlowModel,
        )
        from trellis2.modules.sparse.basic import SparseTensor as RefSparseTensor

        _HAS_REF = True
    except Exception:  # noqa: BLE001
        _HAS_REF = False


# Tiny config for fast shape / parity tests.
_TINY_KW = dict(
    resolution=4,
    in_channels=8,
    model_channels=96,
    cond_channels=64,
    out_channels=8,
    num_blocks=2,
    num_heads=8,
    mlp_ratio=4.0,
    pe_mode="rope",
    share_mod=True,
    qk_rms_norm=True,
    qk_rms_norm_cross=True,
    dtype="float32",
)


def _make_voxels(B: int = 2, N_per: int = 32, C: int = 8, R: int = 4, seed: int = 0) -> Voxels:
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


def _half_linears(module: torch.nn.Module) -> torch.nn.Module:
    for sub in module.modules():
        if isinstance(sub, torch.nn.Linear):
            sub.weight.data = sub.weight.data.half()
            if sub.bias is not None:
                sub.bias.data = sub.bias.data.half()
    return module


# -----------------------------------------------------------------------------
# Construction / state_dict
# -----------------------------------------------------------------------------
def test_slat_flow_state_dict_keys():
    m = SLatFlowModel(**_TINY_KW)
    sd = m.state_dict()
    assert "input_layer.weight" in sd
    assert "out_layer.weight" in sd
    assert "t_embedder.mlp.0.weight" in sd
    assert "blocks.0.self_attn.to_qkv.weight" in sd
    assert "blocks.0.cross_attn.to_kv.weight" in sd
    assert "adaLN_modulation.1.weight" in sd  # share_mod=True
    assert "blocks.0.modulation" in sd  # block-local modulation per share_mod


def test_slat_flow_param_count_for_4b_config():
    """Real config from configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json.

    1.3B target — 30 blocks × 1536 ch × 12 heads × mlp_ratio 5.3334.
    """
    try:
        m = SLatFlowModel(
            resolution=32,
            in_channels=32,
            model_channels=1536,
            cond_channels=1024,
            out_channels=32,
            num_blocks=30,
            num_heads=12,
            mlp_ratio=5.3334,
            pe_mode="rope",
            share_mod=True,
            qk_rms_norm=True,
            qk_rms_norm_cross=True,
            dtype="bfloat16",
        )
    except (RuntimeError, MemoryError) as e:
        pytest.skip(f"insufficient memory: {e}")
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000_000 < n < 1_500_000_000, n


# -----------------------------------------------------------------------------
# CUDA forward
# -----------------------------------------------------------------------------
@_skip_no_cuda
@_skip_no_flash
def test_slat_flow_forward_shape():
    """Run with model.dtype=fp16 so blocks are halved but I/O stays fp32.

    This matches upstream's `convert_to(fp16)` which only converts the
    `blocks` ModuleList; t_embedder / input_layer / out_layer remain fp32.
    """
    torch.manual_seed(0)
    m = SLatFlowModel(**{**_TINY_KW, "dtype": "float16"}).cuda()
    v = _make_voxels(B=2, N_per=16, C=_TINY_KW["in_channels"], R=4).to("cuda")
    t = torch.tensor([500.0, 500.0], device="cuda")
    cond = torch.randn(2, 16, _TINY_KW["cond_channels"], device="cuda")
    out = m(v, t, cond)
    assert isinstance(out, Voxels)
    assert out.feats.shape == v.feats.shape


# -----------------------------------------------------------------------------
# Reference parity (random-init)
# -----------------------------------------------------------------------------
@_skip_no_cuda
@_skip_no_flash
@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_slat_flow_matches_reference():
    torch.manual_seed(0)
    kw = {**_TINY_KW, "dtype": "float16"}
    ours = SLatFlowModel(**kw).cuda().eval()
    ref = RefSLatFlowModel(**kw).cuda().eval()
    ref.load_state_dict(ours.state_dict())

    v_ours = _make_voxels(B=2, N_per=16, C=_TINY_KW["in_channels"], R=4).to("cuda")
    v_ref = RefSparseTensor(v_ours.feats.clone(), v_ours.coords.clone())
    t = torch.tensor([321.0, 321.0], device="cuda")
    cond = torch.randn(2, 16, _TINY_KW["cond_channels"], device="cuda")

    o = ours(v_ours, t, cond).feats
    r = ref(v_ref, t, cond).feats
    torch.testing.assert_close(o, r, rtol=1e-2, atol=1e-2)


# -----------------------------------------------------------------------------
# Real-weight parity (microsoft/TRELLIS.2-4B 1.3B bf16 SLAT flow)
# -----------------------------------------------------------------------------
def _hf_path(repo: str, file: str) -> str:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo_id=repo, filename=file)
    if p is None:
        raise FileNotFoundError(f"{repo}/{file} not in HF cache")
    return p


def _has_slat_weights() -> bool:
    try:
        _hf_path(
            "microsoft/TRELLIS.2-4B",
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        )
        return True
    except Exception:  # noqa: BLE001
        return False


@_skip_no_cuda
@_skip_no_flash
@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
@pytest.mark.skipif(not _has_slat_weights(), reason="slat_flow weights not in HF cache")
def test_slat_flow_real_weights_match_reference():
    """Load the published 1.3B bf16 SLAT flow into both impls; compare."""
    from safetensors.torch import load_file

    cfg_p = _hf_path("microsoft/TRELLIS.2-4B", "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json")
    sft_p = _hf_path(
        "microsoft/TRELLIS.2-4B",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    )
    args = json.load(open(cfg_p))["args"]
    state = load_file(sft_p)

    ours = SLatFlowModel(**args).cuda().eval()
    ours.load_state_dict(state, strict=False)
    ref = RefSLatFlowModel(**args).cuda().eval()
    ref.load_state_dict(state, strict=False)

    torch.manual_seed(0)
    # Tiny voxel set so the test stays fast — ~64 voxels, real channel count.
    v = _make_voxels(B=1, N_per=64, C=args["in_channels"], R=8).to("cuda")
    v_ref = RefSparseTensor(v.feats.clone(), v.coords.clone())
    t = torch.tensor([500.0], device="cuda")
    cond = torch.randn(1, 32, args["cond_channels"], device="cuda")

    with torch.no_grad():
        o = ours(v, t, cond).feats
        r = ref(v_ref, t, cond).feats
    # bf16 internals + 30-block accumulation ⇒ wider tolerance.
    torch.testing.assert_close(o, r, rtol=2e-2, atol=2e-2)
    assert o.shape == v.feats.shape
