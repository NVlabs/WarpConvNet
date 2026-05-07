# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 9b tests: SparseResBlockC2S3d + SparseUnetVaeDecoder + FlexiDualGridVaeDecoder.

CUDA forward tests verify shape; real-weight load test confirms state_dict
parity with the published `microsoft/TRELLIS.2-4B/ckpts/shape_dec_*.safetensors`.
Mesh extraction itself is deferred to phase 10 (o-voxel CUDA build).
"""
import json
import os
import sys

import pytest
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.trellis2.shape_vae import (
    FlexiDualGridVaeDecoder,
    SparseResBlockC2S3d,
    SparseUnetVaeDecoder,
    convert_trellis2_shape_vae_state_dict,
)
from warpconvnet.models.trellis2.sparse_ops import from_feats_coords

_HAS_CUDA = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")


def _make_voxels(B: int = 1, N_per: int = 32, C: int = 32, R: int = 4, seed: int = 0) -> Voxels:
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
# State_dict / construction
# -----------------------------------------------------------------------------
def test_c2s_block_state_dict_keys():
    blk = SparseResBlockC2S3d(channels=64, out_channels=32, pred_subdiv=True)
    keys = set(blk.state_dict().keys())
    assert "norm1.weight" in keys and "norm1.bias" in keys
    assert "conv1.weight" in keys and "conv1.bias" in keys
    assert "conv2.weight" in keys and "conv2.bias" in keys
    assert "to_subdiv.weight" in keys and "to_subdiv.bias" in keys


def test_c2s_block_skip_repeat_factor():
    blk = SparseResBlockC2S3d(channels=64, out_channels=32, pred_subdiv=True)
    assert blk._repeat == 32 // (64 // 8)  # 32 // 8 = 4


def test_shape_vae_state_dict_conversion_to_native_sparse_conv_layout():
    cfg = dict(
        resolution=8,
        model_channels=[64],
        latent_channels=8,
        num_blocks=[1],
        block_type=["SparseConvNeXtBlock3d"],
        up_block_type=[],
        block_args=[{}],
        use_fp16=False,
    )
    m = FlexiDualGridVaeDecoder(**cfg)
    native = m.state_dict()["blocks.0.0.conv.weight"]
    upstream = native.reshape(3, 3, 3, 64, 64).permute(4, 0, 1, 2, 3).contiguous()
    converted = convert_trellis2_shape_vae_state_dict(
        {"blocks.0.0.conv.weight": upstream},
        m,
    )
    torch.testing.assert_close(converted["blocks.0.0.conv.weight"], native)


def test_decoder_constructs_real_shape_dec_config():
    """Real config from microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.json."""
    cfg = dict(
        resolution=256,
        model_channels=[1024, 512, 256, 128, 64],
        latent_channels=32,
        num_blocks=[4, 16, 8, 4, 0],
        block_type=["SparseConvNeXtBlock3d"] * 5,
        up_block_type=["SparseResBlockC2S3d"] * 4,
        block_args=[{}] * 5,
        use_fp16=True,
    )
    try:
        m = FlexiDualGridVaeDecoder(**cfg)
    except (RuntimeError, MemoryError) as e:
        pytest.skip(f"insufficient memory: {e}")
    n = sum(p.numel() for p in m.parameters())
    # Real shape_dec_next_dc_f16c32_fp16: 474.2M params.
    assert 400_000_000 < n < 600_000_000, n


# -----------------------------------------------------------------------------
# CUDA forward (tiny config)
# -----------------------------------------------------------------------------
@_skip_no_cuda
def test_c2s_block_forward_shape():
    torch.manual_seed(0)
    blk = SparseResBlockC2S3d(channels=64, out_channels=32, pred_subdiv=True).cuda()
    v = _make_voxels(B=1, N_per=8, C=64, R=4).to("cuda")
    h, subdiv = blk(v)
    # C2S spreads 1 voxel into up to 8 children based on subdiv mask.
    assert h.feats.shape[1] == 32
    assert subdiv.feats.shape[1] == 8
    assert h.coords.shape[0] == subdiv.feats.gt(0).sum().item()


@_skip_no_cuda
def test_decoder_forward_tiny():
    torch.manual_seed(0)
    cfg = dict(
        resolution=8,
        model_channels=[64, 32, 16],
        latent_channels=8,
        num_blocks=[1, 1, 1],
        block_type=["SparseConvNeXtBlock3d"] * 3,
        up_block_type=["SparseResBlockC2S3d"] * 2,
        block_args=[{}] * 3,
        use_fp16=False,
    )
    dec = FlexiDualGridVaeDecoder(**cfg).cuda()
    v = _make_voxels(B=1, N_per=4, C=cfg["latent_channels"], R=2).to("cuda")
    vertices, intersected, quad_lerp = dec(v)
    assert vertices.feats.shape[1] == 3
    assert intersected.feats.shape[1] == 3
    assert quad_lerp.feats.shape[1] == 1
    assert intersected.feats.dtype == torch.bool


# -----------------------------------------------------------------------------
# Real-weight state_dict load (microsoft/TRELLIS.2-4B shape decoder)
# -----------------------------------------------------------------------------
def _hf_path(repo: str, file: str) -> str:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo_id=repo, filename=file)
    if p is None:
        raise FileNotFoundError(f"{repo}/{file} not in HF cache")
    return p


def _has_shape_dec_weights() -> bool:
    try:
        _hf_path(
            "microsoft/TRELLIS.2-4B",
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        )
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_shape_dec_weights(), reason="shape_dec weights not in HF cache")
def test_shape_decoder_real_weights_load():
    """The published shape-decoder safetensors must load into our module
    without unexpected keys after sparse-conv weight conversion."""
    from safetensors.torch import load_file

    cfg_p = _hf_path("microsoft/TRELLIS.2-4B", "ckpts/shape_dec_next_dc_f16c32_fp16.json")
    sft_p = _hf_path("microsoft/TRELLIS.2-4B", "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")
    args = json.load(open(cfg_p))["args"]
    state = load_file(sft_p)

    m = FlexiDualGridVaeDecoder(**args)
    missing, unexpected = m.load_trellis2_state_dict(state, strict=False)
    assert not unexpected, f"unexpected upstream keys: {unexpected[:5]}"
    # `missing` may contain registered buffers (e.g. running stats) that
    # safetensors strips; only flag missing *parameters*.
    param_names = {n for n, _ in m.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    assert not missing_params, f"missing param keys: {missing_params[:5]}"
