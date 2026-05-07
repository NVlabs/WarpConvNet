# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 tests: SparseStructureFlowModel + FlowEulerSampler.

DinoV3 wrapper is exercised lightly (skipped unless `transformers` + the
hub-cached weight are reachable on the host).
"""
import os
import sys

import pytest
import torch

from warpconvnet.models.trellis2.samplers import (
    FlowEulerCfgSampler,
    FlowEulerGuidanceIntervalSampler,
    FlowEulerSampler,
)
from warpconvnet.models.trellis2.sparse_structure_flow import SparseStructureFlowModel

_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
_HAS_REF = False
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH):
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    if _TRELLIS2_PATH not in sys.path:
        sys.path.insert(0, _TRELLIS2_PATH)
    try:
        from trellis2.models.sparse_structure_flow import (
            SparseStructureFlowModel as RefSparseStructureFlowModel,
        )
        from trellis2.pipelines.samplers import FlowEulerSampler as RefFlowEulerSampler

        _HAS_REF = True
    except Exception:  # noqa: BLE001
        _HAS_REF = False


# -----------------------------------------------------------------------------
# Tiny config — keeps wall-clock manageable while exercising every code path.
# -----------------------------------------------------------------------------
_KW = dict(
    resolution=4,
    in_channels=4,
    model_channels=96,
    cond_channels=64,
    out_channels=4,
    num_blocks=2,
    num_heads=8,
    mlp_ratio=4.0,
    pe_mode="rope",
    share_mod=True,
    qk_rms_norm=True,
    qk_rms_norm_cross=True,
    dtype="float32",
)


@pytest.fixture
def flow_model():
    torch.manual_seed(0)
    return SparseStructureFlowModel(**_KW)


# -----------------------------------------------------------------------------
# SparseStructureFlowModel
# -----------------------------------------------------------------------------
def test_ss_flow_forward_shape(flow_model):
    B = 2
    x = torch.randn(B, _KW["in_channels"], *[_KW["resolution"]] * 3)
    t = torch.tensor([0.5] * B)
    cond = torch.randn(B, 16, _KW["cond_channels"])
    out = flow_model(x, t, cond)
    assert out.shape == x.shape


def test_ss_flow_state_dict_roundtrip(flow_model):
    other = SparseStructureFlowModel(**_KW)
    other.load_state_dict(flow_model.state_dict())
    x = torch.randn(1, _KW["in_channels"], *[_KW["resolution"]] * 3)
    t = torch.tensor([0.3])
    cond = torch.randn(1, 8, _KW["cond_channels"])
    torch.testing.assert_close(flow_model(x, t, cond), other(x, t, cond))


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_ss_flow_matches_reference():
    # CPU + fp32 ⇒ both sides must use torch SDPA.
    from trellis2.modules.attention.config import set_backend as _set_backend

    _set_backend("sdpa")
    os.environ["ATTN_BACKEND"] = "sdpa"
    torch.manual_seed(0)
    ours = SparseStructureFlowModel(**_KW)
    ref = RefSparseStructureFlowModel(**_KW)
    # Our rope_phases is registered persistent=False (complex64 / safetensors
    # incompatible); the upstream still keeps it persistent. Load non-strictly
    # so the freshly-constructed buffer on each side is used as-is.
    ref.load_state_dict(ours.state_dict(), strict=False)
    B = 2
    x = torch.randn(B, _KW["in_channels"], *[_KW["resolution"]] * 3)
    t = torch.tensor([0.42] * B)
    cond = torch.randn(B, 8, _KW["cond_channels"])
    torch.testing.assert_close(ours(x, t, cond), ref(x, t, cond), rtol=1e-4, atol=1e-4)


# -----------------------------------------------------------------------------
# FlowEulerSampler
# -----------------------------------------------------------------------------
class _IdentityModel(torch.nn.Module):
    """Returns zero velocity ⇒ Euler step keeps x_t unchanged."""

    def forward(self, x_t, t, cond=None, **kwargs):
        return torch.zeros_like(x_t)


def test_flow_euler_zero_velocity_preserves_noise():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    noise = torch.randn(1, 4, 4, 4, 4)
    out = sampler.sample(_IdentityModel(), noise, cond=None, steps=4, verbose=False)
    torch.testing.assert_close(out["samples"], noise)


def test_flow_euler_step_count_matches_steps_kw():
    sampler = FlowEulerSampler(sigma_min=1e-5)
    noise = torch.randn(1, 4, 4, 4, 4)
    out = sampler.sample(_IdentityModel(), noise, cond=None, steps=5, verbose=False)
    assert len(out["pred_x_t"]) == 5
    assert len(out["pred_x_0"]) == 5


def test_flow_euler_cfg_strength_one_equals_uncfg():
    """guidance_strength=1 ⇒ purely conditional (== plain euler with cond)."""
    cfg = FlowEulerCfgSampler(sigma_min=1e-5)
    plain = FlowEulerSampler(sigma_min=1e-5)
    noise = torch.randn(1, 4, 4, 4, 4)
    cond = torch.zeros(1, 4)
    out_cfg = cfg.sample(
        _IdentityModel(),
        noise,
        cond=cond,
        neg_cond=cond,
        steps=4,
        guidance_strength=1.0,
        verbose=False,
    )
    out_plain = plain.sample(_IdentityModel(), noise, cond=cond, steps=4, verbose=False)
    torch.testing.assert_close(out_cfg["samples"], out_plain["samples"])


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_flow_euler_matches_reference():
    """Compare full sampling trajectory against upstream sampler."""

    class _LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(7)
            self.w = torch.nn.Parameter(torch.randn(4, 4) * 0.01)

        def forward(self, x_t, t, cond=None, **kwargs):
            return torch.einsum("bcwhd,co->bowhd", x_t, self.w)

    model = _LinearModel().eval()
    ours = FlowEulerSampler(sigma_min=1e-5)
    ref = RefFlowEulerSampler(sigma_min=1e-5)
    torch.manual_seed(0)
    noise = torch.randn(1, 4, 4, 4, 4)
    o = ours.sample(model, noise, cond=None, steps=10, verbose=False)
    r = ref.sample(model, noise, cond=None, steps=10, verbose=False)
    torch.testing.assert_close(o["samples"], r["samples"], rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Param-count sanity for full SS flow config (1.3B).
# -----------------------------------------------------------------------------
def test_ss_flow_param_count_for_4b_config():
    """Real config from configs/gen/ss_flow_img_dit_1_3B_64_bf16.json.

    Rough budget: 30 blocks × ~36M ≈ 1.08B; t_embedder + io ≈ 50M ⇒ ~1.1B–1.3B.
    Build only the model definition (params live on CPU in fp32, ~5GB); skip if
    there's not enough host RAM.
    """
    try:
        m = SparseStructureFlowModel(
            resolution=16,
            in_channels=8,
            model_channels=1536,
            cond_channels=1024,
            out_channels=8,
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
