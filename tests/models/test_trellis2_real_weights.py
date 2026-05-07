# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 — stage-1 end-to-end check on **real** TRELLIS.2 weights.

Loads the published safetensors checkpoints from `microsoft/TRELLIS.2-4B` and
`microsoft/TRELLIS-image-large` into both our re-implementation and the
upstream reference, and asserts bit-exact forward parity on identical inputs.

Skipped automatically without CUDA, the upstream package, or cached weights.
"""
import json
import os
import sys

import pytest
import torch

from warpconvnet.models.trellis2.sparse_structure_flow import SparseStructureFlowModel
from warpconvnet.models.trellis2.sparse_structure_vae import SparseStructureDecoder

_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
_HAS_REF = False
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH):
    # Both ours and upstream prefer flash_attn for fp16/bf16 inputs after the
    # `nn/dit:_sdpa_4d` fix; aligning the env var keeps state_dict-loaded
    # bit-parity intact (sdpa vs flash_attn produce ~10% rtol drift).
    os.environ["ATTN_BACKEND"] = "flash_attn"
    if _TRELLIS2_PATH not in sys.path:
        sys.path.insert(0, _TRELLIS2_PATH)
    try:
        from trellis2.models.sparse_structure_flow import (
            SparseStructureFlowModel as RefSparseStructureFlowModel,
        )
        from trellis2.models.sparse_structure_vae import (
            SparseStructureDecoder as RefSparseStructureDecoder,
        )

        _HAS_REF = True
    except Exception:  # noqa: BLE001
        _HAS_REF = False


def _hf_path(repo: str, file: str) -> str:
    """Resolve a cached HF Hub file path; raises if not present."""
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo_id=repo, filename=file)
    if p is None:
        raise FileNotFoundError(f"{repo}/{file} not in HF cache")
    return p


def _load_safetensors(path: str) -> dict:
    from safetensors.torch import load_file

    return load_file(path)


_HAS_CUDA = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
_skip_no_ref = pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")


def _can_resolve(repo: str, file: str) -> bool:
    try:
        _hf_path(repo, file)
        return True
    except Exception:  # noqa: BLE001
        return False


# -----------------------------------------------------------------------------
# SS decoder (small, fp16, ~50M params): fastest sanity check.
# -----------------------------------------------------------------------------
@_skip_no_cuda
@_skip_no_ref
@pytest.mark.skipif(
    not _can_resolve("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
    reason="ss_dec weights not in HF cache",
)
def test_ss_decoder_real_weights_match_reference():
    json_p = _hf_path("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.json")
    sft_p = _hf_path("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors")
    with open(json_p) as f:
        args = json.load(f)["args"]

    state = _load_safetensors(sft_p)

    ours = SparseStructureDecoder(**args).cuda().eval()
    ours.load_state_dict(state)
    ref = RefSparseStructureDecoder(**args).cuda().eval()
    ref.load_state_dict(state)

    torch.manual_seed(0)
    z = torch.randn(1, args["latent_channels"], 16, 16, 16, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        o = ours(z)
        r = ref(z)
    torch.testing.assert_close(o, r, rtol=1e-3, atol=1e-3)
    assert o.shape == (1, args["out_channels"], 64, 64, 64)


# -----------------------------------------------------------------------------
# SS flow model (1.3B, bf16). Heavyweight — runs in fp32 outer, bf16 inner.
# -----------------------------------------------------------------------------
@_skip_no_cuda
@_skip_no_ref
@pytest.mark.skipif(
    not _can_resolve("microsoft/TRELLIS.2-4B", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
    reason="ss_flow weights not in HF cache",
)
def test_ss_flow_real_weights_match_reference():
    # Force both impls onto the flash_attn backend at runtime — sibling CPU
    # tests may have left upstream config.BACKEND on 'sdpa', which would
    # diverge from our runtime choice (10% rtol drift).
    from trellis2.modules.attention.config import set_backend as _set_backend

    _set_backend("flash_attn")
    os.environ["ATTN_BACKEND"] = "flash_attn"
    json_p = _hf_path("microsoft/TRELLIS.2-4B", "ckpts/ss_flow_img_dit_1_3B_64_bf16.json")
    sft_p = _hf_path("microsoft/TRELLIS.2-4B", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors")
    with open(json_p) as f:
        args = json.load(f)["args"]

    state = _load_safetensors(sft_p)

    ours = SparseStructureFlowModel(**args).cuda().eval()
    ours.load_state_dict(state, strict=False)
    ref = RefSparseStructureFlowModel(**args).cuda().eval()
    # rope_phases is a complex64 buffer recomputed at construction; safetensors
    # cannot store complex tensors, so it is not in the checkpoint. Load
    # non-strictly on both sides.
    ref.load_state_dict(state, strict=False)

    torch.manual_seed(0)
    B = 1
    R = args["resolution"]
    x = torch.randn(B, args["in_channels"], R, R, R, device="cuda")
    t = torch.tensor([500.0], device="cuda")
    cond = torch.randn(B, 32, args["cond_channels"], device="cuda")
    with torch.no_grad():
        o = ours(x, t, cond)
        r = ref(x, t, cond)
    # bf16 internals → looser tolerance.
    torch.testing.assert_close(o, r, rtol=5e-3, atol=5e-3)
    assert o.shape == x.shape
