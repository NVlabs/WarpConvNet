# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression pin: F16-accumulator pcoff tiles at a narrow encoder shape.

At (C_in=32, C_out=32, K=3x3x3, N~250k), pcoff F16-accumulator tiles
54/55/56 produce isolated output cells with `max_rel` up to several hundred
against fp64 reference. `p99` stays at the noise floor (~1e-3), so prior
per_algo_grad_sweep / per-tile tests missed it. This test asserts the failure
mode is NOT silently re-introduced by:

  1. Importing `_AB_MASK_GEMM_PCOFF_F16ACC` and confirming none of the tiles
     enter the auto pool by default (env-default = 0 ceiling).
  2. Running each tile directly at the failure shape and verifying max_rel
     exceeds the pass tolerance the production code would use — confirming
     the tile is genuinely unsafe at this shape, so the gate is load-bearing.

If WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL is ever flipped back to >0 default,
or if a future tile lands in `_AB_MASK_GEMM_PCOFF_F16ACC` without gating,
this test fails at import time / the second arm catches numerical drift.
"""

import os

import pytest
import torch

from warpconvnet.constants import WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
    _AB_MASK_GEMM_PCOFF_F16ACC,
    _get_adaptive_AB_params,
)
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import _execute_forward
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_forward_logic,
)
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Shape that triggers the bug. C=32 satisfies the prior ceiling=32 predicate;
# K=27 satisfies kv<=32; N=250k is the saturation threshold where isolated
# cells start blowing up under fp16 accumulation.
_C = 32
_K = (3, 3, 3)
_N_TARGET = 250_000
_FAIL_TILES = (54, 55, 56)  # tile 57 happened to pass at this exact shape


def _build_problem():
    device = "cuda"
    extent = int(round((_N_TARGET * 8) ** (1 / 3))) + 4
    g = torch.Generator(device=device).manual_seed(0xDEAD)
    coords = torch.randint(
        0,
        extent,
        (int(_N_TARGET * 1.15), 3),
        device=device,
        dtype=torch.int32,
        generator=g,
    )
    feats = torch.zeros(coords.shape[0], 1, device=device)
    offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int32, device=device)
    v = Voxels(coords, feats, offsets=offsets, device=device).unique()
    out, _, kmap = generate_output_coords_and_kernel_map(
        v,
        kernel_size=_K,
        kernel_dilation=(1, 1, 1),
        stride=(1, 1, 1),
        generative=False,
        transposed=False,
    )
    N_in = v.feature_tensor.shape[0]
    N_out = out.shape[0]
    cin = torch.arange(_C, device=device, dtype=torch.float64) / _C
    row = (torch.arange(N_in, device=device, dtype=torch.float64) % 16) / 16.0
    in64 = (row.unsqueeze(1) + cin.unsqueeze(0)) / 2.0
    base = cin.view(1, _C, 1) + cin.view(1, 1, _C)
    rc = torch.arange(27, device=device, dtype=torch.float64) / 27.0
    w64 = (base * (1.0 + rc.view(27, 1, 1))).contiguous() / 4.0
    return in64.half().contiguous(), w64.half().contiguous(), kmap, N_out, in64, w64


def test_pcoff_f16acc_gated_off_by_default():
    """Default pool must NOT contain F16-accum pcoff tiles for narrow-ch shapes."""
    assert WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL == 0, (
        "Default ceiling must be 0 — the prior 32 silently admitted F16-accum pcoff "
        "tiles 54/55/56/57 for C<=32 layers, which corrupted isolated output cells "
        "at training-realistic N."
    )
    params = _get_adaptive_AB_params(
        in_channels=_C,
        out_channels=_C,
        kernel_volume=27,
        num_in_coords=_N_TARGET,
        use_fp16_accum=False,
    )
    pool_tiles = {p[1]["tile_id"] for p in params if p[0] == "mask_gemm"}
    f16_pcoff_tiles = {p[1]["tile_id"] for p in _AB_MASK_GEMM_PCOFF_F16ACC}
    overlap = pool_tiles & f16_pcoff_tiles
    assert not overlap, (
        f"F16-accum pcoff tiles {overlap} entered default pool at C={_C}. "
        f"Likely WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL default was flipped back >0."
    )


def test_pcoff_f16acc_opt_in_admits_tiles():
    """Sanity: opt-in via use_fp16_accum still admits F16-accum pcoff (documented contract)."""
    params = _get_adaptive_AB_params(
        in_channels=_C,
        out_channels=_C,
        kernel_volume=27,
        num_in_coords=_N_TARGET,
        use_fp16_accum=True,
    )
    pool_tiles = {p[1]["tile_id"] for p in params if p[0] == "mask_gemm"}
    f16_pcoff_tiles = {p[1]["tile_id"] for p in _AB_MASK_GEMM_PCOFF_F16ACC}
    assert (
        pool_tiles & f16_pcoff_tiles
    ), "F16-accum pcoff tiles should be admitted when use_fp16_accum=True (opt-in path)."


@pytest.mark.parametrize("tile_id", _FAIL_TILES, ids=lambda t: f"tile{t}")
def test_pcoff_f16acc_genuinely_unsafe_at_failure_shape(tile_id):
    """Confirm the gated tiles ARE numerically broken at this shape — gate is load-bearing."""
    in16, w16, kmap, N_out, in64, w64 = _build_problem()
    ref = _explicit_gemm_forward_logic(in64, w64, kmap, N_out, torch.float64)
    out = _execute_forward(
        algo="mask_gemm",
        params={"tile_id": tile_id},
        in_features=in16,
        weight=w16,
        kernel_map=kmap,
        num_out_coords=N_out,
        compute_dtype=torch.float16,
        fwd_block_size=None,
    )
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-12)
    rel = rel[torch.isfinite(rel)]
    max_rel = rel.max().item()
    # If this fails (max_rel < 1.0), either the kernel was fixed upstream or
    # the failure shape changed — in either case investigate before relaxing
    # the gate. Lower bound of 1.0 is deliberately conservative; observed
    # values were 5 / 12 / 525 for tiles 54 / 55 / 56 respectively.
    assert max_rel > 1.0, (
        f"Tile {tile_id} unexpectedly safe at C={_C} K=3x3x3 N~{_N_TARGET} "
        f"(max_rel={max_rel:.2e}). If the kernel was fixed, update the gate "
        f"and remove this regression pin."
    )
