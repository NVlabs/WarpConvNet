# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression pins: F16-accumulator pcoff tiles at a narrow encoder shape.

History, two eras:

- Era 1 (2026-05): at (C_in=32, C_out=32, K=3x3x3, N~250k) tiles 54/55/56
  produced isolated output cells with max_rel up to several hundred (observed
  5 / 12 / 525) against an fp64 reference. p99 stayed at the noise floor, so
  sweep-style tests missed it. This file originally pinned the tiles as
  GENUINELY BROKEN to prove the small-channel gate was load-bearing.
- Era 2 (2026-07-25): root-caused as the single-k-tile (k_tiles==1, C_in<=32)
  cp.async under-synchronization in the shared warpgemm mainloop emission and
  fixed at the family root (racecheck 0 hazards, cross-arch). The corruption
  arm below is therefore INVERTED: it now pins the FIX at the exact historical
  failure shape, so the race cannot silently return where it hurt most
  (post-fix max_rel observed at the fp16 noise floor, 2.0e-03..3.3e-03, on
  both sm_103 and sm_120).

The pool gate (WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL default 0) is RETAINED
for now: the corruption rationale is gone, but F16-accumulator PRECISION at
training scale is a separate standing concern, and relaxing the default
re-admits F16-accum candidates to auto pools — that requires an N>=3
ScanNet-class training validation (see variance discipline), not just this
kernel-level pass. Arms 1-2 continue to pin the gate's default and opt-in
contract.
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
def test_pcoff_f16acc_fixed_at_historical_failure_shape(tile_id):
    """Pin the 2026-07-25 race fix at the exact shape where the tiles used to corrupt.

    Pre-fix these tiles hit max_rel 5 / 12 / 525 (tiles 54/55/56) here; the
    single-k-tile mainloop race fix brings all three to the fp16 noise floor.
    A regression of the k_tiles==1 synchronization discipline re-fires this
    test at its most sensitive known shape (complementing the racecheck CI
    lane, which catches the hazard even when it does not numerically land).
    """
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
    # Observed post-fix: 2.0e-03..3.3e-03 on sm_103 and sm_120. 0.05 leaves
    # generous fp16 headroom while sitting orders of magnitude below the
    # pre-fix corruption (>=5).
    assert max_rel < 0.05, (
        f"Tile {tile_id} regressed at the historical failure shape C={_C} "
        f"K=3x3x3 N~{_N_TARGET}: max_rel={max_rel:.2e} (post-fix noise floor "
        f"is ~3e-3; pre-fix corruption was >=5). Suspect the k_tiles==1 "
        f"mainloop synchronization was weakened — run the racecheck lane."
    )
