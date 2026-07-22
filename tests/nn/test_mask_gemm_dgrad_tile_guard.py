# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for ``_select_dgrad_tile``'s dgrad_wt tile-id whitelist.

``_select_dgrad_tile`` (mask_gemm.py) resolves the tile_id + routing for the
``use_fwd_for_dgrad=True`` (``mask_gemm_fwd_as_dgrad``) path. Two invariants
protect the binding's dgrad_wt arm from a device-assert that would poison the
CUDA context mid-autotune:

  1. ``params["tile_id"]`` must be a canonical dgrad_wt id in 900-911. A
     foreign id (e.g. a fwd pcoff id 54-63 leaking in via a stale benchmark
     cache) is rejected before it ever reaches the kernel.
  2. No dgrad_wt id can service ``mask_words > 1`` (K > 32): the binding's
     dgrad_wt arm has no MaskWords>1 instantiation, so ALL of 900-911 must
     raise at mask_words>1 -- not just the structurally MW1-only subset.

These are pure-Python decision tests; no CUDA launch, no GPU required.
"""

import pytest

from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
    _AB_MASK_GEMM_FWD_AS_DGRAD,
)
from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
    _DGRAD_WT_TILES,
    _select_dgrad_tile,
)

# Aligned per-group channels at vec_width=8 (fp16/bf16 16-byte vectorization).
_VEC_WIDTH = 8
_ALIGNED_C = 64
_UNALIGNED_C = 60


def test_foreign_tile_id_raises():
    """A tile_id outside 900-911 (e.g. a stale-cache fwd pcoff id) must raise,
    quoting the valid range, for aligned channels with no scalar fallback."""
    with pytest.raises(RuntimeError, match="900-911"):
        _select_dgrad_tile(
            use_fwd_for_dgrad=True,
            params={"tile_id": 58},
            mask_words=1,
            use_f32_out_tile=False,
            vec_width=_VEC_WIDTH,
            C_in_g=_ALIGNED_C,
            C_out_g=_ALIGNED_C,
            use_fp16_accum=False,
        )


@pytest.mark.parametrize("tile_id", sorted(_DGRAD_WT_TILES))
def test_dgrad_wt_tiles_reject_mask_words_gt1(tile_id):
    """Every canonical dgrad_wt id (900-911) must raise at mask_words>1: the
    binding's dgrad_wt arm has no MaskWords>1 instantiation, so this is NOT
    limited to the structurally MW1-only subset (903, 905-911)."""
    with pytest.raises(RuntimeError, match="MaskWords>1"):
        _select_dgrad_tile(
            use_fwd_for_dgrad=True,
            params={"tile_id": tile_id},
            mask_words=2,
            use_f32_out_tile=False,
            vec_width=_VEC_WIDTH,
            C_in_g=_ALIGNED_C,
            C_out_g=_ALIGNED_C,
            use_fp16_accum=False,
        )


@pytest.mark.parametrize("tile_id", sorted(_DGRAD_WT_TILES))
def test_dgrad_wt_tiles_pass_through_at_mask_words_1(tile_id):
    """At mask_words==1, canonical dgrad_wt ids are returned unchanged and do
    not route through the fwd-fallback (scalar) path."""
    resolved_tile, use_fwd_fallback = _select_dgrad_tile(
        use_fwd_for_dgrad=True,
        params={"tile_id": tile_id},
        mask_words=1,
        use_f32_out_tile=False,
        vec_width=_VEC_WIDTH,
        C_in_g=_ALIGNED_C,
        C_out_g=_ALIGNED_C,
        use_fp16_accum=False,
    )
    assert resolved_tile == tile_id
    assert use_fwd_fallback is False


def test_unaligned_channels_route_to_fwd_fallback_even_with_foreign_tile_id():
    """Unaligned per-group channels take the scalar wcn fwd-fallback path
    (tiles 70/71/72) before the whitelist check runs -- so a foreign tile_id
    here must NOT raise; the alignment branch takes precedence."""
    resolved_tile, use_fwd_fallback = _select_dgrad_tile(
        use_fwd_for_dgrad=True,
        params={"tile_id": 58},  # foreign id; must be ignored on this path
        mask_words=1,
        use_f32_out_tile=False,
        vec_width=_VEC_WIDTH,
        C_in_g=_UNALIGNED_C,
        C_out_g=_ALIGNED_C,
        use_fp16_accum=False,
    )
    assert resolved_tile in (70, 71, 72)
    assert use_fwd_fallback is True


def test_fwd_as_dgrad_pool_entries_are_all_canonical_dgrad_wt_ids():
    """Guard against someone adding fwd pcoff ids (54-63) to the fwd_as_dgrad
    pools in algo_params.py: every entry's tile_id must be in 900-911, else
    it would hit the foreign-tile-id raise above at autotune time."""
    for algo, params in _AB_MASK_GEMM_FWD_AS_DGRAD:
        assert algo == "mask_gemm_fwd_as_dgrad"
        tile_id = params["tile_id"]
        assert tile_id in _DGRAD_WT_TILES, (
            f"fwd_as_dgrad pool entry has tile_id={tile_id}, not in 900-911 "
            "(canonical dgrad_wt range) -- would be rejected as a foreign "
            "tile id by _select_dgrad_tile."
        )
