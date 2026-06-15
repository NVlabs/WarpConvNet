# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard between the mask-GEMM tile-selection frozensets and the
checked-in warpgemm tile_metadata snapshot.

mask_gemm.py hardcodes which tiles must NOT be dispatched at MaskWords>1
(_MW1_ONLY_FWD_TILES, _MW1_ONLY_DGRAD_WT_TILES). That fact is OWNED BY WARPGEMM
(see the SOURCE-OF-TRUTH BOUNDARY comment in mask_gemm.py). These tests assert
the literals stay consistent with the snapshot so a warpgemm regen that changes
tile capabilities fails loudly in CI instead of silently mis-routing a kernel
(which would device-assert `K <= MaskWords*32` and kill the CUDA context).

Two layers:
  - STRUCTURAL (always runs, reads the checked-in snapshot): pcoff tiles are
    pcoff, 32x32 tiles are 32x32, MW-capable defaults are excluded, and the
    wcn-only fallback tiles are absent from warpgemm metadata.
  - AUTHORED FIELD (runs once warpgemm emits per-tile ``dispatch_mask_words``):
    the literals must equal {tile : 2 not in dispatch_mask_words}. Skipped until
    the field ships; flip on by bumping the schema gate.
"""

from __future__ import annotations

import pytest

build_tile_metadata = pytest.importorskip(
    "warpconvnet.csrc.mask_gemm.tile_metadata",
    reason="tile_metadata snapshot unavailable",
).build_tile_metadata

from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
    _32X32_FWD_TILES,
    _MASK_WORDS_TIERS,
    _MW1_ONLY_DGRAD_WT_TILES,
    _MW1_ONLY_FWD_TILES,
    _MW4_MAX_FWD_TILES,
    _PCOFF_FWD_TILES,
    _WCN_ONLY_FWD_TILES,
)


def _forward_by_id():
    # active_only=True, no arch filter -> runs on CPU CI (no CUDA needed).
    return {t.tile_id: t for t in build_tile_metadata(active_only=True, ops=("forward",))}


def test_pcoff_fwd_tiles_are_pcoff():
    """Every _PCOFF_FWD_TILES id exists and is structurally a pcoff tile."""
    byid = _forward_by_id()
    for tid in _PCOFF_FWD_TILES:
        assert tid in byid, f"pcoff tile {tid} vanished from warpgemm forward metadata"
        assert "pcoff" in byid[tid].kernel_struct, (
            f"tile {tid} no longer pcoff (struct={byid[tid].kernel_struct}); "
            "its MW1-only smem-ceiling rationale may no longer hold — review "
            "_PCOFF_FWD_TILES in mask_gemm.py."
        )


def test_32x32_fwd_tiles_are_32x32():
    """Every _32X32_FWD_TILES id exists and is a 32x32 tile."""
    byid = _forward_by_id()
    for tid in _32X32_FWD_TILES:
        assert tid in byid, f"32x32 tile {tid} vanished from warpgemm forward metadata"
        t = byid[tid]
        assert t.tile_m == 32 and t.tile_n == 32, (
            f"tile {tid} is no longer 32x32 (got {t.tile_m}x{t.tile_n}); the "
            "instantiation-gap rationale changed — review _32X32_FWD_TILES."
        )


def test_mw_capable_fwd_defaults_excluded():
    """The aligned/scalar MW>1 fwd defaults must NOT be flagged MW1-only.

    41 (aligned K>32 default), 3 (64x128 3s), 2 (128x64) are flat/fused tiles
    the selector relies on for K>32. If any slipped into _MW1_ONLY_FWD_TILES the
    K>32 path would wrongly raise.
    """
    byid = _forward_by_id()
    for tid in (41, 3, 2):
        assert tid in byid, f"MW-capable default tile {tid} missing from metadata"
        assert tid not in _MW1_ONLY_FWD_TILES
        assert "pcoff" not in byid[tid].kernel_struct
        assert not (byid[tid].tile_m == 32 and byid[tid].tile_n == 32)


def test_dgrad_wt_mw1_only_split():
    """dgrad_wt aliases 900-911: MW1-only iff pcoff or 32x32; flat/fused excluded."""
    byid = _forward_by_id()
    for tid in (900, 901, 902, 904):
        if tid in byid:
            assert (
                tid not in _MW1_ONLY_DGRAD_WT_TILES
            ), f"flat/fused dgrad_wt alias {tid} wrongly flagged MW1-only"
    for tid in _MW1_ONLY_DGRAD_WT_TILES:
        assert tid in byid, f"dgrad_wt MW1-only alias {tid} missing from metadata"
        t = byid[tid]
        assert "pcoff" in t.kernel_struct or (
            t.tile_m == 32 and t.tile_n == 32
        ), f"dgrad_wt alias {tid} is neither pcoff nor 32x32 yet flagged MW1-only"


def test_mask_words_tiers_match_snapshot_union():
    """_MASK_WORDS_TIERS (mirroring the binding DISPATCH_MW ladder) must equal the
    union of every tile's dispatch_mask_words in the snapshot. If warpgemm adds a
    higher tier (e.g. MW16) this fires so the binding macro + _MASK_WORDS_TIERS get
    extended together. Skipped until dispatch_mask_words ships."""
    tiles = []
    for op in ("forward", "dgrad", "wgrad"):
        tiles += build_tile_metadata(active_only=True, ops=(op,))
    if not tiles or not hasattr(tiles[0], "dispatch_mask_words"):
        pytest.skip("dispatch_mask_words not present yet")
    union = set()
    for t in tiles:
        union.update(t.dispatch_mask_words)
    assert set(_MASK_WORDS_TIERS) == union, (
        f"_MASK_WORDS_TIERS {sorted(_MASK_WORDS_TIERS)} != snapshot tier union "
        f"{sorted(union)}. warpgemm changed the tier ladder — update "
        "_MASK_WORDS_TIERS and the DISPATCH_MW macro in mask_gemm_bindings.cu."
    )


def test_mw4_max_guard_disjoint_from_mw1_only():
    """A tile is either MW1-only or MW4-capped, never both."""
    assert not (_MW4_MAX_FWD_TILES & _MW1_ONLY_FWD_TILES), (
        f"tiles in both MW4-max and MW1-only guards: "
        f"{sorted(_MW4_MAX_FWD_TILES & _MW1_ONLY_FWD_TILES)}"
    )


def test_mw4_max_tiles_authorized_at_mw4_not_mw8():
    """The MW4-capped tiles (32x32 tile 28) must be authorized at MW4 but NOT at
    MW8 in the field — i.e. promoting them to MW2/4 is field-backed, and the MW4
    cap (reject K>128) matches that there is no MW8 authorization. Skipped until
    dispatch_mask_words ships."""
    byid = _forward_by_id()
    sample = next(iter(byid.values()), None)
    if sample is None or not hasattr(sample, "dispatch_mask_words"):
        pytest.skip("dispatch_mask_words not present yet")
    for tid in _MW4_MAX_FWD_TILES:
        assert tid in byid, f"MW4-max tile {tid} missing from metadata"
        dmw = byid[tid].dispatch_mask_words
        assert 4 in dmw, f"tile {tid} promoted to MW4 but field {dmw} lacks 4"
        assert 8 not in dmw, (
            f"tile {tid} field {dmw} authorizes MW8 — the MW4 cap is now too "
            "conservative; instantiate MW8 and relax _MW4_MAX_FWD_TILES."
        )


def test_wcn_only_tiles_absent_from_metadata():
    """wcn-only fallback tiles (70-72 scalar, 80-82 f32-out, 300-307 strided)
    must not appear in warpgemm metadata. If warpgemm absorbs them, migrate the
    selector to derive from the snapshot instead of treating them binding-private.
    """
    byid = _forward_by_id()
    leaked = sorted(tid for tid in _WCN_ONLY_FWD_TILES if tid in byid)
    assert not leaked, (
        f"wcn-only tiles now present in warpgemm metadata: {leaked}. "
        "Migrate mask_gemm.py to derive these from the snapshot."
    )


@pytest.mark.parametrize(
    "id_lo,id_hi,authored",
    [
        (0, 399, _MW1_ONLY_FWD_TILES),
        (900, 911, _MW1_ONLY_DGRAD_WT_TILES),
    ],
)
def test_mw1_authorized_tiles_are_guarded(id_lo, id_hi, authored):
    """Correctness invariant once warpgemm emits per-tile ``dispatch_mask_words``:
    every tile AUTHORIZED only at MW1 (2 not in dispatch_mask_words) MUST be in
    the MW1-only guard, else the selector could dispatch it at MW>1 and
    device-assert.

    This is a SUBSET check, not equality. ``dispatch_mask_words`` is an
    authorization + correctness signal, NOT an availability guarantee: a tile
    authorized at MW2/4 may legitimately stay in the guard until THIS build's
    DISPATCH_MW macro actually instantiates its MW2/4 kernel (dropping it must be
    simultaneous with that binding change, not with field presence). So
    guard >= {MW1-only-authorized}; over-guarding is safe, under-guarding crashes.

    Skipped until the field ships."""
    byid = _forward_by_id()
    sample = next(iter(byid.values()), None)
    if sample is None or not hasattr(sample, "dispatch_mask_words"):
        pytest.skip(
            "warpgemm tile_metadata has no dispatch_mask_words yet; structural "
            "tests cover the interim."
        )
    mw1_authorized = frozenset(
        tid for tid, t in byid.items() if id_lo <= tid <= id_hi and 2 not in t.dispatch_mask_words
    )
    missing = mw1_authorized - frozenset(authored)
    assert not missing, (
        f"tiles authorized only at MW1 but absent from the guard for ids "
        f"[{id_lo},{id_hi}]: {sorted(missing)}. The selector could dispatch them "
        "at MW>1 and crash — add them to the MW1-only set in mask_gemm.py."
    )
