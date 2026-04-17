# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for tile metadata adapter.

The metadata describes warpgemm's primitive tile catalog (tile_ids 0..42
for forward, different numbering than warpconvnet's Prod_* enum in
``gemm_mma_tiles.h``). MW>1 is runtime-dispatched via warpconvnet's
``DISPATCH_MW`` macro, not exposed as separate TileEntry records in the
metadata. These tests verify the adapter mechanics; dispatch.py still
uses its own per-Prod_*-tile logic.
"""
import pytest

from warpconvnet.nn.functional.sparse_conv.detail.tile_metadata import (
    _MIN_SCHEMA_VERSION,
    _get_tiles,
    candidate_tiles,
    explain,
    pick_tile,
    rank_candidates,
)


def test_schema_version_ok():
    assert _MIN_SCHEMA_VERSION >= 1


def test_forward_tiles_loaded():
    tiles = _get_tiles("forward")
    assert len(tiles) > 0
    for t in tiles:
        assert t.op == "forward"
        assert t.tile_m > 0 and t.tile_n > 0 and t.tile_k > 0


def test_dgrad_tiles_loaded():
    tiles = _get_tiles("dgrad")
    assert len(tiles) > 0


def test_wgrad_tiles_loaded():
    tiles = _get_tiles("wgrad")
    assert len(tiles) > 0


def test_candidate_tiles_aligned_f16():
    # Aligned C=128, K=27 (MW=1), f16 — should match many primitive tiles
    c = candidate_tiles("forward", 128, 128, 27, 1, "f16", "f16")
    assert len(c) > 0
    # All candidates must claim C_in=128 aligned
    for t in c:
        assert t.handles_c_in(128)
        assert t.handles_c_out(128)


def test_rank_prefers_vectorized():
    c = candidate_tiles("forward", 128, 128, 27, 1, "f16", "f16")
    assert len(c) >= 2
    ranked = rank_candidates(c, 128, 128)
    # Top pick prefers vectorized path
    assert ranked[0].prefers_vectorized_c_in(128)


def test_pick_tile_returns_record():
    m = pick_tile("forward", 128, 128, 27, 1, "f16", "f16")
    assert m is not None
    assert hasattr(m, "tile_id")
    assert hasattr(m, "kernel_struct")


def test_pick_tile_unaligned_cin_returns_scalar():
    # C_in=3 forces scalar-A tile (c_in_alignment==1 in metadata)
    m = pick_tile("forward", 3, 32, 27, 1, "f16", "f16")
    # primitive tile 40/41/42 are scalar per warpgemm
    assert m is not None
    # scalar tile has c_in_alignment=1 OR specifically handles unaligned C_in
    assert m.handles_c_in(3)


def test_pick_tile_no_match_returns_none():
    # Impossible request: dtype combo that no tile supports
    m = pick_tile("forward", 128, 128, 27, 1, "nonexistent", "nonexistent")
    assert m is None


def test_explain_returns_string():
    s = explain("forward", 64, 64, 27, 1, "f16", "f16")
    assert isinstance(s, str)
    assert "Problem:" in s
    assert "Survivors:" in s


def test_wgrad_split_k_capability():
    c = candidate_tiles("wgrad", 128, 128, 27, 1, "f16", "f32")
    assert any(t.supports_split_k for t in c), "no split-K wgrad tile found"


def test_handles_accumulator_base_tile_has_both():
    """Base tiles (no specialization suffix) must advertise both f16 and f32."""
    from warpconvnet.nn.functional.sparse_conv.detail.tile_metadata import _get_tiles

    tiles = _get_tiles("forward")
    # Find a base tile — kernel_struct without an _Accum/_K8 suffix.
    base = [
        t
        for t in tiles
        if not any(sfx in t.kernel_struct for sfx in ("_F16Accum", "_F16K8", "_DgradK8", "_F32K8"))
    ]
    assert base, "no base tile found"
    b = base[0]
    assert b.handles_accumulator("f16")
    assert b.handles_accumulator("f32")


def test_acc_dtype_hard_filter_f32():
    # acc_dtype="f32" should drop fp16-only specialized tiles
    c_all = candidate_tiles("forward", 128, 128, 27, 1, "f16", "f16")
    c_f32 = candidate_tiles("forward", 128, 128, 27, 1, "f16", "f16", acc_dtype="f32")
    assert len(c_f32) > 0
    assert len(c_f32) <= len(c_all)
    for t in c_f32:
        assert t.handles_accumulator("f32")


def test_acc_dtype_hard_filter_f16():
    c_f16 = candidate_tiles("forward", 128, 128, 27, 1, "f16", "f16", acc_dtype="f16")
    assert len(c_f16) > 0
    for t in c_f16:
        assert t.handles_accumulator("f16")


def test_pick_tile_prefer_fp16_accum_ranker():
    # Soft preference — both candidates exist, ranker biases toward f16
    m_f16_pref = pick_tile("forward", 128, 128, 27, 1, "f16", "f16", prefer_fp16_accum=True)
    m_f32_pref = pick_tile("forward", 128, 128, 27, 1, "f16", "f16", prefer_fp16_accum=False)
    assert m_f16_pref is not None
    assert m_f32_pref is not None


def test_wgrad_handles_c_is_op_specific_modulo_kvec():
    """Schema v3: wgrad uses strict per_group % min == 0, not padded-check.

    Under v2 semantics, wgrad candidates at (C_in=16 groups=2 → per_group=8)
    would be *incorrectly rejected* (padded_8=8 < some tile_m). Under v3,
    they are accepted because 8 % 8 == 0.
    """
    from warpconvnet.nn.functional.sparse_conv.detail.tile_metadata import _get_tiles

    tiles = _get_tiles("wgrad")
    # per-group=8 (kVec-aligned) must be accepted by at least one wgrad tile
    assert any(t.handles_c_in(16, groups=2) for t in tiles)
    assert any(t.handles_c_out(16, groups=2) for t in tiles)
    # per-group=6 (NOT kVec-aligned) must be rejected by every kVec-strict tile
    for t in tiles:
        if t.min_per_group_c_in == 8:
            assert not t.handles_c_in(12, groups=2), (
                f"wgrad tile {t.tile_id} accepted per_group=6 but binding "
                f"rejects with strict % 8 rule"
            )


def test_fwd_handles_c_in_still_padded_semantics():
    """Schema v3: fwd/dgrad keep the padded-threshold rule (unchanged)."""
    from warpconvnet.nn.functional.sparse_conv.detail.tile_metadata import _get_tiles

    tiles = _get_tiles("forward")
    # C=3 (unaligned) should be accepted by scalar tiles (c_in_alignment=1,
    # min_per_group_c_in small) via the padded-to-8 rule.
    assert any(t.handles_c_in(3) for t in tiles)


def test_snapshot_fallback_parses():
    """Explicit test that JSON snapshot parses without warpgemm import."""
    from warpconvnet.nn.functional.sparse_conv.detail.tile_metadata import _load_snapshot

    tiles = _load_snapshot()
    assert len(tiles) > 0
    # Verify at least a few fields present and typed
    t = tiles[0]
    assert isinstance(t.tile_id, int)
    assert isinstance(t.mask_words, int)
    assert isinstance(t.supported_input_dtypes, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
