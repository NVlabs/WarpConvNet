# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard: mask-GEMM dispatch truth table vs canonical warpgemm metadata.

Background
----------
wcn's mask-GEMM launcher templates are keyed on (ElemIn, TileTag, ElemOut), NOT
on the kernel class, so a binding arm can silently launch a kernel whose name
differs from the ``kernel_struct`` its canonical warpgemm metadata advertises.
Autotune/selection gates read that canonical metadata, so a silent mismatch lets
an upstream consumer trust a lie (e.g. tile 41 advertises the scalar-A kernel's
permissive alignment while the aligned 2s_pipelined kernel actually runs).

The binding now carries a wcn-OWNED truth table (``kMaskGemmDispatchTruth`` in
mask_gemm_bindings.cu), exposed as:
  * ``mask_gemm.dispatched_kernel(op, tile_id, mask_words)`` — kernel actually launched
  * ``mask_gemm.canonical_kernel_struct(op, tile_id)`` — kernel canonical metadata names
  * ``mask_gemm.dispatch_truth()`` — the full table with a per-row deviation ``note``

These tests enforce, for EVERY launchable arm, that the actual kernel equals the
canonical kernel UNLESS the divergence is explicitly documented (``note != ""``)
AND allow-listed here. So a re-keyed arm (or a new silent mismatch) fails CI:
either the actual kernel stops matching canonical for a supposed-match tile, or a
new deviation appears that is not in the allow-list below.

The three documented deviation classes (canonical single-field ``kernel_struct``
cannot express any of them):
  * STALE NAME — fwd 2, dgrad 0/1/24: canonical names a ``_2s`` kernel that does
    not exist in-tree; the fused / 1s_flat(_direpi) kernel is its evolved
    replacement and the only kernel for that shape.
  * MW-DEPENDENT — fwd 41: 2s_pipelined at MW1, 1s_flat at MW>1; canonical names
    the scalar-A variant (which executes only via wcn tile 71).
  * ID OVERLOAD — fwd 70/71/72: warpgemm assigns these ids to experimental
    128x128 8-warp tiles; wcn reuses them as scalar fallbacks handled before the
    canonical switch.
  * WT ALIAS — dgrad 900-911: launch the forward kernel with a pre-transposed
    weight; canonical metadata is op=forward at tile_id-500.
"""

from __future__ import annotations

import pytest

_C = pytest.importorskip(
    "warpconvnet._C",
    reason="warpconvnet._C (compiled) unavailable",
)
mask_gemm = getattr(_C, "mask_gemm", None)
if mask_gemm is None or not hasattr(mask_gemm, "dispatch_truth"):
    pytest.skip(
        "mask_gemm dispatch introspection not present — rebuild the extension",
        allow_module_level=True,
    )


# (op, tile_id) whose actual kernel intentionally differs from canonical metadata.
# Adding/removing an arm deviation must update this set — the test asserts it
# equals exactly the set of deviating rows observed at runtime.
EXPECTED_DEVIATIONS = frozenset(
    {
        ("forward", 2),  # stale name: non-fused 128x64 fwd kernel does not exist
        ("forward", 41),  # MW-dependent: 2s_pipelined (MW1) / 1s_flat (MW>1)
        ("forward", 70),  # id overload: canonical = 128x128 8warp experimental
        ("forward", 71),  # id overload
        ("forward", 72),  # id overload
        ("dgrad", 0),  # stale name: dgrad _2s does not exist -> 1s_flat
        ("dgrad", 1),  # stale name: dgrad _2s does not exist -> 1s_flat_direpi
        ("dgrad", 24),  # stale name (F16Accum) -> 1s_flat_direpi
    }
    | {("dgrad", t) for t in range(900, 912)}  # dgrad_wt aliases launch fwd kernels
)

# (op, tile_id) that are wcn-only (absent from canonical metadata entirely).
# Note the asymmetry for ids 70/71/72: in the FORWARD namespace they collide with
# experimental 128x128 8-warp tiles (so they are DEVIATIONS above), but the dgrad
# namespace has no canonical 70/71/72 record, so the dgrad scalar tiles are truly
# wcn-only here.
EXPECTED_WCN_ONLY = frozenset(
    {("forward", t) for t in range(300, 308)}  # strided downsample
    | {("forward", 80), ("forward", 82)}  # fwd f32-output
    | {("dgrad", 70), ("dgrad", 71), ("dgrad", 72)}  # dgrad scalar fallbacks
    | {("dgrad", 81)}  # dgrad f32-output
    | {("wgrad", 73)}  # scalar wgrad
)


def _rows():
    return mask_gemm.dispatch_truth()


def test_truth_table_nonempty_and_well_formed():
    rows = _rows()
    assert rows, "dispatch_truth() returned no rows"
    for r in rows:
        assert r["op"] in ("forward", "dgrad", "wgrad"), r
        assert 1 <= r["mask_words_lo"] <= r["mask_words_hi"] <= 12, r
        assert r["kernel_struct"], r  # every arm names a kernel


def test_actual_matches_canonical_unless_documented():
    """The core invariant: actual kernel == canonical kernel for every launchable
    arm, EXCEPT the documented deviations. A silently re-keyed arm breaks this."""
    observed_dev = set()
    observed_wcn = set()
    for r in _rows():
        op, tile, ks = r["op"], r["tile_id"], r["kernel_struct"]
        canon = mask_gemm.canonical_kernel_struct(op, tile)
        key = (op, tile)
        if canon == ks:
            # Clean match — must not be flagged as a deviation.
            assert r["note"] == "", (
                f"{op} tile {tile} matches canonical ({ks}) but carries a "
                f"deviation note: {r['note']!r}"
            )
            continue
        # Not a clean match -> must be documented.
        assert r["note"] != "", (
            f"{op} tile {tile} launches {ks} but canonical metadata says "
            f"{canon!r} with NO deviation note. Either an arm was re-keyed without "
            f"updating kMaskGemmDispatchTruth, or a new silent mismatch appeared."
        )
        if op == "dgrad" and tile >= 900:
            # wt alias: canonical lives under op=forward at tile-500.
            fwd_canon = mask_gemm.canonical_kernel_struct("forward", tile - 500)
            assert fwd_canon, f"dgrad_wt {tile} alias target forward {tile - 500} missing"
            observed_dev.add(key)
        elif canon == "":
            observed_wcn.add(key)
        else:
            observed_dev.add(key)

    assert observed_dev == EXPECTED_DEVIATIONS, (
        "documented-deviation set drifted.\n"
        f"  unexpected (new, undocumented in test): {sorted(observed_dev - EXPECTED_DEVIATIONS)}\n"
        f"  missing (documented here but no longer deviating): "
        f"{sorted(EXPECTED_DEVIATIONS - observed_dev)}"
    )
    assert observed_wcn == EXPECTED_WCN_ONLY, (
        "wcn-only (metadata-absent) set drifted.\n"
        f"  unexpected: {sorted(observed_wcn - EXPECTED_WCN_ONLY)}\n"
        f"  missing: {sorted(EXPECTED_WCN_ONLY - observed_wcn)}"
    )


def test_headline_fwd_2_and_41_deviations():
    """The two forward EXECUTED-set collisions this change-set is about."""
    # tile 2: metadata names the (non-existent) non-fused kernel; fused runs.
    assert mask_gemm.dispatched_kernel("forward", 2, 1) == "MaskGemm_forward_128x64x32_2s_fused"
    assert mask_gemm.canonical_kernel_struct("forward", 2) == "MaskGemm_forward_128x64x32_2s"

    # tile 41: MW-dependent, and canonical names the scalar-A variant.
    assert (
        mask_gemm.dispatched_kernel("forward", 41, 1) == "MaskGemm_forward_64x64x32_2s_pipelined"
    )
    assert mask_gemm.dispatched_kernel("forward", 41, 2) == "MaskGemm_forward_64x64x32_1s_flat"
    assert (
        mask_gemm.canonical_kernel_struct("forward", 41) == "MaskGemm_forward_64x64x32_1s_flat_sa"
    )


def test_flat_sa_never_runs_for_the_aligned_default():
    """The scalar-A kernel executes ONLY via wcn tile 71 (misaligned C), never as
    the aligned 64x64 default (tile 41). Guards against a future re-key that would
    ship the unproven scalar kernel on the hot path."""
    assert mask_gemm.dispatched_kernel("forward", 71, 1) == "MaskGemm_forward_64x64x32_1s_flat_sa"
    for mw in (1, 2, 4, 8, 12):
        assert "flat_sa" not in mask_gemm.dispatched_kernel("forward", 41, mw), (
            f"tile 41 dispatches flat_sa at mask_words={mw} — the aligned 64x64 "
            "default must never route to the scalar-A kernel"
        )


def test_dgrad_native_stale_2s_names():
    """Native dgrad tiles 0/1/24 name a '_2s' kernel that does not exist; the
    1s_flat/1s_flat_direpi kernel runs by design."""
    assert mask_gemm.dispatched_kernel("dgrad", 0, 1) == "MaskGemm_dgrad_64x64x32_1s_flat"
    assert mask_gemm.canonical_kernel_struct("dgrad", 0) == "MaskGemm_dgrad_64x64x32_2s"
    assert mask_gemm.dispatched_kernel("dgrad", 1, 1) == "MaskGemm_dgrad_64x128x32_1s_flat_direpi"
    assert mask_gemm.canonical_kernel_struct("dgrad", 1) == "MaskGemm_dgrad_64x128x32_2s"


def test_production_pool_tiles_are_all_tabled():
    """Every tile id the autotune pools can select must have a truth-table row, so
    no pool-active arm is un-audited."""
    from warpconvnet.nn.functional.sparse_conv.detail import algo_params as ap

    def _pool_ids(pool):
        out = []
        for entry in pool:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
                tid = entry[1].get("tile_id")
                if tid is not None:
                    out.append(tid)
        return out

    fwd_ids = set(_pool_ids(getattr(ap, "_AB_MASK_GEMM", [])))
    tabled_fwd = {r["tile_id"] for r in _rows() if r["op"] == "forward"}
    tabled_dgrad = {r["tile_id"] for r in _rows() if r["op"] == "dgrad"}
    missing = {t for t in fwd_ids if t not in tabled_fwd and t not in tabled_dgrad}
    assert (
        not missing
    ), f"autotune fwd pool tile ids absent from dispatch truth table: {sorted(missing)}"
