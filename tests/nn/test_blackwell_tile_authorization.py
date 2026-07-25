# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blackwell handoff §3 regression tests: exact-arch + backend authorization.

Every candidate path — normal selection, pinned tile ids, and cached autotune
records — must validate ``device_arch in compile_archs`` (exact membership, no
``>=`` inference) and a launchable backend before entering the C++ dispatch
switch. The sm100_umma backend is a nonlaunchable scaffold and must never be
selected, on any device.
"""
import pytest

from warpconvnet.nn.functional.sparse_conv.detail import tile_metadata as tm
from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
    _METADATA_ABSENT_DGRAD_LAUNCH_IDS,
    _METADATA_ABSENT_FWD_LAUNCH_IDS,
    _METADATA_ABSENT_WGRAD_LAUNCH_IDS,
    _require_launchable,
)

_SM100_SCAFFOLD = {
    "forward": (1000, 1001),
    "dgrad": (1100, 1101),
    "wgrad": (1200,),
}
_SM120_EXPERIMENTAL = {
    "forward": (2000, 2001),
    "dgrad": (2100, 2101),
    "wgrad": (2200,),
}


@pytest.fixture
def force_arch(monkeypatch):
    """Pin the cached device arch code for the duration of a test."""

    def _set(arch: int | None):
        monkeypatch.setattr(tm, "_DEVICE_ARCH", arch)

    return _set


def test_schema_gate_requires_v7():
    assert tm._MIN_SCHEMA_VERSION == 7


def test_active_tiles_never_include_sm100_umma():
    for op in ("forward", "dgrad", "wgrad"):
        backends = {t.backend for t in tm._get_tiles(op, filter_arch=False)}
        assert backends <= tm._LAUNCHABLE_BACKENDS


def test_sm100_scaffold_rejected_regardless_of_arch(force_arch):
    # Even if the device WERE sm_100, the backend is not launchable here.
    for arch in (100, 120, 89):
        force_arch(arch)
        for op, ids in _SM100_SCAFFOLD.items():
            for tid in ids:
                reason = tm.tile_launch_rejection(op, tid)
                assert reason is not None and "sm100_umma" in reason


def test_sm120_pinned_tiles_rejected_on_non_sm120(force_arch):
    force_arch(89)
    for op, ids in _SM120_EXPERIMENTAL.items():
        for tid in ids:
            reason = tm.tile_launch_rejection(op, tid)
            assert reason is not None and "compile_archs" in reason


def test_exact_membership_not_ordering(force_arch):
    # sm_121 / sm_103 must not alias into sm_120 / sm_100 pins: membership is
    # exact, so a hypothetical newer device code is rejected, not inherited.
    for arch in (103, 121):
        force_arch(arch)
        assert tm.tile_launch_rejection("forward", 2000) is not None


def test_production_compat_tiles_pass_on_all_certified_archs(force_arch):
    # MW-tier compat family 60-62/500-504 compiles for every certified arch.
    for arch in (80, 89, 90, 100, 120):
        force_arch(arch)
        for op in ("forward", "dgrad"):
            for tid in (60, 61, 62, 500, 501, 502, 503, 504):
                assert tm.tile_launch_rejection(op, tid) is None


def test_foreign_id_rejected():
    assert tm.tile_launch_rejection("forward", 999) is not None
    assert tm.tile_launch_rejection("wgrad", 424242) is not None


def test_require_launchable_raises_for_pinned_sm100():
    with pytest.raises(RuntimeError, match="sm100_umma"):
        _require_launchable("forward", 1000, _METADATA_ABSENT_FWD_LAUNCH_IDS)
    with pytest.raises(RuntimeError, match="sm100_umma"):
        _require_launchable("wgrad", 1200, _METADATA_ABSENT_WGRAD_LAUNCH_IDS)


def test_require_launchable_allows_wcn_only_carveouts():
    for tid in sorted(_METADATA_ABSENT_FWD_LAUNCH_IDS):
        _require_launchable("forward", tid, _METADATA_ABSENT_FWD_LAUNCH_IDS)
    for tid in sorted(_METADATA_ABSENT_DGRAD_LAUNCH_IDS):
        _require_launchable("dgrad", tid, _METADATA_ABSENT_DGRAD_LAUNCH_IDS)
    for tid in sorted(_METADATA_ABSENT_WGRAD_LAUNCH_IDS):
        _require_launchable("wgrad", tid, _METADATA_ABSENT_WGRAD_LAUNCH_IDS)


def test_candidate_pool_excludes_experimental_by_default():
    ids = {t.tile_id for t in tm.candidate_tiles("forward", 128, 128, 27)}
    assert ids
    assert not ids & set(_SM120_EXPERIMENTAL["forward"])
    assert not ids & set(_SM100_SCAFFOLD["forward"])


def test_cache_version_invalidates_pre_blackwell_records():
    from warpconvnet.constants import WARPCONVNET_BENCHMARK_CACHE_VERSION

    assert WARPCONVNET_BENCHMARK_CACHE_VERSION >= 15.0
