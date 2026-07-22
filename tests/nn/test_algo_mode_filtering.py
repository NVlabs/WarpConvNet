# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sparse conv autotune algorithm mode filtering.

Verifies that "auto", "all", single-algo, and list-of-algo modes
produce the correct candidate sets for both the AB (gather-scatter,
forward + dgrad) and AtB (gather-gather, wgrad) passes.
"""

import pytest


@pytest.fixture(autouse=True)
def _import_algo_params():
    """Import algo_params module once; skip if unavailable."""
    try:
        from warpconvnet.nn.functional.sparse_conv.detail import (
            algo_params,
        )  # noqa: F401
    except Exception as e:
        pytest.skip(f"Cannot import algo_params module: {e}")


def _import():
    from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
        _filter_benchmark_params_by_env_config,
        _get_adaptive_AB_params,
        _get_adaptive_AtB_params,
        _get_filtered_AB_params,
        _get_filtered_AtB_params,
        _AB_PARAMS_AUTO,
        _ATB_PARAMS_AUTO,
        _ALL_AB_PARAMS,
        _ALL_ATB_PARAMS,
    )

    return dict(
        _filter=_filter_benchmark_params_by_env_config,
        _adaptive_ab=_get_adaptive_AB_params,
        _adaptive_atb=_get_adaptive_AtB_params,
        _filtered_ab=_get_filtered_AB_params,
        _filtered_atb=_get_filtered_AtB_params,
        _auto_ab=_AB_PARAMS_AUTO,
        _auto_atb=_ATB_PARAMS_AUTO,
        _all_ab=_ALL_AB_PARAMS,
        _all_atb=_ALL_ATB_PARAMS,
    )


def _key(params):
    """Canonical hashable key set for a list of (algo, params) candidates."""
    return {(a, tuple(sorted(p.items()))) for a, p in params}


# ---------------------------------------------------------------------------
# Candidate set structure
# ---------------------------------------------------------------------------


class TestCandidateSetSizes:
    """Verify reduced (adaptive) vs full ("all") candidate relationships.

    Exact candidate counts are backend/hardware dependent (mask_gemm, cutlass,
    cute, SM90 availability), so these assert structural relationships instead
    of hard-coded magic numbers.
    """

    def test_all_forward_nonempty_and_superset_of_auto_pool(self):
        m = _import()
        assert len(m["_all_ab"]) > 0
        # The static "auto" superset must be contained in the exhaustive set.
        assert _key(m["_auto_ab"]).issubset(_key(m["_all_ab"]))

    def test_all_backward_nonempty_and_superset_of_auto_pool(self):
        m = _import()
        assert len(m["_all_atb"]) > 0
        assert _key(m["_auto_atb"]).issubset(_key(m["_all_atb"]))

    def test_adaptive_forward_is_subset_of_all(self):
        m = _import()
        all_set = _key(m["_all_ab"])
        for c_in, c_out, kv in [(32, 32, 27), (128, 256, 27), (384, 256, 27)]:
            adaptive = m["_adaptive_ab"](c_in, c_out, kv)
            missing = _key(adaptive) - all_set
            assert (
                not missing
            ), f"adaptive AB({c_in},{c_out},{kv}) not in _ALL_AB_PARAMS: {missing}"

    def test_adaptive_backward_is_subset_of_all(self):
        m = _import()
        all_set = _key(m["_all_atb"])
        for c_in, c_out, kv in [(32, 32, 27), (256, 256, 27)]:
            adaptive = m["_adaptive_atb"](c_in, c_out, kv)
            missing = _key(adaptive) - all_set
            assert (
                not missing
            ), f"adaptive AtB({c_in},{c_out},{kv}) not in _ALL_ATB_PARAMS: {missing}"

    def test_adaptive_forward_is_strictly_smaller_than_all(self):
        m = _import()
        # Even the largest adaptive set (channel-rich shape) must trim the
        # exhaustive pool.
        adaptive = m["_adaptive_ab"](384, 256, 27)
        assert len(adaptive) < len(m["_all_ab"])

    def test_adaptive_backward_is_strictly_smaller_than_all(self):
        m = _import()
        adaptive = m["_adaptive_atb"](256, 256, 27)
        assert len(adaptive) < len(m["_all_atb"])


# ---------------------------------------------------------------------------
# Adaptive forward params (channel-dependent gating)
# ---------------------------------------------------------------------------


class TestAdaptiveForwardParams:
    """Verify channel-dependent candidate selection.

    Current gating (SM 8.9 pool): mask_gemm + cutlass_implicit_gemm form the
    always-present core; cutlass_grouped_hybrid is added for max_ch in 129-256;
    cute_grouped is added only for max_ch > 256.
    """

    def test_core_algos_present_all_channels(self):
        """mask_gemm and cutlass_implicit_gemm appear in every adaptive config."""
        m = _import()
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
            _HAS_MASK_GEMM,
            _HAS_CUTLASS_BACKEND,
        )

        for c_in, c_out in [(32, 32), (128, 256), (384, 256)]:
            algos = {a for a, _ in m["_adaptive_ab"](c_in, c_out, 27)}
            if _HAS_MASK_GEMM:
                assert "mask_gemm" in algos, f"mask_gemm missing at ({c_in},{c_out})"
            if _HAS_CUTLASS_BACKEND:
                assert (
                    "cutlass_implicit_gemm" in algos
                ), f"cutlass_implicit_gemm missing at ({c_in},{c_out})"

    def test_large_channels_include_cute_grouped(self):
        m = _import()
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
            _HAS_CUTE_GROUPED,
        )

        if not _HAS_CUTE_GROUPED:
            pytest.skip("cute_grouped backend unavailable")
        algos = {a for a, _ in m["_adaptive_ab"](384, 256, 27)}
        assert "cute_grouped" in algos

    def test_small_channels_exclude_cute_grouped(self):
        m = _import()
        # cute_grouped is gated to max_ch > 256; small channels must not include it.
        for c_in, c_out in [(32, 32), (64, 128)]:
            algos = {a for a, _ in m["_adaptive_ab"](c_in, c_out, 27)}
            assert "cute_grouped" not in algos

    def test_mid_channels_include_cutlass_grouped(self):
        m = _import()
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
            _HAS_CUTLASS_BACKEND,
        )

        if not _HAS_CUTLASS_BACKEND:
            pytest.skip("cutlass backend unavailable")
        algos = {a for a, _ in m["_adaptive_ab"](256, 256, 27)}
        assert "cutlass_grouped_hybrid" in algos

    def test_small_channels_exclude_cutlass_grouped(self):
        m = _import()
        algos = {a for a, _ in m["_adaptive_ab"](32, 32, 27)}
        assert "cutlass_grouped_hybrid" not in algos

    def test_cute_grouped_boundary_256_vs_257(self):
        """max_ch == 256 excludes cute_grouped; max_ch == 257 includes it."""
        m = _import()
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
            _HAS_CUTE_GROUPED,
        )

        if not _HAS_CUTE_GROUPED:
            pytest.skip("cute_grouped backend unavailable")
        algos_256 = {a for a, _ in m["_adaptive_ab"](256, 256, 27)}
        algos_257 = {a for a, _ in m["_adaptive_ab"](257, 256, 27)}
        assert "cute_grouped" not in algos_256
        assert "cute_grouped" in algos_257


# ---------------------------------------------------------------------------
# _filter_benchmark_params_by_env_config
# ---------------------------------------------------------------------------


class TestFilterByEnvConfig:
    """Test the filter function for auto/all/single/list modes."""

    def test_auto_returns_all_params_passed_in(self):
        m = _import()
        dummy = [("cutlass_implicit_gemm", {}), ("explicit_gemm", {})]
        result = m["_filter"](dummy, "auto", is_forward=True)
        assert len(result) == len(dummy)
        assert result[0][0] == "cutlass_implicit_gemm"
        assert result[1][0] == "explicit_gemm"

    def test_all_forward_returns_full_set(self):
        m = _import()
        dummy = [("cutlass_implicit_gemm", {})]  # small input, should be ignored
        result = m["_filter"](dummy, "all", is_forward=True)
        assert len(result) == len(m["_all_ab"])

    def test_all_backward_returns_full_set(self):
        m = _import()
        dummy = [("cutlass_implicit_gemm", {})]
        result = m["_filter"](dummy, "all", is_forward=False)
        assert len(result) == len(m["_all_atb"])

    def test_single_algo_string_filters(self):
        m = _import()
        params = list(m["_all_ab"])  # use full set as input
        result = m["_filter"](params, "cutlass_implicit_gemm", is_forward=True)
        assert all(a == "cutlass_implicit_gemm" for a, _ in result)
        assert len(result) >= 1

    def test_list_of_algos_filters(self):
        m = _import()
        params = list(m["_all_ab"])
        result = m["_filter"](params, ["cutlass_implicit_gemm", "cute_grouped"], is_forward=True)
        algo_names = {a for a, _ in result}
        assert algo_names <= {"cutlass_implicit_gemm", "cute_grouped"}
        assert len(result) >= 2  # at least one of each

    def test_unknown_algo_raises(self):
        # Hardened behavior: an unknown algo name is a hard error (typo guard),
        # no longer a silent fall-back to the passed-in pool.
        m = _import()
        dummy = [("cutlass_implicit_gemm", {}), ("explicit_gemm", {})]
        with pytest.raises(ValueError):
            m["_filter"](dummy, "nonexistent_algo", is_forward=True)

    def test_empty_list_falls_back(self):
        m = _import()
        dummy = [("cutlass_implicit_gemm", {})]
        result = m["_filter"](dummy, [], is_forward=True)
        assert len(result) == len(dummy)


# ---------------------------------------------------------------------------
# _get_filtered_AB_params / _get_filtered_AtB_params
# ---------------------------------------------------------------------------


class TestGetFilteredParams:
    """Test the top-level filtered param getters (use env default = 'auto')."""

    def test_filtered_forward_returns_nonempty(self):
        m = _import()
        result = m["_filtered_ab"]()
        assert len(result) > 0

    def test_filtered_backward_returns_nonempty(self):
        m = _import()
        result = m["_filtered_atb"]()
        assert len(result) > 0

    def test_filtered_forward_smaller_than_all(self):
        m = _import()
        result = m["_filtered_ab"]()
        assert len(result) < len(m["_all_ab"])

    def test_filtered_backward_smaller_than_all(self):
        m = _import()
        result = m["_filtered_atb"]()
        assert len(result) < len(m["_all_atb"])


# ---------------------------------------------------------------------------
# Algorithm name consistency
# ---------------------------------------------------------------------------


class TestAlgoNameConsistency:
    """Ensure all algo names in adaptive sets appear in the full sets."""

    def test_forward_algo_names_valid(self):
        m = _import()
        all_algo_names = {a for a, _ in m["_all_ab"]}
        for c_in, c_out in [(3, 32), (32, 32), (64, 128), (256, 256), (384, 256)]:
            for algo, _ in m["_adaptive_ab"](c_in, c_out, 27):
                assert algo in all_algo_names, f"Unknown fwd algo: {algo}"

    def test_backward_algo_names_valid(self):
        m = _import()
        all_algo_names = {a for a, _ in m["_all_atb"]}
        for c_in, c_out in [(32, 32), (128, 128), (256, 256)]:
            for algo, _ in m["_adaptive_atb"](c_in, c_out, 27):
                assert algo in all_algo_names, f"Unknown bwd algo: {algo}"

    def test_no_duplicate_candidates_in_adaptive_fwd(self):
        m = _import()
        for c_in, c_out in [(32, 32), (256, 256), (384, 256)]:
            keys = [(a, tuple(sorted(p.items()))) for a, p in m["_adaptive_ab"](c_in, c_out, 27)]
            assert len(keys) == len(set(keys)), f"Duplicate candidates in adaptive({c_in},{c_out})"

    def test_no_duplicate_candidates_in_adaptive_bwd(self):
        m = _import()
        for c_in, c_out in [(32, 32), (256, 256)]:
            keys = [(a, tuple(sorted(p.items()))) for a, p in m["_adaptive_atb"](c_in, c_out, 27)]
            assert len(keys) == len(set(keys)), f"Duplicate candidates in adaptive({c_in},{c_out})"

    def test_no_duplicate_candidates_in_all_fwd(self):
        m = _import()
        keys = [(a, tuple(sorted(p.items()))) for a, p in m["_all_ab"]]
        assert len(keys) == len(set(keys))

    def test_no_duplicate_candidates_in_all_bwd(self):
        m = _import()
        keys = [(a, tuple(sorted(p.items()))) for a, p in m["_all_atb"]]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Enum consistency
# ---------------------------------------------------------------------------


class TestEnumConsistency:
    """Verify enum values match the algo names used in param lists."""

    def test_ab_enum_covers_all_algos(self):
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
            SPARSE_CONV_AB_ALGO_MODE,
        )

        m = _import()
        enum_values = {e.value for e in SPARSE_CONV_AB_ALGO_MODE}
        all_algo_names = {a for a, _ in m["_all_ab"]}
        # Every algo in the full set should have a corresponding enum
        for algo in all_algo_names:
            assert algo in enum_values, f"AB algo '{algo}' has no enum entry"

    def test_atb_enum_covers_all_algos(self):
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
            SPARSE_CONV_ATB_ALGO_MODE,
        )

        m = _import()
        enum_values = {e.value for e in SPARSE_CONV_ATB_ALGO_MODE}
        all_algo_names = {a for a, _ in m["_all_atb"]}
        for algo in all_algo_names:
            assert algo in enum_values, f"AtB algo '{algo}' has no enum entry"

    def test_auto_and_all_in_ab_enum(self):
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
            SPARSE_CONV_AB_ALGO_MODE,
        )

        assert SPARSE_CONV_AB_ALGO_MODE.AUTO.value == "auto"
        assert SPARSE_CONV_AB_ALGO_MODE.ALL.value == "all"

    def test_auto_and_all_in_atb_enum(self):
        from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
            SPARSE_CONV_ATB_ALGO_MODE,
        )

        assert SPARSE_CONV_ATB_ALGO_MODE.AUTO.value == "auto"
        assert SPARSE_CONV_ATB_ALGO_MODE.ALL.value == "all"


# ---------------------------------------------------------------------------
# constants.py validation
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constants.py VALID_ALGOS includes auto and all."""

    def test_valid_algos_has_auto(self):
        from warpconvnet.constants import VALID_ALGOS

        assert "auto" in VALID_ALGOS

    def test_valid_algos_has_all(self):
        from warpconvnet.constants import VALID_ALGOS

        assert "all" in VALID_ALGOS

    def test_default_fwd_mode_is_auto(self):
        from warpconvnet.constants import WARPCONVNET_FWD_ALGO_MODE

        # Default (no env var) should be "auto"
        assert WARPCONVNET_FWD_ALGO_MODE == "auto"

    def test_default_dgrad_mode_is_auto(self):
        from warpconvnet.constants import WARPCONVNET_DGRAD_ALGO_MODE

        assert WARPCONVNET_DGRAD_ALGO_MODE == "auto"

    def test_default_wgrad_mode_is_auto(self):
        from warpconvnet.constants import WARPCONVNET_WGRAD_ALGO_MODE

        assert WARPCONVNET_WGRAD_ALGO_MODE == "auto"
