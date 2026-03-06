"""Tests for sparse conv autotune algorithm mode filtering.

Verifies that "auto", "all", single-algo, and list-of-algo modes
produce the correct candidate sets for both forward and backward passes.
"""

import pytest


@pytest.fixture(autouse=True)
def _import_unified():
    """Import unified module symbols once; skip if unavailable."""
    try:
        from warpconvnet.nn.functional.sparse_conv.detail import unified  # noqa: F401
    except Exception as e:
        pytest.skip(f"Cannot import unified module: {e}")


def _import():
    from warpconvnet.nn.functional.sparse_conv.detail.unified import (
        _filter_benchmark_params_by_env_config,
        _get_adaptive_forward_params,
        _get_filtered_forward_params,
        _get_filtered_backward_params,
        _BENCHMARK_FORWARD_PARAMS_BASE,
        _BENCHMARK_FORWARD_PARAMS_SMALL_CH,
        _BENCHMARK_FORWARD_PARAMS_GROUPED,
        _BENCHMARK_BACKWARD_PARAMS,
        _ALL_BENCHMARK_FORWARD_PARAMS,
        _ALL_BENCHMARK_BACKWARD_PARAMS,
    )
    return dict(
        _filter=_filter_benchmark_params_by_env_config,
        _adaptive_fwd=_get_adaptive_forward_params,
        _filtered_fwd=_get_filtered_forward_params,
        _filtered_bwd=_get_filtered_backward_params,
        _base_fwd=_BENCHMARK_FORWARD_PARAMS_BASE,
        _small_ch_fwd=_BENCHMARK_FORWARD_PARAMS_SMALL_CH,
        _grouped_fwd=_BENCHMARK_FORWARD_PARAMS_GROUPED,
        _reduced_bwd=_BENCHMARK_BACKWARD_PARAMS,
        _all_fwd=_ALL_BENCHMARK_FORWARD_PARAMS,
        _all_bwd=_ALL_BENCHMARK_BACKWARD_PARAMS,
    )


# ---------------------------------------------------------------------------
# Candidate set sizes
# ---------------------------------------------------------------------------

class TestCandidateSetSizes:
    """Verify reduced vs full candidate counts."""

    def test_all_forward_has_19_candidates(self):
        m = _import()
        assert len(m["_all_fwd"]) == 19

    def test_all_backward_has_32_candidates(self):
        m = _import()
        assert len(m["_all_bwd"]) == 32

    def test_reduced_forward_base_is_subset_of_all(self):
        m = _import()
        all_set = {(a, tuple(sorted(p.items()))) for a, p in m["_all_fwd"]}
        for algo, params in m["_base_fwd"]:
            assert (algo, tuple(sorted(params.items()))) in all_set, (
                f"Base fwd candidate ({algo}, {params}) not in _ALL_BENCHMARK_FORWARD_PARAMS"
            )

    def test_reduced_backward_is_subset_of_all(self):
        m = _import()
        all_set = {(a, tuple(sorted(p.items()))) for a, p in m["_all_bwd"]}
        for algo, params in m["_reduced_bwd"]:
            assert (algo, tuple(sorted(params.items()))) in all_set, (
                f"Reduced bwd candidate ({algo}, {params}) not in _ALL_BENCHMARK_BACKWARD_PARAMS"
            )

    def test_reduced_forward_is_strictly_smaller_than_all(self):
        m = _import()
        # Even for small channels (maximum adaptive set), should be < 19
        adaptive = m["_adaptive_fwd"](32, 32, 27)
        assert len(adaptive) < len(m["_all_fwd"])

    def test_reduced_backward_is_strictly_smaller_than_all(self):
        m = _import()
        assert len(m["_reduced_bwd"]) < len(m["_all_bwd"])


# ---------------------------------------------------------------------------
# Adaptive forward params (channel-dependent)
# ---------------------------------------------------------------------------

class TestAdaptiveForwardParams:
    """Verify channel-dependent candidate selection."""

    def test_large_channels_exclude_implicit_gemm(self):
        m = _import()
        params = m["_adaptive_fwd"](256, 256, 27)
        algos = [a for a, _ in params]
        assert "implicit_gemm" not in algos
        assert "implicit_gemm_grouped" not in algos
        assert "cute_implicit_gemm" not in algos

    def test_small_channels_include_implicit_gemm(self):
        m = _import()
        params = m["_adaptive_fwd"](32, 32, 27)
        algos = [a for a, _ in params]
        assert "implicit_gemm" in algos

    def test_small_channels_include_grouped(self):
        m = _import()
        params = m["_adaptive_fwd"](32, 64, 27)
        algos = [a for a, _ in params]
        assert "implicit_gemm_grouped" in algos

    def test_boundary_64_is_small(self):
        m = _import()
        params_64 = m["_adaptive_fwd"](32, 64, 27)
        algos_64 = [a for a, _ in params_64]
        assert "implicit_gemm" in algos_64

    def test_boundary_65_is_large(self):
        m = _import()
        params_65 = m["_adaptive_fwd"](32, 65, 27)
        algos_65 = [a for a, _ in params_65]
        assert "implicit_gemm" not in algos_65

    def test_all_adaptive_contain_base_algos(self):
        """Both large and small channel configs should contain the base algos."""
        m = _import()
        base_set = {(a, tuple(sorted(p.items()))) for a, p in m["_base_fwd"]}
        for c_in, c_out in [(32, 32), (128, 256), (384, 256)]:
            params = m["_adaptive_fwd"](c_in, c_out, 27)
            param_set = {(a, tuple(sorted(p.items()))) for a, p in params}
            assert base_set.issubset(param_set), (
                f"Base fwd not subset of adaptive({c_in},{c_out}): "
                f"missing {base_set - param_set}"
            )


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
        assert len(result) == len(m["_all_fwd"])

    def test_all_backward_returns_full_set(self):
        m = _import()
        dummy = [("cutlass_implicit_gemm", {})]
        result = m["_filter"](dummy, "all", is_forward=False)
        assert len(result) == len(m["_all_bwd"])

    def test_single_algo_string_filters(self):
        m = _import()
        params = list(m["_all_fwd"])  # use full set as input
        result = m["_filter"](params, "cutlass_implicit_gemm", is_forward=True)
        assert all(a == "cutlass_implicit_gemm" for a, _ in result)
        assert len(result) >= 1

    def test_list_of_algos_filters(self):
        m = _import()
        params = list(m["_all_fwd"])
        result = m["_filter"](
            params, ["cutlass_implicit_gemm", "cute_grouped"], is_forward=True
        )
        algo_names = {a for a, _ in result}
        assert algo_names <= {"cutlass_implicit_gemm", "cute_grouped"}
        assert len(result) >= 2  # at least one of each

    def test_unknown_algo_falls_back_to_all_params(self):
        m = _import()
        dummy = [("cutlass_implicit_gemm", {}), ("explicit_gemm", {})]
        result = m["_filter"](dummy, "nonexistent_algo", is_forward=True)
        # Should fall back to all_params when no match found
        assert len(result) == len(dummy)

    def test_empty_list_falls_back(self):
        m = _import()
        dummy = [("cutlass_implicit_gemm", {})]
        result = m["_filter"](dummy, [], is_forward=True)
        assert len(result) == len(dummy)


# ---------------------------------------------------------------------------
# _get_filtered_forward_params / _get_filtered_backward_params
# ---------------------------------------------------------------------------

class TestGetFilteredParams:
    """Test the top-level filtered param getters (use env default = 'auto')."""

    def test_filtered_forward_returns_nonempty(self):
        m = _import()
        result = m["_filtered_fwd"]()
        assert len(result) > 0

    def test_filtered_backward_returns_nonempty(self):
        m = _import()
        result = m["_filtered_bwd"]()
        assert len(result) > 0

    def test_filtered_forward_smaller_than_all(self):
        m = _import()
        result = m["_filtered_fwd"]()
        assert len(result) < len(m["_all_fwd"])

    def test_filtered_backward_smaller_than_all(self):
        m = _import()
        result = m["_filtered_bwd"]()
        assert len(result) < len(m["_all_bwd"])


# ---------------------------------------------------------------------------
# Algorithm name consistency
# ---------------------------------------------------------------------------

class TestAlgoNameConsistency:
    """Ensure all algo names in reduced sets appear in the full sets."""

    def test_forward_algo_names_valid(self):
        m = _import()
        all_algo_names = {a for a, _ in m["_all_fwd"]}
        for c_in, c_out in [(3, 32), (32, 32), (64, 128), (256, 256)]:
            for algo, _ in m["_adaptive_fwd"](c_in, c_out, 27):
                assert algo in all_algo_names, f"Unknown fwd algo: {algo}"

    def test_backward_algo_names_valid(self):
        m = _import()
        all_algo_names = {a for a, _ in m["_all_bwd"]}
        for algo, _ in m["_reduced_bwd"]:
            assert algo in all_algo_names, f"Unknown bwd algo: {algo}"

    def test_no_duplicate_candidates_in_adaptive(self):
        m = _import()
        for c_in, c_out in [(32, 32), (256, 256)]:
            params = m["_adaptive_fwd"](c_in, c_out, 27)
            keys = [(a, tuple(sorted(p.items()))) for a, p in params]
            assert len(keys) == len(set(keys)), (
                f"Duplicate candidates in adaptive({c_in},{c_out})"
            )

    def test_no_duplicate_candidates_in_reduced_bwd(self):
        m = _import()
        keys = [(a, tuple(sorted(p.items()))) for a, p in m["_reduced_bwd"]]
        assert len(keys) == len(set(keys))

    def test_no_duplicate_candidates_in_all_fwd(self):
        m = _import()
        keys = [(a, tuple(sorted(p.items()))) for a, p in m["_all_fwd"]]
        assert len(keys) == len(set(keys))

    def test_no_duplicate_candidates_in_all_bwd(self):
        m = _import()
        keys = [(a, tuple(sorted(p.items()))) for a, p in m["_all_bwd"]]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Enum consistency
# ---------------------------------------------------------------------------

class TestEnumConsistency:
    """Verify enum values match the algo names used in param lists."""

    def test_fwd_enum_covers_all_algos(self):
        from warpconvnet.nn.functional.sparse_conv.detail.unified import (
            SPARSE_CONV_FWD_ALGO_MODE,
        )
        m = _import()
        enum_values = {e.value for e in SPARSE_CONV_FWD_ALGO_MODE}
        all_algo_names = {a for a, _ in m["_all_fwd"]}
        # Every algo in the full set should have a corresponding enum
        for algo in all_algo_names:
            assert algo in enum_values, f"Fwd algo '{algo}' has no enum entry"

    def test_bwd_enum_covers_all_algos(self):
        from warpconvnet.nn.functional.sparse_conv.detail.unified import (
            SPARSE_CONV_BWD_ALGO_MODE,
        )
        m = _import()
        enum_values = {e.value for e in SPARSE_CONV_BWD_ALGO_MODE}
        all_algo_names = {a for a, _ in m["_all_bwd"]}
        for algo in all_algo_names:
            assert algo in enum_values, f"Bwd algo '{algo}' has no enum entry"

    def test_auto_and_all_in_fwd_enum(self):
        from warpconvnet.nn.functional.sparse_conv.detail.unified import (
            SPARSE_CONV_FWD_ALGO_MODE,
        )
        assert SPARSE_CONV_FWD_ALGO_MODE.AUTO.value == "auto"
        assert SPARSE_CONV_FWD_ALGO_MODE.ALL.value == "all"

    def test_auto_and_all_in_bwd_enum(self):
        from warpconvnet.nn.functional.sparse_conv.detail.unified import (
            SPARSE_CONV_BWD_ALGO_MODE,
        )
        assert SPARSE_CONV_BWD_ALGO_MODE.AUTO.value == "auto"
        assert SPARSE_CONV_BWD_ALGO_MODE.ALL.value == "all"


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

    def test_default_bwd_mode_is_auto(self):
        from warpconvnet.constants import WARPCONVNET_BWD_ALGO_MODE
        assert WARPCONVNET_BWD_ALGO_MODE == "auto"
