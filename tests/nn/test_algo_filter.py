# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Algorithm filter strictness.

When the caller names specific algorithms (``fwd_algo=['explicit_gemm']``),
the benchmarker must run exactly those — no silent fallback to the full
adaptive pool. Catching this matters for verification recipes that use
``explicit_gemm`` as the ground truth reference.
"""
import pytest

from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
    _filter_benchmark_params_by_env_config,
    _get_adaptive_AB_params,
    _get_adaptive_AtB_params,
)


def _adaptive(c=128, k=27, n=50000):
    return _get_adaptive_AB_params(c, c, k, num_in_coords=n)


def test_explicit_gemm_only_returns_one_entry():
    r = _filter_benchmark_params_by_env_config(_adaptive(), ["explicit_gemm"], is_forward=True)
    assert r == [("explicit_gemm", {})]


def test_implicit_gemm_expands_from_exhaustive_pool():
    # implicit_gemm is not in adaptive pool but is in _ALL_AB_PARAMS
    r = _filter_benchmark_params_by_env_config(_adaptive(), ["implicit_gemm"], is_forward=True)
    assert len(r) >= 1
    for algo, p in r:
        assert algo == "implicit_gemm"


def test_multiple_algos():
    r = _filter_benchmark_params_by_env_config(
        _adaptive(), ["explicit_gemm", "implicit_gemm"], is_forward=True
    )
    algos = {algo for algo, _ in r}
    assert "explicit_gemm" in algos
    assert "implicit_gemm" in algos


def test_unknown_algo_raises():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        _filter_benchmark_params_by_env_config(
            _adaptive(), ["totally_made_up_algo"], is_forward=True
        )


def test_auto_passthrough_unchanged():
    r = _filter_benchmark_params_by_env_config(_adaptive(), "auto", is_forward=True)
    # Should return adaptive pool as-is
    assert len(r) == len(_adaptive())


def test_all_returns_exhaustive():
    r = _filter_benchmark_params_by_env_config([], "all", is_forward=True)
    # Exhaustive pool has explicit_gemm + many others
    algos = {algo for algo, _ in r}
    assert "explicit_gemm" in algos


def test_backward_explicit_gemm_only():
    wgrad_adaptive = _get_adaptive_AtB_params(128, 128, 27, num_in_coords=50000)
    r = _filter_benchmark_params_by_env_config(wgrad_adaptive, ["explicit_gemm"], is_forward=False)
    assert r == [("explicit_gemm", {})]


def test_spatially_sparse_conv_explicit_gemm_end_to_end():
    """Verify dispatcher actually runs explicit_gemm when requested (not silently auto)."""
    import torch

    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.nn.functional.sparse_conv import spatially_sparse_conv
    from warpconvnet.nn.functional.sparse_conv.detail import unified as _unified

    torch.manual_seed(0)
    coords = [torch.randint(0, 60, (1500, 3), dtype=torch.int32)]
    feats = [torch.randn(1500, 16)]
    v = Voxels(coords, feats, device="cuda").unique()
    w = torch.nn.Parameter(torch.randn(27, 16, 16, device="cuda"))

    # Clear benchmark cache so we don't read a prior production entry
    _unified._BENCHMARK_AB_RESULTS.clear()

    # Forcing explicit_gemm: the cached tuple should be explicit_gemm.
    out = spatially_sparse_conv(v, w, kernel_size=(3, 3, 3), fwd_algo="explicit_gemm")
    assert out.feature_tensor.shape == (v.feature_tensor.shape[0], 16)

    # Verify the cache now holds explicit_gemm
    cached = list(_unified._BENCHMARK_AB_RESULTS.values())
    assert len(cached) >= 1
    top = cached[0] if isinstance(cached[0], tuple) else cached[0][0]
    assert top[0] == "explicit_gemm", (
        f"expected explicit_gemm cached; got {top[0]} " f"(silent fallback regressed)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
