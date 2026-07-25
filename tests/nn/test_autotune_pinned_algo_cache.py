# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pinned-algorithm autotune resolution cache.

When a caller pins ``fwd_algo`` to an algorithm that cannot service a shape
(every candidate is rejected), the sweep falls back to ``explicit_gemm``. The
fallback winner does not satisfy the pinned filter's membership test, so before
the fix every forward re-ran the full autotune sweep forever. The fix records a
negative resolution keyed by the exact (config, filter) identity so subsequent
forwards short-circuit to the fallback with no benchmarking. See
warpconvnet/nn/functional/sparse_conv/detail/unified.py and .../autotune.py.
"""

import logging

import pytest
import torch

from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.types.voxels import Voxels

import warpconvnet.utils.benchmark_cache as bc
import warpconvnet.nn.functional.sparse_conv.detail.autotune as autotune
import warpconvnet.nn.functional.sparse_conv.detail.backends as backends
import warpconvnet.nn.functional.sparse_conv.detail.unified as unified
from warpconvnet.nn.functional.sparse_conv.detail.unified import (
    UnifiedSpatiallySparseConvFunction,
)

# A fake forward backend name that is always "unsupported" for the shape.
_UNVIABLE_ALGO = "zzz_unviable_fwd"


@pytest.fixture
def scoped_benchmark_cache(monkeypatch, tmp_path):
    """Point the benchmark cache at an empty tmp dir and reset the lazy singleton
    plus the in-memory autotune dicts so sweep counting is isolated."""
    monkeypatch.setattr(bc, "WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE", str(tmp_path))
    monkeypatch.setattr(bc, "_generic_benchmark_cache", None)

    saved_ab = dict(autotune._BENCHMARK_AB_RESULTS)
    saved_fb = dict(autotune._BENCHMARK_AB_FALLBACK_RESULTS)
    autotune._BENCHMARK_AB_RESULTS.clear()
    autotune._BENCHMARK_AB_FALLBACK_RESULTS.clear()
    try:
        yield tmp_path
    finally:
        bc._generic_benchmark_cache = None
        autotune._BENCHMARK_AB_RESULTS.clear()
        autotune._BENCHMARK_AB_RESULTS.update(saved_ab)
        autotune._BENCHMARK_AB_FALLBACK_RESULTS.clear()
        autotune._BENCHMARK_AB_FALLBACK_RESULTS.update(saved_fb)


@pytest.fixture
def fwd_sweep_counter():
    """Count 'Auto-tuning forward' sweeps from the autotune logger directly (it
    sets propagate=False, so pytest's caplog root handler never sees them)."""
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            if "Auto-tuning forward" in record.getMessage():
                records.append(record.getMessage())

    target = autotune.logger.logger
    handler = _Collector(level=logging.INFO)
    prev_level = target.level
    target.setLevel(logging.INFO)
    target.addHandler(handler)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(prev_level)


@pytest.fixture
def unviable_backend(monkeypatch):
    """Register a fake forward backend that always reports an unsupported status,
    and make the algo filter route a pin to it. Every candidate for the pinned
    filter is then rejected, so the sweep falls back to explicit_gemm."""
    patched = dict(backends.FORWARD_BACKENDS)
    patched[_UNVIABLE_ALGO] = lambda ctx: 7  # nonzero GEMM status -> rejected
    monkeypatch.setattr(backends, "FORWARD_BACKENDS", patched)

    orig_filter = unified._filter_benchmark_params_by_env_config

    def _patched_filter(all_params, env_config, is_forward=True):
        if isinstance(env_config, list) and _UNVIABLE_ALGO in env_config:
            return [(_UNVIABLE_ALGO, {})]
        return orig_filter(all_params, env_config, is_forward=is_forward)

    monkeypatch.setattr(unified, "_filter_benchmark_params_by_env_config", _patched_filter)
    return _UNVIABLE_ALGO


def _build_forward_probe(C_in: int = 32, C_out: int = 32):
    """Small stride-1 3x3x3 sparse conv forward probe (in_features, weight,
    kernel_map, num_out_coords)."""
    torch.manual_seed(0)
    device = "cuda"
    coords = [(torch.rand((400, 3)) / 0.1).int()]
    features = [torch.rand((400, C_in))]
    voxels = Voxels(coords, features, device=device).unique()

    weight = (torch.randn(27, C_in, C_out, device=device) * 0.05).contiguous()
    bic = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    kernel_map = generate_kernel_map(bic, bic, (1, 1, 1), (3, 3, 3))
    num_out = bic.shape[0]
    in_feats = voxels.feature_tensor.contiguous()
    return in_feats, weight, kernel_map, num_out


def _run_forward(in_feats, weight, kernel_map, num_out, fwd_algo):
    out = UnifiedSpatiallySparseConvFunction.apply(
        in_feats,
        weight,
        kernel_map,
        num_out,
        fwd_algo,  # fwd_algo
        "auto",  # dgrad_algo
        "auto",  # wgrad_algo
        None,  # compute_dtype
        None,  # fwd_block_size
        None,  # bwd_block_size
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_unviable_sweeps_once(scoped_benchmark_cache, unviable_backend, fwd_sweep_counter):
    # A pin whose every candidate is rejected must sweep exactly once; every
    # subsequent forward short-circuits to the recorded fallback.
    in_feats, weight, kernel_map, num_out = _build_forward_probe()

    N = 5
    for _ in range(N):
        _run_forward(in_feats, weight, kernel_map, num_out, [unviable_backend])

    assert (
        len(fwd_sweep_counter) == 1
    ), f"expected 1 sweep across {N} forwards, got {len(fwd_sweep_counter)}"
    # A negative-resolution marker was recorded for exactly this pin.
    assert len(autotune._BENCHMARK_AB_FALLBACK_RESULTS) == 1
    (_, filter_key), (algo, _) = next(iter(autotune._BENCHMARK_AB_FALLBACK_RESULTS.items()))
    assert filter_key == unviable_backend
    assert algo == "explicit_gemm"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_unviable_survives_cache_reload(
    scoped_benchmark_cache, unviable_backend, fwd_sweep_counter
):
    # The negative resolution persists through the generic benchmark cache: a
    # simulated fresh process (singleton reset + reload from disk) does not
    # re-sweep.
    in_feats, weight, kernel_map, num_out = _build_forward_probe()
    _run_forward(in_feats, weight, kernel_map, num_out, [unviable_backend])
    assert len(fwd_sweep_counter) == 1

    # Force the marker to disk, then simulate a fresh process: drop the in-memory
    # autotune dicts and the cache singleton, and reload the cache from disk.
    cache = bc.get_generic_benchmark_cache()
    cache.save_cache(cache._results, force=True)
    autotune._BENCHMARK_AB_RESULTS.clear()
    autotune._BENCHMARK_AB_FALLBACK_RESULTS.clear()
    bc._generic_benchmark_cache = None
    autotune._initialize_benchmark_cache()

    # The marker must have been reloaded from disk.
    assert len(autotune._BENCHMARK_AB_FALLBACK_RESULTS) == 1

    fwd_sweep_counter.clear()
    for _ in range(3):
        _run_forward(in_feats, weight, kernel_map, num_out, [unviable_backend])
    assert len(fwd_sweep_counter) == 0, "reload must not trigger a re-sweep"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_viable_sweeps_once_and_reuses(scoped_benchmark_cache, fwd_sweep_counter):
    # A pin that IS viable (explicit_gemm always services the shape) sweeps once,
    # caches the winner under the config, and reuses it -- no negative marker.
    in_feats, weight, kernel_map, num_out = _build_forward_probe()

    for _ in range(4):
        _run_forward(in_feats, weight, kernel_map, num_out, ["explicit_gemm"])

    assert len(fwd_sweep_counter) == 1
    # A viable pin caches its winner in the config dict, not the fallback cache.
    assert len(autotune._BENCHMARK_AB_FALLBACK_RESULTS) == 0
    assert len(autotune._BENCHMARK_AB_RESULTS) == 1
    winner = next(iter(autotune._BENCHMARK_AB_RESULTS.values()))
    best_algo = winner[0][0] if isinstance(winner, list) else winner[0]
    assert best_algo == "explicit_gemm"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_switching_pin_retunes_once(scoped_benchmark_cache, unviable_backend, fwd_sweep_counter):
    # Resolving one pin must not mask a different pin: switching the filter
    # re-tunes exactly once, scoped to the new (config, filter) identity.
    in_feats, weight, kernel_map, num_out = _build_forward_probe()

    # First pin (unviable) resolves once.
    _run_forward(in_feats, weight, kernel_map, num_out, [unviable_backend])
    assert len(fwd_sweep_counter) == 1

    # Switch to a viable pin: distinct filter_key -> re-tunes once, then reuses.
    fwd_sweep_counter.clear()
    for _ in range(3):
        _run_forward(in_feats, weight, kernel_map, num_out, ["explicit_gemm"])
    assert len(fwd_sweep_counter) == 1, "a different pin must re-tune exactly once"
