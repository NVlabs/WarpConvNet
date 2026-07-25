# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the backward auto-tune numeric self-check.

The self-check compares each backward candidate's grad_in/grad_weight against an
explicit_gemm reference computed once per config and disqualifies candidates
that silently return a zero (or garbage) gradient -- the reported cin64 dgrad
zero-grad failure mode. See warpconvnet/nn/functional/sparse_conv/detail/autotune.py.
"""

import logging

import pytest
import torch

from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.types.voxels import Voxels

import warpconvnet.nn.functional.sparse_conv.detail.autotune as autotune
import warpconvnet.nn.functional.sparse_conv.detail.backends as backends
from warpconvnet.nn.functional.sparse_conv.detail.autotune import (
    _grad_pair_disqualified,
    _backward_numeric_disqualifies,
    _forward_numeric_disqualifies,
    _run_backward_benchmarks,
    _NUMERIC_MAX_ELEM_RDIFF,
)

# ---------------------------------------------------------------------------
# Unit tests for the comparison / disqualification helper
# ---------------------------------------------------------------------------


def _ref_tensor(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Nonzero reference with a realistic magnitude spread.
    return torch.randn(128, 64, generator=g) * 0.05


def test_helper_zero_candidate_disqualified():
    ref = _ref_tensor()
    cand = torch.zeros_like(ref)
    reason, rdiff, zero_ratio = _grad_pair_disqualified(ref, cand)
    assert reason is not None
    assert "zero" in reason.lower()
    assert zero_ratio < 1e-3


def test_helper_none_candidate_disqualified():
    ref = _ref_tensor()
    reason, _, _ = _grad_pair_disqualified(ref, None)
    assert reason is not None


def test_helper_small_noise_passes():
    ref = _ref_tensor()
    # 1% relative noise -- a legitimate, numerically-correct candidate.
    cand = ref + 0.01 * ref.abs().mean() * torch.randn_like(ref)
    reason, rdiff, _ = _grad_pair_disqualified(ref, cand)
    assert reason is None
    assert rdiff < 0.25


def test_helper_garbage_disqualified():
    ref = _ref_tensor()
    # 10x off everywhere -- garbage.
    cand = ref * 10.0
    reason, rdiff, _ = _grad_pair_disqualified(ref, cand)
    assert reason is not None
    assert rdiff > 0.25


def test_helper_fp16_accum_like_passes():
    ref = _ref_tensor()
    # ~2e-2 mean-relative difference, the ballpark of an fp16-accumulator tile.
    scale = 2e-2 * ref.abs().mean()
    cand = ref + scale * torch.randn_like(ref)
    reason, rdiff, _ = _grad_pair_disqualified(ref, cand)
    assert reason is None
    assert rdiff < 0.25


def test_helper_nan_candidate_disqualified():
    ref = _ref_tensor()
    cand = ref.clone()
    cand[0, 0] = float("nan")
    reason, _, _ = _grad_pair_disqualified(ref, cand)
    assert reason is not None


def test_helper_zero_reference_skips():
    # No signal in the reference -> nothing to check, candidate always passes.
    ref = torch.zeros(32, 8)
    cand = torch.randn(32, 8)
    reason, _, _ = _grad_pair_disqualified(ref, cand)
    assert reason is None


def test_combined_only_checks_requested_direction():
    ref = _ref_tensor()
    bad = torch.zeros_like(ref)
    # dgrad sweep: grad_in checked, grad_weight ignored (garbage weight slot ok).
    reason = _backward_numeric_disqualifies(
        (ref, torch.zeros_like(ref)), (bad, ref), needs_input_grad=(True, False)
    )
    assert reason is not None and reason.startswith("grad_in")
    # wgrad sweep: grad_weight checked, grad_in ignored.
    reason = _backward_numeric_disqualifies(
        (torch.zeros_like(ref), ref), (ref, bad), needs_input_grad=(False, True)
    )
    assert reason is not None and reason.startswith("grad_weight")


# ---------------------------------------------------------------------------
# Forward numeric self-check
#
# A GEMM tile can be wrong rather than merely slow, and timing alone cannot
# tell the difference. The forward check exists because three mask_gemm
# production tiles were measured returning wrong forward output on GB300.
# ---------------------------------------------------------------------------


def test_forward_zero_candidate_disqualified():
    ref = _ref_tensor()
    assert _forward_numeric_disqualifies(ref, torch.zeros_like(ref)) is not None


def test_forward_nan_candidate_disqualified():
    ref = _ref_tensor()
    bad = ref.clone()
    bad[3, 7] = float("nan")
    assert _forward_numeric_disqualifies(ref, bad) is not None


def test_forward_inf_candidate_disqualified():
    ref = _ref_tensor()
    bad = ref.clone()
    bad[5, 1] = float("inf")
    assert _forward_numeric_disqualifies(ref, bad) is not None


def test_forward_small_noise_passes():
    ref = _ref_tensor()
    # 1% relative perturbation — a legitimate difference in accumulation order.
    cand = ref * 1.01
    assert _forward_numeric_disqualifies(ref, cand) is None


def test_forward_sparse_corruption_disqualified():
    # The GB300 tile-41 failure mode: a handful of rows read many times the
    # reference while the rest are exact. The mean-relative criterion inherited
    # from the backward check cannot see this, so the max-element bound must.
    ref = _ref_tensor()
    cand = ref.clone()
    cand[0, 0] = ref.abs().max() * 11.0
    assert _grad_pair_disqualified(ref, cand)[0] is None, "mean criterion should miss this"
    reason = _forward_numeric_disqualifies(ref, cand)
    assert reason is not None and "max element" in reason


def test_forward_max_elem_threshold_is_loose_enough_for_fp16():
    # fp16 kernels differ from the reference by ~1e-2 at worst; the bound must
    # sit far above that or good candidates get disqualified.
    assert _NUMERIC_MAX_ELEM_RDIFF >= 0.1
    ref = _ref_tensor()
    cand = ref + torch.randn_like(ref) * ref.abs().max() * 0.02
    assert _forward_numeric_disqualifies(ref, cand) is None


def test_forward_zero_reference_skips():
    ref = torch.zeros(32, 16)
    # No signal to validate against — must not disqualify anything.
    assert _forward_numeric_disqualifies(ref, torch.randn(32, 16)) is None


# ---------------------------------------------------------------------------
# Integration test: a real backward autotune sweep with an injected zero candidate
# ---------------------------------------------------------------------------


@pytest.fixture
def scoped_benchmark_cache(monkeypatch, tmp_path):
    """Point the benchmark cache at an empty tmp dir so tests never touch
    ~/.cache/warpconvnet, and reset the lazy singleton around the test."""
    import warpconvnet.utils.benchmark_cache as bc

    monkeypatch.setattr(bc, "WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE", str(tmp_path))
    monkeypatch.setattr(bc, "_generic_benchmark_cache", None)
    yield tmp_path
    bc._generic_benchmark_cache = None


def _build_backward_probe(C_in: int = 64, C_out: int = 64):
    """Small stride-1 3x3x3 sparse conv backward probe."""
    torch.manual_seed(0)
    device = "cuda"
    coords = [(torch.rand((400, 3)) / 0.1).int()]
    features = [torch.rand((400, C_in))]
    voxels = Voxels(coords, features, device=device).unique()

    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weight = (torch.randn(num_kernels, C_in, C_out, device=device) * 0.05).contiguous()

    bic = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    kernel_map = generate_kernel_map(bic, bic, stride, kernel_size)
    num_out_coords = bic.shape[0]

    in_features = voxels.feature_tensor.contiguous()
    grad_output = torch.randn(num_out_coords, C_out, device=device) * 0.05
    return (
        grad_output,
        in_features,
        weight,
        kernel_map,
        num_out_coords,
        torch.device(device),
    )


_ZERO_ALGO = "zzz_injected_zero_dgrad"


@pytest.fixture
def autotune_warnings():
    """Capture WARNING+ records from the autotune logger directly (it has
    propagate=False, so pytest's caplog root handler never sees them)."""
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collector(level=logging.WARNING)
    target = autotune.logger.logger
    target.addHandler(handler)
    try:
        yield records
    finally:
        target.removeHandler(handler)


@pytest.fixture
def inject_zero_dgrad_backend(monkeypatch):
    """Register a fake backward backend that runs fine but returns an all-zero
    grad_in -- the silent zero-grad failure mode. Test-scoped only."""

    def _fake_zero_backward(ctx):
        grad_in = torch.zeros_like(ctx.in_features) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(ctx.weight) if ctx.needs_input_grad[1] else None
        return grad_in, grad_weight

    patched = dict(backends.BACKWARD_BACKENDS)
    patched[_ZERO_ALGO] = _fake_zero_backward
    monkeypatch.setattr(backends, "BACKWARD_BACKENDS", patched)
    return _ZERO_ALGO


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zero_candidate_disqualified_in_sweep(
    scoped_benchmark_cache, inject_zero_dgrad_backend, autotune_warnings, monkeypatch
):
    monkeypatch.setattr(autotune, "WARPCONVNET_AUTOTUNE_NUMERIC_CHECK", True)
    grad_output, in_features, weight, kernel_map, num_out_coords, device = _build_backward_probe()

    # dgrad-only sweep. The injected zero candidate returns instantly, so on
    # timing alone it would win -- the numeric check must reject it.
    custom_params = [("explicit_gemm", {}), (_ZERO_ALGO, {})]
    results = _run_backward_benchmarks(
        grad_output,
        in_features,
        weight,
        kernel_map,
        num_out_coords,
        None,
        device,
        custom_params=custom_params,
        needs_input_grad=(True, False),
    )

    algos = [algo for algo, _, _ in results]
    # Zero candidate must not appear at all (disqualified == skipped).
    assert _ZERO_ALGO not in algos
    # A good candidate still wins and is available to be cached.
    assert len(results) >= 1
    assert results[0][0] == "explicit_gemm"
    # Disqualification was logged at WARNING with the algo name.
    assert any(
        _ZERO_ALGO in rec.getMessage() and "DISQUALIFIED" in rec.getMessage()
        for rec in autotune_warnings
        if rec.levelno == logging.WARNING
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_env_escape_hatch_disables_check(
    scoped_benchmark_cache, inject_zero_dgrad_backend, monkeypatch
):
    # With the check disabled, the zero candidate is no longer filtered out and
    # (being instant) survives into the results.
    monkeypatch.setattr(autotune, "WARPCONVNET_AUTOTUNE_NUMERIC_CHECK", False)
    grad_output, in_features, weight, kernel_map, num_out_coords, device = _build_backward_probe()

    custom_params = [("explicit_gemm", {}), (_ZERO_ALGO, {})]
    results = _run_backward_benchmarks(
        grad_output,
        in_features,
        weight,
        kernel_map,
        num_out_coords,
        None,
        device,
        custom_params=custom_params,
        needs_input_grad=(True, False),
    )
    algos = [algo for algo, _, _ in results]
    assert _ZERO_ALGO in algos


def _build_wgrad_overflow_probe(scale: float = 50.0):
    """Small stride-1 3x3x3 wgrad probe whose fp16 accumulation overflows to
    inf: coherent-sign (all-positive) grad_output scaled up so the per-tap sum
    over tens of thousands of terms exceeds fp16's ~65504 range -- the real AMP
    failure mode where the explicit_gemm reference overflows identically to
    every candidate."""
    torch.manual_seed(0)
    device = "cuda"
    N, C_in, C_out = 40_000, 32, 32
    coords = [(torch.rand((N, 3)) / 0.05).int()]
    features = [torch.rand((N, C_in))]
    voxels = Voxels(coords, features, device=device).unique()

    kernel_size = (3, 3, 3)
    num_kernels = 27
    weight = (torch.randn(num_kernels, C_in, C_out, device=device) * 0.05).half()

    bic = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    kernel_map = generate_kernel_map(bic, bic, (1, 1, 1), kernel_size)
    num_out_coords = bic.shape[0]

    in_features = voxels.feature_tensor.contiguous().half()
    # Coherent (all-positive) grad_output so the accumulation does not cancel.
    grad_output = (torch.ones(num_out_coords, C_out, device=device) * scale).half()
    return (
        grad_output,
        in_features,
        weight,
        kernel_map,
        num_out_coords,
        torch.device(device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_wgrad_fp16_overflow_no_false_positive(
    scoped_benchmark_cache, autotune_warnings, monkeypatch
):
    # Regression: the guard must NOT disqualify healthy wgrad candidates when
    # fp16 accumulation legitimately overflows to inf (normal AMP behaviour).
    monkeypatch.setattr(autotune, "WARPCONVNET_AUTOTUNE_NUMERIC_CHECK", True)
    grad_output, in_features, weight, kernel_map, num_out_coords, device = (
        _build_wgrad_overflow_probe()
    )

    # Precondition: the explicit_gemm reference really does overflow here.
    ref_ctx = backends.BwdCtx(
        grad_output=grad_output,
        in_features=in_features,
        weight=weight,
        kernel_map=kernel_map,
        num_out_coords=num_out_coords,
        compute_dtype=torch.float16,
        device=device,
        needs_input_grad=(False, True),
        params={},
    )
    _, ref_gw = backends.run_backward("explicit_gemm", ref_ctx)
    assert not torch.isfinite(ref_gw).all().item(), "probe did not overflow; adjust scale"

    custom_params = [("explicit_gemm", {}), ("mask_gemm", {})]
    results = _run_backward_benchmarks(
        grad_output,
        in_features,
        weight,
        kernel_map,
        num_out_coords,
        torch.float16,
        device,
        custom_params=custom_params,
        needs_input_grad=(False, True),
    )

    # No candidate disqualified, and a real candidate still wins.
    assert not any("DISQUALIFIED" in rec.getMessage() for rec in autotune_warnings), [
        rec.getMessage() for rec in autotune_warnings
    ]
    assert len(results) >= 1
    assert results[0][0] in ("explicit_gemm", "mask_gemm")
    # The check should have logged exactly one "disabled for this sweep" notice
    # citing the non-finite reference.
    disabled = [
        rec.getMessage()
        for rec in autotune_warnings
        if "self-check disabled for this sweep" in rec.getMessage()
    ]
    assert len(disabled) == 1 and "non-finite" in disabled[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fail_open_when_all_candidates_disqualified(
    scoped_benchmark_cache, inject_zero_dgrad_backend, autotune_warnings, monkeypatch
):
    # Structural fail-open: even with a healthy (finite, nonzero) reference, if
    # EVERY candidate disqualifies the check disables itself and the rejected
    # candidates are re-timed so selection never collapses to nothing.
    monkeypatch.setattr(autotune, "WARPCONVNET_AUTOTUNE_NUMERIC_CHECK", True)
    grad_output, in_features, weight, kernel_map, num_out_coords, device = _build_backward_probe()

    # Only zero-returning candidates in the pool (the reference explicit_gemm is
    # computed internally and is healthy). All candidates disqualify.
    custom_params = [(_ZERO_ALGO, {})]
    results = _run_backward_benchmarks(
        grad_output,
        in_features,
        weight,
        kernel_map,
        num_out_coords,
        None,
        device,
        custom_params=custom_params,
        needs_input_grad=(True, False),
    )

    # Fell open: the zero candidate was re-timed rather than dropped, so we get a
    # non-empty result set instead of collapsing to the explicit_gemm fallback.
    algos = [algo for algo, _, _ in results]
    assert _ZERO_ALGO in algos
    disabled = [
        rec.getMessage()
        for rec in autotune_warnings
        if "self-check disabled for this sweep" in rec.getMessage()
    ]
    assert len(disabled) == 1 and "disqualified" in disabled[0].lower()
