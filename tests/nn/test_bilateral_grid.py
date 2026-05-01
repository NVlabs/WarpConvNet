# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the regular bilateral grid + Fast Bilateral Solver."""

import pytest
import torch

from warpconvnet.nn.functional.bilateral_grid import (
    BilateralGrid,
    bilateral_filter_grid,
    bilateral_solver,
    fast_bilateral_solver,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_constant_input_filter(device):
    """A constant feature should filter to itself (with `normalize=True`)."""
    torch.manual_seed(0)
    N, d = 256, 4
    p = torch.randn(N, d, device=device).float()
    f = torch.full((N, 3), 7.5, device=device)

    grid = BilateralGrid.build(p)
    out = grid.filter(f, normalize=True)
    torch.testing.assert_close(out, f, atol=1e-3, rtol=1e-3)


def test_splat_slice_partition_of_unity(device):
    """Splatting all-ones — d-linear weights per row sum to 1, so total mass = N."""
    torch.manual_seed(1)
    N, d = 200, 3
    p = torch.randn(N, d, device=device).float()
    grid = BilateralGrid.build(p)
    ones_in = torch.ones(N, 1, device=device)
    splatted = grid.splat(ones_in)  # (V, 1)  — counts in cells
    expected = torch.tensor(float(N), device=device)
    torch.testing.assert_close(splatted.sum().float(), expected, atol=1e-2, rtol=1e-3)
    # Slicing the splat: for each input, (sum over corners w_c * count_c).
    # For a single point: sum_c w_c * w_c = sum w_c² ≠ 1 in general.
    # That is not a partition-of-unity test on slice alone — we test the
    # combination filter with normalize=True separately.


def test_filter_close_to_brute_force(device):
    """For low d and tight bandwidth, normalized filter ≈ Gaussian bilateral on small N."""
    torch.manual_seed(2)
    N, d = 200, 2
    p = torch.randn(N, d, device=device).float()
    f = torch.randn(N, 3, device=device).float()

    sigma = 1.0
    grid = BilateralGrid.build(p / sigma)
    out = grid.filter(f, normalize=True)

    pos_s = p / sigma
    d_sq = ((pos_s.unsqueeze(0) - pos_s.unsqueeze(1)) ** 2).sum(-1)
    w = torch.exp(-0.5 * d_sq)
    ref = (w @ f) / w.sum(-1, keepdim=True).clamp_min(1e-20)

    err = (out - ref).norm() / ref.norm().clamp_min(1e-6)
    # Regular grid is a fairly coarse approximation of the Gaussian; loose threshold
    assert err < 0.5, f"grid bilateral too far from reference: rel-err={err:.3f}"


def test_solver_matches_observation_with_high_confidence(device):
    """High confidence + low lam: solver should recover target.

    Each point gets its own widely-spaced voxel so cells aren't shared.
    With conf >> lam, A ≈ diag(C̄), y ≈ t̄ = target (per cell), slice → target.
    """
    torch.manual_seed(3)
    N, d = 50, 3
    # Spread points across a 50-unit-wide cube, voxel size 1 → each point
    # alone in its voxel.
    p = torch.randn(N, d, device=device).float() * 10.0
    target = torch.randn(N, 2, device=device).float()
    conf = torch.full((N,), 1e6, device=device)

    grid = BilateralGrid.build(p)
    # Skip bistochastization at extreme confidence (Sinkhorn iteration would
    # blow up before the data term dominates).
    out = bilateral_solver(
        grid,
        target=target,
        confidence=conf,
        lam=1.0,
        max_iters=80,
        bistochastize=False,
    )
    rel = (out - target).norm() / target.norm().clamp_min(1e-6)
    assert rel < 0.2, f"high-confidence solver did not recover target: rel-err={rel:.3f}"


def test_solver_smooths_low_confidence(device):
    """A single high-confidence outlier should bleed into low-confidence neighbors."""
    torch.manual_seed(4)
    N, d = 100, 2
    # All points clustered tightly together (one neighborhood)
    p = torch.randn(N, d, device=device).float() * 0.1
    target = torch.zeros(N, 1, device=device).float()
    target[0] = 10.0
    conf = torch.full((N,), 1e-3, device=device)
    conf[0] = 1.0  # only the outlier is confident
    out = fast_bilateral_solver(
        src_xyz=p[:, :d],
        src_feat=torch.zeros(N, 0, device=device).float(),
        target=target,
        confidence=conf,
        sigma_xyz=1.0,
        sigma_feat=1.0,  # not used since src_feat is empty
        lam=1.0,
        max_iters=80,
    )
    # All points should converge toward similar (smoothed) values, not the
    # constant 0 of low-conf neighbors. Variance should be small relative to
    # the outlier value.
    spread = out.std()
    assert out.mean() > 0.1, f"Expected outlier to bleed into neighbors, got mean={out.mean()}"
    assert spread < 4.0, f"Expected smooth output, got spread={spread}"


def test_bilateral_filter_grid_shape(device):
    torch.manual_seed(5)
    N = 500
    xyz = torch.rand(N, 3, device=device).float()
    rgb = torch.rand(N, 3, device=device).float() * 255.0
    val = torch.randn(N, 4, device=device).float()
    out = bilateral_filter_grid(xyz, rgb, val, sigma_xyz=0.05, sigma_feat=30.0)
    assert out.shape == (N, 4)
    assert torch.isfinite(out).all()
