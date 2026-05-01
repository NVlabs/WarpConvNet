# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permutohedral lattice filter tests.

We compare against a brute-force Gaussian filter on small inputs:
    out[i] = sum_j w_ij * f_j / sum_j w_ij
    w_ij   = exp(-||(p_i - p_j)/sigma||^2 / 2)

Because the permutohedral filter is an *approximation* (3-tap blur per axis,
plus piecewise-linear barycentric reconstruction), we use loose tolerances
relative to the reference. The test is correctness-of-shape, not bitwise
equivalence.
"""

import numpy as np
import pytest
import torch

from warpconvnet.nn.functional.permutohedral import (
    PermutohedralLattice,
    permutohedral_filter,
    _embed_lattice,
    _find_enclosing_simplex,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _ref_gaussian_filter(positions, features, sigma):
    """Brute-force normalized gaussian filter."""
    pos_s = positions / sigma
    d_sq = ((pos_s.unsqueeze(0) - pos_s.unsqueeze(1)) ** 2).sum(-1)  # (N, N)
    w = torch.exp(-0.5 * d_sq)  # (N, N)
    num = w @ features
    den = w.sum(-1, keepdim=True).clamp_min(1e-20)
    return num / den


def test_embed_sums_to_zero(device):
    torch.manual_seed(0)
    N, d = 100, 5
    p = torch.randn(N, d, device=device)
    e = _embed_lattice(p)
    assert e.shape == (N, d + 1)
    # By construction sum should be exactly 0 (up to float error).
    s = e.sum(-1)
    torch.testing.assert_close(s, torch.zeros_like(s), atol=1e-4, rtol=0.0)


def test_simplex_consistency(device):
    """Greedy + barycentric should reproduce a valid simplex assignment."""
    torch.manual_seed(0)
    N, d = 200, 4
    p = torch.randn(N, d, device=device).float() * 2.0
    elevated = _embed_lattice(p)
    greedy, rank, bary = _find_enclosing_simplex(elevated)

    # Sum of greedy should equal 0 along the lattice axis (so it lies on H_d).
    sum_g = greedy.sum(-1)
    assert (sum_g == 0).all(), f"greedy sum != 0 for some inputs: {sum_g}"

    # Rank should be a permutation of 0..d for each point.
    sorted_rank, _ = rank.sort(-1)
    expected = torch.arange(d + 1, device=device).unsqueeze(0).expand(N, d + 1)
    assert (sorted_rank == expected).all(), "rank is not a permutation"

    # Barycentric weights should sum to 1.
    bsum = bary.sum(-1)
    torch.testing.assert_close(bsum, torch.ones_like(bsum), atol=1e-4, rtol=1e-3)
    # All barys in [0, 1].
    assert (bary >= -1e-4).all() and (
        bary <= 1 + 1e-4
    ).all(), f"barycentric out of range, min={bary.min()}, max={bary.max()}"


def test_filter_constant_input_is_constant(device):
    """A constant feature filtered by the permutohedral lattice should still be
    that constant. This catches normalization bugs."""
    torch.manual_seed(1)
    N, d = 300, 3
    p = torch.randn(N, d, device=device).float()
    f = torch.full((N, 4), 2.5, device=device)

    # Run with both pre-built lattice and one-shot helper.
    out = permutohedral_filter(p, f, sigma=1.0)
    torch.testing.assert_close(out, f, atol=2e-2, rtol=2e-2)


def test_filter_close_to_reference_low_d(device):
    """For d=2 and a tight bandwidth, output should be close to brute-force."""
    torch.manual_seed(2)
    N, d = 200, 2
    p = torch.randn(N, d, device=device).float()
    f = torch.randn(N, 3, device=device).float()

    sigma = 0.7
    out = permutohedral_filter(p, f, sigma=sigma)
    ref = _ref_gaussian_filter(p, f, sigma)

    # Permutohedral is an approximation; allow ~10-20% RMS error vs reference.
    err = (out - ref).norm() / ref.norm().clamp_min(1e-6)
    assert err < 0.25, f"permutohedral too far from reference: rel-err={err:.3f}"


def test_lattice_query_at_different_positions(device):
    """build() at src positions, slice at dst positions sampled near src.

    The permutohedral cross-position slice can only gather from vertices
    populated by the src splat. Queries far outside src coverage get zero
    contributions, so for a fair correctness check we sample dst nearby.
    """
    torch.manual_seed(3)
    N, M, d = 200, 50, 3
    p = torch.randn(N, d, device=device).float()
    # Sample dst from the same distribution near src points so cross-coverage
    # is tight; otherwise unpopulated vertices bias the output.
    q = p[torch.randint(0, N, (M,))] + torch.randn(M, d, device=device).float() * 0.05
    f = torch.randn(N, 4, device=device).float()

    sigma = 0.5
    lat = PermutohedralLattice.build(p / sigma)
    out = lat.filter(f, query_positions=q / sigma)
    assert out.shape == (M, 4)

    # Cross-check vs brute force at the new positions.
    pos_s_p = p / sigma
    pos_s_q = q / sigma
    d_sq = ((pos_s_q.unsqueeze(1) - pos_s_p.unsqueeze(0)) ** 2).sum(-1)  # (M, N)
    w = torch.exp(-0.5 * d_sq)
    ref = (w @ f) / w.sum(-1, keepdim=True).clamp_min(1e-20)

    err = (out - ref).norm() / ref.norm().clamp_min(1e-6)
    # Looser threshold: cross-position lattice slicing has more approximation
    # error than self-filtering because of partial vertex coverage.
    assert err < 0.5, f"cross-position permutohedral too far: rel-err={err:.3f}"


def test_bilateral_permutohedral_color_disambiguates(device):
    """Two co-located points with very different colors should NOT be blended.

    With sigma_xyz large (so spatial proximity is dominant) but sigma_feat
    small (color edge preserved), the high-color and low-color clusters
    should retain near-original feature values.
    """
    from warpconvnet.nn.functional.permutohedral import bilateral_permutohedral_filter

    torch.manual_seed(0)
    N = 1000
    # All points clustered tight in xyz (spatial sigma=1 makes them neighbors)
    xyz = torch.randn(N, 3, device=device).float() * 0.1
    # Half red, half blue — well-separated in color (sigma_feat=10 keeps them apart)
    rgb = torch.zeros(N, 3, device=device).float()
    rgb[: N // 2, 0] = 200.0  # red
    rgb[N // 2 :, 2] = 200.0  # blue
    # Use rgb itself as the value to filter — easy to inspect
    val = rgb.clone()

    out = bilateral_permutohedral_filter(
        src_xyz=xyz,
        src_feat=rgb,
        src_value=val,
        sigma_xyz=1.0,
        sigma_feat=10.0,
    )

    # Red half stays mostly red; blue half stays mostly blue.
    red_half_red = out[: N // 2, 0].mean().item()
    red_half_blue = out[: N // 2, 2].mean().item()
    blue_half_red = out[N // 2 :, 0].mean().item()
    blue_half_blue = out[N // 2 :, 2].mean().item()
    assert (
        red_half_red > 150 and red_half_blue < 30
    ), f"red half bled into blue: r={red_half_red}, b={red_half_blue}"
    assert (
        blue_half_blue > 150 and blue_half_red < 30
    ), f"blue half bled into red: r={blue_half_red}, b={blue_half_blue}"


def test_bilateral_permutohedral_xyz_only_blurs_colors(device):
    """Sanity counter-test: with sigma_feat huge (color ignored), the filter
    becomes a plain spatial blur and red/blue clusters DO mix when they sit
    in the same xyz neighborhood."""
    from warpconvnet.nn.functional.permutohedral import bilateral_permutohedral_filter

    torch.manual_seed(0)
    N = 1000
    xyz = torch.randn(N, 3, device=device).float() * 0.1
    rgb = torch.zeros(N, 3, device=device).float()
    rgb[: N // 2, 0] = 200.0
    rgb[N // 2 :, 2] = 200.0
    val = rgb.clone()

    # sigma_feat=1e6 effectively turns the bilateral into pure spatial blur.
    out = bilateral_permutohedral_filter(
        src_xyz=xyz,
        src_feat=rgb,
        src_value=val,
        sigma_xyz=1.0,
        sigma_feat=1e6,
    )
    # Red and blue should both pull toward the cluster average ~100.
    assert out[:, 0].mean().item() < 150, "red did not blur"
    assert out[:, 2].mean().item() < 150, "blue did not blur"


def test_high_dim_bilateral(device):
    """5D bilateral (2D xy + 3D rgb) — the use case shape. Just check it runs
    and the output is finite and shape-correct."""
    torch.manual_seed(4)
    N = 500
    xy = torch.rand(N, 2, device=device)
    rgb = torch.rand(N, 3, device=device)
    pos = torch.cat([xy / 0.05, rgb / 0.1], dim=-1)
    f = rgb.clone()
    lat = PermutohedralLattice.build(pos)
    out = lat.filter(f)
    assert out.shape == (N, 3)
    assert torch.isfinite(out).all(), "non-finite output"
    # Output should still be roughly within rgb range (bilateral preserves intensity)
    assert out.min() >= -0.3 and out.max() <= 1.3, f"out range: [{out.min()}, {out.max()}]"
