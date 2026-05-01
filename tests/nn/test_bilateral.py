# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bilateral filter correctness tests.

We compare against a brute-force O(N*M) reference implementation in numpy/torch.
"""

import numpy as np
import pytest
import torch

from warpconvnet.nn.functional.bilateral import bilateral_filter, bilateral_label_propagate


def _ref_bilateral(
    src_xyz,
    src_feat,
    src_value,
    query_xyz,
    query_feat,
    sigma_xyz,
    sigma_feat,
):
    """Brute-force reference (no neighbor truncation)."""
    M = query_xyz.shape[0]
    V = src_value.shape[1]
    out = torch.zeros((M, V), dtype=torch.float32, device=src_xyz.device)
    inv2sx2 = 1.0 / (2 * sigma_xyz * sigma_xyz)
    inv2sf2 = 1.0 / (2 * sigma_feat * sigma_feat)
    for i in range(M):
        d_xyz_sq = ((src_xyz - query_xyz[i : i + 1]) ** 2).sum(-1)
        d_f_sq = ((src_feat - query_feat[i : i + 1]) ** 2).sum(-1)
        w = torch.exp(-d_xyz_sq * inv2sx2 - d_f_sq * inv2sf2)
        out[i] = (w.unsqueeze(-1) * src_value).sum(0) / w.sum().clamp_min(1e-20)
    return out


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def test_bilateral_small_self_filter_matches_reference(device):
    """KNN bilateral with k=N-1 should match a reference that uses the same
    KNN-derived neighborhood. (KNN returns the K closest including self at
    d=0; the reference here mimics that exactly.)
    """
    torch.manual_seed(0)
    N = 64
    xyz = torch.randn(N, 3, device=device)
    rgb = torch.rand(N, 3, device=device) * 255.0
    val = torch.randn(N, 5, device=device)

    sigma_xyz = 0.3
    sigma_feat = 50.0
    k = N - 1

    out = bilateral_filter(
        src_xyz=xyz,
        src_feat=rgb,
        src_value=val,
        sigma_xyz=sigma_xyz,
        sigma_feat=sigma_feat,
        k=k,
        mode="knn",
    )

    # Reference: replicate KNN-then-bilateral exactly.
    inv2sx2 = 1.0 / (2 * sigma_xyz * sigma_xyz)
    inv2sf2 = 1.0 / (2 * sigma_feat * sigma_feat)
    d_xyz_full = ((xyz.unsqueeze(0) - xyz.unsqueeze(1)) ** 2).sum(-1)  # (N, N)
    _, knn_ref = torch.topk(d_xyz_full, k, dim=1, largest=False)
    ref = torch.zeros((N, val.shape[1]), dtype=val.dtype, device=device)
    for i in range(N):
        nbr = knn_ref[i]
        d_xyz_sq = ((xyz[nbr] - xyz[i : i + 1]) ** 2).sum(-1)
        d_f_sq = ((rgb[nbr] - rgb[i : i + 1]) ** 2).sum(-1)
        w = torch.exp(-d_xyz_sq * inv2sx2 - d_f_sq * inv2sf2)
        ref[i] = (w.unsqueeze(-1) * val[nbr]).sum(0) / w.sum().clamp_min(1e-20)

    torch.testing.assert_close(out.float(), ref, atol=1e-4, rtol=1e-3)


def test_bilateral_radius_mode_self_filter(device):
    torch.manual_seed(1)
    N = 200
    xyz = torch.randn(N, 3, device=device).float() * 0.2
    rgb = torch.rand(N, 3, device=device).float() * 255.0
    val = torch.randn(N, 4, device=device).float()

    sigma_xyz = 0.5  # large radius so we capture all points
    sigma_feat = 200.0
    out_radius = bilateral_filter(
        src_xyz=xyz,
        src_feat=rgb,
        src_value=val,
        sigma_xyz=sigma_xyz,
        sigma_feat=sigma_feat,
        mode="radius",
        radius_mult=10.0,  # 10*sigma → contains everything
    )
    ref = _ref_bilateral(xyz, rgb, val, xyz, rgb, sigma_xyz, sigma_feat)

    torch.testing.assert_close(out_radius.float(), ref, atol=1e-3, rtol=1e-2)


def test_bilateral_label_propagate_basic(device):
    """Tight clusters at distinct colors → labels propagate to NN within color."""
    torch.manual_seed(2)
    # Two color-separated clusters at the same xyz location range
    n_per = 32
    xyz_a = torch.randn(n_per, 3, device=device) * 0.1
    xyz_b = torch.randn(n_per, 3, device=device) * 0.1 + 1.0
    rgb_a = torch.full((n_per, 3), 10.0, device=device)
    rgb_b = torch.full((n_per, 3), 240.0, device=device)
    src_xyz = torch.cat([xyz_a, xyz_b], 0)
    src_rgb = torch.cat([rgb_a, rgb_b], 0)
    src_lbl = torch.cat(
        [
            torch.zeros(n_per, dtype=torch.long, device=device),
            torch.ones(n_per, dtype=torch.long, device=device),
        ]
    )

    # Densify: 5x dst points sampled around each cluster (color matched to cluster)
    n_dst = 64
    dst_xyz_a = (
        xyz_a[torch.randint(0, n_per, (n_dst,))] + torch.randn(n_dst, 3, device=device) * 0.01
    )
    dst_xyz_b = (
        xyz_b[torch.randint(0, n_per, (n_dst,))] + torch.randn(n_dst, 3, device=device) * 0.01
    )
    dst_rgb_a = rgb_a[:1].expand(n_dst, 3)
    dst_rgb_b = rgb_b[:1].expand(n_dst, 3)
    dst_xyz = torch.cat([dst_xyz_a, dst_xyz_b], 0)
    dst_rgb = torch.cat([dst_rgb_a, dst_rgb_b], 0)
    expected = torch.cat(
        [
            torch.zeros(n_dst, dtype=torch.long, device=device),
            torch.ones(n_dst, dtype=torch.long, device=device),
        ]
    )

    out = bilateral_label_propagate(
        src_xyz=src_xyz,
        src_feat=src_rgb,
        src_labels=src_lbl,
        dst_xyz=dst_xyz,
        dst_feat=dst_rgb,
        sigma_xyz=0.05,
        sigma_feat=20.0,
        k=8,
    )
    assert (out == expected).all(), f"label propagation mismatch: {out} vs {expected}"


def test_bilateral_label_propagate_color_disambiguates(device):
    """Two overlapping clusters at the SAME xyz, different color. Pure-NN can't
    disambiguate, but bilateral with sigma_feat << color gap can."""
    torch.manual_seed(3)
    n = 50
    xyz_shared = torch.randn(n, 3, device=device) * 0.1
    src_xyz = torch.cat([xyz_shared, xyz_shared], 0)  # exactly overlapping
    src_rgb = torch.cat(
        [
            torch.full((n, 3), 0.0, device=device),  # black
            torch.full((n, 3), 255.0, device=device),  # white
        ]
    )
    src_lbl = torch.cat(
        [
            torch.zeros(n, dtype=torch.long, device=device),
            torch.ones(n, dtype=torch.long, device=device),
        ]
    )

    dst_xyz = xyz_shared.clone()
    dst_rgb_black = torch.full((n, 3), 5.0, device=device)
    dst_rgb_white = torch.full((n, 3), 250.0, device=device)
    out_black = bilateral_label_propagate(
        src_xyz=src_xyz,
        src_feat=src_rgb,
        src_labels=src_lbl,
        dst_xyz=dst_xyz,
        dst_feat=dst_rgb_black,
        sigma_xyz=0.5,
        sigma_feat=10.0,
        k=20,
    )
    out_white = bilateral_label_propagate(
        src_xyz=src_xyz,
        src_feat=src_rgb,
        src_labels=src_lbl,
        dst_xyz=dst_xyz,
        dst_feat=dst_rgb_white,
        sigma_xyz=0.5,
        sigma_feat=10.0,
        k=20,
    )
    assert (out_black == 0).all(), f"black-color queries should get label 0, got {out_black}"
    assert (out_white == 1).all(), f"white-color queries should get label 1, got {out_white}"
