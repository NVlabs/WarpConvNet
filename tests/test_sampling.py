# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warpconvnet
from warpconvnet.ops.sampling import farthest_point_sampling


def farthest_point_sampling_py_packed(points, offsets, K):
    """
    Slow Python implementation for verification (packed inputs).
    """
    B = offsets.shape[0] - 1
    device = points.device

    out_idxs = []

    # Loop over batch
    for b in range(B):
        start = offsets[b].item()
        end = offsets[b + 1].item()
        N_i = end - start

        if N_i == 0:
            # Should not happen in tests usually, but handle gracefullly?
            # Our CUDA implementation just does nothing if N_i <= 0, so output uninitialized?
            # We assume N_i > 0
            continue

        pts = points[start:end]
        temp_dists = torch.full((N_i,), float("inf"), device=device)

        batch_out_idxs = []

        # Initialize
        old_local = 0
        batch_out_idxs.append(old_local + start)

        for k in range(1, K):
            dists = torch.sum((pts - pts[old_local]) ** 2, dim=1)
            temp_dists = torch.min(temp_dists, dists)

            old_local = torch.argmax(temp_dists).item()
            batch_out_idxs.append(old_local + start)

        out_idxs.extend(batch_out_idxs)

    return torch.tensor(out_idxs, dtype=torch.long, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_farthest_point_sampling_packed():
    # Construct packed batch
    # B=2, varying N_i
    N1 = 128
    N2 = 200
    K = 32

    pts1 = torch.randn(N1, 3, device="cuda", dtype=torch.float32)
    pts2 = (
        torch.randn(N2, 3, device="cuda", dtype=torch.float32) + 10.0
    )  # Shift to ensure distinctness

    points = torch.cat([pts1, pts2], dim=0)
    offsets = torch.tensor([0, N1, N1 + N2], device="cuda", dtype=torch.int32)
    B = 2

    # Run CUDA implementation
    idxs_cuda = farthest_point_sampling(points, offsets, K)

    # Run Python implementation
    idxs_ref = farthest_point_sampling_py_packed(points, offsets, K)

    assert idxs_cuda.shape == (B * K,)
    assert idxs_cuda.dtype == torch.int32

    # Check if indices match (mostly)
    # Using allclose might not work for indices, check equality
    assert torch.allclose(idxs_cuda.long(), idxs_ref, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fps_packed_shapes():
    B = 3
    Ns = [10, 20, 15]
    K = 5
    pts_list = [torch.rand(n, 3, device="cuda") for n in Ns]
    points = torch.cat(pts_list, dim=0)

    offsets_list = [0]
    curr = 0
    for n in Ns:
        curr += n
        offsets_list.append(curr)
    offsets = torch.tensor(offsets_list, device="cuda", dtype=torch.int32)

    idxs = farthest_point_sampling(points, offsets, K)

    assert idxs.shape == (B * K,)

    # Check bounds per batch
    for b in range(B):
        start = offsets[b].item()
        end = offsets[b + 1].item()
        batch_idxs = idxs[b * K : (b + 1) * K]

        assert batch_idxs.max() < end
        assert batch_idxs.min() >= start

        # Check uniqueness
        assert len(torch.unique(batch_idxs)) == K
