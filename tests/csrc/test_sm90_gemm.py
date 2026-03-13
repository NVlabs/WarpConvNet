# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SM90 WGMMA GEMM kernels (single and grouped)."""

import pytest
import torch
import warpconvnet._C as _C

# Skip entire module if SM90 hardware or compiled support not available
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9),
    reason="Requires SM90+ hardware",
)

_has_sm90_gemm = hasattr(_C.gemm, "cute_gemm_sm90_AD_gather_scatter")
_has_sm90_grouped = hasattr(_C.gemm, "cute_gemm_sm90_grouped_AD_gather_scatter")


def _rand(rows, cols, dtype=torch.float16, device="cuda"):
    return torch.randn(rows, cols, device=device, dtype=dtype) * 0.1


def _reference_gather_scatter_gemm(A, B, in_map, out_map, M_out, alpha=1.0, beta=1.0):
    """Reference: D[out_map] = alpha * A[in_map] @ B (direct write, unique out indices)"""
    gathered = A[in_map].float()
    result = alpha * (gathered @ B.float())
    D = torch.zeros(M_out, B.shape[1], device=A.device, dtype=torch.float32)
    D[out_map.long()] = result
    return D


# ============================================================================
# SM90 Single GEMM tests
# ============================================================================


@pytest.mark.skipif(not _has_sm90_gemm, reason="SM90 GEMM not compiled")
@pytest.mark.parametrize("tile", [100, 101, 103, 104])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,gather_size",
    [
        (512, 64, 64, 256),
        (1024, 128, 128, 512),
        (2048, 64, 256, 1024),
        (4096, 256, 64, 2048),  # Multi k-tile (K=256 > tK=64 → 4 k-tiles)
    ],
)
def test_sm90_single_gemm(tile, dtype, M, K, N, gather_size):
    """Test SM90 single GEMM with gather/scatter against reference."""
    A = _rand(M, K, dtype)
    B = _rand(K, N, dtype)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    D = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    in_map = torch.randint(0, M, (gather_size,), device="cuda", dtype=torch.int32)
    # Unique output indices (single GEMM does direct write, not accumulate)
    out_map = torch.randperm(M, device="cuda")[:gather_size].int()

    status = _C.gemm.cute_gemm_sm90_AD_gather_scatter(
        A, B, C, D, in_map, out_map, mma_tile=tile, alpha=1.0, beta=0.0
    )
    assert status == 0, f"SM90 GEMM failed with status {status}"

    ref = _reference_gather_scatter_gemm(A, B, in_map, out_map, M, alpha=1.0, beta=0.0)
    torch.testing.assert_close(D, ref, atol=0.05, rtol=0.05)


# ============================================================================
# SM90 Grouped GEMM tests
# ============================================================================


@pytest.mark.skipif(not _has_sm90_grouped, reason="SM90 grouped GEMM not compiled")
@pytest.mark.parametrize("tile", [100, 101, 104])
@pytest.mark.parametrize("use_atomic", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_sm90_grouped_gemm(tile, use_atomic, dtype):
    """Test SM90 grouped GEMM with atomic and non-atomic epilogue."""
    torch.manual_seed(42)
    M_A = 2048
    K = 64
    N = 128
    num_groups = 3
    group_sizes_list = [300, 500, 200]

    A = _rand(M_A, K, dtype)
    B_list = [_rand(K, N, dtype) for _ in range(num_groups)]
    D = torch.zeros(M_A, N, device="cuda", dtype=torch.float32)

    # Build maps: for non-atomic test, ensure no duplicate output rows across ALL groups
    in_maps_list = []
    out_maps_list = []
    total_gs = sum(group_sizes_list)
    if not use_atomic:
        # Non-atomic: output rows must be unique across all groups (no race condition)
        all_out = torch.randperm(M_A, device="cuda")[:total_gs].int()
    offset_acc = 0
    for g, gs in enumerate(group_sizes_list):
        in_maps_list.append(torch.randint(0, M_A, (gs,), device="cuda", dtype=torch.int32))
        if use_atomic:
            out_maps_list.append(torch.randint(0, M_A, (gs,), device="cuda", dtype=torch.int32))
        else:
            out_maps_list.append(all_out[offset_acc : offset_acc + gs])
            offset_acc += gs

    in_map = torch.cat(in_maps_list)
    out_map = torch.cat(out_maps_list)

    # Tile M size from tile tag
    tile_m_sizes = {100: 64, 101: 128, 102: 128, 103: 256, 104: 64}
    tile_m = tile_m_sizes[tile]

    # Build GroupedGemmParams tensors
    weight_ptrs = torch.tensor(
        [b.data_ptr() for b in B_list], device="cuda", dtype=torch.int64
    )
    tile_offsets_list = [0]
    map_offsets_list = [0]
    total_m_tiles = 0
    for gs in group_sizes_list:
        ntiles = (gs + tile_m - 1) // tile_m
        total_m_tiles += ntiles
        tile_offsets_list.append(total_m_tiles)
        map_offsets_list.append(map_offsets_list[-1] + gs)

    tile_offsets = torch.tensor(tile_offsets_list, device="cuda", dtype=torch.int32)
    group_sizes_t = torch.tensor(group_sizes_list, device="cuda", dtype=torch.int32)
    map_offsets = torch.tensor(map_offsets_list[:-1], device="cuda", dtype=torch.int32)

    status = _C.gemm.cute_gemm_sm90_grouped_AD_gather_scatter(
        A,
        D,
        in_map,
        out_map,
        weight_ptrs,
        tile_offsets,
        group_sizes_t,
        map_offsets,
        total_m_tiles,
        tile,
        1.0,
        use_atomic,
    )
    assert status == 0, f"SM90 grouped GEMM failed with status {status}"

    # Reference
    D_ref = torch.zeros(M_A, N, device="cuda", dtype=torch.float32)
    offset = 0
    for g, gs in enumerate(group_sizes_list):
        im = in_map[offset : offset + gs].long()
        om = out_map[offset : offset + gs].long()
        gathered = A[im].float()
        result = gathered @ B_list[g].float()
        D_ref.index_add_(0, om, result)
        offset += gs

    torch.testing.assert_close(D, D_ref, atol=0.1, rtol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
