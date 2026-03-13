# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Correctness tests for the cute_grouped_sm90 sparse convolution algorithm.
#
# Compares SM90 WGMMA grouped forward and backward against explicit_gemm reference.

import pytest
import torch

try:
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped_sm90 import (
        _cute_grouped_sm90_forward_logic,
        _cute_grouped_sm90_backward_logic,
    )
    from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
        _explicit_gemm_forward_logic,
        _explicit_gemm_backward_logic,
    )
    from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
    from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
    from warpconvnet.geometry.types.voxels import Voxels

    HAS_SM90 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
    )
except ImportError:
    HAS_SM90 = False

pytestmark = pytest.mark.skipif(
    not HAS_SM90,
    reason="Requires SM90+ hardware and cute_grouped_sm90 backend",
)


def _make_test_data(N, C_in, C_out, kernel_size=3, dtype=torch.float32, seed=42):
    """Create test voxels, weights, and kernel map."""
    torch.manual_seed(seed)
    coords = torch.randint(0, max(20, N // 5), (N, 3), device="cuda", dtype=torch.int32)
    coords = torch.unique(coords, dim=0)
    N_actual = coords.shape[0]
    feats = torch.randn(N_actual, C_in, device="cuda", dtype=dtype)
    offsets = torch.tensor([0, N_actual], dtype=torch.long, device="cuda")
    vox = Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        voxel_size=1.0,
    )
    K = kernel_size**3
    weight = torch.randn(K, C_in, C_out, device="cuda", dtype=dtype) * 0.01
    batch_coords = batch_indexed_coordinates(vox.coordinate_tensor, offsets)
    stride = tuple([1] * 3)
    ksize = tuple([kernel_size] * 3)
    kmap = generate_kernel_map(batch_coords, batch_coords, stride, ksize)
    return feats, weight, kmap, N_actual


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mma_tile", [100, 101, 104])
@pytest.mark.parametrize("N,C", [(5, 8), (100, 32), (500, 64)])
def test_sm90_grouped_forward_vs_explicit(N, C, mma_tile):
    """SM90 grouped forward must match explicit_gemm reference."""
    feats, weight, kmap, N_actual = _make_test_data(N, C, C)
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    out = _cute_grouped_sm90_forward_logic(feats, weight, kmap, N_actual, mma_tile=mma_tile)
    assert not isinstance(out, int), f"SM90 grouped returned error status {out}"
    assert not out.isnan().any(), "SM90 grouped output contains NaN"
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("C", [8, 16, 32, 64, 128])
def test_sm90_grouped_forward_channel_widths(C):
    """Test across different channel widths."""
    feats, weight, kmap, N_actual = _make_test_data(200, C, C)
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    out = _cute_grouped_sm90_forward_logic(feats, weight, kmap, N_actual, mma_tile=100)
    assert not isinstance(out, int), f"SM90 grouped returned error status {out}"
    assert not out.isnan().any(), "SM90 grouped output contains NaN"
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)


def test_sm90_grouped_forward_asymmetric_channels():
    """Test C_in != C_out."""
    feats, weight, kmap, N_actual = _make_test_data(200, 32, 64)
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    out = _cute_grouped_sm90_forward_logic(feats, weight, kmap, N_actual, mma_tile=100)
    assert not isinstance(out, int), f"SM90 grouped returned error status {out}"
    assert not out.isnan().any(), "SM90 grouped output contains NaN"
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)


def test_sm90_grouped_forward_large_scale():
    """Large-scale test to catch NaN that only appears at scale."""
    feats, weight, kmap, N_actual = _make_test_data(5000, 64, 64)
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    out = _cute_grouped_sm90_forward_logic(feats, weight, kmap, N_actual, mma_tile=101)
    assert not isinstance(out, int), f"SM90 grouped returned error status {out}"
    assert not out.isnan().any(), f"NaN in output ({out.isnan().sum().item()} NaN values)"
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mma_tile", [100, 101, 104])
def test_sm90_grouped_backward_grad_input(mma_tile):
    """SM90 grouped backward input gradient vs explicit_gemm reference."""
    feats, weight, kmap, N_actual = _make_test_data(200, 32, 32)
    ref_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    grad_output = torch.randn_like(ref_out)

    result = _cute_grouped_sm90_backward_logic(
        grad_output, feats, weight, kmap, (True, False), mma_tile=mma_tile
    )
    assert not isinstance(result[0], int), f"SM90 backward failed: {result}"
    grad_in, _ = result

    ref_grad_in, _ = _explicit_gemm_backward_logic(grad_output, feats, weight, kmap)

    assert not grad_in.isnan().any(), "Backward grad_input contains NaN"
    torch.testing.assert_close(grad_in, ref_grad_in, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("mma_tile", [100, 101, 104])
def test_sm90_grouped_backward_grad_weight(mma_tile):
    """SM90 grouped backward weight gradient vs explicit_gemm reference."""
    feats, weight, kmap, N_actual = _make_test_data(200, 32, 32)
    ref_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    grad_output = torch.randn_like(ref_out)

    result = _cute_grouped_sm90_backward_logic(
        grad_output, feats, weight, kmap, (False, True), mma_tile=mma_tile
    )
    assert not isinstance(result[1], int), f"SM90 backward failed: {result}"
    _, grad_w = result

    _, ref_grad_w = _explicit_gemm_backward_logic(grad_output, feats, weight, kmap)

    assert not grad_w.isnan().any(), "Backward grad_weight contains NaN"
    torch.testing.assert_close(grad_w, ref_grad_w, atol=0.05, rtol=0.05)


def test_sm90_grouped_backward_both_grads():
    """SM90 grouped backward with both input and weight gradients."""
    feats, weight, kmap, N_actual = _make_test_data(200, 32, 32)
    ref_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    grad_output = torch.randn_like(ref_out)

    result = _cute_grouped_sm90_backward_logic(
        grad_output, feats, weight, kmap, (True, True), mma_tile=100
    )
    assert not isinstance(result[0], int), f"SM90 backward failed: {result}"
    grad_in, grad_w = result

    ref_grad_in, ref_grad_w = _explicit_gemm_backward_logic(grad_output, feats, weight, kmap)

    assert not grad_in.isnan().any(), "Backward grad_input contains NaN"
    assert not grad_w.isnan().any(), "Backward grad_weight contains NaN"
    torch.testing.assert_close(grad_in, ref_grad_in, atol=0.05, rtol=0.05)
    torch.testing.assert_close(grad_w, ref_grad_w, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# Backward at larger channel widths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("C", [32, 64, 128])
def test_sm90_grouped_backward_channel_widths(C):
    """Test backward across different channel widths."""
    feats, weight, kmap, N_actual = _make_test_data(200, C, C)
    ref_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    grad_output = torch.randn_like(ref_out)

    result = _cute_grouped_sm90_backward_logic(
        grad_output, feats, weight, kmap, (True, True), mma_tile=100
    )
    assert not isinstance(result[0], int), f"SM90 backward failed: {result}"
    grad_in, grad_w = result

    ref_grad_in, ref_grad_w = _explicit_gemm_backward_logic(grad_output, feats, weight, kmap)

    assert not grad_in.isnan().any(), "Backward grad_input contains NaN"
    assert not grad_w.isnan().any(), "Backward grad_weight contains NaN"
    torch.testing.assert_close(grad_in, ref_grad_in, atol=0.05, rtol=0.05)
    torch.testing.assert_close(grad_w, ref_grad_w, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# Large-scale backward
# ---------------------------------------------------------------------------

def test_sm90_grouped_backward_large_scale():
    """Large-scale backward test."""
    feats, weight, kmap, N_actual = _make_test_data(5000, 64, 64)
    ref_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    grad_output = torch.randn_like(ref_out)

    result = _cute_grouped_sm90_backward_logic(
        grad_output, feats, weight, kmap, (True, True), mma_tile=101
    )
    assert not isinstance(result[0], int), f"SM90 backward failed: {result}"
    grad_in, grad_w = result

    ref_grad_in, ref_grad_w = _explicit_gemm_backward_logic(grad_output, feats, weight, kmap)

    assert not grad_in.isnan().any(), "Backward grad_input contains NaN"
    assert not grad_w.isnan().any(), "Backward grad_weight contains NaN"
    torch.testing.assert_close(grad_in, ref_grad_in, atol=0.1, rtol=0.1)
    torch.testing.assert_close(grad_w, ref_grad_w, atol=0.1, rtol=0.1)


# ---------------------------------------------------------------------------
# Multi-batch
# ---------------------------------------------------------------------------

def test_sm90_grouped_forward_backward_multi_batch():
    """Test with multiple batches for both forward and backward."""
    torch.manual_seed(42)
    coords1 = torch.randint(0, 20, (100, 3), device="cuda", dtype=torch.int32)
    coords1 = torch.unique(coords1, dim=0)
    coords2 = torch.randint(0, 20, (150, 3), device="cuda", dtype=torch.int32)
    coords2 = torch.unique(coords2, dim=0)
    N1, N2 = coords1.shape[0], coords2.shape[0]
    N = N1 + N2
    C = 32

    coords = torch.cat([coords1, coords2], dim=0)
    feats = torch.randn(N, C, device="cuda")
    offsets = torch.tensor([0, N1, N], dtype=torch.long, device="cuda")
    vox = Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        voxel_size=1.0,
    )
    weight = torch.randn(27, C, C, device="cuda") * 0.01
    batch_coords = batch_indexed_coordinates(vox.coordinate_tensor, offsets)
    kmap = generate_kernel_map(batch_coords, batch_coords, (1, 1, 1), (3, 3, 3))

    # Forward
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N)
    out = _cute_grouped_sm90_forward_logic(feats, weight, kmap, N, mma_tile=100)
    assert not isinstance(out, int), f"SM90 grouped returned error status {out}"
    assert not out.isnan().any(), "SM90 grouped output contains NaN"
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)

    # Backward
    grad_output = torch.randn_like(ref)
    result = _cute_grouped_sm90_backward_logic(
        grad_output, feats, weight, kmap, (True, True), mma_tile=100
    )
    assert not isinstance(result[0], int), f"SM90 backward failed: {result}"
    ref_grad_in, ref_grad_w = _explicit_gemm_backward_logic(grad_output, feats, weight, kmap)
    torch.testing.assert_close(result[0], ref_grad_in, atol=0.05, rtol=0.05)
    torch.testing.assert_close(result[1], ref_grad_w, atol=0.05, rtol=0.05)
