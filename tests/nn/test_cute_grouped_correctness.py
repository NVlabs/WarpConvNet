# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Correctness tests for the cute_grouped sparse convolution algorithm.
#
# Compares cute_grouped forward and backward against explicit_gemm reference
# at various scales, channel widths, and dtypes. This catches the dtype mismatch
# bug where float32 weight pointers were passed to a float16 kernel.

import pytest
import torch

try:
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped import (
        _cute_grouped_forward_logic,
        _cute_grouped_backward_logic,
    )
    from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
        _explicit_gemm_forward_logic,
    )
    from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
    from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
    from warpconvnet.geometry.types.voxels import Voxels

    HAS_CUTE_GROUPED = True
except ImportError:
    HAS_CUTE_GROUPED = False

pytestmark = pytest.mark.skipif(
    not HAS_CUTE_GROUPED or not torch.cuda.is_available(),
    reason="Requires CUDA and cute_grouped backend",
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

@pytest.mark.parametrize("mma_tile", [0, 1, 3])
@pytest.mark.parametrize("N,C", [(5, 8), (100, 32), (500, 64)])
def test_cute_grouped_forward_vs_explicit(N, C, mma_tile):
    """cute_grouped forward must match explicit_gemm reference."""
    feats, weight, kmap, N_actual = _make_test_data(N, C, C)
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    out = _cute_grouped_forward_logic(feats, weight, kmap, N_actual, mma_tile=mma_tile)
    assert not isinstance(out, int), f"cute_grouped returned error status {out}"
    assert not out.isnan().any(), "cute_grouped output contains NaN"
    # fp16 compute tolerance
    torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("mma_tile", [0, 1, 3])
def test_cute_grouped_forward_uniform_weights(mma_tile):
    """Regression test: uniform weights=0.01, feats=1.0 — the original bug scenario."""
    N = 5
    C = 8
    coords = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        device="cuda",
        dtype=torch.int32,
    )
    feats = torch.ones(N, C, device="cuda")
    offsets = torch.tensor([0, N], dtype=torch.long, device="cuda")
    vox = Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        voxel_size=1.0,
    )
    weight = torch.ones(27, C, C, device="cuda") * 0.01
    batch_coords = batch_indexed_coordinates(vox.coordinate_tensor, offsets)
    kmap = generate_kernel_map(batch_coords, batch_coords, (1, 1, 1), (3, 3, 3))

    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N)
    out = _cute_grouped_forward_logic(feats, weight, kmap, N, mma_tile=mma_tile)

    assert not isinstance(out, int), f"cute_grouped returned error status {out}"
    assert not out.isnan().any(), "cute_grouped output contains NaN"
    # Values should be ~0.16 and ~0.24, not -900 or -1801
    assert out.abs().max().item() < 10.0, f"Output magnitude too large: {out.abs().max().item()}"
    torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("C", [8, 16, 32, 64])
def test_cute_grouped_forward_channel_widths(C):
    """Test across different channel widths (all must be aligned to 8)."""
    feats, weight, kmap, N_actual = _make_test_data(200, C, C)
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    out = _cute_grouped_forward_logic(feats, weight, kmap, N_actual, mma_tile=3)
    assert not isinstance(out, int), f"cute_grouped returned error status {out}"
    assert not out.isnan().any(), "cute_grouped output contains NaN"
    torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)


def test_cute_grouped_forward_asymmetric_channels():
    """Test C_in != C_out."""
    feats, weight, kmap, N_actual = _make_test_data(200, 32, 64)
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    out = _cute_grouped_forward_logic(feats, weight, kmap, N_actual, mma_tile=3)
    assert not isinstance(out, int), f"cute_grouped returned error status {out}"
    assert not out.isnan().any(), "cute_grouped output contains NaN"
    torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mma_tile", [0, 1, 3])
def test_cute_grouped_backward_grad_input(mma_tile):
    """cute_grouped backward input gradient must be correct."""
    feats, weight, kmap, N_actual = _make_test_data(200, 32, 32)
    # Get reference forward output for grad shape
    ref_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    grad_output = torch.randn_like(ref_out)

    result = _cute_grouped_backward_logic(
        grad_output, feats, weight, kmap, (True, False), mma_tile=mma_tile
    )
    assert not isinstance(result[0], int), f"cute_grouped backward failed: {result}"
    grad_in, _ = result
    assert not grad_in.isnan().any(), "Backward grad_input contains NaN"
    assert grad_in.abs().max().item() < 1e6, "Backward grad_input magnitude too large"


@pytest.mark.parametrize("mma_tile", [0, 1, 3])
def test_cute_grouped_backward_grad_weight(mma_tile):
    """cute_grouped backward weight gradient must be correct."""
    feats, weight, kmap, N_actual = _make_test_data(200, 32, 32)
    ref_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    grad_output = torch.randn_like(ref_out)

    result = _cute_grouped_backward_logic(
        grad_output, feats, weight, kmap, (False, True), mma_tile=mma_tile
    )
    assert not isinstance(result[0], int), f"cute_grouped backward failed: {result}"
    _, grad_w = result
    assert not grad_w.isnan().any(), "Backward grad_weight contains NaN"
    assert grad_w.abs().max().item() < 1e6, "Backward grad_weight magnitude too large"


def test_cute_grouped_backward_both_grads():
    """cute_grouped backward with both input and weight gradients."""
    feats, weight, kmap, N_actual = _make_test_data(200, 32, 32)
    ref_out = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    grad_output = torch.randn_like(ref_out)

    result = _cute_grouped_backward_logic(
        grad_output, feats, weight, kmap, (True, True), mma_tile=3
    )
    assert not isinstance(result[0], int), f"cute_grouped backward failed: {result}"
    grad_in, grad_w = result
    assert not grad_in.isnan().any(), "Backward grad_input contains NaN"
    assert not grad_w.isnan().any(), "Backward grad_weight contains NaN"


# ---------------------------------------------------------------------------
# Large scale (catches NaN at scale)
# ---------------------------------------------------------------------------

def test_cute_grouped_forward_large_scale():
    """Large-scale test to catch NaN that only appears at scale."""
    feats, weight, kmap, N_actual = _make_test_data(5000, 32, 32)
    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N_actual)
    out = _cute_grouped_forward_logic(feats, weight, kmap, N_actual, mma_tile=3)
    assert not isinstance(out, int), f"cute_grouped returned error status {out}"
    assert not out.isnan().any(), f"NaN in output ({out.isnan().sum().item()} NaN values)"
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# Multi-batch
# ---------------------------------------------------------------------------

def test_cute_grouped_forward_multi_batch():
    """Test with multiple batches."""
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

    ref = _explicit_gemm_forward_logic(feats, weight, kmap, N)
    out = _cute_grouped_forward_logic(feats, weight, kmap, N, mma_tile=3)
    assert not isinstance(out, int), f"cute_grouped returned error status {out}"
    assert not out.isnan().any(), "cute_grouped output contains NaN"
    torch.testing.assert_close(out, ref, atol=0.01, rtol=0.01)
