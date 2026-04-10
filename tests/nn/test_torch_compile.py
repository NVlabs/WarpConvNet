# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for torch.compile support.

Verifies that torch.compile(model) works without errors and produces
numerically correct results for the key sparse convolution modules.
"""

import pytest
import torch
import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv


def _make_voxels(B=2, N_range=(500, 1000), C=16, device="cuda"):
    """Create small test Voxels for fast torch.compile tests."""
    torch.manual_seed(42)
    Ns = torch.randint(*N_range, (B,))
    coords = [(torch.rand((int(N.item()), 3)) / 0.02).int() for N in Ns]
    features = [torch.rand((int(N.item()), C)) for N in Ns]
    return Voxels(coords, features, device=device).unique()


class SimpleSparseConvModel(nn.Module):
    """Minimal model: conv -> bias -> relu."""

    def __init__(self, C_in, C_out, kernel_size=3):
        super().__init__()
        self.conv = SpatiallySparseConv(
            C_in,
            C_out,
            kernel_size=kernel_size,
            bias=True,
            fwd_algo="explicit_gemm",
            dgrad_algo="explicit_gemm",
            wgrad_algo="explicit_gemm",
        )
        self.bn = nn.BatchNorm1d(C_out)

    def forward(self, x: Voxels) -> Voxels:
        x = self.conv(x)
        # Apply BN directly to feature tensor and reconstruct
        out_features = self.bn(x.feature_tensor)
        out_features = torch.relu(out_features)
        return x.replace(batched_features=out_features)


class TwoLayerSparseConvModel(nn.Module):
    """Two-layer model with skip connection."""

    def __init__(self, C):
        super().__init__()
        self.conv1 = SpatiallySparseConv(
            C,
            C,
            kernel_size=3,
            bias=True,
            fwd_algo="explicit_gemm",
            dgrad_algo="explicit_gemm",
            wgrad_algo="explicit_gemm",
        )
        self.conv2 = SpatiallySparseConv(
            C,
            C,
            kernel_size=3,
            bias=True,
            fwd_algo="explicit_gemm",
            dgrad_algo="explicit_gemm",
            wgrad_algo="explicit_gemm",
        )

    def forward(self, x: Voxels) -> Voxels:
        residual_features = x.feature_tensor
        x = self.conv1(x)
        x = self.conv2(x)
        # Skip connection on features
        out_features = x.feature_tensor + residual_features
        return x.replace(batched_features=out_features)


@pytest.fixture
def small_voxels():
    return _make_voxels(B=2, N_range=(500, 1000), C=16, device="cuda")


# --------------------------------------------------------------------------
# Test: torch.compile does not crash
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_sparse_conv_no_crash(small_voxels):
    """torch.compile(model) should not raise errors during forward pass."""
    model = SimpleSparseConvModel(16, 32).cuda()
    compiled = torch.compile(model, fullgraph=False)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = compiled(small_voxels)
    assert out.feature_tensor.shape[1] == 32
    assert out.feature_tensor.device.type == "cuda"


# --------------------------------------------------------------------------
# Test: compiled forward matches eager forward
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_matches_eager(small_voxels):
    """Compiled model should produce the same output as eager model."""
    model = SimpleSparseConvModel(16, 32).cuda().eval()
    compiled = torch.compile(model, fullgraph=False)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        eager_out = model(small_voxels)
        compiled_out = compiled(small_voxels)

    torch.testing.assert_close(
        eager_out.feature_tensor,
        compiled_out.feature_tensor,
        atol=1e-3,
        rtol=1e-3,
    )


# --------------------------------------------------------------------------
# Test: compiled backward works
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_backward(small_voxels):
    """Backward pass should work through compiled model."""
    model = SimpleSparseConvModel(16, 32).cuda()
    compiled = torch.compile(model, fullgraph=False)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = compiled(small_voxels)
        loss = out.feature_tensor.sum()
    loss.backward()

    # Check that gradients exist
    assert model.conv.weight.grad is not None
    assert model.conv.weight.grad.shape == model.conv.weight.shape


# --------------------------------------------------------------------------
# Test: two-layer model with skip connection compiles
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_two_layer_model(small_voxels):
    """Multi-layer model with skip connections should compile."""
    model = TwoLayerSparseConvModel(16).cuda()
    compiled = torch.compile(model, fullgraph=False)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = compiled(small_voxels)
        loss = out.feature_tensor.sum()
    loss.backward()

    assert model.conv1.weight.grad is not None
    assert model.conv2.weight.grad is not None


# --------------------------------------------------------------------------
# Test: stride-2 convolution compiles
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_stride2_conv(small_voxels):
    """Stride-2 convolution should work under torch.compile."""
    model = SpatiallySparseConv(
        16,
        32,
        kernel_size=3,
        stride=2,
        bias=True,
        fwd_algo="explicit_gemm",
        dgrad_algo="explicit_gemm",
        wgrad_algo="explicit_gemm",
    ).cuda()
    compiled_model = torch.compile(model, fullgraph=False)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = compiled_model(small_voxels)

    # Stride-2 should reduce the number of output voxels
    assert out.feature_tensor.shape[0] < small_voxels.feature_tensor.shape[0]
    assert out.feature_tensor.shape[1] == 32


# --------------------------------------------------------------------------
# Test: IntSearchResult pytree registration
# --------------------------------------------------------------------------
def test_intsearchresult_pytree():
    """IntSearchResult should be correctly flattened/unflattened by pytree."""
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from warpconvnet.geometry.coords.search.search_results import IntSearchResult

    in_maps = torch.arange(10, device="cuda")
    out_maps = torch.arange(10, device="cuda")
    offsets = torch.tensor([0, 3, 7, 10])
    isr = IntSearchResult(in_maps, out_maps, offsets, identity_map_index=1)

    flat, spec = tree_flatten(isr)
    assert len(flat) == 3  # in_maps, out_maps, offsets

    reconstructed = tree_unflatten(flat, spec)
    assert isinstance(reconstructed, IntSearchResult)
    assert torch.equal(reconstructed.in_maps, in_maps)
    assert torch.equal(reconstructed.out_maps, out_maps)
    assert reconstructed.identity_map_index == 1


# --------------------------------------------------------------------------
# Test: repeated compilation doesn't cause issues
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_multiple_inputs(small_voxels):
    """Compiled model should handle multiple different inputs."""
    model = SimpleSparseConvModel(16, 32).cuda().eval()
    compiled = torch.compile(model, fullgraph=False)

    # First input
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out1 = compiled(small_voxels)

    # Second input with different data
    voxels2 = _make_voxels(B=2, N_range=(500, 1000), C=16, device="cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        out2 = compiled(voxels2)

    assert out1.feature_tensor.shape[1] == 32
    assert out2.feature_tensor.shape[1] == 32
