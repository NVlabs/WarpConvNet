# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import pytest
import torch
import warp as wp

from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.attention import (
    Attention,
    NestedAttention,
    PatchAttention,
    SpatialFeatureAttention,
    ToAttention,
    TransformerBlock,
    ZeroOutPoints,
)
from warpconvnet.nn.encodings import FourierEncoding
from warpconvnet.nn.modules.mlp import Linear


@pytest.fixture
def setup_points():
    """Setup test points with random coordinates."""
    wp.init()
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    B, min_N, max_N, C = 3, 1000, 10000, 128
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features).to(device), B, Ns, C


def test_basic_attention(setup_points):
    """Test basic attention mechanism."""
    pc: Points = setup_points[0]
    B, Ns, C = setup_points[1:]
    device = pc.device

    # Setup attention layers
    dim = C * 8
    num_heads = 8
    lift = Linear(C, dim).to(device)
    to_attn = ToAttention(
        out_channels=dim,
        num_heads=num_heads,
        num_encoding_channels=32,
        encoding_range=1.0,
        concat_input=True,
        num_spatial_features=3,
    ).to(device)
    attn = Attention(dim, num_heads=num_heads, qkv_bias=True).to(device)
    zero_out = ZeroOutPoints()

    # Forward pass
    pc = lift(pc)
    features, pos_enc, mask, num_points = to_attn(pc)
    x = attn(features, pos_enc, mask)
    x = zero_out(x, num_points)

    assert x.shape == (B, Ns.max(), dim)


@pytest.mark.parametrize("use_encoding", [True, False])
def test_to_attention(setup_points, use_encoding):
    """Test ToAttention module with and without positional encoding."""
    pc: Points = setup_points[0]
    B, Ns, C = setup_points[1:]
    device = pc.device

    # Setup layers
    dim = C * 8
    num_heads = 8
    lift = Linear(C, dim).to(device)
    to_attn = ToAttention(
        out_channels=dim,
        use_encoding=use_encoding,
        num_heads=num_heads,
        num_encoding_channels=32,
        encoding_range=1.0,
        concat_input=True,
        num_spatial_features=3,
    ).to(device)

    # Forward pass
    pc = lift(pc)
    features, pos_enc, mask, num_points = to_attn(pc)

    # Verify shapes
    max_N = Ns.max()
    assert features.shape == (B, max_N, dim)
    assert mask.shape == (B, 1, max_N, max_N)

    # Verify mask properties
    for b in range(B):
        assert torch.all(mask[b, :, : num_points[b], : num_points[b]]).item()
        assert not torch.any(mask[b, :, :, num_points[b] :]).item()


def test_patch_attention(setup_points):
    """Test patch-based attention."""
    pc: Points = setup_points[0]
    _, _, C = setup_points[1:]
    device = pc.device

    # Setup layers
    patch_size = 32
    dim = C * 8
    num_heads = 8
    lift = Linear(C, dim).to(device)
    patch_attn = PatchAttention(
        dim=dim,
        patch_size=patch_size,
        num_heads=num_heads,
        qkv_bias=True,
        use_sdpa=False,
    ).to(device)

    # Forward pass
    pc = lift(pc)
    out = patch_attn(pc)

    # Verify output
    assert isinstance(out, Points)
    assert out.feature_tensor.shape[-1] == dim
    assert len(out) == len(pc)


def test_patch_transformer_block(setup_points):
    """Test transformer block with patch attention."""
    pc: Points = setup_points[0]
    _, _, C = setup_points[1:]
    device = pc.device

    # Setup layers
    patch_size = 32
    dim = C * 8
    num_heads = 8
    lift = Linear(C, dim).to(device)
    patch_attn = TransformerBlock(
        dim=dim,
        num_heads=num_heads,
        ffn_multiplier=4,
        qkv_bias=True,
        attn_fn=functools.partial(PatchAttention, patch_size=patch_size),
    ).to(device)

    # Test with points
    pc = lift(pc)
    out = patch_attn(pc)
    assert out.feature_tensor.shape[-1] == dim
    assert len(out) == len(pc)

    # Test with voxels
    st = pc.to_voxels(voxel_size=0.02)
    out = patch_attn(st)
    assert out.feature_tensor.shape[-1] == dim
    assert len(out) == len(st)

    # Test gradient flow
    pc = setup_points[0]
    pc.batched_features.batched_tensor.requires_grad_(True)
    pc = lift(pc)
    out = patch_attn(pc)
    loss = out.feature_tensor.sum()
    loss.backward()

    # Verify gradients
    assert patch_attn.attention.qkv.weight.grad is not None
    assert patch_attn.attention_norm.norm.weight.grad is not None
    assert lift.block.weight.grad is not None


@pytest.mark.inference_mode()
def test_nested_transformer_block(setup_points):
    """Test transformer block with nested attention."""
    pc: Points = setup_points[0]
    _, _, C = setup_points[1:]
    device = pc.device

    # Setup layers
    dim = C * 8
    num_heads = 8
    lift = Linear(C, dim).to(device)
    transf = TransformerBlock(
        dim=dim,
        num_heads=num_heads,
        ffn_multiplier=4,
        qkv_bias=True,
        attn_fn=NestedAttention,
    ).to(device)

    # Forward pass
    with torch.inference_mode():
        pc = lift(pc)
        out = transf(pc)
    assert isinstance(out, Points)


@pytest.mark.inference_mode()
def test_nested_attention(setup_points):
    """Test nested attention mechanism."""
    pc: Points = setup_points[0]
    _, _, C = setup_points[1:]
    device = pc.device

    # Setup layers
    dim = C * 8
    lift = Linear(C, dim).to(device)
    attn = NestedAttention(
        dim=dim,
        num_heads=8,
        pos_enc=FourierEncoding(3, dim),
    ).to(device)

    # Forward pass
    with torch.inference_mode():
        pc = lift(pc)
        out = attn(pc)
    assert out.features.shape == (pc.offsets[-1], dim)


@pytest.mark.inference_mode()
def test_cross_nested_attention(setup_points):
    """Test cross nested attention with queries."""
    pc: Points = setup_points[0]
    B, _, C = setup_points[1:]
    device = pc.device

    # Setup layers
    dim = C * 8
    lift = Linear(C, dim).to(device)
    attn = NestedAttention(dim=dim, num_heads=8).to(device)

    # Create queries and run attention
    N = 100
    queries = torch.randn(B, N, dim).to(device)
    pc = lift(pc)
    out = attn(queries, pc)
    assert out.shape == (B, N, dim)


def test_spatial_feature_attention(setup_points):
    """Test spatial feature attention."""
    pc: Points = setup_points[0]
    _, _, C = setup_points[1:]
    device = pc.device

    # Setup layers
    lift = Linear(C, C * 8).to(device)
    attn = SpatialFeatureAttention(dim=C * 8, num_heads=8).to(device)

    # Forward pass
    pc = lift(pc)
    out = attn(pc)
    assert isinstance(out, Points)
