# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import pytest
import torch
import warp as wp

from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.attention import (
    Attention,
    PatchAttention,
    SpatialFeatureAttention,
    ToAttention,
    TransformerBlock,
    ZeroOutPoints,
)
from warpconvnet.nn.modules.mlp import Linear

try:
    import flash_attn

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


@pytest.fixture
def setup_points():
    """Setup test points with random coordinates."""
    wp.init()
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B, min_N, max_N, C = 3, 1000, 10000, 128
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand(int(N), 3) * 100 for N in Ns]
    features = [torch.rand(int(N), C) for N in Ns]
    return Points(coords, features).to(device), B, Ns, C


@pytest.fixture
def setup_batch_tensor():
    """Setup batch tensor data for attention testing."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, N, C = 2, 1024, 256
    x = torch.randn(B, N, C, device=device)
    num_points = torch.tensor([512, 768], device=device)
    return x, num_points, B, N, C


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
    attn = Attention(dim, num_heads=num_heads, qkv_bias=True, enable_flash=False).to(device)
    zero_out = ZeroOutPoints()

    # Forward pass
    pc = lift(pc)
    features, pos_enc, mask, num_points = to_attn(pc)
    x = attn(features, pos_enc, mask)
    x = zero_out(x, num_points)

    assert x.shape == (B, Ns.max(), dim)


@pytest.mark.parametrize("enable_flash", [True, False])
def test_attention_with_flash(setup_batch_tensor, enable_flash):
    """Test attention with and without flash attention."""
    x, num_points, B, N, C = setup_batch_tensor

    # Skip flash attention test if not available
    if enable_flash and not FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn not available")

    # Test attention
    attn = Attention(
        dim=C,
        num_heads=8,
        qkv_bias=True,
        enable_flash=enable_flash,
        attn_drop=0.1,
        proj_drop=0.1,
    ).to(x.device)

    # Forward pass
    if enable_flash and FLASH_ATTN_AVAILABLE:
        # Flash attention path
        out = attn(x, num_points=num_points)
    else:
        # Standard attention path
        out = attn(x, num_points=num_points)

    assert out.shape == (B, N, C)

    # Test gradient flow
    x.requires_grad_(True)
    out = attn(x, num_points=num_points)
    loss = out.sum()
    loss.backward()
    assert attn.qkv.weight.grad is not None


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
    if not FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn not available")

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
    ).to(device)

    # Forward pass
    pc = lift(pc)
    out = patch_attn(pc, order=POINT_ORDERING.MORTON_XYZ)

    # Verify output
    assert isinstance(out, Points)
    assert out.feature_tensor.shape[-1] == dim
    assert len(out) == len(pc)


def test_patch_transformer_block(setup_points):
    """Test transformer block with patch attention."""
    if not FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn not available")

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

    # Verify gradients exist
    assert patch_attn.attention.qkv.weight.grad is not None
    assert patch_attn.attention_norm.norm.weight.grad is not None
    assert lift.block.weight.grad is not None


@pytest.mark.parametrize("enable_flash", [True, False])
def test_spatial_feature_attention(setup_points, enable_flash):
    """Test spatial feature attention with and without flash attention."""
    pc: Points = setup_points[0]
    _, _, C = setup_points[1:]
    device = pc.device

    # Skip flash attention test if not available
    if enable_flash and not FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn not available")

    # Setup layers
    lift = Linear(C, C * 8).to(device)
    attn = SpatialFeatureAttention(
        dim=C * 8,
        num_heads=8,
        enable_flash=enable_flash,
    ).to(device)

    # Forward pass
    pc = lift(pc)
    out = attn(pc)
    assert isinstance(out, Points)


def test_attention_consistency():
    """Test that flash attention and standard attention produce similar results."""
    if not FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn not available")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for flash attention comparison")

    # Create test data
    B, N, C = 2, 128, 256  # Small size for exact comparison
    num_heads = 8
    x = torch.randn(B, N, C, device=device)

    # Create both attention modules with same initialization
    torch.manual_seed(42)
    attn_flash = Attention(
        dim=C,
        num_heads=num_heads,
        enable_flash=True,
        attn_drop=0.0,  # Disable dropout for exact comparison
        proj_drop=0.0,
    ).to(device)

    torch.manual_seed(42)
    attn_standard = Attention(
        dim=C,
        num_heads=num_heads,
        enable_flash=False,
        attn_drop=0.0,  # Disable dropout for exact comparison
        proj_drop=0.0,
    ).to(device)

    # Ensure same parameters
    attn_standard.load_state_dict(attn_flash.state_dict())

    # Forward pass
    with torch.no_grad():
        out_flash = attn_flash(x)
        out_standard = attn_standard(x)

    # Check that outputs are close (allow for numerical differences)
    assert torch.allclose(out_flash, out_standard, atol=1e-3, rtol=1e-3)
