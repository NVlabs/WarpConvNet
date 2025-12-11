# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.normalizations import LayerNorm, RMSNorm, SegmentedLayerNorm
from warpconvnet.nn.functional.normalizations import segmented_layer_norm


# Note: setup_points fixture is now imported from tests/conftest.py


def test_rms_norm_points(setup_points):
    """Test RMSNorm with point cloud input."""
    points: Points = setup_points[0]
    device = points.device

    # Create normalization layer
    rms_norm = RMSNorm(points.num_channels).to(device)

    # Forward pass
    normed_pc = rms_norm(points)

    # Verify output properties
    assert normed_pc.batch_size == points.batch_size
    assert normed_pc.num_channels == points.num_channels

    # Test gradient flow
    normed_pc.features.sum().backward()
    assert rms_norm.norm.weight.grad is not None


def test_rms_norm_voxels(setup_voxels):
    """Test RMSNorm with voxel input."""
    voxels: Voxels = setup_voxels
    device = voxels.device

    # Create normalization layer
    rms_norm = RMSNorm(voxels.num_channels).to(device)

    # Forward pass
    normed_voxels = rms_norm(voxels)

    # Verify output properties
    assert normed_voxels.batch_size == voxels.batch_size
    assert normed_voxels.num_channels == voxels.num_channels

    # Test gradient flow
    normed_voxels.features.sum().backward()
    assert rms_norm.norm.weight.grad is not None


def test_layer_norm_voxels(setup_voxels):
    """Test LayerNorm with voxel input."""
    voxels: Voxels = setup_voxels
    device = voxels.device

    # Create normalization layer
    layer_norm = LayerNorm([voxels.num_channels]).to(device)

    # Forward pass
    normed_voxels = layer_norm(voxels)

    # Verify output properties
    assert normed_voxels.batch_size == voxels.batch_size
    assert normed_voxels.num_channels == voxels.num_channels

    # Test gradient flow
    normed_voxels.features.sum().backward()
    assert layer_norm.norm.weight.grad is not None


def test_segmented_layer_norm_function():
    """Test SegmentedLayerNormFunction with voxel input."""
    # Test with your function directly
    N, C = 10, 5
    x = torch.randn(N, C, requires_grad=True, device="cuda")
    offsets = torch.tensor([0, 5, 10], device="cuda")
    gamma = torch.randn(C, requires_grad=True, device="cuda")
    beta = torch.randn(C, requires_grad=True, device="cuda")

    output = segmented_layer_norm(x, offsets, gamma, beta)
    loss = output.sum()
    loss.backward()

    assert gamma.grad is not None
    assert beta.grad is not None


def test_segmented_layer_norm(setup_voxels):
    """Test SegmentedLayerNorm with voxel input."""
    voxels: Voxels = setup_voxels
    device = voxels.device

    # Create normalization layer
    layer_norm = SegmentedLayerNorm(voxels.num_channels, elementwise_affine=True).to(device)

    # Set the features to require gradients
    voxels.feature_tensor.requires_grad = True

    # Forward pass
    normed_voxels = layer_norm(voxels)

    # Verify output properties
    assert normed_voxels.batch_size == voxels.batch_size
    assert normed_voxels.num_channels == voxels.num_channels


def test_segmented_range_norm():
    """Test segmented_range_norm."""
    from warpconvnet.nn.functional.normalizations import segmented_range_norm

    # Setup
    features = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [5.0, 50.0], [5.0, 50.0], [10.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    # 3 segments: rows 0-2 (size 3), rows 3-4 (size 2), row 5 (size 1)
    splits = torch.tensor([0, 3, 5, 6], dtype=torch.int32)

    # Expected:
    # Seg 0: min=[1, 10], max=[3, 30], range=[2, 20]
    #   Row 0: (1-1)/2=0, (10-10)/20=0
    #   Row 1: (2-1)/2=0.5, (20-10)/20=0.5
    #   Row 2: (3-1)/2=1, (30-10)/20=1

    # Seg 1: min=[5, 50], max=[5, 50], range=[0, 0]
    #   numerator = 5-5=0. denominator = eps. result = 0.

    # Seg 2: min=[10, 0], max=[10, 0], range=[0, 0]
    #   Row 5: 0.

    out = segmented_range_norm(features, splits, eps=1e-5)

    expected = torch.tensor(
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )

    assert torch.allclose(out, expected, atol=1e-5)

    # Test backward
    loss = out.sum()
    loss.backward()
    assert features.grad is not None


def test_segmented_range_norm_cuda():
    """Test segmented_range_norm on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from warpconvnet.nn.functional.normalizations import segmented_range_norm

    features = torch.rand(100, 16, device="cuda", requires_grad=True)
    splits = torch.tensor([0, 50, 100], device="cuda", dtype=torch.int32)

    out = segmented_range_norm(features, splits)
    assert out.shape == features.shape
    # Values should be roughly in [0, 1], but eps might affect it slightly if range is small
    # If range is huge, eps is negligible.

    # Check bounds
    assert (out >= 0).all()
    # If range is 0, output is 0.
    # If range > 0, output is (x-min)/(range+eps) < (x-min)/range <= 1
    assert (out <= 1.0).all()

    loss = out.sum()
    loss.backward()
    assert features.grad is not None


def test_segmented_range_norm_gradcheck():
    """Test gradients for segmented_range_norm."""
    from torch.autograd import gradcheck
    from warpconvnet.nn.functional.normalizations import segmented_range_norm

    # Use double precision for gradcheck
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create inputs
    # Use small range to ensure gradients are not too small
    features = torch.randn(10, 4, dtype=dtype, device=device, requires_grad=True)
    # Segments: [0, 3, 6, 10] -> sizes 3, 3, 4
    splits = torch.tensor([0, 3, 6, 10], dtype=torch.int32, device=device)

    # Functional wrapper for gradcheck
    def func(f):
        return segmented_range_norm(f, splits, eps=1e-6)

    assert gradcheck(func, (features,), eps=1e-6, atol=1e-4)


def test_segmented_range_norm_gradcheck_small_segments():
    """Test gradients for segmented_range_norm with small segments."""
    from torch.autograd import gradcheck
    from warpconvnet.nn.functional.normalizations import segmented_range_norm

    # Test with single-element segments where min=max
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 5 elements, each in its own segment
    features = torch.randn(5, 2, dtype=dtype, device=device, requires_grad=True)
    splits = torch.arange(6, dtype=torch.int32, device=device)

    def func(f):
        return segmented_range_norm(f, splits, eps=1e-6)

    assert gradcheck(func, (features,), eps=1e-6, atol=1e-4)
