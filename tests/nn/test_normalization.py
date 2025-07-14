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
