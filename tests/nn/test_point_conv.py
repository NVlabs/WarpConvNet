# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig, RealSearchMode
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.ops.reductions import REDUCTIONS


@pytest.fixture
def setup_points():
    """Setup test points with random coordinates."""
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features).to(device)


def test_point_conv_radius(setup_points):
    """Test point convolution with radius-based neighbor search."""
    pc: Points = setup_points
    device = pc.device

    # Create conv layer
    in_channels, out_channels = pc.num_channels, 16
    search_arg = RealSearchConfig(
        mode=RealSearchMode.RADIUS,
        radius=0.1,
    )
    conv = PointConv(
        in_channels,
        out_channels,
        neighbor_search_args=search_arg,
    ).to(device)

    # Forward pass
    out = conv(pc)
    out.feature_tensor.mean().backward()

    # Verify gradients exist
    for name, param in conv.named_parameters():
        if param.numel() > 0:
            assert param.grad is not None, f"{name} has no gradient"


def test_point_conv_knn(setup_points):
    """Test point convolution with k-nearest neighbors search."""
    pc: Points = setup_points
    device = pc.device

    # Create conv layer
    in_channels, out_channels = pc.num_channels, 16
    search_args = RealSearchConfig(
        mode=RealSearchMode.KNN,
        knn_k=16,
    )
    conv = PointConv(
        in_channels,
        out_channels,
        neighbor_search_args=search_args,
    ).to(device)

    # Forward pass
    out = conv(pc)
    assert out.num_channels == out_channels


def test_point_conv_downsample(setup_points):
    """Test point convolution with downsampling."""
    pc: Points = setup_points
    device = pc.device

    # Create conv layer
    in_channels, out_channels = pc.num_channels, 16
    search_args = RealSearchConfig(
        mode=RealSearchMode.RADIUS,
        radius=0.1,
    )
    conv = PointConv(
        in_channels,
        out_channels,
        neighbor_search_args=search_args,
        pooling_reduction=REDUCTIONS.MEAN,
        pooling_voxel_size=0.1,
        out_point_type="downsample",
    ).to(device)

    # Forward pass
    out = conv(pc)
    assert out.voxel_size is not None

    # Test backward pass
    out.feature_tensor.mean().backward()

    # Verify gradients exist
    for name, param in conv.named_parameters():
        if param.numel() > 0:
            assert param.grad is not None, f"{name} has no gradient"
