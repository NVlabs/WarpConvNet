# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.features.grid import GridFeatures, GridMemoryFormat


@pytest.fixture
def setup_device():
    """Setup test device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def setup_grid_features(setup_device):
    """Setup test grid features with all memory formats."""
    device = setup_device
    batch_size = 2
    grid_shape = (8, 8, 8)
    num_channels = 16
    H, W, D = grid_shape

    # Create standard format tensor (b_x_y_z_c)
    standard_tensor = torch.rand(batch_size, *grid_shape, num_channels, device=device)
    offsets = torch.tensor(
        [
            0,
            grid_shape[0] * grid_shape[1] * grid_shape[2],
            2 * grid_shape[0] * grid_shape[1] * grid_shape[2],
        ],
        device=device,
    )

    # Create grid features with different memory formats
    formats = {
        "standard": GridMemoryFormat.b_x_y_z_c,
        "channel_first": GridMemoryFormat.b_c_x_y_z,
        "factorized_z": GridMemoryFormat.b_zc_x_y,
        "factorized_x": GridMemoryFormat.b_xc_y_z,
        "factorized_y": GridMemoryFormat.b_yc_x_z,
    }

    # Create tensors in different memory formats
    tensors = {}
    tensors["standard"] = standard_tensor
    tensors["channel_first"] = standard_tensor.permute(0, 4, 1, 2, 3)
    tensors["factorized_z"] = standard_tensor.permute(0, 3, 4, 1, 2).reshape(
        batch_size, D * num_channels, H, W
    )
    tensors["factorized_x"] = standard_tensor.permute(0, 1, 4, 2, 3).reshape(
        batch_size, H * num_channels, W, D
    )
    tensors["factorized_y"] = standard_tensor.permute(0, 2, 4, 1, 3).reshape(
        batch_size, W * num_channels, H, D
    )

    # Create corresponding grid features
    features = {}
    for name, fmt in formats.items():
        tensor = tensors[name]
        if fmt in [GridMemoryFormat.b_x_y_z_c, GridMemoryFormat.b_c_x_y_z]:
            features[name] = GridFeatures(tensor, offsets, memory_format=fmt)
        else:
            features[name] = GridFeatures(
                tensor,
                offsets,
                memory_format=fmt,
                grid_shape=grid_shape,
                num_channels=num_channels,
            )

    return features, standard_tensor, grid_shape, num_channels, device


def test_grid_features_all_formats(setup_device):
    """Test creation of grid features with all memory formats."""
    device = setup_device
    batch_size = 2
    grid_shape = (4, 6, 8)
    num_channels = 16
    H, W, D = grid_shape

    # Create offsets
    offsets = torch.tensor([0, H * W * D, 2 * H * W * D], device=device)

    # Test standard format (b_x_y_z_c)
    tensor = torch.rand(batch_size, H, W, D, num_channels, device=device)
    features = GridFeatures(tensor, offsets, GridMemoryFormat.b_x_y_z_c)

    assert features.grid_shape == grid_shape
    assert features.num_channels == num_channels
    assert features.memory_format == GridMemoryFormat.b_x_y_z_c
    assert torch.allclose(features.batched_tensor, tensor)

    # Test channel-first format (b_c_x_y_z)
    tensor = torch.rand(batch_size, num_channels, H, W, D, device=device)
    features = GridFeatures(tensor, offsets, GridMemoryFormat.b_c_x_y_z)

    assert features.grid_shape == grid_shape
    assert features.num_channels == num_channels
    assert features.memory_format == GridMemoryFormat.b_c_x_y_z
    assert torch.allclose(features.batched_tensor, tensor)

    # Test factorized Z format (b_zc_x_y)
    tensor = torch.rand(batch_size, D * num_channels, H, W, device=device)
    features = GridFeatures(
        tensor,
        offsets,
        GridMemoryFormat.b_zc_x_y,
        grid_shape=grid_shape,
        num_channels=num_channels,
    )

    assert features.grid_shape == grid_shape
    assert features.num_channels == num_channels
    assert features.memory_format == GridMemoryFormat.b_zc_x_y
    assert torch.allclose(features.batched_tensor, tensor)

    # Test factorized X format (b_xc_y_z)
    tensor = torch.rand(batch_size, H * num_channels, W, D, device=device)
    features = GridFeatures(
        tensor,
        offsets,
        GridMemoryFormat.b_xc_y_z,
        grid_shape=grid_shape,
        num_channels=num_channels,
    )

    assert features.grid_shape == grid_shape
    assert features.num_channels == num_channels
    assert features.memory_format == GridMemoryFormat.b_xc_y_z
    assert torch.allclose(features.batched_tensor, tensor)

    # Test factorized Y format (b_yc_x_z)
    tensor = torch.rand(batch_size, W * num_channels, H, D, device=device)
    features = GridFeatures(
        tensor,
        offsets,
        GridMemoryFormat.b_yc_x_z,
        grid_shape=grid_shape,
        num_channels=num_channels,
    )

    assert features.grid_shape == grid_shape
    assert features.num_channels == num_channels
    assert features.memory_format == GridMemoryFormat.b_yc_x_z
    assert torch.allclose(features.batched_tensor, tensor)


def test_all_memory_format_conversions(setup_grid_features):
    """Test conversion between all memory formats."""
    features_dict, standard_tensor, grid_shape, num_channels, device = setup_grid_features

    # Test conversion from each format to all other formats
    for src_name, src_features in features_dict.items():
        # First convert to standard format
        standard_converted = src_features.to_memory_format(GridMemoryFormat.b_x_y_z_c)
        assert standard_converted.memory_format == GridMemoryFormat.b_x_y_z_c
        assert torch.allclose(
            standard_converted.batched_tensor,
            features_dict["standard"].batched_tensor,
            rtol=1e-5,
            atol=1e-5,
        )

        # Then test conversion to all other formats
        for dst_name, dst_features in features_dict.items():
            if src_name == dst_name:
                continue

            # Convert from source to destination format
            converted = src_features.to_memory_format(dst_features.memory_format)

            # Check properties
            assert converted.memory_format == dst_features.memory_format
            assert converted.grid_shape == dst_features.grid_shape
            assert converted.num_channels == dst_features.num_channels

            # Check tensor shape based on format
            if dst_features.memory_format == GridMemoryFormat.b_x_y_z_c:
                assert converted.batched_tensor.shape == (2, *grid_shape, num_channels)
            elif dst_features.memory_format == GridMemoryFormat.b_c_x_y_z:
                assert converted.batched_tensor.shape == (2, num_channels, *grid_shape)
            elif dst_features.memory_format == GridMemoryFormat.b_zc_x_y:
                H, W, D = grid_shape
                assert converted.batched_tensor.shape == (2, D * num_channels, H, W)
            elif dst_features.memory_format == GridMemoryFormat.b_xc_y_z:
                H, W, D = grid_shape
                assert converted.batched_tensor.shape == (2, H * num_channels, W, D)
            elif dst_features.memory_format == GridMemoryFormat.b_yc_x_z:
                H, W, D = grid_shape
                assert converted.batched_tensor.shape == (2, W * num_channels, H, D)

            # Check values are equal by converting both to standard format
            std1 = converted.to_standard_format()
            std2 = dst_features.to_standard_format()
            assert torch.allclose(std1, std2, rtol=1e-5, atol=1e-5)


def test_create_empty_grid_features(setup_device):
    """Test creating empty grid features with any memory format."""
    device = setup_device
    grid_shape = (4, 6, 8)
    num_channels = 16
    batch_size = 2
    H, W, D = grid_shape

    for fmt in GridMemoryFormat:
        # Create empty grid features
        features = GridFeatures.create_empty(
            grid_shape, num_channels, batch_size, memory_format=fmt, device=device
        )

        # Check properties
        assert features.grid_shape == grid_shape
        assert features.num_channels == num_channels
        assert features.memory_format == fmt
        assert features.offsets.shape == (batch_size + 1,)
        assert features.offsets[0] == 0

        # Check tensor shape based on format
        if fmt == GridMemoryFormat.b_x_y_z_c:
            assert features.batched_tensor.shape == (batch_size, H, W, D, num_channels)
        elif fmt == GridMemoryFormat.b_c_x_y_z:
            assert features.batched_tensor.shape == (batch_size, num_channels, H, W, D)
        elif fmt == GridMemoryFormat.b_zc_x_y:
            assert features.batched_tensor.shape == (batch_size, D * num_channels, H, W)
        elif fmt == GridMemoryFormat.b_xc_y_z:
            assert features.batched_tensor.shape == (batch_size, H * num_channels, W, D)
        elif fmt == GridMemoryFormat.b_yc_x_z:
            assert features.batched_tensor.shape == (batch_size, W * num_channels, H, D)


def test_create_factorized_formats(setup_device):
    """Test creating factorized grid features from a single grid feature."""
    device = setup_device
    grid_shape = (4, 6, 8)
    num_channels = 16
    batch_size = 2

    # Create a standard grid feature
    standard = GridFeatures.create_empty(
        grid_shape,
        num_channels,
        batch_size,
        memory_format=GridMemoryFormat.b_x_y_z_c,
        device=device,
    )

    # Fill with random values
    standard.batched_tensor.data.normal_()

    # Create factorized formats
    factorized = GridFeatures.create_factorized_formats(
        standard,
        formats=[GridMemoryFormat.b_zc_x_y, GridMemoryFormat.b_xc_y_z, GridMemoryFormat.b_yc_x_z],
    )

    # Check we got 3 formats
    assert len(factorized) == 3

    # Check each format
    for i, fmt in enumerate(
        [GridMemoryFormat.b_zc_x_y, GridMemoryFormat.b_xc_y_z, GridMemoryFormat.b_yc_x_z]
    ):
        assert factorized[i].memory_format == fmt
        assert factorized[i].grid_shape == grid_shape
        assert factorized[i].num_channels == num_channels

        # Convert back to standard and check it matches original
        reconverted = factorized[i].to_memory_format(GridMemoryFormat.b_x_y_z_c)
        assert torch.allclose(
            reconverted.batched_tensor, standard.batched_tensor, rtol=1e-5, atol=1e-5
        )
