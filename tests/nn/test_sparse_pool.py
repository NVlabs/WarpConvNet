# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.nn.functional.sparse_pool import sparse_reduce, sparse_unpool
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


@pytest.fixture
def setup_voxels():
    """Setup test voxels with random coordinates."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    coords = [(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Voxels(coords, features, device=device).unique()


def test_generate_output_coords(setup_voxels):
    """Test generation of output coordinates for pooling."""
    voxels: Voxels = setup_voxels

    # Test coordinate striding
    batch_indexed_coords = batch_indexed_coordinates(
        voxels.coordinate_tensor,
        voxels.offsets,
    )
    output_coords, offsets = stride_coords(batch_indexed_coords, stride=(2, 2, 2))

    assert output_coords.shape[0] < batch_indexed_coords.shape[0]
    assert offsets.shape == (voxels.batch_size + 1,)


@pytest.mark.parametrize("reduction", ["max", "random"])
def test_sparse_reduce(setup_voxels, reduction):
    """Test sparse reduction with different reduction methods."""
    voxels: Voxels = setup_voxels
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)

    # Perform reduction
    voxels_downsampled = sparse_reduce(
        voxels, kernel_size=kernel_size, stride=stride, reduction=reduction
    )

    # Verify output
    assert voxels_downsampled.coordinate_tensor.shape[0] < voxels.coordinate_tensor.shape[0]
    assert voxels_downsampled.num_channels == voxels.num_channels
    assert voxels_downsampled.batch_size == voxels.batch_size


def test_sparse_reduce_consistency(setup_voxels):
    """Test consistency between different reduction methods."""
    voxels: Voxels = setup_voxels
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)

    # Perform reductions with different methods
    voxels_max = sparse_reduce(voxels, kernel_size, stride, reduction="max")
    voxels_random = sparse_reduce(voxels, kernel_size, stride, reduction="random")

    # Verify outputs have same shape
    assert voxels_max.coordinate_tensor.shape == voxels_random.coordinate_tensor.shape
    assert voxels_max.num_channels == voxels_random.num_channels
    assert voxels_max.batch_size == voxels_random.batch_size


@pytest.mark.parametrize("concat_unpooled", [True, False])
def test_sparse_unpool(setup_voxels, concat_unpooled):
    """Test sparse unpooling with and without feature concatenation."""
    voxels: Voxels = setup_voxels
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)

    # First reduce the voxels
    voxels_downsampled = sparse_reduce(voxels, kernel_size, stride, reduction="max")

    # Then unpool
    voxels_unpooled = sparse_unpool(
        voxels_downsampled,
        voxels,
        kernel_size=kernel_size,
        stride=stride,
        concat_unpooled_voxels=concat_unpooled,
    )

    # Verify output shape
    assert voxels_unpooled.coordinate_tensor.shape[0] == voxels.coordinate_tensor.shape[0]

    if concat_unpooled:
        assert voxels_unpooled.num_channels == voxels.num_channels * 2
    else:
        assert voxels_unpooled.num_channels == voxels_downsampled.num_channels

    assert voxels_unpooled.batch_size == voxels.batch_size
