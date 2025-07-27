# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.coords.search.torch_hashmap import HashMethod
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.utils.timer import Timer
from warpconvnet.utils.unique import unique_hashmap, unique_torch


@pytest.fixture
def setup_voxels():
    """Setup test voxels with random coordinates and features."""
    wp.init()
    torch.manual_seed(0)

    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    coords = [(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Voxels(coords, features), Ns, voxel_size


def test_voxel_construction(setup_voxels):
    """Test voxel tensor construction and properties."""
    voxels, Ns, _ = setup_voxels
    Ns_cumsum = Ns.cumsum(dim=0).tolist()

    # Test basic properties
    assert voxels.batched_coordinates.batch_size == len(Ns)
    assert torch.equal(voxels.batched_coordinates.offsets, torch.IntTensor([0] + Ns_cumsum))
    assert voxels.batched_coordinates.batched_tensor.shape == (Ns_cumsum[-1], 3)
    assert voxels.batched_features.batch_size == len(Ns)
    assert voxels.batched_features.batched_tensor.shape == (Ns_cumsum[-1], 7)

    # Test device movement
    device = torch.device("cuda:0")
    voxels_gpu = voxels.to(device)
    assert voxels_gpu.batched_coordinates.batched_tensor.device == device


@pytest.mark.benchmark(group="unique_methods")
@pytest.mark.parametrize(
    "hash_method",
    [
        HashMethod.CITY,
        HashMethod.MURMUR,
        HashMethod.FNV1A,
    ],
)
def test_hashmap_unique(setup_voxels, benchmark, hash_method):
    """Benchmark different hash methods for unique operation."""
    voxels, _, _ = setup_voxels
    device = "cuda:0"
    voxels = voxels.to(device)
    coords = voxels.batched_coordinates
    bcoords = batch_indexed_coordinates(coords.batched_tensor, coords.offsets)

    result = benchmark.pedantic(
        lambda: unique_hashmap(bcoords, hash_method),
        iterations=20,
        rounds=3,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="unique_methods")
def test_torch_unique(setup_voxels, benchmark):
    """Benchmark torch unique operation."""
    voxels, _, _ = setup_voxels
    device = "cuda:0"
    voxels = voxels.to(device)
    coords = voxels.batched_coordinates
    bcoords = batch_indexed_coordinates(coords.batched_tensor, coords.offsets)

    result = benchmark.pedantic(
        lambda: unique_torch(bcoords, dim=0, return_to_unique_indices=True),
        iterations=20,
        rounds=3,
        warmup_rounds=1,
    )


def test_unique_results(setup_voxels):
    """Test that both unique methods give same results."""
    voxels, _, _ = setup_voxels
    device = "cuda:0"
    voxels = voxels.to(device)
    coords = voxels.batched_coordinates
    bcoords = batch_indexed_coordinates(coords.batched_tensor, coords.offsets)

    # Get results from both methods
    unique_index, _ = unique_hashmap(bcoords, HashMethod.FNV1A)
    unique_coords, to_orig_indices, _, _, to_unique_indices = unique_torch(
        bcoords, dim=0, return_to_unique_indices=True
    )

    # Validate results
    assert len(unique_index) == len(
        to_unique_indices
    ), f"Unique index and to_unique_indices have different lengths: {len(unique_index)} != {len(to_unique_indices)}"
    assert torch.equal(bcoords[unique_index].unique(dim=0), unique_coords)


def test_sort(setup_voxels):
    """Test voxel sorting."""
    voxels, _, _ = setup_voxels
    device = torch.device("cuda:0")
    voxels = voxels.to(device)
    sorted_voxels = voxels.sort(ordering=POINT_ORDERING.MORTON_XYZ)
    assert sorted_voxels is not None


def test_unique(setup_voxels):
    """Test voxel uniqueness operations."""
    voxels, _, _ = setup_voxels
    device = torch.device("cuda:0")
    voxels = voxels.to(device)

    unique_voxels = voxels.unique()
    bcoords = batch_indexed_coordinates(voxels.batched_coordinates.batched_tensor, voxels.offsets)

    assert torch.unique(bcoords, dim=0).shape[0] == len(unique_voxels.batched_coordinates)


def test_dense_conversion():
    """Test conversion between dense and sparse representations."""
    dense_tensor = torch.rand(16, 3, 128, 128)
    # Empty out 80% of the elements
    dense_tensor[dense_tensor < 0.8] = 0

    # Test sparse conversion
    voxels = Voxels.from_dense(dense_tensor, dense_tensor_channel_dim=1)
    assert voxels.batch_size == 16

    # Test dense conversion
    dense_tensor2 = voxels.to_dense(channel_dim=1)
    assert torch.equal(dense_tensor2, dense_tensor)


def test_sparse_dense_conversion(setup_voxels):
    """Test sparse-dense-sparse conversion with convolution."""
    voxels, _, _ = setup_voxels
    device = torch.device("cuda:0")
    voxels = voxels.to(device)

    # Convert to dense
    dense_tensor = voxels.to_dense(channel_dim=1)

    # Apply convolution
    out_channels = 13
    conv = torch.nn.Conv3d(7, out_channels, 3, padding=1, bias=False).to(device)
    dense_tensor = conv(dense_tensor)

    # Convert back to sparse
    voxels2 = Voxels.from_dense(
        dense_tensor,
        target_spatial_sparse_tensor=voxels,
    )
    assert voxels2.num_channels == out_channels


def test_to_point(setup_voxels):
    """Test conversion to point representation."""
    voxels, _, voxel_size = setup_voxels
    device = torch.device("cuda:0")
    voxels = voxels.to(device)

    voxels.set_tensor_stride((2, 2, 2))
    points = voxels.to_point(voxel_size)
    assert points.features.shape[1] == 7


@pytest.mark.parametrize(
    "in_dtype,amp_dtype",
    [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
)
def test_voxel_amp(setup_voxels, in_dtype, amp_dtype):
    """Test voxel behavior with Automatic Mixed Precision."""
    voxels, _, _ = setup_voxels
    device = torch.device("cuda:0")
    voxels = voxels.to(device)

    # Convert input to target dtype
    if in_dtype is not None:
        voxels = voxels.to(dtype=in_dtype)

    # Test feature access with autocast
    with torch.cuda.amp.autocast(dtype=amp_dtype):
        # Features should be in amp_dtype inside autocast
        features = voxels.features
        assert features.dtype == amp_dtype

        # Original features should maintain their dtype
        assert voxels.batched_features.batched_tensor.dtype == in_dtype

        # Test some operations
        result = features + 1.0
        assert result.dtype == features.dtype

    # Outside autocast, should be back to original dtype
    features = voxels.features
    assert features.dtype == in_dtype


@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ],
)
def test_voxel_dtype_conversion(setup_voxels, in_dtype, out_dtype):
    """Test voxel dtype conversion."""
    voxels, _, _ = setup_voxels
    device = torch.device("cuda:0")
    voxels = voxels.to(device, dtype=in_dtype)

    converted = voxels.to(dtype=out_dtype)
    assert converted.features.dtype == out_dtype
    assert voxels.features.dtype == in_dtype  # Original unchanged

    # Test operations maintain dtype
    result = converted + 1.0
    assert result.features.dtype == out_dtype


def test_extra_attributes():
    # Test for extra attribute in the Geometry base class
    # Critical for X.replace() to work
    device = "cuda:0"
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    coords = [torch.floor(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    voxels = Voxels(coords, features, device=device, test_attribute="test", voxel_size=voxel_size)
    # Add extra attribute
    replaced = voxels.replace(batched_features=voxels.batched_features + 1)
    # Check that the extra attribute is present
    assert replaced.extra_attributes["test_attribute"] == "test"
    assert replaced.voxel_size == voxel_size


def test_from_dense():
    B, C, H, W, D = 2, 3, 100, 100, 100
    rand_img = torch.randn(B, C, H, W, D)
    mask = torch.rand(B, 1, H, W, D) < 0.5
    rand_img[mask.expand_as(rand_img)] = 0
    voxels = Voxels.from_dense(rand_img, voxel_size=0.05).to("cuda:0")

    assert voxels.offsets[-1] == voxels.batched_coordinates.batched_tensor.shape[0]
    assert voxels.offsets[-1] == voxels.batched_features.batched_tensor.shape[0]
