# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv import (
    SPARSE_CONV_FWD_ALGO_MODE,
    SPARSE_CONV_BWD_ALGO_MODE,
    STRIDED_CONV_MODE,
    SpatiallySparseConvBatchedExplicitGEMMFunction,
    SpatiallySparseConvExplicitGEMMFunction,
    SpatiallySparseConvImplicitGEMMFunction,
    UnifiedSpatiallySparseConvFunction,
    spatially_sparse_conv,
)
from warpconvnet.nn.functional.sparse_pool import sparse_max_pool
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


@pytest.fixture
def setup_voxels():
    """Setup test voxels with random coordinates."""
    wp.init()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    coords = [(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Voxels(coords, features, device=device).unique()


@pytest.fixture
def setup_small_voxels():
    """Setup small voxels for gradient checking."""
    wp.init()
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    B, min_N, max_N, C = 3, 10, 20, 7
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    coords = [(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Voxels(coords, features, device=device).unique()


def test_generate_output_coords(setup_voxels):
    """Test generation of output coordinates."""
    voxels = setup_voxels
    batch_indexed_coords = batch_indexed_coordinates(
        voxels.coordinate_tensor,
        voxels.offsets,
    )
    output_coords, offsets = stride_coords(batch_indexed_coords, stride=(2, 2, 2))

    assert output_coords.shape[0] < batch_indexed_coords.shape[0]
    assert offsets.shape == (voxels.batch_size + 1,)


def test_generate_kernel_map(setup_voxels):
    """Test kernel map generation and validation."""
    voxels = setup_voxels
    device = voxels.device

    # Setup coordinates
    batch_indexed_in_coords = batch_indexed_coordinates(
        voxels.coordinate_tensor,
        voxels.offsets,
    )
    batch_indexed_output_coords, offsets = stride_coords(batch_indexed_in_coords, stride=(2, 2, 2))

    # Generate kernel map
    assert batch_indexed_in_coords.dtype == torch.int32
    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_output_coords,
        in_to_out_stride_ratio=(2, 2, 2),
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
    )

    # Verify kernel map properties
    tot_kernel_map = kernel_map.offsets[-1].item()
    assert tot_kernel_map == kernel_map.in_maps.shape[0]
    assert tot_kernel_map == kernel_map.out_maps.shape[0]

    # Verify map sizes match
    for _, (in_map, out_map) in enumerate(kernel_map):
        assert in_map.shape[0] == out_map.shape[0]

    # Manual verification with hashmap
    in_hashmap = TorchHashTable.from_keys(batch_indexed_in_coords)
    kernel_offsets = _generate_kernel_offsets((3, 3, 3), (1, 1, 1), device)

    batch_indexed_output_coords = batch_indexed_output_coords * torch.tensor(
        [1, 2, 2, 2], dtype=torch.int32, device=device
    )

    N_in = batch_indexed_in_coords.shape[0]
    N_out = batch_indexed_output_coords.shape[0]

    for i, (in_map, out_map) in enumerate(kernel_map):
        offseted_out_coords = batch_indexed_output_coords + kernel_offsets[i]
        indices = in_hashmap.search(offseted_out_coords)
        valid_bool = (indices >= 0).to(device)
        num_valid = valid_bool.sum().item()
        found_in_map = indices[valid_bool]

        assert num_valid == in_map.shape[0]
        assert in_map.max().item() < N_in
        assert out_map.max().item() < N_out
        assert found_in_map.max().item() <= N_in

        unique_found_in_map = found_in_map.unique(sorted=True)
        unique_in_map = in_map.unique(sorted=True)
        assert torch.all(unique_found_in_map == unique_in_map)


def test_sparse_conv(setup_voxels):
    """Test basic sparse convolution."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 13

    # Create weights and bias
    kernel_size = (3, 3, 3)
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)
    bias = torch.randn(C_out).to(voxels.device)

    # Forward pass
    out_implicit = spatially_sparse_conv(
        voxels,
        weight=weights,
        bias=bias,
        kernel_size=kernel_size,
        stride=(2, 2, 2),
        fwd_algo=SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM,
    )
    out_explicit = spatially_sparse_conv(
        voxels,
        weight=weights,
        bias=bias,
        kernel_size=kernel_size,
        stride=(2, 2, 2),
        fwd_algo=SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM,
    )
    # out_batched_explicit = spatially_sparse_conv(
    #     voxels,
    #     weight=weights,
    #     bias=bias,
    #     kernel_size=kernel_size,
    #     stride=(2, 2, 2),
    #     fwd_algo=SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM_BATCHED,
    # )
    assert out_implicit.num_channels == C_out
    assert out_explicit.num_channels == C_out
    assert torch.allclose(out_implicit.feature_tensor, out_explicit.feature_tensor)
    # assert torch.allclose(out_implicit.feature_tensor, out_batched_explicit.feature_tensor)


def test_sparse_conv_explicit_backward(setup_small_voxels):
    """Test sparse convolution gradients."""
    voxels = setup_small_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    skip_symmetric_kernel_map = False

    # Setup convolution parameters
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)

    # Generate kernel map
    batch_indexed_in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    batch_indexed_out_coords, offsets = stride_coords(batch_indexed_in_coords, stride=stride)
    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        stride,
        kernel_size,
    )

    # Prepare for gradient check
    feature_tensor = voxels.feature_tensor.detach().requires_grad_(True)

    # Run gradient check
    torch.autograd.gradcheck(
        SpatiallySparseConvExplicitGEMMFunction.apply,
        (
            feature_tensor,
            weights,
            kernel_map,
            skip_symmetric_kernel_map,
            batch_indexed_out_coords.shape[0],
        ),
        eps=1e-3,
        atol=1e-3,
        rtol=1e-3,
    )


def test_sparse_conv_implicit_backward(setup_small_voxels):
    """Test sparse convolution gradients."""
    voxels = setup_small_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    skip_symmetric_kernel_map = False

    # Setup convolution parameters
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)

    # Generate kernel map
    batch_indexed_in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    batch_indexed_out_coords, offsets = stride_coords(batch_indexed_in_coords, stride=stride)
    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        stride,
        kernel_size,
    )

    # Prepare for gradient check
    feature_tensor = voxels.feature_tensor.detach().requires_grad_(True)

    # Run gradient check
    torch.autograd.gradcheck(
        SpatiallySparseConvImplicitGEMMFunction.apply,
        (
            feature_tensor,
            weights,
            kernel_map,
            skip_symmetric_kernel_map,
            batch_indexed_out_coords.shape[0],
        ),
        eps=1e-3,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "stride_mode", [STRIDED_CONV_MODE.REDUCE_AND_STRIDE, STRIDED_CONV_MODE.STRIDE_ONLY]
)
def test_sparse_conv_stride_modes(setup_voxels, stride_mode):
    """Test different striding modes for sparse convolution."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)

    conv = SpatiallySparseConv(
        C_in,
        C_out,
        kernel_size,
        stride,
        stride_mode=stride_mode,
    ).to(voxels.device)

    out = conv(voxels)
    assert out.num_channels == C_out


@pytest.mark.parametrize("generative", [True, False])
def test_sparse_conv_generative(setup_voxels, generative):
    """Test generative sparse convolution."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 13

    conv = SpatiallySparseConv(
        C_in,
        C_out,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        generative=generative,
    ).to(voxels.device)

    out = conv(voxels)
    assert out.num_channels == C_out

    if generative:
        assert out.coordinate_tensor.shape[0] > voxels.coordinate_tensor.shape[0]


def test_sparse_conv_amp(setup_voxels):
    """Test sparse convolution with automatic mixed precision."""
    voxels: Voxels = setup_voxels.to("cuda:0").sort()
    C_in, C_out = voxels.num_channels, 13

    conv = SpatiallySparseConv(
        C_in,
        C_out,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        out_code_backend="morton",
    ).to(voxels.device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = conv(voxels)
    assert out.num_channels == C_out


def _generate_kernel_offsets(kernel_size, kernel_dilation, device):
    """Helper function to generate kernel offsets."""
    i, j, k = torch.meshgrid(
        torch.arange(kernel_size[0], dtype=torch.int32),
        torch.arange(kernel_size[1], dtype=torch.int32),
        torch.arange(kernel_size[2], dtype=torch.int32),
        indexing="ij",
    )
    i, j, k = i.flatten(), j.flatten(), k.flatten()
    return torch.stack(
        [
            torch.zeros_like(i),
            (i - kernel_size[0] // 2) * kernel_dilation[0],
            (j - kernel_size[1] // 2) * kernel_dilation[1],
            (k - kernel_size[2] // 2) * kernel_dilation[2],
        ],
        dim=1,
    ).to(device)
