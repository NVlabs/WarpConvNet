# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable
from warpconvnet.geometry.coords.search.torch_discrete import (
    generate_kernel_map,
    kernel_offsets_from_size,
)
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv import (
    SPARSE_CONV_FWD_ALGO_MODE,
    SPARSE_CONV_BWD_ALGO_MODE,
    STRIDED_CONV_MODE,
    _implicit_gemm_forward_logic,
    _implicit_gemm_backward_logic,
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
    _cutlass_implicit_gemm_forward_logic,
    _cutlass_implicit_gemm_backward_logic,
    SpatiallySparseConvExplicitGEMMFunction,
    SpatiallySparseConvImplicitGEMMFunction,
    UnifiedSpatiallySparseConvFunction,
    spatially_sparse_conv,
)
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


@pytest.fixture
def setup_voxels():
    """Setup test voxels with random coordinates."""
    wp.init()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, min_N, max_N, C = 3, 100000, 1000000, 16
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
    voxel_size = 0.2
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
    kernel_offsets = kernel_offsets_from_size((3, 3, 3), (1, 1, 1), device=device)

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


def test_generate_kernel_map_with_skip_symmetric_kernel_map(setup_voxels):
    """Test kernel map generation with skip symmetric kernel map."""
    voxels = setup_voxels
    device = voxels.device

    # Setup coordinates
    batch_indexed_in_coords = batch_indexed_coordinates(
        voxels.coordinate_tensor,
        voxels.offsets,
    )

    # Generate kernel map
    assert batch_indexed_in_coords.dtype == torch.int32
    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_in_coords,
        in_to_out_stride_ratio=(1, 1, 1),
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
    )

    # Skip symmetric kernel map
    kernel_map_skip = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_in_coords,
        in_to_out_stride_ratio=(1, 1, 1),
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        skip_symmetric_kernel_map=True,
    )

    assert (
        kernel_map_skip.identity_map_index is not None
        and kernel_map_skip.identity_map_index == 27 // 2
    )

    # Verify kernel map properties
    num_skip_offsets = len(kernel_map_skip.offsets)
    assert torch.all(kernel_map.offsets[:num_skip_offsets] == kernel_map_skip.offsets)
    assert torch.all(
        kernel_map.in_maps[: kernel_map.offsets[num_skip_offsets - 1]] == kernel_map_skip.in_maps
    )
    assert torch.all(
        kernel_map.out_maps[: kernel_map.offsets[num_skip_offsets - 1]] == kernel_map_skip.out_maps
    )


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


def test_sparse_conv_forward_backward_with_cutlass(setup_voxels):
    """Test sparse convolution forward backward with cutlass."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 32
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)

    # Setup convolution parameters
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.clamp(torch.randn(num_kernels, C_in, C_out).to(voxels.device), min=-1, max=1)
    # Generate kernel map
    batch_indexed_in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)

    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_in_coords,
        stride,
        kernel_size,
    )

    # Explicit GEMM
    out_explicit = _explicit_gemm_forward_logic(
        voxels.feature_tensor,
        weights,
        kernel_map,
        batch_indexed_in_coords.shape[0],
    )
    out_implicit = _cutlass_implicit_gemm_forward_logic(
        voxels.feature_tensor,
        weights,
        kernel_map,
        batch_indexed_in_coords.shape[0],
        accumulator_type=torch.float32,
        split_k_slices=1,
    )
    assert torch.allclose(out_explicit, out_implicit, atol=1e-1, rtol=1e-3)

    # Backward pass
    grad_out = torch.clamp(torch.randn_like(out_explicit), min=-1, max=1)
    grad_in_explicit, grad_weight_explicit = _explicit_gemm_backward_logic(
        grad_out.clone(),
        voxels.feature_tensor.clone(),
        weights.clone(),
        kernel_map,
    )

    grad_in, grad_weight = _cutlass_implicit_gemm_backward_logic(
        grad_out.clone(),
        voxels.feature_tensor.clone(),
        weights.clone(),
        kernel_map,
        accumulator_type=torch.float32,
    )
    assert torch.allclose(grad_in, grad_in_explicit, atol=1e-1, rtol=1e-3)
    assert torch.allclose(grad_weight, grad_weight_explicit, atol=1e-1, rtol=1e-2)


def test_sparse_conv_forward_backward_implicit_explicit_gemm(setup_voxels):
    """Test sparse convolution forward backward with implicit and explicit gemm."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 32
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)

    # Setup convolution parameters
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)
    # Generate kernel map
    batch_indexed_in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)

    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_in_coords,
        stride,
        kernel_size,
    )

    # Explicit GEMM
    out_explicit = _explicit_gemm_forward_logic(
        voxels.feature_tensor,
        weights,
        kernel_map,
        batch_indexed_in_coords.shape[0],
    )
    out_implicit = _implicit_gemm_forward_logic(
        voxels.feature_tensor,
        weights,
        kernel_map,
        batch_indexed_in_coords.shape[0],
        compute_dtype=None,
        fwd_block_size=16,
    )
    assert torch.allclose(out_explicit, out_implicit, atol=1e-3, rtol=1e-5)

    # Backward pass
    grad_out = torch.randn_like(out_explicit)
    grad_in_explicit, grad_weight_explicit = _explicit_gemm_backward_logic(
        grad_out.clone(),
        voxels.feature_tensor.clone(),
        weights.clone(),
        kernel_map,
    )

    grad_in, grad_weight = _implicit_gemm_backward_logic(
        grad_out.clone(),
        voxels.feature_tensor.clone(),
        weights.clone(),
        kernel_map,
        num_out_coords=out_explicit.shape[0],
        compute_dtype=None,
        gemm_block_size=16,
        split_k_threads_per_block=128,
        split_k_factor=4,
    )
    assert torch.allclose(grad_in, grad_in_explicit, atol=1e-3, rtol=1e-5)
    assert torch.allclose(grad_weight, grad_weight_explicit, atol=1e-3, rtol=1e-5)


def test_sparse_conv_forward_with_skip_symmetric(setup_small_voxels):
    """Test sparse convolution forward with skip symmetric kernel map."""
    voxels = setup_small_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)

    # Setup convolution parameters
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)
    # Set the weights after the identity to be zero
    iden_map_idx = kernel_size[0] * kernel_size[1] * kernel_size[2] // 2
    weights[iden_map_idx + 1 :] = 0

    # Generate kernel map
    batch_indexed_in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)

    # Forward pass with skip symmetric kernel map
    kernel_map_skip = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_in_coords,
        stride,
        kernel_size,
        skip_symmetric_kernel_map=True,
    )
    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_in_coords,
        stride,
        kernel_size,
    )

    # Implicit GEMM
    out_skip = _implicit_gemm_forward_logic(
        voxels.feature_tensor,
        weights,
        kernel_map_skip,
        batch_indexed_in_coords.shape[0],
        compute_dtype=torch.float32,
        fwd_block_size=16,
    )
    out = _implicit_gemm_forward_logic(
        voxels.feature_tensor,
        weights,
        kernel_map,
        batch_indexed_in_coords.shape[0],
        compute_dtype=torch.float32,
        fwd_block_size=16,
    )

    assert torch.allclose(out_skip, out)

    # Explicit GEMM
    out_skip_explicit = _explicit_gemm_forward_logic(
        voxels.feature_tensor,
        weights,
        kernel_map_skip,
        batch_indexed_in_coords.shape[0],
    )
    out_explicit = _explicit_gemm_forward_logic(
        voxels.feature_tensor,
        weights,
        kernel_map,
        batch_indexed_in_coords.shape[0],
    )
    assert torch.allclose(out_skip_explicit, out_explicit)


def test_sparse_conv_explicit_backward(setup_small_voxels):
    """Test sparse convolution gradients."""
    voxels = setup_small_voxels
    C_in, C_out = 7, 13
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)

    # Setup convolution parameters
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)

    # Generate kernel map
    batch_indexed_in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    batch_indexed_out_coords, offsets = stride_coords(batch_indexed_in_coords, stride=stride)
    # Prepare for gradient check
    feature_tensor = voxels.feature_tensor.detach().requires_grad_(True)

    # Run gradient check
    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        stride,
        kernel_size,
    )
    torch.autograd.gradcheck(
        SpatiallySparseConvExplicitGEMMFunction.apply,
        (
            feature_tensor,
            weights,
            kernel_map,
            batch_indexed_out_coords.shape[0],
        ),
        eps=1e-3,
        atol=1e-3,
        rtol=1e-3,
    )

    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        stride,
        kernel_size,
    )
    torch.autograd.gradcheck(
        SpatiallySparseConvExplicitGEMMFunction.apply,
        (
            feature_tensor,
            weights,
            kernel_map,
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
    stride = (1, 1, 1)

    # Setup convolution parameters
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)

    # Generate kernel map
    batch_indexed_in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    batch_indexed_out_coords, offsets = stride_coords(batch_indexed_in_coords, stride=stride)

    # Prepare for gradient check
    feature_tensor = voxels.feature_tensor.detach().requires_grad_(True)

    # Run gradient check
    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        stride,
        kernel_size,
    )
    torch.autograd.gradcheck(
        SpatiallySparseConvImplicitGEMMFunction.apply,
        (
            feature_tensor,
            weights,
            kernel_map,
            batch_indexed_out_coords.shape[0],
        ),
        eps=1e-3,
        atol=1e-3,
        rtol=1e-3,
    )

    kernel_map = generate_kernel_map(
        batch_indexed_in_coords,
        batch_indexed_out_coords,
        stride,
        kernel_size,
    )
    torch.autograd.gradcheck(
        SpatiallySparseConvImplicitGEMMFunction.apply,
        (
            feature_tensor,
            weights,
            kernel_map,
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
        order=POINT_ORDERING.MORTON_XYZ,
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
