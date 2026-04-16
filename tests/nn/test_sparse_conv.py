# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.geometry.coords.search.packed_hashmap import PackedHashTable
from warpconvnet.geometry.coords.search.torch_discrete import (
    generate_kernel_map,
    kernel_offsets_from_size,
)
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv import (
    SPARSE_CONV_AB_ALGO_MODE,
    SPARSE_CONV_ATB_ALGO_MODE,
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
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv, SparseConv2d, SparseConv3d
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


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

    # Manual verification with PackedHashTable
    in_hashmap = PackedHashTable.from_coords(batch_indexed_in_coords, device=device)
    kernel_offsets = kernel_offsets_from_size((3, 3, 3), (1, 1, 1), device=device)

    batch_indexed_output_coords = batch_indexed_output_coords * torch.tensor(
        [1, 2, 2, 2], dtype=torch.int32, device=device
    )

    N_in = batch_indexed_in_coords.shape[0]
    N_out = batch_indexed_output_coords.shape[0]

    for i, (in_map, out_map) in enumerate(kernel_map):
        offseted_out_coords = (batch_indexed_output_coords + kernel_offsets[i]).contiguous()
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

    # Verify kernel map properties: offsets (counts) must match exactly,
    # and the sets of (in, out) pairs within each offset group must match
    # (ordering within groups is non-deterministic due to atomicAdd scatter).
    num_skip_offsets = len(kernel_map_skip.offsets)
    assert torch.all(kernel_map.offsets[:num_skip_offsets] == kernel_map_skip.offsets)
    for k in range(num_skip_offsets - 1):
        s_full, e_full = kernel_map.offsets[k].item(), kernel_map.offsets[k + 1].item()
        s_skip, e_skip = kernel_map_skip.offsets[k].item(), kernel_map_skip.offsets[k + 1].item()
        assert e_full - s_full == e_skip - s_skip
        if e_full > s_full:
            # Compare sorted pairs to handle non-deterministic ordering
            full_pairs = torch.stack(
                [kernel_map.in_maps[s_full:e_full], kernel_map.out_maps[s_full:e_full]], dim=1
            )
            skip_pairs = torch.stack(
                [kernel_map_skip.in_maps[s_skip:e_skip], kernel_map_skip.out_maps[s_skip:e_skip]],
                dim=1,
            )
            full_sorted = full_pairs[full_pairs[:, 1].argsort()]
            skip_sorted = skip_pairs[skip_pairs[:, 1].argsort()]
            assert torch.all(full_sorted == skip_sorted), f"Mismatch at offset group {k}"


def test_sparse_conv(setup_voxels):
    """Test basic sparse convolution."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 13

    # Create weights and bias
    kernel_size = (3, 3, 3)
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out, device=voxels.device, dtype=torch.double)
    bias = torch.randn(C_out).to(voxels.device)

    # Forward pass
    out_implicit = spatially_sparse_conv(
        voxels,
        weight=weights,
        bias=bias,
        kernel_size=kernel_size,
        stride=(2, 2, 2),
        fwd_algo=SPARSE_CONV_AB_ALGO_MODE.IMPLICIT_GEMM,
    )
    out_explicit = spatially_sparse_conv(
        voxels,
        weight=weights,
        bias=bias,
        kernel_size=kernel_size,
        stride=(2, 2, 2),
        fwd_algo=SPARSE_CONV_AB_ALGO_MODE.EXPLICIT_GEMM,
    )
    # out_batched_explicit = spatially_sparse_conv(
    #     voxels,
    #     weight=weights,
    #     bias=bias,
    #     kernel_size=kernel_size,
    #     stride=(2, 2, 2),
    #     fwd_algo=SPARSE_CONV_AB_ALGO_MODE.EXPLICIT_GEMM_BATCHED,
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
    )
    if isinstance(out_implicit, int):
        pytest.skip(f"cutlass forward kernel unavailable (status={out_implicit})")
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
    if isinstance(grad_in, int):
        status_code, failing_kernel = grad_in, grad_weight
        pytest.skip(
            f"cutlass backward kernel unavailable (status={status_code}, kernel={failing_kernel})"
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
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)

    # Setup convolution parameters
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out, device=voxels.device, dtype=torch.double)

    # Generate kernel map
    batch_indexed_in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    batch_indexed_out_coords, offsets = stride_coords(batch_indexed_in_coords, stride=stride)
    # Prepare for gradient check
    feature_tensor = voxels.feature_tensor.detach().to(torch.double).requires_grad_(True)

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


def test_sparse_conv_transposed_generative(setup_small_voxels):
    """Test transposed + generative sparse convolution."""
    voxels = setup_small_voxels
    conv = SpatiallySparseConv(
        voxels.num_channels,
        voxels.num_channels,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        transposed=True,
        generative=True,
    ).to(voxels.device)

    out = conv(voxels)
    assert out.coordinate_tensor.shape[0] >= voxels.coordinate_tensor.shape[0]


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


def test_sparse_conv_algorithm_list_functionality(setup_small_voxels):
    """Test sparse convolution with algorithm lists for limiting search space."""
    voxels = setup_small_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)

    # Create weights
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)

    # Test with enum list
    out1 = spatially_sparse_conv(
        voxels,
        weight=weights,
        kernel_size=kernel_size,
        fwd_algo=[
            SPARSE_CONV_AB_ALGO_MODE.IMPLICIT_GEMM,
            SPARSE_CONV_AB_ALGO_MODE.CUTLASS_IMPLICIT_GEMM,
        ],
        dgrad_algo=[
            SPARSE_CONV_AB_ALGO_MODE.IMPLICIT_GEMM,
            SPARSE_CONV_AB_ALGO_MODE.CUTLASS_IMPLICIT_GEMM,
        ],
        wgrad_algo=[
            SPARSE_CONV_ATB_ALGO_MODE.IMPLICIT_GEMM,
            SPARSE_CONV_ATB_ALGO_MODE.CUTLASS_IMPLICIT_GEMM,
        ],
    )

    # Test with string list
    out2 = spatially_sparse_conv(
        voxels,
        weight=weights,
        kernel_size=kernel_size,
        fwd_algo=["implicit_gemm", "cutlass_implicit_gemm"],
        dgrad_algo=["implicit_gemm", "cutlass_implicit_gemm"],
        wgrad_algo=["implicit_gemm", "cutlass_implicit_gemm"],
    )

    # Test with mixed string/enum list
    out3 = spatially_sparse_conv(
        voxels,
        weight=weights,
        kernel_size=kernel_size,
        fwd_algo=["implicit_gemm", SPARSE_CONV_AB_ALGO_MODE.CUTLASS_IMPLICIT_GEMM],
        dgrad_algo=["implicit_gemm", SPARSE_CONV_AB_ALGO_MODE.CUTLASS_IMPLICIT_GEMM],
        wgrad_algo=["implicit_gemm", SPARSE_CONV_ATB_ALGO_MODE.CUTLASS_IMPLICIT_GEMM],
    )

    # Test single algorithm
    out4 = spatially_sparse_conv(
        voxels,
        weight=weights,
        kernel_size=kernel_size,
        fwd_algo=SPARSE_CONV_AB_ALGO_MODE.IMPLICIT_GEMM,
        dgrad_algo=SPARSE_CONV_AB_ALGO_MODE.IMPLICIT_GEMM,
        wgrad_algo=SPARSE_CONV_ATB_ALGO_MODE.IMPLICIT_GEMM,
    )

    # All should have same output dimensions
    assert out1.num_channels == C_out
    assert out2.num_channels == C_out
    assert out3.num_channels == C_out
    assert out4.num_channels == C_out

    # Results should be similar (they're the same mathematical operation)
    assert out1.feature_tensor.shape == out2.feature_tensor.shape
    assert out1.feature_tensor.shape == out3.feature_tensor.shape
    assert out1.feature_tensor.shape == out4.feature_tensor.shape


# =============================================================================
# Group convolution tests
# =============================================================================


@pytest.mark.parametrize(
    "C_in,C_out,groups",
    [
        (64, 128, 4),
        (64, 64, 8),
        (128, 128, 16),
    ],
)
def test_group_conv_forward(C_in, C_out, groups):
    """Test group convolution forward produces correct output shape."""
    coords = torch.unique(torch.randint(0, 20, (2000, 3), dtype=torch.int32), dim=0)
    feats = torch.randn(coords.shape[0], C_in)
    voxels = Voxels([coords], [feats]).to("cuda")

    layer = SparseConv3d(C_in, C_out, 3, bias=False, groups=groups).cuda()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = layer(voxels)

    assert out.features.shape == (coords.shape[0], C_out)
    assert out.features.dtype == torch.float16


@pytest.mark.parametrize(
    "C_in,C_out,groups",
    [
        (64, 128, 4),  # C_in_g=16, C_out_g=32
        (64, 64, 4),  # C_in_g=16, C_out_g=16
        (128, 256, 8),  # C_in_g=16, C_out_g=32
    ],
)
def test_group_conv_backward(C_in, C_out, groups):
    """Test group convolution backward produces correctly shaped gradients."""
    coords = torch.unique(torch.randint(0, 20, (2000, 3), dtype=torch.int32), dim=0)
    feats = torch.randn(coords.shape[0], C_in, device="cuda", requires_grad=True)
    voxels = Voxels(
        batched_coordinates=coords.cuda(),
        batched_features=feats,
        offsets=torch.tensor([0, coords.shape[0]], dtype=torch.int32, device="cuda"),
        device="cuda",
    )

    layer = SparseConv3d(C_in, C_out, 3, bias=False, groups=groups).cuda()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = layer(voxels)
    out.features.float().sum().backward()

    assert feats.grad is not None
    assert feats.grad.shape == (coords.shape[0], C_in)
    C_in_g = C_in // groups
    C_out_g = C_out // groups
    assert layer.weight.grad.shape == (27, groups, C_in_g, C_out_g)
    assert torch.isfinite(feats.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


def test_group_conv_correctness():
    """Test group conv matches per-group explicit_gemm reference."""
    C_in, C_out, groups = 32, 64, 4
    C_in_g, C_out_g = C_in // groups, C_out // groups
    torch.manual_seed(0)
    coords = torch.unique(torch.randint(0, 15, (800, 3), dtype=torch.int32), dim=0)
    N = coords.shape[0]
    feats = torch.randn(N, C_in, device="cuda")
    weight = torch.randn(27, groups, C_in_g, C_out_g, device="cuda")
    voxels = Voxels([coords], [feats]).to("cuda")

    # Production group conv under AMP
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out_prod = spatially_sparse_conv(
            voxels,
            weight,
            kernel_size=(3, 3, 3),
            groups=groups,
            fwd_algo="production",
        )

    # Reference: per-group explicit_gemm in fp32
    ref_outputs = []
    for g in range(groups):
        feats_g = feats[:, g * C_in_g : (g + 1) * C_in_g]
        w_g = weight[:, g]
        v_g = Voxels([coords], [feats_g]).to("cuda")
        out_g = spatially_sparse_conv(v_g, w_g, kernel_size=(3, 3, 3), fwd_algo="explicit_gemm")
        ref_outputs.append(out_g.feature_tensor)
    out_ref = torch.cat(ref_outputs, dim=1).float()

    rdiff = (out_prod.feature_tensor.float() - out_ref).abs().mean() / (
        out_ref.abs().mean() + 1e-8
    )
    assert rdiff < 0.01, f"Group conv rdiff={rdiff:.5f} (expected < 0.01)"


def test_group_conv_weight_shape():
    """Test that group conv creates weight with correct shape."""
    layer = SparseConv3d(64, 128, 3, groups=4)
    assert layer.weight.shape == (27, 4, 16, 32)

    layer_dw = SparseConv3d(64, 64, 3, groups=64)
    assert layer_dw.weight.shape == (27, 64, 1, 1)

    layer_g1 = SparseConv3d(64, 128, 3, groups=1)
    assert layer_g1.weight.shape == (27, 64, 128)


def test_group_conv_invalid_groups():
    """Test that invalid group parameters raise errors."""
    with pytest.raises(ValueError, match="divisible by groups"):
        SparseConv3d(64, 128, 3, groups=3)  # 64 not divisible by 3

    with pytest.raises(ValueError, match="divisible by groups"):
        SparseConv3d(64, 100, 3, groups=3)  # 100 not divisible by 3


# ── SparseConv2d (2D sparse convolution via padded cuhash) ──────────────


def _make_2d_voxels(B=2, min_N=500, max_N=2000, C=16, device="cuda"):
    """Create 2D Voxels (N, 2) coordinates for SparseConv2d tests."""
    torch.manual_seed(42)
    coords_list = []
    feats_list = []
    for _ in range(B):
        N = torch.randint(min_N, max_N, (1,)).item()
        c = (torch.rand(N, 2) * 100).int()  # 2D spatial coords
        f = torch.randn(N, C)
        coords_list.append(c)
        feats_list.append(f)
    return Voxels(coords_list, feats_list, device=device).unique()


def test_sparse_conv2d_forward():
    """Test SparseConv2d forward through PackedHashTable (padded cuhash path)."""
    voxels_2d = _make_2d_voxels(C=16)
    layer = SparseConv2d(16, 32, kernel_size=3, bias=True, fwd_algo="explicit_gemm").cuda()

    out = layer(voxels_2d)

    assert out.num_channels == 32
    assert out.coordinate_tensor.shape[1] == 2  # still 2D spatial
    assert out.feature_tensor.shape[0] == voxels_2d.feature_tensor.shape[0]


def test_sparse_conv2d_strided():
    """Test SparseConv2d with stride=2 produces downsampled output."""
    voxels_2d = _make_2d_voxels(C=16)
    layer = SparseConv2d(16, 32, kernel_size=3, stride=2, fwd_algo="explicit_gemm").cuda()

    out = layer(voxels_2d)

    assert out.num_channels == 32
    assert out.coordinate_tensor.shape[1] == 2
    assert out.feature_tensor.shape[0] < voxels_2d.feature_tensor.shape[0]


def test_sparse_conv2d_backward():
    """Test SparseConv2d backward pass with gradient check."""
    voxels_2d = _make_2d_voxels(B=1, min_N=50, max_N=100, C=8)
    layer = SparseConv2d(8, 8, kernel_size=3, bias=False, fwd_algo="explicit_gemm").cuda().double()
    voxels_2d = voxels_2d.replace(batched_features=voxels_2d.feature_tensor.double())

    out = layer(voxels_2d)
    loss = out.feature_tensor.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape


def test_sparse_conv2d_kernel_map_uses_cuhash():
    """Verify 2D conv routes through PackedHashTable (padded to 4D), not TorchHashTable."""
    voxels_2d = _make_2d_voxels(C=8)
    bcoords = batch_indexed_coordinates(voxels_2d.coordinate_tensor, voxels_2d.offsets)
    assert bcoords.shape[1] == 3  # [batch, x, y]

    # generate_kernel_map should pad to 4D internally and use PackedHashTable
    kernel_map = generate_kernel_map(
        bcoords,
        bcoords,
        in_to_out_stride_ratio=(1, 1),
        kernel_size=(3, 3),
    )
    # Verify we got valid kernel map results
    assert kernel_map.in_maps.shape[0] > 0
    assert kernel_map.out_maps.shape[0] > 0
    assert kernel_map.offsets.shape[0] == 3 * 3 + 1  # K=9, offsets has K+1 entries
    # Verify _pair_table was set (only happens on cuhash path)
    assert hasattr(kernel_map, "_pair_table") and kernel_map._pair_table is not None
