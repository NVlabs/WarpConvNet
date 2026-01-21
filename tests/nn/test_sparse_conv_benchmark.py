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
    SpatiallySparseConvExplicitGEMMFunction,
    SpatiallySparseConvImplicitGEMMFunction,
    UnifiedSpatiallySparseConvFunction,
    spatially_sparse_conv,
    _BENCHMARK_FORWARD_RESULTS,
    _BENCHMARK_BACKWARD_RESULTS,
    SpatiallySparseConvConfig,
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


def wrapped_spatially_sparse_conv_with_sync(*args, **kwargs):
    torch.cuda.synchronize()
    result = spatially_sparse_conv(*args, **kwargs)
    torch.cuda.synchronize()
    return result


# Renamed and split from the original test_sparse_conv
def test_sparse_conv_explicit_benchmark(setup_voxels, benchmark):
    """Benchmark EXPLICIT_GEMM sparse convolution."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)
    bias = torch.randn(C_out).to(voxels.device)

    benchmark.pedantic(
        wrapped_spatially_sparse_conv_with_sync,
        args=(
            voxels,
            weights,
            kernel_size,
        ),
        kwargs={
            "bias": bias,
            "stride": stride,
            "fwd_algo": SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM,
            "bwd_algo": SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM,
            "implicit_matmul_fwd_block_size": None,
            "implicit_matmul_bwd_block_size": None,
        },
        rounds=5,
        iterations=1,
    )


def test_sparse_conv_implicit_benchmark(setup_voxels, benchmark):
    """Benchmark IMPLICIT_GEMM sparse convolution."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)
    bias = torch.randn(C_out).to(voxels.device)

    fixed_implicit_fwd_block_size = 16
    fixed_implicit_bwd_block_size = 16

    benchmark.pedantic(
        wrapped_spatially_sparse_conv_with_sync,
        args=(
            voxels,
            weights,
            kernel_size,
        ),
        kwargs={
            "bias": bias,
            "stride": stride,
            "fwd_algo": SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM,
            "bwd_algo": SPARSE_CONV_BWD_ALGO_MODE.IMPLICIT_GEMM,
            "implicit_matmul_fwd_block_size": fixed_implicit_fwd_block_size,
            "implicit_matmul_bwd_block_size": fixed_implicit_bwd_block_size,
        },
        rounds=5,
        iterations=1,
    )


def test_sparse_conv_auto_benchmark_and_correctness(setup_voxels, benchmark):
    """Benchmark AUTO mode, check correctness and cache population."""
    voxels = setup_voxels
    C_in, C_out = voxels.num_channels, 13

    _BENCHMARK_FORWARD_RESULTS.clear()

    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    num_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    weights = torch.randn(num_kernels, C_in, C_out).to(voxels.device)
    bias = torch.randn(C_out).to(voxels.device)

    # Run explicit once for correctness comparison (not benchmarked here, but could be)
    out_explicit_features = spatially_sparse_conv(
        voxels,
        weights,
        kernel_size,
        bias=bias,
        stride=stride,
        fwd_algo=SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM,
        bwd_algo=SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM,
    ).feature_tensor

    out_auto_result = benchmark.pedantic(
        wrapped_spatially_sparse_conv_with_sync,
        args=(
            voxels,
            weights,
            kernel_size,
        ),
        kwargs={
            "bias": bias,
            "stride": stride,
            "fwd_algo": SPARSE_CONV_FWD_ALGO_MODE.AUTO,
            "bwd_algo": SPARSE_CONV_BWD_ALGO_MODE.AUTO,
            "implicit_matmul_fwd_block_size": None,
            "implicit_matmul_bwd_block_size": None,
        },
        rounds=5,
        iterations=1,
    )
    out_auto_features = out_auto_result.feature_tensor

    assert out_auto_features.shape[1] == C_out
    assert torch.allclose(out_auto_features, out_explicit_features, atol=1e-6)

    found_config_in_cache = False
    for config_key in _BENCHMARK_FORWARD_RESULTS:
        if (
            config_key.in_channels == C_in
            and config_key.out_channels == C_out
            and config_key.kernel_volume == num_kernels
        ):
            found_config_in_cache = True
            assert _BENCHMARK_FORWARD_RESULTS[config_key] is not None
            assert isinstance(_BENCHMARK_FORWARD_RESULTS[config_key][0], SPARSE_CONV_FWD_ALGO_MODE)
            break
    assert (
        found_config_in_cache
    ), "Benchmark cache was not populated by AUTO mode or key not found."


def test_sparse_conv_explicit_backward(setup_small_voxels):
    """Test sparse convolution gradients."""
    voxels = setup_small_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)

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
        (feature_tensor, weights, kernel_map, batch_indexed_out_coords.shape[0]),
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
            batch_indexed_out_coords.shape[0],
        ),
        eps=1e-3,
        atol=1e-3,
        rtol=1e-3,
    )


def test_sparse_conv_unified_auto_gradcheck(setup_small_voxels):
    """Test UnifiedSpatiallySparseConvFunction gradients with AUTO algo mode."""
    voxels = setup_small_voxels
    C_in, C_out = voxels.num_channels, 13
    kernel_size_val = (3, 3, 3)
    stride_val = (2, 2, 2)

    # Clear benchmark caches
    _BENCHMARK_FORWARD_RESULTS.clear()
    _BENCHMARK_BACKWARD_RESULTS.clear()

    num_kernels = kernel_size_val[0] * kernel_size_val[1] * kernel_size_val[2]
    weights = torch.randn(num_kernels, C_in, C_out, device=voxels.device, dtype=torch.float64)

    feature_tensor = (
        voxels.feature_tensor.detach().clone().to(dtype=torch.float64).requires_grad_(True)
    )

    # Generate kernel_map and num_out_coords (as UnifiedSpatiallySparseConvFunction expects them)
    # This part mimics what spatially_sparse_conv would do internally.
    from warpconvnet.nn.functional.sparse_conv import generate_output_coords_and_kernel_map

    # Effective compute_dtype for gradcheck should be float64
    effective_compute_dtype = torch.float64

    # Make a temporary voxels object for this internal call if necessary, to avoid side effects
    temp_voxels_for_map = Voxels(
        batched_coordinates=voxels.coordinate_tensor,
        batched_features=voxels.feature_tensor,
        offsets=voxels.offsets,
        device=voxels.device,
    ).unique()

    batch_indexed_out_coords, _, kernel_map_generated = generate_output_coords_and_kernel_map(
        input_sparse_tensor=temp_voxels_for_map,
        kernel_size=kernel_size_val,
        kernel_dilation=(1, 1, 1),
        stride=stride_val,
        generative=False,
        transposed=False,
        output_spatially_sparse_tensor=None,
        kernel_search_batch_size=max(num_kernels // kernel_size_val[0], 8),
        stride_mode=STRIDED_CONV_MODE.STRIDE_ONLY,
        out_code_backend="hashmap",
    )
    num_out_coords_generated = batch_indexed_out_coords.shape[0]

    # Inputs for UnifiedSpatiallySparseConvFunction.apply
    fn_inputs = (
        feature_tensor,
        weights,
        kernel_map_generated,
        num_out_coords_generated,
        SPARSE_CONV_FWD_ALGO_MODE.AUTO,
        SPARSE_CONV_BWD_ALGO_MODE.AUTO,
        effective_compute_dtype,
        None,
        None,
    )

    torch.autograd.gradcheck(
        UnifiedSpatiallySparseConvFunction.apply,
        fn_inputs,
        eps=1e-5,
        atol=1e-4,
        rtol=1e-4,
    )

    # Check if benchmark caches were populated
    assert (
        len(_BENCHMARK_FORWARD_RESULTS) > 0
    ), "Forward benchmark cache not populated after AUTO gradcheck"
    assert (
        len(_BENCHMARK_BACKWARD_RESULTS) > 0
    ), "Backward benchmark cache not populated after AUTO gradcheck"

    # Simplified check as before for test_sparse_conv
    found_fwd_config_in_cache_grad = False
    for config_key in _BENCHMARK_FORWARD_RESULTS:
        if (
            config_key.in_channels == C_in
            and config_key.out_channels == C_out
            and config_key.kernel_volume == num_kernels
        ):
            found_fwd_config_in_cache_grad = True
            break
    assert (
        found_fwd_config_in_cache_grad
    ), "AUTO mode did not populate forward cache correctly during gradcheck."

    found_bwd_config_in_cache_grad = False
    for config_key in _BENCHMARK_BACKWARD_RESULTS:
        if (
            config_key.in_channels == C_in
            and config_key.out_channels == C_out
            and config_key.kernel_volume == num_kernels
        ):
            found_bwd_config_in_cache_grad = True
            break
    assert (
        found_bwd_config_in_cache_grad
    ), "AUTO mode did not populate backward cache correctly during gradcheck."
