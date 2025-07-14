# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.nn.functional.sparse_conv_depth import (
    _explicit_depthwise_forward_logic,
    _implicit_depthwise_forward_logic,
    _explicit_depthwise_backward_logic,
    _implicit_depthwise_backward_logic,
    spatially_sparse_depthwise_conv,
    UnifiedSpatiallySparseDepthwiseConvFunction,
    SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE,
    SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE,
    _run_depthwise_forward_benchmarks,
    _run_depthwise_backward_benchmarks,
)
from warpconvnet.utils.ntuple import ntuple


@pytest.fixture
def setup_small_depthwise_conv_data():
    """Setup small test data for gradient checking (faster execution)."""
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create smaller test parameters for faster gradient checking
    B, N, C = 1, 50, 4
    kernel_size = (3, 3, 3)
    kernel_volume = int(np.prod(kernel_size))

    # Create small coordinates and features
    coords = [torch.randint(0, 10, (N, 3), dtype=torch.int32, device=device)]
    features = [
        torch.randn(N, C, device=device, dtype=torch.float64)
    ]  # Use float64 for better precision

    # Create voxels
    voxels = Voxels(coords, features, device=str(device)).unique()

    # Create depthwise weight (K, C) with float64 for better precision
    weight = torch.randn(kernel_volume, C, device=device, dtype=torch.float64, requires_grad=True)

    # Generate kernel map
    batch_indexed_coords = batch_indexed_coordinates(
        voxels.coordinate_tensor,
        voxels.offsets,
    )
    kernel_map = generate_kernel_map(
        batch_indexed_coords,
        batch_indexed_coords,
        (1, 1, 1),  # in_to_out_stride_ratio
        kernel_size,
        (1, 1, 1),  # kernel_dilation
        skip_symmetric_kernel_map=False,
    )

    return {
        "voxels": voxels,
        "weight": weight,
        "kernel_map": kernel_map,
        "kernel_size": kernel_size,
        "device": device,
    }


@pytest.fixture
def setup_depthwise_conv_data():
    """Setup test data for depthwise convolution."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test parameters
    B, min_N, max_N, C = 2, 1000, 5000, 8
    kernel_size = (3, 3, 3)
    kernel_volume = int(np.prod(kernel_size))

    # Generate random batch sizes
    Ns = torch.randint(min_N, max_N, (B,))

    # Create coordinates and features
    coords = []
    features = []
    for N in Ns:
        n_int = int(N.item())
        coord = torch.randint(0, 100, (n_int, 3), dtype=torch.int32, device=device)
        feat = torch.randn(n_int, C, device=device)
        coords.append(coord)
        features.append(feat)

    # Create voxels
    voxels = Voxels(coords, features, device=str(device)).unique()

    # Create depthwise weight (K, C) - each channel has its own kernel
    weight = torch.randn(kernel_volume, C, device=device, requires_grad=True)

    # Generate kernel map
    batch_indexed_coords = batch_indexed_coordinates(
        voxels.coordinate_tensor,
        voxels.offsets,
    )
    kernel_map = generate_kernel_map(
        batch_indexed_coords,
        batch_indexed_coords,
        (1, 1, 1),  # in_to_out_stride_ratio
        kernel_size,
        (1, 1, 1),  # kernel_dilation
        skip_symmetric_kernel_map=False,
    )

    return {
        "voxels": voxels,
        "weight": weight,
        "kernel_map": kernel_map,
        "kernel_size": kernel_size,
        "device": device,
    }


def test_explicit_depthwise_forward(setup_depthwise_conv_data):
    """Test explicit depthwise forward pass."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Test forward pass
    output = _explicit_depthwise_forward_logic(
        voxels.feature_tensor,
        weight,
        kernel_map,
        voxels.coordinate_tensor.shape[0],
        compute_dtype=None,
    )

    # Verify output shape
    assert output.shape == (voxels.coordinate_tensor.shape[0], voxels.num_channels)
    assert output.dtype == voxels.feature_tensor.dtype
    assert output.device == voxels.feature_tensor.device


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available - skipping implicit depthwise tests"
)
def test_implicit_depthwise_forward(setup_depthwise_conv_data):
    """Test implicit depthwise forward pass."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    try:
        # Test forward pass
        output = _implicit_depthwise_forward_logic(
            voxels.feature_tensor,
            weight,
            kernel_map,
            voxels.coordinate_tensor.shape[0],
            compute_dtype=None,
        )

        # Verify output shape
        assert output.shape == (voxels.coordinate_tensor.shape[0], voxels.num_channels)
        assert output.dtype == voxels.feature_tensor.dtype
        assert output.device == voxels.feature_tensor.device
    except ImportError:
        pytest.skip("warpconvnet._C not available - skipping implicit tests")


def test_explicit_vs_implicit_forward_consistency(setup_depthwise_conv_data):
    """Test that explicit and implicit forward passes produce similar results."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Explicit forward
    output_explicit = _explicit_depthwise_forward_logic(
        voxels.feature_tensor,
        weight,
        kernel_map,
        voxels.coordinate_tensor.shape[0],
        compute_dtype=None,
    )

    try:
        # Implicit forward
        output_implicit = _implicit_depthwise_forward_logic(
            voxels.feature_tensor,
            weight,
            kernel_map,
            voxels.coordinate_tensor.shape[0],
            compute_dtype=None,
        )

        # Check consistency (allow for small numerical differences)
        assert torch.allclose(output_explicit, output_implicit, atol=1e-5, rtol=1e-4)
    except ImportError:
        pytest.skip("warpconvnet._C not available - skipping implicit comparison")


def test_explicit_depthwise_backward(setup_depthwise_conv_data):
    """Test explicit depthwise backward pass."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Create gradient output
    grad_output = torch.randn_like(voxels.feature_tensor)

    # Test backward pass
    grad_in, grad_weight = _explicit_depthwise_backward_logic(
        grad_output,
        voxels.feature_tensor,
        weight,
        kernel_map,
        compute_dtype=None,
        device=data["device"],
    )

    # Verify gradient shapes
    assert grad_in.shape == voxels.feature_tensor.shape
    assert grad_weight.shape == weight.shape
    assert grad_in.dtype == voxels.feature_tensor.dtype
    assert grad_weight.dtype == weight.dtype


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available - skipping implicit depthwise tests"
)
def test_implicit_depthwise_backward(setup_depthwise_conv_data):
    """Test implicit depthwise backward pass."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    try:
        # Create gradient output
        grad_output = torch.randn_like(voxels.feature_tensor)

        # Test backward pass
        grad_in, grad_weight = _implicit_depthwise_backward_logic(
            grad_output,
            voxels.feature_tensor,
            weight,
            kernel_map,
            compute_dtype=None,
            device=data["device"],
        )

        # Verify gradient shapes
        assert grad_in.shape == voxels.feature_tensor.shape
        assert grad_weight.shape == weight.shape
        assert grad_in.dtype == voxels.feature_tensor.dtype
        assert grad_weight.dtype == weight.dtype
    except ImportError:
        pytest.skip("warpconvnet._C not available - skipping implicit tests")


@pytest.mark.parametrize(
    "fwd_algo",
    [
        SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT,
        SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT,
    ],
)
@pytest.mark.parametrize(
    "bwd_algo",
    [
        SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
        SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT,
    ],
)
def test_unified_depthwise_conv_function(setup_depthwise_conv_data, fwd_algo, bwd_algo):
    """Test unified depthwise convolution function with different algorithms."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Skip implicit tests if CUDA not available
    if not torch.cuda.is_available() and (
        fwd_algo == SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT
        or bwd_algo == SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT
    ):
        pytest.skip("CUDA not available - skipping implicit algorithm tests")

    try:
        # Test the unified function
        output = UnifiedSpatiallySparseDepthwiseConvFunction.apply(
            voxels.feature_tensor,
            weight,
            kernel_map,
            voxels.coordinate_tensor.shape[0],
            fwd_algo,
            bwd_algo,
            None,  # compute_dtype
        )

        # Verify output is not None
        assert output is not None
        assert output.shape == (voxels.coordinate_tensor.shape[0], voxels.num_channels)
        assert output.dtype == voxels.feature_tensor.dtype
        assert output.device == voxels.feature_tensor.device

        # Test gradient computation
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        assert weight.grad is not None
        assert weight.grad.shape == weight.shape

    except ImportError:
        if (
            fwd_algo == SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT
            or bwd_algo == SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT
        ):
            pytest.skip("warpconvnet._C not available - skipping implicit tests")
        else:
            raise


def test_spatially_sparse_depthwise_conv_api(setup_depthwise_conv_data):
    """Test the public API function for spatially sparse depthwise convolution."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Test the public API
    output = spatially_sparse_depthwise_conv(
        voxels.feature_tensor,
        weight,
        kernel_map,
        voxels.coordinate_tensor.shape[0],
        fwd_algo=SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT,
        bwd_algo=SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
        compute_dtype=None,
    )

    # Verify output
    assert output.shape == (voxels.coordinate_tensor.shape[0], voxels.num_channels)
    assert output.dtype == voxels.feature_tensor.dtype
    assert output.device == voxels.feature_tensor.device


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available - skipping benchmark tests"
)
def test_depthwise_forward_benchmarks(setup_depthwise_conv_data):
    """Test depthwise forward benchmarking functionality."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Run benchmarks with minimal iterations for testing
    results = _run_depthwise_forward_benchmarks(
        voxels.feature_tensor,
        weight,
        kernel_map,
        voxels.coordinate_tensor.shape[0],
        compute_dtype=None,
        warmup_iters=1,
        benchmark_iters=1,
    )

    # Verify benchmark results
    assert len(results) > 0
    assert all(len(result) == 3 for result in results)  # (algo, params, time)
    assert all(isinstance(result[2], float) for result in results)  # time is float

    # Results should be sorted by time
    times = [result[2] for result in results]
    assert times == sorted(times)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available - skipping benchmark tests"
)
def test_depthwise_backward_benchmarks(setup_depthwise_conv_data):
    """Test depthwise backward benchmarking functionality."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Create gradient output
    grad_output = torch.randn_like(voxels.feature_tensor)

    # Run benchmarks with minimal iterations for testing
    results = _run_depthwise_backward_benchmarks(
        grad_output,
        voxels.feature_tensor,
        weight,
        kernel_map,
        compute_dtype=None,
        device=data["device"],
        warmup_iters=1,
        benchmark_iters=1,
    )

    # Verify benchmark results
    assert len(results) > 0
    assert all(len(result) == 3 for result in results)  # (algo, params, time)
    assert all(isinstance(result[2], float) for result in results)  # time is float

    # Results should be sorted by time
    times = [result[2] for result in results]
    assert times == sorted(times)


def test_auto_algorithm_selection(setup_depthwise_conv_data):
    """Test automatic algorithm selection."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Test AUTO mode for forward algorithm
    output_auto = spatially_sparse_depthwise_conv(
        voxels.feature_tensor,
        weight,
        kernel_map,
        voxels.coordinate_tensor.shape[0],
        fwd_algo=SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.AUTO,
        bwd_algo=SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
        compute_dtype=None,
    )

    # Test explicit mode for comparison
    output_explicit = spatially_sparse_depthwise_conv(
        voxels.feature_tensor,
        weight,
        kernel_map,
        voxels.coordinate_tensor.shape[0],
        fwd_algo=SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT,
        bwd_algo=SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
        compute_dtype=None,
    )

    # Verify outputs have same shape and are numerically close
    assert output_auto.shape == output_explicit.shape
    # Allow for small numerical differences due to different algorithms
    assert torch.allclose(output_auto, output_explicit, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("compute_dtype", [None, torch.float32, torch.float16])
def test_different_compute_dtypes(setup_depthwise_conv_data, compute_dtype):
    """Test depthwise convolution with different compute dtypes."""
    data = setup_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Skip float16 if not supported
    if compute_dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("Float16 requires CUDA")

    # Test with specified compute dtype
    output = spatially_sparse_depthwise_conv(
        voxels.feature_tensor,
        weight,
        kernel_map,
        voxels.coordinate_tensor.shape[0],
        fwd_algo=SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT,
        bwd_algo=SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
        compute_dtype=compute_dtype,
    )

    # Verify output
    assert output.shape == (voxels.coordinate_tensor.shape[0], voxels.num_channels)
    # Output should maintain input dtype regardless of compute_dtype
    assert output.dtype == voxels.feature_tensor.dtype


def test_explicit_depthwise_gradcheck(setup_small_depthwise_conv_data):
    """Test gradient checking for explicit depthwise convolution."""
    data = setup_small_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    def depthwise_conv_func(in_features, weight_tensor):
        return UnifiedSpatiallySparseDepthwiseConvFunction.apply(
            in_features,
            weight_tensor,
            kernel_map,
            voxels.coordinate_tensor.shape[0],
            SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT,
            SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
            None,  # compute_dtype
        )

    # Prepare inputs for gradcheck
    input_features = voxels.feature_tensor.clone().detach().requires_grad_(True)
    weight_tensor = weight.clone().detach().requires_grad_(True)

    # Run gradient check
    assert torch.autograd.gradcheck(
        depthwise_conv_func,
        (input_features, weight_tensor),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        check_undefined_grad=False,  # Some gradients might be undefined for sparse operations
    ), "Explicit depthwise convolution failed gradient check"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available - skipping implicit gradient check"
)
def test_implicit_depthwise_gradcheck(setup_small_depthwise_conv_data):
    """Test gradient checking for implicit depthwise convolution."""
    data = setup_small_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    try:

        def depthwise_conv_func(in_features, weight_tensor):
            return UnifiedSpatiallySparseDepthwiseConvFunction.apply(
                in_features,
                weight_tensor,
                kernel_map,
                voxels.coordinate_tensor.shape[0],
                SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.IMPLICIT,
                SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.IMPLICIT,
                None,  # compute_dtype
            )

        # Prepare inputs for gradcheck
        input_features = voxels.feature_tensor.clone().detach().requires_grad_(True)
        weight_tensor = weight.clone().detach().requires_grad_(True)

        # Run gradient check
        assert torch.autograd.gradcheck(
            depthwise_conv_func,
            (input_features, weight_tensor),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            check_undefined_grad=False,
        ), "Implicit depthwise convolution failed gradient check"

    except ImportError:
        pytest.skip("warpconvnet._C not available - skipping implicit gradient check")


def test_depthwise_conv_manual_grad_check(setup_small_depthwise_conv_data):
    """Manual gradient check to verify specific gradient computations."""
    data = setup_small_depthwise_conv_data
    voxels = data["voxels"]
    weight = data["weight"]
    kernel_map = data["kernel_map"]

    # Forward pass with gradient tracking
    input_features = voxels.feature_tensor.clone().detach().requires_grad_(True)
    weight_tensor = weight.clone().detach().requires_grad_(True)

    output = UnifiedSpatiallySparseDepthwiseConvFunction.apply(
        input_features,
        weight_tensor,
        kernel_map,
        voxels.coordinate_tensor.shape[0],
        SPARSE_DEPTHWISE_CONV_FWD_ALGO_MODE.EXPLICIT,
        SPARSE_DEPTHWISE_CONV_BWD_ALGO_MODE.EXPLICIT,
        None,
    )

    # Create a simple loss
    loss = output.sum()
    loss.backward()

    # Verify gradients exist and have correct shapes
    assert input_features.grad is not None, "Input features gradient is None"
    assert weight_tensor.grad is not None, "Weight gradient is None"

    assert input_features.grad.shape == input_features.shape, "Input gradient shape mismatch"
    assert weight_tensor.grad.shape == weight_tensor.shape, "Weight gradient shape mismatch"

    # Verify gradients are not all zeros (indicating computation occurred)
    assert not torch.allclose(
        input_features.grad, torch.zeros_like(input_features.grad)
    ), "Input gradients are all zeros"
    assert not torch.allclose(
        weight_tensor.grad, torch.zeros_like(weight_tensor.grad)
    ), "Weight gradients are all zeros"

    # Verify gradients have reasonable magnitudes
    input_grad_norm = torch.norm(input_features.grad)
    weight_grad_norm = torch.norm(weight_tensor.grad)

    assert input_grad_norm > 1e-6, f"Input gradient norm too small: {input_grad_norm}"
    assert weight_grad_norm > 1e-6, f"Weight gradient norm too small: {weight_grad_norm}"
    assert input_grad_norm < 1e3, f"Input gradient norm too large: {input_grad_norm}"
    assert weight_grad_norm < 1e3, f"Weight gradient norm too large: {weight_grad_norm}"
