# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest
import numpy as np
import torch

import warpconvnet._C as _C


def create_test_data(device="cuda", dtype=torch.float32, seed=42):
    """Create test data for segmented arithmetic testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Test parameters
    N = 100  # Total number of rows in B and D
    C = 32  # Number of channels
    K = 5  # Number of segments

    # Create segment sizes (variable lengths)
    segment_sizes = [15, 25, 30, 20, 10]  # Must sum to N
    assert sum(segment_sizes) == N, f"Segment sizes must sum to N={N}"

    # Create segment offsets [0, 15, 40, 70, 90, 100]
    segment_offsets = torch.tensor(
        [0] + [sum(segment_sizes[: i + 1]) for i in range(len(segment_sizes))],
        device=device,
        dtype=torch.int32,
    )

    # Create input tensors
    B = torch.randn(N, C, device=device, dtype=dtype)
    C_segments = torch.randn(K, C, device=device, dtype=dtype)
    D = torch.zeros(N, C, device=device, dtype=dtype)

    return B, C_segments, D, segment_offsets, segment_sizes


def create_simple_test_data(device="cuda", dtype=torch.float32):
    """Create simple test data for easy verification."""
    # Simple test: 2 segments of size 2 each, 2 channels
    N, C, K = 4, 2, 2

    # Create predictable data
    B = torch.tensor(
        [
            [1.0, 2.0],  # Segment 0
            [3.0, 4.0],  # Segment 0
            [5.0, 6.0],  # Segment 1
            [7.0, 8.0],  # Segment 1
        ],
        device=device,
        dtype=dtype,
    )

    C_segments = torch.tensor(
        [
            [10.0, 20.0],  # For segment 0
            [30.0, 40.0],  # For segment 1
        ],
        device=device,
        dtype=dtype,
    )

    D = torch.zeros(N, C, device=device, dtype=dtype)

    segment_offsets = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)

    return B, C_segments, D, segment_offsets


def compute_reference_result(B, C_segments, segment_offsets, operation):
    """Compute reference result using CPU operations."""
    N, C = B.shape
    K = len(segment_offsets) - 1

    result = torch.zeros_like(B)

    for k in range(K):
        start = int(segment_offsets[k].item())
        end = int(segment_offsets[k + 1].item())

        # Apply operation to segment
        if operation == "add":
            result[start:end] = B[start:end] + C_segments[k : k + 1]
        elif operation in ["subtract", "sub"]:
            result[start:end] = B[start:end] - C_segments[k : k + 1]
        elif operation in ["multiply", "mul"]:
            result[start:end] = B[start:end] * C_segments[k : k + 1]
        elif operation in ["divide", "div"]:
            result[start:end] = B[start:end] / C_segments[k : k + 1]
        else:
            raise ValueError(f"Unknown operation: {operation}")

    return result


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_segmented_arithmetic_basic(operation, dtype):
    """Test basic functionality of segmented arithmetic operations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Create test data
    B, C_segments, D, segment_offsets = create_simple_test_data(device="cuda", dtype=dtype)

    # Compute reference result
    expected = compute_reference_result(B, C_segments, segment_offsets, operation)

    # Run kernel
    _C.utils.segmented_arithmetic(
        tensor_b=B,
        tensor_c=C_segments,
        tensor_d=D,
        offsets=segment_offsets,
        operation=operation,
        kernel_type="basic",
    )

    # Compare results
    if dtype == torch.float16:
        tolerance = 1e-2  # Lower precision for float16
    else:
        tolerance = 1e-5

    assert torch.allclose(
        D, expected, atol=tolerance, rtol=tolerance
    ), f"Operation {operation} failed. Max diff: {torch.max(torch.abs(D - expected))}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_segmented_arithmetic_dtypes(dtype):
    """Test different data types."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Skip float64 for half precision operations in simple test
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BFloat16 not supported on this device")

    device = torch.device("cuda")

    # Create test data
    B, C_segments, D, segment_offsets, _ = create_test_data(device="cuda", dtype=dtype)

    # Test addition operation
    expected = compute_reference_result(B, C_segments, segment_offsets, "add")

    # Run kernel
    _C.utils.segmented_arithmetic(
        tensor_b=B, tensor_c=C_segments, tensor_d=D, offsets=segment_offsets, operation="add"
    )

    # Set tolerance based on dtype
    if dtype == torch.float16:
        tolerance = 1e-2
    elif dtype == torch.float64:
        tolerance = 1e-10
    else:
        tolerance = 1e-5

    assert torch.allclose(D, expected, atol=tolerance, rtol=tolerance)


def test_segmented_arithmetic_large_input():
    """Test with larger input sizes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Large test parameters
    N = 10000
    C = 256
    K = 100

    # Create segment sizes
    base_size = N // K
    segment_sizes = [base_size] * (K - 1)
    segment_sizes.append(N - sum(segment_sizes))  # Handle remainder

    # Create segment offsets
    segment_offsets = torch.tensor(
        [0] + [sum(segment_sizes[: i + 1]) for i in range(len(segment_sizes))],
        device=device,
        dtype=torch.int32,
    )

    # Create tensors
    B = torch.randn(N, C, device=device, dtype=torch.float32)
    C_segments = torch.randn(K, C, device=device, dtype=torch.float32)
    D = torch.zeros(N, C, device=device, dtype=torch.float32)

    # Run kernel
    _C.utils.segmented_arithmetic(
        tensor_b=B, tensor_c=C_segments, tensor_d=D, offsets=segment_offsets, operation="multiply"
    )

    # Verify a few random segments
    for _ in range(5):
        k = torch.randint(0, K, (1,)).item()
        start = int(segment_offsets[k].item())
        end = int(segment_offsets[k + 1].item())

        expected_segment = B[start:end] * C_segments[k : k + 1]
        actual_segment = D[start:end]

        assert torch.allclose(actual_segment, expected_segment, atol=1e-4, rtol=1e-4)


def test_segmented_arithmetic_edge_cases():
    """Test edge cases and boundary conditions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Test case 1: Single segment
    B = torch.randn(10, 5, device=device, dtype=torch.float32)
    C_segments = torch.randn(1, 5, device=device, dtype=torch.float32)
    D = torch.zeros_like(B)
    segment_offsets = torch.tensor([0, 10], device=device, dtype=torch.int32)

    _C.utils.segmented_arithmetic(B, C_segments, D, segment_offsets, "add")
    expected = B + C_segments[0:1]
    assert torch.allclose(D, expected, atol=1e-5)

    # Test case 2: Many small segments (size 1)
    N = 20
    B = torch.randn(N, 3, device=device, dtype=torch.float32)
    C_segments = torch.randn(N, 3, device=device, dtype=torch.float32)
    D = torch.zeros_like(B)
    segment_offsets = torch.arange(N + 1, device=device, dtype=torch.int32)

    _C.utils.segmented_arithmetic(B, C_segments, D, segment_offsets, "subtract")
    expected = B - C_segments
    assert torch.allclose(D, expected, atol=1e-5)


def test_segmented_arithmetic_error_cases():
    """Test error handling and validation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    B = torch.randn(10, 5, device=device, dtype=torch.float32)
    C_segments = torch.randn(2, 5, device=device, dtype=torch.float32)
    D = torch.zeros_like(B)
    segment_offsets = torch.tensor([0, 5, 10], device=device, dtype=torch.int32)

    # Test invalid operation
    with pytest.raises(Exception):
        _C.utils.segmented_arithmetic(B, C_segments, D, segment_offsets, "invalid_op")

    # Test dimension mismatch
    C_wrong = torch.randn(2, 3, device=device, dtype=torch.float32)  # Wrong channels
    with pytest.raises(Exception):
        _C.utils.segmented_arithmetic(B, C_wrong, D, segment_offsets, "add")

    # Test wrong offset size
    wrong_offsets = torch.tensor([0, 10], device=device, dtype=torch.int32)  # K+1 != 3
    with pytest.raises(Exception):
        _C.utils.segmented_arithmetic(B, C_segments, D, wrong_offsets, "add")

    # Test CPU tensors (should fail)
    B_cpu = torch.randn(10, 5, dtype=torch.float32)
    with pytest.raises(Exception):
        _C.utils.segmented_arithmetic(B_cpu, C_segments, D, segment_offsets, "add")


def test_segmented_arithmetic_inplace_modification():
    """Test that D tensor is modified in-place correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    B, C_segments, D, segment_offsets = create_simple_test_data(device=device, dtype=torch.float32)

    # Store initial D tensor ID
    D_id = id(D)
    D_initial = D.clone()

    # Run operation
    _C.utils.segmented_arithmetic(B, C_segments, D, segment_offsets, "add")

    # Verify tensor is modified in-place
    assert id(D) == D_id, "Tensor D should be modified in-place"
    assert not torch.equal(D, D_initial), "Tensor D should be modified"


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_segmented_arithmetic_operation_aliases(operation):
    """Test that operation aliases work correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    B, C_segments, D, segment_offsets = create_simple_test_data(device=device, dtype=torch.float32)

    # This should not raise an exception
    _C.utils.segmented_arithmetic(B, C_segments, D, segment_offsets, operation)

    # Verify result is not all zeros (operation was applied)
    assert not torch.allclose(D, torch.zeros_like(D))


def test_segmented_arithmetic_memory_layout():
    """Test with non-contiguous tensors to verify contiguity handling."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Create non-contiguous tensors
    B_large = torch.randn(20, 10, device=device, dtype=torch.float32)
    B = B_large[:10, :5]  # Non-contiguous slice

    C_segments = torch.randn(2, 5, device=device, dtype=torch.float32)
    D = torch.zeros(10, 5, device=device, dtype=torch.float32)
    segment_offsets = torch.tensor([0, 5, 10], device=device, dtype=torch.int32)

    # Should work (tensors will be made contiguous internally)
    _C.utils.segmented_arithmetic(B, C_segments, D, segment_offsets, "add")

    # Verify result is meaningful
    expected = compute_reference_result(B, C_segments, segment_offsets, "add")
    assert torch.allclose(D, expected, atol=1e-5)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
