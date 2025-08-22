#!/usr/bin/env python3
"""
Test script for the templated implicit reduction CUDA kernel.

This test suite validates the clean template-based implementation following
the CUTLASS pattern. Features:
- Single templated kernel implementation for all data types
- Optional B parameter with compile-time optimization
- Clean namespace organization
- Multiple data types (float32, float16, bfloat16, float64)
- Status-based error handling

Operation: result[c] = ∑_i A[a_indices[i], c] * B[b_indices[i], c]
If B is None, treated as all ones: result[c] = ∑_i A[a_indices[i], c]
"""

import torch

from tests.common import compare_results, rand_clamped, rand_indices


def test_implicit_reduction_basic_with_b():
    """Test basic functionality with B matrix."""
    print("Testing implicit reduction kernel with B matrix...")

    # Import the warpconvnet module
    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        print("Make sure the module is compiled and available")
        return False

    # Test parameters
    N_A = 12  # Number of rows in A
    N_B = 8  # Number of rows in B
    C = 6  # Number of columns/channels
    M = 8  # Number of operations

    # Create test data
    device = "cuda"

    # Create input tensors
    A = rand_clamped((N_A, C), torch.float32, device)
    B = rand_clamped((N_B, C), torch.float32, device)
    result = torch.zeros((C,), device=device, dtype=torch.float32)

    # Create indices
    a_indices = rand_indices(N_A, M, device)
    b_indices = rand_indices(N_B, M, device)

    print(f"Input shapes: A={A.shape}, B={B.shape}, result={result.shape}")
    print(f"Indices: a_indices={a_indices.shape}, b_indices={b_indices.shape}")
    print(f"M={M}, C={C}, N_A={N_A}, N_B={N_B}")
    print(f"a_indices: {a_indices.cpu().numpy()}")
    print(f"b_indices: {b_indices.cpu().numpy()}")

    # Run implicit reduction kernel
    warpconvnet_c.fma.implicit_reduction(A, a_indices, B, b_indices, result, "basic")
    print("Kernel execution successful!")

    # Verify results using CPU computation
    result_expected = torch.zeros((C,), device=device, dtype=torch.float32)
    for i in range(M):
        a_idx = int(a_indices[i].item())
        b_idx = int(b_indices[i].item())
        result_expected += A[a_idx] * B[b_idx]

    # Check results
    max_abs_diff, max_rel_diff = compare_results(result, result_expected)
    assert max_abs_diff < 1e-5, f"Max absolute difference {max_abs_diff} exceeds tolerance 1e-5"
    assert max_rel_diff < 1e-5, f"Max relative difference {max_rel_diff} exceeds tolerance 1e-5"


def test_implicit_reduction_basic_without_b():
    """Test basic functionality without B matrix (B treated as all ones)."""
    print("Testing implicit reduction kernel without B matrix...")

    # Import the warpconvnet module
    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        return False

    # Test parameters
    N_A = 10  # Number of rows in A
    C = 4  # Number of columns/channels
    M = 8  # Number of operations

    # Create test data
    device = "cuda"

    # Create input tensors
    A = rand_clamped((N_A, C), torch.float32, device)
    result = torch.zeros((C,), device=device, dtype=torch.float32)

    # Create indices for A only
    a_indices = rand_indices(N_A, M, device)

    print(f"Input shapes: A={A.shape}, result={result.shape}")
    print(f"Indices: a_indices={a_indices.shape}")
    print(f"M={M}, C={C}, N_A={N_A}")

    # Run implicit reduction kernel without B (pass None for B and b_indices)
    warpconvnet_c.fma.implicit_reduction(A, a_indices, None, None, result, "basic")
    print("Kernel execution successful!")

    # Verify results using CPU computation (B treated as all ones)
    result_expected = torch.zeros((C,), device=device, dtype=torch.float32)
    for i in range(M):
        a_idx = int(a_indices[i].item())
        result_expected += A[a_idx]  # * 1 (B treated as ones)

    # Check results
    max_abs_diff, max_rel_diff = compare_results(result, result_expected)
    assert max_abs_diff < 1e-5, f"Max absolute difference {max_abs_diff} exceeds tolerance 1e-5"
    assert max_rel_diff < 1e-5, f"Max relative difference {max_rel_diff} exceeds tolerance 1e-5"


def test_implicit_reduction_dtypes():
    """Test different data types (float32, float16, float64)."""
    print("Testing different data types...")

    # Import the warpconvnet module
    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        return False

    device = "cuda"

    # Test parameters
    N_A = 8
    N_B = 6
    C = 4
    M = 6

    # Test different data types
    dtypes = [
        (torch.float32, "float32", 1e-5),
        (torch.float16, "float16", 5e-3),
        (torch.float64, "float64", 1e-10),
    ]

    for dtype, dtype_name, tolerance in dtypes:
        print(f"Testing {dtype_name}...")

        # Create test data
        A = rand_clamped((N_A, C), dtype, device)
        B = rand_clamped((N_B, C), dtype, device)
        result = torch.zeros((C,), device=device, dtype=dtype)

        # Create indices
        a_indices = rand_indices(N_A, M, device)
        b_indices = rand_indices(N_B, M, device)

        # Run kernel
        warpconvnet_c.fma.implicit_reduction(A, a_indices, B, b_indices, result, "basic")
        print(f"✓ {dtype_name} kernel executed successfully")

        # Verify results
        result_expected = torch.zeros((C,), device=device, dtype=dtype)
        for i in range(M):
            a_idx = int(a_indices[i].item())
            b_idx = int(b_indices[i].item())
            result_expected += A[a_idx] * B[b_idx]

        # Check results
        max_abs_diff, max_rel_diff = compare_results(result, result_expected)
        assert (
            max_abs_diff < tolerance
        ), f"Max absolute difference {max_abs_diff} exceeds tolerance {tolerance}"
        assert (
            max_rel_diff < tolerance
        ), f"Max relative difference {max_rel_diff} exceeds tolerance {tolerance}"
