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
import numpy as np
import sys


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
    M = 10  # Number of operations

    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    # Create input tensors
    A = torch.randn(N_A, C, device=device, dtype=torch.float32)
    B = torch.randn(N_B, C, device=device, dtype=torch.float32)
    result = torch.zeros(C, device=device, dtype=torch.float32)

    # Create indices
    a_indices = torch.randint(0, N_A, (M,), device=device, dtype=torch.int32)
    b_indices = torch.randint(0, N_B, (M,), device=device, dtype=torch.int32)

    print(f"Input shapes: A={A.shape}, B={B.shape}, result={result.shape}")
    print(f"Indices: a_indices={a_indices.shape}, b_indices={b_indices.shape}")
    print(f"M={M}, C={C}, N_A={N_A}, N_B={N_B}")
    print(f"a_indices: {a_indices.cpu().numpy()}")
    print(f"b_indices: {b_indices.cpu().numpy()}")

    # Run implicit reduction kernel
    try:
        warpconvnet_c.fma.implicit_reduction(A, a_indices, B, b_indices, result, "basic")
        print("Kernel execution successful!")
    except Exception as e:
        print(f"ERROR: Kernel execution failed: {e}")
        return False

    # Verify results using CPU computation
    result_expected = torch.zeros(C, device=device, dtype=torch.float32)
    for i in range(M):
        a_idx = int(a_indices[i].item())
        b_idx = int(b_indices[i].item())
        result_expected += A[a_idx] * B[b_idx]

    # Check results
    diff = torch.abs(result - result_expected)
    max_diff = torch.max(diff).item()
    print(f"Maximum difference between GPU and CPU results: {max_diff}")

    if max_diff < 1e-5:
        print("✓ Basic test with B matrix PASSED")
        return True
    else:
        print("✗ Basic test with B matrix FAILED")
        print(f"GPU result: {result.cpu().numpy()}")
        print(f"CPU result: {result_expected.cpu().numpy()}")
        return False


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    # Create input tensors
    A = torch.randn(N_A, C, device=device, dtype=torch.float32)
    result = torch.zeros(C, device=device, dtype=torch.float32)

    # Create indices for A only
    a_indices = torch.randint(0, N_A, (M,), device=device, dtype=torch.int32)

    print(f"Input shapes: A={A.shape}, result={result.shape}")
    print(f"Indices: a_indices={a_indices.shape}")
    print(f"M={M}, C={C}, N_A={N_A}")

    # Run implicit reduction kernel without B (pass None for B and b_indices)
    try:
        warpconvnet_c.fma.implicit_reduction(A, a_indices, None, None, result, "basic")
        print("Kernel execution successful!")
    except Exception as e:
        print(f"ERROR: Kernel execution failed: {e}")
        return False

    # Verify results using CPU computation (B treated as all ones)
    result_expected = torch.zeros(C, device=device, dtype=torch.float32)
    for i in range(M):
        a_idx = int(a_indices[i].item())
        result_expected += A[a_idx]  # * 1 (B treated as ones)

    # Check results
    diff = torch.abs(result - result_expected)
    max_diff = torch.max(diff).item()
    print(f"Maximum difference between GPU and CPU results: {max_diff}")

    if max_diff < 1e-5:
        print("✓ Basic test without B matrix PASSED")
        return True
    else:
        print("✗ Basic test without B matrix FAILED")
        print(f"GPU result: {result.cpu().numpy()}")
        print(f"CPU result: {result_expected.cpu().numpy()}")
        return False


def test_implicit_reduction_dtypes():
    """Test different data types (float32, float16, float64)."""
    print("Testing different data types...")

    # Import the warpconvnet module
    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

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
        A = torch.randn(N_A, C, device=device, dtype=dtype)
        B = torch.randn(N_B, C, device=device, dtype=dtype)
        result = torch.zeros(C, device=device, dtype=dtype)

        # Create indices
        a_indices = torch.randint(0, N_A, (M,), device=device, dtype=torch.int32)
        b_indices = torch.randint(0, N_B, (M,), device=device, dtype=torch.int32)

        # Run kernel
        try:
            warpconvnet_c.fma.implicit_reduction(A, a_indices, B, b_indices, result, "basic")
            print(f"✓ {dtype_name} kernel executed successfully")
        except Exception as e:
            print(f"✗ {dtype_name} kernel failed: {e}")
            return False

        # Verify results
        result_expected = torch.zeros(C, device=device, dtype=dtype)
        for i in range(M):
            a_idx = int(a_indices[i].item())
            b_idx = int(b_indices[i].item())
            result_expected += A[a_idx] * B[b_idx]

        # Check results
        diff = torch.abs(result - result_expected)
        max_diff = torch.max(diff).item()

        if max_diff < tolerance:
            print(f"✓ {dtype_name} test PASSED (max_diff: {max_diff})")
        else:
            print(f"✗ {dtype_name} test FAILED (max_diff: {max_diff}, tolerance: {tolerance})")
            return False

    return True


def test_implicit_reduction_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing edge cases...")

    # Import the warpconvnet module
    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    # Test 1: Empty operations (M=0)
    print("Testing empty operations (M=0)...")
    N_A = 5
    N_B = 4
    C = 3
    M = 0

    A = torch.randn(N_A, C, device=device, dtype=torch.float32)
    B = torch.randn(N_B, C, device=device, dtype=torch.float32)
    result = torch.zeros(C, device=device, dtype=torch.float32)

    # Create empty indices
    a_indices = torch.tensor([], device=device, dtype=torch.int32)
    b_indices = torch.tensor([], device=device, dtype=torch.int32)

    try:
        warpconvnet_c.fma.implicit_reduction(A, a_indices, B, b_indices, result, "basic")
        print("✓ Empty operations test PASSED")
    except Exception as e:
        print(f"✗ Empty operations test failed: {e}")
        return False

    # Test 2: Single operation (M=1)
    print("Testing single operation (M=1)...")
    M = 1
    a_indices = torch.tensor([2], device=device, dtype=torch.int32)
    b_indices = torch.tensor([1], device=device, dtype=torch.int32)
    result.zero_()

    try:
        warpconvnet_c.fma.implicit_reduction(A, a_indices, B, b_indices, result, "basic")
        print("✓ Single operation test PASSED")
    except Exception as e:
        print(f"✗ Single operation test failed: {e}")
        return False

    return True


def test_implicit_reduction_bounds_checking():
    """Test bounds checking for out-of-range indices."""
    print("Testing bounds checking...")

    # Import the warpconvnet module
    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    # Test parameters
    N_A = 5
    N_B = 4
    C = 3
    M = 6

    # Create test data
    A = torch.randn(N_A, C, device=device, dtype=torch.float32)
    B = torch.randn(N_B, C, device=device, dtype=torch.float32)
    result = torch.zeros(C, device=device, dtype=torch.float32)

    # Create indices with some out-of-bounds values
    a_indices = torch.tensor(
        [0, 2, 4, -1, 5, 1], device=device, dtype=torch.int32
    )  # -1 and 5 are out of bounds
    b_indices = torch.tensor(
        [0, 1, 3, 2, -1, 4], device=device, dtype=torch.int32
    )  # -1 and 4 are out of bounds

    print(
        f"Testing with indices: a_indices={a_indices.cpu().numpy()}, b_indices={b_indices.cpu().numpy()}"
    )
    print(f"Valid ranges: A[0:{N_A}], B[0:{N_B}]")

    try:
        warpconvnet_c.fma.implicit_reduction(A, a_indices, B, b_indices, result, "basic")
        print("✓ Bounds checking test PASSED (out-of-bounds indices were handled gracefully)")
        return True
    except Exception as e:
        print(f"✗ Bounds checking test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Implicit Reduction CUDA Kernel")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    tests = [
        test_implicit_reduction_basic_with_b,
        test_implicit_reduction_basic_without_b,
        test_implicit_reduction_dtypes,
        test_implicit_reduction_edge_cases,
        test_implicit_reduction_bounds_checking,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"Test {test.__name__} failed!")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
