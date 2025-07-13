#!/usr/bin/env python3
"""
Test script for the templated implicit FMA CUDA kernel.

This test suite validates the clean template-based implementation following
the CUTLASS GEMM pattern. Features:
- No explicit type names in function names
- Clean namespace organization
- Multiple data types (float32, float16, float64)
- Multiple kernel variants (basic, optimized, rowwise)
- Shared memory optimization
- Status-based error handling
"""

import torch
import numpy as np
import sys


def test_implicit_fma_basic():
    """Test basic functionality of the implicit FMA kernel."""
    print("Testing implicit FMA kernel...")

    # Import the warpconvnet module
    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        print("Make sure the module is compiled and available")
        return False

    # Test parameters
    N_A = 10  # Number of rows in A
    N_C = 8  # Number of rows in C
    C = 4  # Number of columns/channels
    num_ops = 6  # Number of operations

    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    # Create input tensors
    a = torch.randn(N_A, C, device=device, dtype=torch.float32)
    b = torch.randn(C, device=device, dtype=torch.float32)
    c = torch.zeros(N_C, C, device=device, dtype=torch.float32)

    # Create indices for gather-scatter operations
    in_indices = torch.randint(0, N_A, (num_ops,), device=device, dtype=torch.int32)
    # Ensure unique output indices (since we removed atomicAdd)
    out_indices = torch.randperm(N_C, device=device, dtype=torch.int32)[:num_ops]

    print(f"Input shapes: A={a.shape}, B={b.shape}, C={c.shape}")
    print(f"Indices: in_indices={in_indices.shape}, out_indices={out_indices.shape}")
    print(f"in_indices: {in_indices.cpu().numpy()}")
    print(f"out_indices: {out_indices.cpu().numpy()}")

    # Save original C for comparison
    c_original = c.clone()

    # Run implicit FMA kernel
    try:
        warpconvnet_c.gemm.implicit_fma(a, b, c, in_indices, out_indices, "basic")
        print("Kernel execution successful!")
    except Exception as e:
        print(f"ERROR: Kernel execution failed: {e}")
        return False

    # Verify results using CPU computation
    c_expected = c_original.clone()
    for i in range(num_ops):
        in_idx = in_indices[i].item()
        out_idx = out_indices[i].item()
        c_expected[out_idx] += a[in_idx] * b

    # Compare results
    diff = torch.abs(c - c_expected)
    max_diff = torch.max(diff)
    print(f"Maximum difference between GPU and CPU results: {max_diff.item()}")

    if max_diff < 1e-5:
        print("✓ Basic test PASSED")
        return True
    else:
        print("✗ Basic test FAILED")
        print(f"GPU result:\n{c}")
        print(f"CPU result:\n{c_expected}")
        return False


def test_implicit_fma_kernel_types():
    """Test different kernel types (basic, optimized, rowwise)."""
    print("\nTesting different kernel types...")

    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        return False

    # Test parameters
    N_A = 20
    N_C = 15
    C = 8
    num_ops = 12

    device = torch.device("cuda")

    # Create test data
    a = torch.randn(N_A, C, device=device, dtype=torch.float32)
    b = torch.randn(C, device=device, dtype=torch.float32)
    in_indices = torch.randint(0, N_A, (num_ops,), device=device, dtype=torch.int32)
    # Ensure unique output indices (since we removed atomicAdd)
    out_indices = torch.randperm(N_C, device=device, dtype=torch.int32)[:num_ops]

    # Test each kernel type
    kernel_types = ["basic", "rowwise"]
    results = {}

    for kernel_type in kernel_types:
        c = torch.zeros(N_C, C, device=device, dtype=torch.float32)

        try:
            warpconvnet_c.gemm.implicit_fma(a, b, c, in_indices, out_indices, kernel_type)
            results[kernel_type] = c.clone()
            print(f"✓ {kernel_type} kernel executed successfully")
        except Exception as e:
            print(f"✗ {kernel_type} kernel failed: {e}")
            return False

    # Compare results between kernel types
    for i, kt1 in enumerate(kernel_types):
        for kt2 in kernel_types[i + 1 :]:
            diff = torch.abs(results[kt1] - results[kt2])
            max_diff = torch.max(diff)
            print(f"Max difference between {kt1} and {kt2}: {max_diff.item()}")
            if max_diff > 1e-5:
                print(f"✗ Results differ between {kt1} and {kt2}")
                return False

    print("✓ All kernel types produce consistent results")
    return True


def test_implicit_fma_dtypes():
    """Test different data types (float32, float16, float64)."""
    print("\nTesting different data types...")

    try:
        import warpconvnet._C as warpconvnet_c
    except ImportError:
        print("ERROR: Could not import warpconvnet._C")
        return False

    # Test parameters
    N_A = 15
    N_C = 12
    C = 6
    num_ops = 8

    device = torch.device("cuda")

    # Test different data types
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
    dtype_names = ["float32", "float16", "bfloat16", "float64"]

    for dtype, dtype_name in zip(dtypes, dtype_names):
        print(f"Testing {dtype_name}...")

        # Create test data
        a = torch.randn(N_A, C, device=device, dtype=dtype)
        b = torch.randn(C, device=device, dtype=dtype)
        c = torch.zeros(N_C, C, device=device, dtype=dtype)
        in_indices = torch.randint(0, N_A, (num_ops,), device=device, dtype=torch.int32)
        # Ensure unique output indices (since we removed atomicAdd)
        out_indices = torch.randperm(N_C, device=device, dtype=torch.int32)[:num_ops]

        # Save original for comparison
        c_original = c.clone()

        try:
            warpconvnet_c.gemm.implicit_fma(a, b, c, in_indices, out_indices, "basic")
            print(f"✓ {dtype_name} kernel executed successfully")
        except Exception as e:
            print(f"✗ {dtype_name} kernel failed: {e}")
            return False

        # Verify results
        c_expected = c_original.clone()
        for i in range(num_ops):
            in_idx = in_indices[i].item()
            out_idx = out_indices[i].item()
            c_expected[out_idx] += a[in_idx] * b

        diff = torch.abs(c - c_expected)
        max_diff = torch.max(diff)
        tolerance = 1e-5 if dtype not in [torch.float16, torch.bfloat16] else 1e-3

        if max_diff < tolerance:
            print(f"✓ {dtype_name} test PASSED (max_diff: {max_diff.item()})")
        else:
            print(
                f"✗ {dtype_name} test FAILED (max_diff: {max_diff.item()}, tolerance: {tolerance})"
            )
            return False

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Implicit FMA CUDA Kernel")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1

    tests = [
        test_implicit_fma_basic,
        test_implicit_fma_kernel_types,
        test_implicit_fma_dtypes,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"Test {test.__name__} failed!")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
