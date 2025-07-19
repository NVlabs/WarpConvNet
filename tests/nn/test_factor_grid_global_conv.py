#!/usr/bin/env python3
"""
Test script for the improved FactorGridGlobalConv implementation.
"""

import torch
import torch.nn as nn
from typing import List, Literal, Optional, Tuple, Type, Union

# Import the new modules
from warpconvnet.geometry.types.factor_grid import FactorGrid
from warpconvnet.geometry.features.grid import GridMemoryFormat, COMPRESSED_FORMATS
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.nn.modules.factor_grid import FactorGridGlobalConv


def create_test_factor_grid(batch_size=1, num_channels=64):
    """Create a test FactorGrid with different memory formats."""

    # Create grids with different factorized formats
    grids = []

    # Grid 1: b_zc_x_y format (Z=2, X=64, Y=64)
    grid1 = Grid.from_shape(
        grid_shape=(64, 64, 2),  # H, W, D
        num_channels=num_channels,
        memory_format=GridMemoryFormat.b_zc_x_y,
        batch_size=batch_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    grids.append(grid1)

    # Grid 2: b_xc_y_z format (X=2, Y=64, Z=64)
    grid2 = Grid.from_shape(
        grid_shape=(2, 64, 64),  # H, W, D
        num_channels=num_channels,
        memory_format=GridMemoryFormat.b_xc_y_z,
        batch_size=batch_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    grids.append(grid2)

    # Grid 3: b_yc_x_z format (Y=2, X=64, Z=64)
    grid3 = Grid.from_shape(
        grid_shape=(64, 2, 64),  # H, W, D
        num_channels=num_channels,
        memory_format=GridMemoryFormat.b_yc_x_z,
        batch_size=batch_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    grids.append(grid3)

    return FactorGrid(grids)


def test_factor_grid_global_conv_basic():
    """Test the basic FactorGridGlobalConv functionality."""

    print("Testing FactorGridGlobalConv (basic)...")

    # Create test FactorGrid
    factor_grid = create_test_factor_grid(batch_size=2, num_channels=32)
    print(f"Created FactorGrid with {len(factor_grid)} grids")

    # Print grid shapes and formats
    for i, grid in enumerate(factor_grid):
        print(
            f"Grid {i}: shape={grid.grid_features.batched_tensor.shape}, format={grid.memory_format}"
        )

    # Create FactorGridGlobalConv module
    global_conv = FactorGridGlobalConv(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        compressed_spatial_dims=(2, 2, 2),  # Z=2, X=2, Y=2 for the three grids
        compressed_memory_formats=(
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ),
        stride=1,
        communication_types=["sum"],
        norm=nn.BatchNorm2d,
        activation=nn.GELU,
    )

    print("Created FactorGridGlobalConv module")

    # Test forward pass
    try:
        output = global_conv(factor_grid)
        print("Forward pass successful!")
        print(f"Output FactorGrid has {len(output)} grids")

        # Print output shapes
        for i, grid in enumerate(output):
            print(
                f"Output Grid {i}: shape={grid.grid_features.batched_tensor.shape}, format={grid.memory_format}"
            )

        return True

    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_grid_global_conv_multiple_communication():
    """Test FactorGridGlobalConv with multiple communication types."""

    print("\nTesting FactorGridGlobalConv with multiple communication types...")

    # Create test FactorGrid
    factor_grid = create_test_factor_grid(batch_size=1, num_channels=16)

    # Create FactorGridGlobalConv module with multiple communication types
    global_conv = FactorGridGlobalConv(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        compressed_spatial_dims=(2, 2, 2),
        compressed_memory_formats=(
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ),
        stride=1,
        communication_types=["sum", "mul"],
        norm=nn.BatchNorm2d,
        activation=nn.GELU,
    )

    # Test forward pass
    try:
        output = global_conv(factor_grid)
        print("Multiple communication forward pass successful!")
        print(f"Output FactorGrid has {len(output)} grids")

        # Print output shapes
        for i, grid in enumerate(output):
            print(f"Output Grid {i}: shape={grid.grid_features.batched_tensor.shape}")

        return True

    except Exception as e:
        print(f"Multiple communication forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_grid_global_conv_stride():
    """Test FactorGridGlobalConv with stride."""

    print("\nTesting FactorGridGlobalConv with stride...")

    # Create test FactorGrid
    factor_grid = create_test_factor_grid(batch_size=1, num_channels=16)

    # Create FactorGridGlobalConv module with stride
    global_conv = FactorGridGlobalConv(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        compressed_spatial_dims=(2, 2, 2),
        compressed_memory_formats=(
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ),
        stride=2,
        communication_types=["sum"],
        norm=nn.BatchNorm2d,
        activation=nn.GELU,
    )

    # Test forward pass
    try:
        output = global_conv(factor_grid)
        print("Stride forward pass successful!")
        print(f"Output FactorGrid has {len(output)} grids")

        # Print output shapes
        for i, grid in enumerate(output):
            print(f"Output Grid {i}: shape={grid.grid_features.batched_tensor.shape}")

        return True

    except Exception as e:
        print(f"Stride forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_grid_global_conv_up_stride():
    """Test FactorGridGlobalConv with up_stride (transposed convolution)."""

    print("\nTesting FactorGridGlobalConv with up_stride...")

    # Create test FactorGrid
    factor_grid = create_test_factor_grid(batch_size=1, num_channels=16)

    # Create FactorGridGlobalConv module with up_stride
    global_conv = FactorGridGlobalConv(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        compressed_spatial_dims=(2, 2, 2),
        compressed_memory_formats=(
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ),
        stride=1,
        up_stride=2,
        communication_types=["sum"],
        norm=nn.BatchNorm2d,
        activation=nn.GELU,
    )

    # Test forward pass
    try:
        output = global_conv(factor_grid)
        print("Up-stride forward pass successful!")
        print(f"Output FactorGrid has {len(output)} grids")

        # Print output shapes
        for i, grid in enumerate(output):
            print(f"Output Grid {i}: shape={grid.grid_features.batched_tensor.shape}")

        return True

    except Exception as e:
        print(f"Up-stride forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_grid_global_conv_different_norm():
    """Test FactorGridGlobalConv with different normalization."""

    print("\nTesting FactorGridGlobalConv with LayerNorm...")

    # Create test FactorGrid
    factor_grid = create_test_factor_grid(batch_size=1, num_channels=16)

    # Create FactorGridGlobalConv module with LayerNorm
    global_conv = FactorGridGlobalConv(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        compressed_spatial_dims=(2, 2, 2),
        compressed_memory_formats=(
            GridMemoryFormat.b_zc_x_y,
            GridMemoryFormat.b_xc_y_z,
            GridMemoryFormat.b_yc_x_z,
        ),
        stride=1,
        communication_types=["sum"],
        norm=nn.LayerNorm,
        activation=nn.ReLU,
    )

    # Test forward pass
    try:
        output = global_conv(factor_grid)
        print("LayerNorm forward pass successful!")
        print(f"Output FactorGrid has {len(output)} grids")

        # Print output shapes
        for i, grid in enumerate(output):
            print(f"Output Grid {i}: shape={grid.grid_features.batched_tensor.shape}")

        return True

    except Exception as e:
        print(f"LayerNorm forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing improved FactorGridGlobalConv implementation...")

    # Run all tests
    tests = [
        test_factor_grid_global_conv_basic,
        test_factor_grid_global_conv_multiple_communication,
        test_factor_grid_global_conv_stride,
        test_factor_grid_global_conv_up_stride,
        test_factor_grid_global_conv_different_norm,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        for i, (test, result) in enumerate(zip(tests, results)):
            status = "✅" if result else "❌"
            print(f"{status} {test.__name__}")

    print(f"{'='*50}")
