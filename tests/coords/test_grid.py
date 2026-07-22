# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import gc
import sys
import time

from warpconvnet.geometry.coords.grid import GridCoords
from warpconvnet.geometry.coords.ops.grid import create_grid_coordinates
from warpconvnet.geometry.features.grid import GridFeatures, GridMemoryFormat
from warpconvnet.geometry.types.grid import Grid


@pytest.fixture
def setup_device():
    """Setup test device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def measure_memory(func, *args, **kwargs):
    """Helper function to measure peak memory usage of a function call."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()

    # Time the function execution
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    # Get peak memory usage
    peak_mem = torch.cuda.max_memory_allocated()

    return result, peak_mem - start_mem, end_time - start_time


def test_lazy_initialization(setup_device):
    """Test that GridCoords lazily initializes tensors only when needed."""
    device = setup_device
    grid_shape = (64, 64, 64)  # Medium size grid

    # Create GridCoords with lazy initialization
    coords = GridCoords.from_shape(grid_shape, batch_size=2, device=device)

    # Check initial state
    assert not coords._is_initialized
    assert coords.grid_shape == grid_shape
    assert coords.batch_size == 2
    assert coords.device == device

    # Access repr/str - should not trigger initialization
    repr_str = repr(coords)
    str_str = str(coords)
    assert "lazy=True" in repr_str
    assert not coords._is_initialized

    # Get shape - should not trigger initialization
    shape = coords.shape
    assert not coords._is_initialized
    assert shape[1] == 3  # Should have 3D coordinates

    # Now access batched_tensor which should trigger initialization
    tensor = coords.batched_tensor
    assert coords._is_initialized
    assert tensor.shape == (2 * 64 * 64 * 64, 3)  # Flattened tensor shape


def test_memory_savings(setup_device):
    """Test that lazy initialization saves memory until tensor is needed."""
    device = setup_device
    if device.type != "cuda":
        pytest.skip("Memory testing requires CUDA")

    # Use a large grid to make memory savings more apparent
    grid_shape = (128, 128, 128)

    # Create standard GridCoords (eager)
    def create_eager():
        # Create tensor first then wrap in GridCoords
        coords_tensor, offsets = create_grid_coordinates(grid_shape, batch_size=2, device=device)
        return GridCoords.from_tensor(coords_tensor, offsets, grid_shape)

    # Create lazy GridCoords
    def create_lazy():
        return GridCoords.from_shape(grid_shape, batch_size=2, device=device)

    # Measure memory for eager initialization
    _, eager_mem, eager_time = measure_memory(create_eager)

    # Measure memory for lazy initialization (without access)
    lazy_coords, lazy_init_mem, lazy_init_time = measure_memory(create_lazy)

    # Measure memory when lazy coords are accessed
    def access_lazy(coords):
        return coords.batched_tensor

    _, lazy_access_mem, _ = measure_memory(access_lazy, lazy_coords)

    # Print memory usage
    print(f"Eager initialization: {eager_mem / 1024**2:.2f} MB, {eager_time:.4f}s")
    print(f"Lazy initialization: {lazy_init_mem / 1024**2:.2f} MB, {lazy_init_time:.4f}s")
    print(f"Lazy access: {lazy_access_mem / 1024**2:.2f} MB")

    # Verify memory savings
    assert lazy_init_mem < eager_mem

    # After access, memory should be similar to eager
    assert (
        abs(lazy_access_mem - eager_mem) / eager_mem < 0.3
    )  # Within 30% (more flexible for test stability)


def test_device_transfer(setup_device):
    """Test device transfer with lazy initialization."""
    original_device = setup_device
    target_device = torch.device("cpu") if original_device.type == "cuda" else setup_device

    grid_shape = (32, 32, 32)

    # Create lazy coordinates
    coords = GridCoords.from_shape(grid_shape, batch_size=1, device=original_device)

    # Transfer to new device without accessing tensor
    new_coords = coords.to(target_device)

    # Check that it's still lazy
    assert not coords._is_initialized
    assert not new_coords._is_initialized

    # Check that device was updated
    assert new_coords.device == target_device

    # Now access the tensor on new device
    tensor = new_coords.batched_tensor
    assert new_coords._is_initialized
    assert tensor.device == target_device


def test_compatibility_with_grid_features(setup_device):
    """Test compatibility with GridFeatures."""
    device = setup_device
    grid_shape = (16, 16, 16)
    num_channels = 4

    # Create lazy coordinates
    coords = GridCoords.from_shape(grid_shape, batch_size=1, device=device)

    # Create grid features
    features = GridFeatures.create_empty(
        grid_shape,
        num_channels,
        batch_size=1,
        memory_format=GridMemoryFormat.b_x_y_z_c,
        device=device,
    )

    # Create grid with lazy coordinates
    grid = Grid(coords, features)

    # Check that coords weren't initialized yet
    assert not coords._is_initialized

    # Access the grid's batched_tensor
    grid_tensor = grid.grid_coords.batched_tensor

    # Now coords should be initialized
    assert coords._is_initialized

    # Check grid properties
    assert grid.grid_shape == grid_shape
    assert grid.num_channels == num_channels


def test_gridcoords_methods(setup_device):
    """Test that all GridCoords methods work correctly with lazy initialization."""
    device = setup_device
    grid_shape = (4, 6, 8)

    # Create lazy grid
    coords = GridCoords.from_shape(grid_shape, batch_size=2, device=device)

    # Test various methods without triggering initialization
    assert coords.grid_shape == grid_shape
    assert coords.batch_size == 2
    assert coords.device == device
    assert coords.numel() == 2 * 4 * 6 * 8 * 3
    assert len(coords) == 2 * 4 * 6 * 8

    # Test data type conversion - lazily updates dtype (like .to()), without
    # initializing either object; accessing the tensor materializes it
    float_coords = coords.float()
    assert not coords._is_initialized
    assert not float_coords._is_initialized
    assert float_coords.batched_tensor.dtype == torch.float32
    assert float_coords._is_initialized

    # Reset for next test
    coords = GridCoords.from_shape(grid_shape, batch_size=2, device=device)

    # Test get_spatial_indices - should not trigger initialization
    flat_indices = torch.tensor([0, 1, 2], device=device)
    h, w, d = coords.get_spatial_indices(flat_indices)
    assert not coords._is_initialized

    # Test get_flattened_indices - should not trigger initialization since it only uses grid_shape
    h_indices = torch.tensor([0, 1, 2], device=device)
    w_indices = torch.tensor([0, 1, 2], device=device)
    d_indices = torch.tensor([0, 1, 2], device=device)
    flat = coords.get_flattened_indices(h_indices, w_indices, d_indices)
    assert not coords._is_initialized
