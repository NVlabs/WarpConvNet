# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp
from pytest_benchmark.fixture import BenchmarkFixture

from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.geometry.coords.search.torch_discrete import (
    _kernel_map_from_direct_queries,
    _kernel_map_from_offsets,
    _kernel_map_from_size,
    kernel_offsets_from_size,
    generate_kernel_map,
)
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.utils.timer import Timer


@pytest.fixture
def setup_voxels():
    """Setup test voxels with random coordinates and features."""
    wp.init()
    device = torch.device("cuda:0")
    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    voxel_size = 0.025
    voxel_coords = [torch.floor(coords / voxel_size).int() for coords in coords]
    return Voxels(voxel_coords, features, device=device).unique()


def test_kernel_map_from_offset(setup_voxels):
    """Test kernel map generation using offset method."""
    device = torch.device("cuda:0")
    voxels: Voxels = setup_voxels.to(device)

    kernel_offsets = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=torch.int32,
        device=device,
    )

    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    voxel_hashmap = voxels.coordinate_hashmap

    kernel_map: IntSearchResult = _kernel_map_from_offsets(
        voxel_hashmap,
        bcoords,
        kernel_offsets,
    )

    tot_num_maps = kernel_map.offsets[-1].item()
    assert tot_num_maps == len(kernel_map.in_maps)
    assert tot_num_maps == len(kernel_map.out_maps)


def test_kernel_map_from_size(setup_voxels):
    """Test kernel map generation using size method."""
    device = torch.device("cuda:0")
    voxels: Voxels = setup_voxels.to(device)

    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    voxel_hashmap = voxels.coordinate_hashmap
    kernel_sizes = (3, 3, 3)

    kernel_map: IntSearchResult = _kernel_map_from_size(
        voxel_hashmap,
        bcoords,
        kernel_sizes,
    )

    tot_num_maps = kernel_map.offsets[-1].item()
    assert tot_num_maps == len(kernel_map.in_maps)
    assert tot_num_maps == len(kernel_map.out_maps)


def test_compare_kernel_map_methods(setup_voxels):
    """Compare results from different kernel map generation methods."""
    device = torch.device("cuda:0")
    voxels: Voxels = setup_voxels.to(device)
    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    voxel_hashmap = voxels.coordinate_hashmap

    # Test parameters
    kernel_size = (3, 3, 3)
    kernel_dilation = (1, 1, 1)

    # Generate kernel offsets
    kernel_offsets = kernel_offsets_from_size(
        kernel_size=kernel_size,
        kernel_dilation=kernel_dilation,
    ).to(device)

    # Get results from all three methods
    kernel_map_offsets = _kernel_map_from_offsets(
        voxel_hashmap,
        bcoords,
        kernel_offsets,
    )

    kernel_map_size = _kernel_map_from_size(
        voxel_hashmap,
        bcoords,
        kernel_size,
    )

    kernel_map_direct = _kernel_map_from_direct_queries(
        voxel_hashmap,
        bcoords,
        kernel_size=kernel_size,
        kernel_dilation=kernel_dilation,
    )

    # Compare results
    assert len(kernel_map_offsets) == len(kernel_map_size)
    assert len(kernel_map_size) == len(kernel_map_direct)

    for i in range(len(kernel_map_offsets)):
        in_map_o, out_map_o = kernel_map_offsets[i]
        in_map_s, out_map_s = kernel_map_size[i]
        in_map_d, out_map_d = kernel_map_direct[i]

        # Check sizes match
        assert len(in_map_o) == len(in_map_s)
        assert len(in_map_s) == len(in_map_d)
        assert len(out_map_o) == len(out_map_s)
        assert len(out_map_s) == len(out_map_d)

        # Check values match (after sorting)
        assert torch.equal(torch.sort(in_map_o)[0], torch.sort(in_map_s)[0])
        assert torch.equal(torch.sort(in_map_s)[0], torch.sort(in_map_d)[0])
        assert torch.equal(torch.sort(out_map_o)[0], torch.sort(out_map_s)[0])
        assert torch.equal(torch.sort(out_map_s)[0], torch.sort(out_map_d)[0])


@pytest.mark.parametrize("kernel_size", [(3, 3, 3), (5, 5, 5)])
@pytest.mark.parametrize("method", ["query", "offset", "size"])
def test_symmetric_kernel_map_correctness(setup_voxels, kernel_size, method):
    """Test that symmetric kernel map produces correct results compared to non-symmetric."""
    device = torch.device("cuda:0")
    voxels: Voxels = setup_voxels.to(device)

    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    in_to_out_stride_ratio = (1, 1, 1)

    # Generate non-symmetric kernel map
    kernel_map_full = generate_kernel_map(
        batch_indexed_in_coords=bcoords,
        batch_indexed_out_coords=bcoords,
        in_to_out_stride_ratio=in_to_out_stride_ratio,
        kernel_size=kernel_size,
        method=method,
        skip_symmetric_kernel_map=False,
    )

    # Generate symmetric kernel map
    kernel_map_symmetric = generate_kernel_map(
        batch_indexed_in_coords=bcoords,
        batch_indexed_out_coords=bcoords,
        in_to_out_stride_ratio=in_to_out_stride_ratio,
        kernel_size=kernel_size,
        method=method,
        skip_symmetric_kernel_map=True,
    )

    total_kernels = kernel_size[0] * kernel_size[1] * kernel_size[2]
    expected_symmetric_kernels = total_kernels // 2

    # Check that symmetric version has half the kernel offsets
    assert len(kernel_map_symmetric) == expected_symmetric_kernels
    assert len(kernel_map_full) == total_kernels

    # Compare the first half of non-symmetric with symmetric results
    for i in range(expected_symmetric_kernels):
        full_in_map, full_out_map = kernel_map_full[i]
        sym_in_map, sym_out_map = kernel_map_symmetric[i]

        # Sort both arrays for comparison
        full_in_sorted, full_in_indices = torch.sort(full_in_map)
        full_out_sorted = full_out_map[full_in_indices]

        sym_in_sorted, sym_in_indices = torch.sort(sym_in_map)
        sym_out_sorted = sym_out_map[sym_in_indices]

        # Check that they match
        assert torch.equal(
            full_in_sorted, sym_in_sorted
        ), f"Mismatch in kernel {i} in_maps for {method} method"
        assert torch.equal(
            full_out_sorted, sym_out_sorted
        ), f"Mismatch in kernel {i} out_maps for {method} method"


def test_symmetric_kernel_map_offset_method(setup_voxels):
    """Test symmetric kernel map using offset method directly."""
    device = torch.device("cuda:0")
    voxels: Voxels = setup_voxels.to(device)

    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    voxel_hashmap = voxels.coordinate_hashmap
    kernel_size = (3, 3, 3)

    # Generate full kernel offsets
    kernel_offsets_full = kernel_offsets_from_size(kernel_size, (1, 1, 1)).to(device)
    total_kernels = kernel_offsets_full.shape[0]
    expected_symmetric_kernels = total_kernels // 2

    # Generate non-symmetric kernel map
    kernel_map_full = _kernel_map_from_offsets(
        voxel_hashmap,
        bcoords,
        kernel_offsets_full,
    )

    # Generate symmetric kernel map
    kernel_map_symmetric = _kernel_map_from_offsets(
        voxel_hashmap,
        bcoords,
        kernel_offsets_full[:expected_symmetric_kernels],
    )

    # Check dimensions
    assert len(kernel_map_symmetric) == expected_symmetric_kernels
    assert len(kernel_map_full) == total_kernels

    # Compare first half
    for i in range(expected_symmetric_kernels):
        full_in_map, full_out_map = kernel_map_full[i]
        sym_in_map, sym_out_map = kernel_map_symmetric[i]

        # Sort for comparison
        full_in_sorted, full_in_indices = torch.sort(full_in_map)
        full_out_sorted = full_out_map[full_in_indices]

        sym_in_sorted, sym_in_indices = torch.sort(sym_in_map)
        sym_out_sorted = sym_out_map[sym_in_indices]

        # Verify they match
        assert torch.equal(
            full_in_sorted, sym_in_sorted
        ), f"Offset method: Mismatch in kernel {i} in_maps"
        assert torch.equal(
            full_out_sorted, sym_out_sorted
        ), f"Offset method: Mismatch in kernel {i} out_maps"


def test_symmetric_kernel_map_size_method(setup_voxels):
    """Test symmetric kernel map using size method directly."""
    device = torch.device("cuda:0")
    voxels: Voxels = setup_voxels.to(device)

    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    voxel_hashmap = voxels.coordinate_hashmap
    kernel_size = (3, 3, 3)

    # Generate non-symmetric kernel map
    kernel_map_full = _kernel_map_from_size(
        voxel_hashmap,
        bcoords,
        kernel_size,
        skip_symmetric_kernel_map=False,
    )

    # Generate symmetric kernel map
    kernel_map_symmetric = _kernel_map_from_size(
        voxel_hashmap,
        bcoords,
        kernel_size,
        skip_symmetric_kernel_map=True,
    )

    total_kernels = 27  # 3x3x3
    expected_symmetric_kernels = total_kernels // 2  # 13

    # Check dimensions
    assert len(kernel_map_symmetric) == expected_symmetric_kernels
    assert len(kernel_map_full) == total_kernels

    # Compare first half
    for i in range(expected_symmetric_kernels):
        full_in_map, full_out_map = kernel_map_full[i]
        sym_in_map, sym_out_map = kernel_map_symmetric[i]

        # Sort for comparison
        full_in_sorted, full_in_indices = torch.sort(full_in_map)
        full_out_sorted = full_out_map[full_in_indices]

        sym_in_sorted, sym_in_indices = torch.sort(sym_in_map)
        sym_out_sorted = sym_out_map[sym_in_indices]

        # Verify they match
        assert torch.equal(
            full_in_sorted, sym_in_sorted
        ), f"Size method: Mismatch in kernel {i} in_maps"
        assert torch.equal(
            full_out_sorted, sym_out_sorted
        ), f"Size method: Mismatch in kernel {i} out_maps"


def test_symmetric_kernel_map_non_cubic_kernels(setup_voxels):
    """Test that symmetric skipping only applies to odd cubic kernels."""
    device = torch.device("cuda:0")
    voxels: Voxels = setup_voxels.to(device)

    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    in_to_out_stride_ratio = (1, 1, 1)

    # Test non-cubic kernel (should not apply symmetric skipping)
    kernel_size_non_cubic = (3, 5, 3)
    kernel_map_full = generate_kernel_map(
        batch_indexed_in_coords=bcoords,
        batch_indexed_out_coords=bcoords,
        in_to_out_stride_ratio=in_to_out_stride_ratio,
        kernel_size=kernel_size_non_cubic,
        method="size",
        skip_symmetric_kernel_map=False,
    )

    kernel_map_symmetric = generate_kernel_map(
        batch_indexed_in_coords=bcoords,
        batch_indexed_out_coords=bcoords,
        in_to_out_stride_ratio=in_to_out_stride_ratio,
        kernel_size=kernel_size_non_cubic,
        method="size",
        skip_symmetric_kernel_map=True,
    )

    # Should have same length since symmetric skipping doesn't apply to non-cubic kernels
    assert len(kernel_map_full) == len(kernel_map_symmetric)

    # Test even kernel (should not apply symmetric skipping)
    kernel_size_even = (4, 4, 4)
    kernel_map_full_even = generate_kernel_map(
        batch_indexed_in_coords=bcoords,
        batch_indexed_out_coords=bcoords,
        in_to_out_stride_ratio=in_to_out_stride_ratio,
        kernel_size=kernel_size_even,
        method="size",
        skip_symmetric_kernel_map=False,
    )

    kernel_map_symmetric_even = generate_kernel_map(
        batch_indexed_in_coords=bcoords,
        batch_indexed_out_coords=bcoords,
        in_to_out_stride_ratio=in_to_out_stride_ratio,
        kernel_size=kernel_size_even,
        method="size",
        skip_symmetric_kernel_map=True,
    )

    # Should have same length since symmetric skipping doesn't apply to even kernels
    assert len(kernel_map_full_even) == len(kernel_map_symmetric_even)


@pytest.mark.benchmark(group="kernel_map")
@pytest.mark.parametrize("kernel_size", [(3, 3, 3), (5, 5, 5), (7, 7, 7), (9, 9, 9)])
class TestKernelMapPerformance:
    def test_offsets_method(self, benchmark: BenchmarkFixture, setup_voxels, kernel_size):
        device = torch.device("cuda:0")
        voxels: Voxels = setup_voxels.to(device)
        bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        voxel_hashmap = voxels.coordinate_hashmap
        kernel_dilation = (1, 1, 1)
        kernel_offsets = kernel_offsets_from_size(kernel_size, kernel_dilation).to(device)

        def run_benchmark():
            return _kernel_map_from_offsets(
                voxel_hashmap,
                bcoords,
                kernel_offsets,
                return_type="offsets",
            )

        benchmark.pedantic(run_benchmark, iterations=4, rounds=3, warmup_rounds=1)

    def test_size_method(self, benchmark: BenchmarkFixture, setup_voxels, kernel_size):
        device = torch.device("cuda:0")
        voxels: Voxels = setup_voxels.to(device)
        bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        voxel_hashmap = voxels.coordinate_hashmap

        def run_benchmark():
            return _kernel_map_from_size(
                voxel_hashmap,
                bcoords,
                kernel_size,
                return_type="offsets",
            )

        benchmark.pedantic(run_benchmark, iterations=4, rounds=3, warmup_rounds=1)

    def test_direct_method(self, benchmark: BenchmarkFixture, setup_voxels, kernel_size):
        device = torch.device("cuda:0")
        voxels: Voxels = setup_voxels.to(device)
        bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        voxel_hashmap = voxels.coordinate_hashmap
        kernel_dilation = (1, 1, 1)

        def run_benchmark():
            return _kernel_map_from_direct_queries(
                voxel_hashmap,
                bcoords,
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
            )

        benchmark.pedantic(run_benchmark, iterations=4, rounds=3, warmup_rounds=1)

    def test_symmetric_vs_full_performance(
        self, benchmark: BenchmarkFixture, setup_voxels, kernel_size
    ):
        """Benchmark symmetric vs full kernel map performance."""
        device = torch.device("cuda:0")
        voxels: Voxels = setup_voxels.to(device)
        bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        in_to_out_stride_ratio = (1, 1, 1)

        def run_symmetric():
            return generate_kernel_map(
                batch_indexed_in_coords=bcoords,
                batch_indexed_out_coords=bcoords,
                in_to_out_stride_ratio=in_to_out_stride_ratio,
                kernel_size=kernel_size,
                method="size",
                skip_symmetric_kernel_map=True,
            )

        # Only benchmark the symmetric version
        result = benchmark.pedantic(run_symmetric, iterations=2, rounds=2, warmup_rounds=1)

        # Verify it produces correct results
        assert len(result) == (kernel_size[0] * kernel_size[1] * kernel_size[2]) // 2
        print(f"\nKernel size {kernel_size}: Symmetric kernel map benchmark completed")
