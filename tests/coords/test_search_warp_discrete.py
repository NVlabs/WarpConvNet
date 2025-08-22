# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.geometry.coords.search.warp_discrete import (
    _kernel_map_from_direct_queries,
    _kernel_map_from_offsets,
    _kernel_map_from_size,
    kernel_offsets_from_size,
)
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


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
        voxel_hashmap._hash_struct,
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
        voxel_hashmap._hash_struct,
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
        voxel_hashmap._hash_struct,
        bcoords,
        kernel_offsets,
    )

    kernel_map_size = _kernel_map_from_size(
        voxel_hashmap._hash_struct,
        bcoords,
        kernel_size,
    )

    kernel_map_direct = _kernel_map_from_direct_queries(
        voxel_hashmap._hash_struct,
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
                voxel_hashmap._hash_struct,
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
                voxel_hashmap._hash_struct,
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
                voxel_hashmap._hash_struct,
                bcoords,
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
            )

        benchmark.pedantic(run_benchmark, iterations=4, rounds=3, warmup_rounds=1)
