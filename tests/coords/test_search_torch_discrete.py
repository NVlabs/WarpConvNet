# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.ravel import ravel_multi_index
from warpconvnet.geometry.coords.search.torch_discrete import (
    generate_kernel_map,
    kernel_offsets_from_size,
    _kernel_map_from_offsets,
    _kernel_map_from_size,
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

    # Compare results
    assert len(kernel_map_offsets) == len(kernel_map_size)

    for i in range(len(kernel_map_offsets)):
        in_map_o, out_map_o = kernel_map_offsets[i]
        in_map_s, out_map_s = kernel_map_size[i]

        # Check sizes match
        assert len(in_map_o) == len(in_map_s)
        assert len(out_map_o) == len(out_map_s)

        # Check values match (after sorting)
        assert torch.equal(torch.sort(in_map_o)[0], torch.sort(in_map_s)[0])
        assert torch.equal(torch.sort(out_map_o)[0], torch.sort(out_map_s)[0])


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


def test_kernel_map_correctness(setup_voxels):
    """Test kernel map correctness by verifying coordinate relationships."""
    device = torch.device("cuda:0")
    voxels: Voxels = setup_voxels.to(device)

    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    voxel_hashmap = voxels.coordinate_hashmap

    # Define a kernel
    kernel_size = (3, 3, 3)
    # Generate kernel offsets
    # Note: kernel_offsets_from_size returns (K, D+1) where D+1 includes batch dim which is 0
    kernel_offsets = kernel_offsets_from_size(
        kernel_size=kernel_size,
        kernel_dilation=(1, 1, 1),
        device=device,
    )

    # Get kernel map
    kernel_map: IntSearchResult = _kernel_map_from_offsets(
        voxel_hashmap,
        bcoords,
        kernel_offsets,
    )

    in_maps = kernel_map.in_maps
    out_maps = kernel_map.out_maps
    offsets = kernel_map.offsets

    # Verify a random sample of the mappings
    # Total number of mappings could be large, so we sample.
    num_mappings = in_maps.shape[0]
    if num_mappings > 0:
        # Use a larger sample size for robustness, but keep it reasonable
        num_samples = min(1000, num_mappings)
        # Use deterministic sampling for reproducibility if needed, or random
        # simple random sample
        sample_indices = torch.randint(0, num_mappings, (num_samples,), device=device)

        # Get the global indices into bcoords
        sample_in_indices = in_maps[sample_indices].long()  # Indices into bcoords
        sample_out_indices = out_maps[sample_indices].long()  # Indices into bcoords

        # Get the actual coordinates
        # bcoords is (N, 4) -> (batch_idx, z, y, x)
        in_coords_actual = bcoords[sample_in_indices]
        out_coords_actual = bcoords[sample_out_indices]

        # We need to find the specific kernel offset for each sampled mapping to verify correctness.
        # This is tricky with random sampling across all mappings because we don't know k easily without search.
        # Alternatively, we can iterate over kernel offsets and sample within each block.

    # Better approach: Iterate over kernel offsets and verifies a few points for each
    for k in range(kernel_offsets.shape[0]):
        start = offsets[k].item()
        end = offsets[k + 1].item()

        if start == end:
            continue

        # Get mappings for this kernel offset
        # We only check a subset to keep the test fast
        num_k = end - start
        if num_k > 10:
            # Sample 10 indices relative to start
            subset_rel = torch.randint(0, num_k, (10,), device=device)
            subset_indices = start + subset_rel
        else:
            subset_indices = torch.arange(start, end, device=device)

        k_in_indices = in_maps[subset_indices].long()
        k_out_indices = out_maps[subset_indices].long()

        k_in_coords = bcoords[k_in_indices]
        k_out_coords = bcoords[k_out_indices]

        # Convolution logic: output[p] = input[p + offset]
        # input_coord = output_coord + kernel_offset
        # kernel_offsets includes the batch dimension (which is 0).
        # bcoords: (batch, z, y, x)
        k_offset = kernel_offsets[k]

        # Expected input coordinate based on output and offset
        expected_in_coords = k_out_coords + k_offset

        # Check for non-negative coordinates before raveling if necessary,
        # but here they should be valid coordinates from bcoords.

        # Determine spatial shape for ravel
        # We can just use a large enough shape.
        # bcoords are (batch, z, y, x)
        # We treat batch as just another dimension for unique integer purposes
        max_vals = bcoords.max(dim=0).values
        # Add buffer
        spatial_shape = (max_vals + 5).cpu().tolist()  # +5 to be safe

        # Ravel actual and expected input coordinates
        raveled_actual = ravel_multi_index(k_in_coords, spatial_shape)
        raveled_expected = ravel_multi_index(expected_in_coords, spatial_shape)

        assert torch.equal(
            raveled_actual, raveled_expected
        ), f"Kernel map mismatch for offset k={k}, offset={k_offset.tolist()}"
