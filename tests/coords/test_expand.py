# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import warp as wp

from warpconvnet.geometry.coords.ops.expand import expand_coords
from warpconvnet.geometry.coords.search.torch_discrete import kernel_offsets_from_size
from warpconvnet.geometry.types.voxels import Voxels


@pytest.fixture
def setup_voxels():
    """Setup test voxels with random coordinates."""
    wp.init()
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, min_N, max_N, C = 3, 100000, 1000000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    coords = [(torch.rand((N, 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    voxels = Voxels(coords, features, device=device).unique()
    return voxels


def test_expand_coords(setup_voxels):
    """Test coordinate expansion functionality."""
    voxels = setup_voxels

    up_coords, offsets = expand_coords(
        voxels.batch_indexed_coordinates,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
    )

    # Test output properties
    assert up_coords.shape[0] > voxels.coordinate_tensor.shape[0]
    assert offsets.shape == (voxels.batch_size + 1,)


@pytest.mark.parametrize("kernel_batch", [None, 9, 27])
def test_expand_kernel_batch(setup_voxels, kernel_batch):
    """Test expansion with different kernel batch sizes."""
    voxels = setup_voxels

    up_coords, offsets = expand_coords(
        voxels.batch_indexed_coordinates,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        kernel_batch=kernel_batch,
    )

    # Results should be the same regardless of batch size
    ref_coords, ref_offsets = expand_coords(
        voxels.batch_indexed_coordinates,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
    )

    assert torch.equal(offsets, ref_offsets)
    assert up_coords.shape[0] == ref_coords.shape[0]


def _cpu_expand(
    coords: torch.Tensor, kernel_size: tuple[int, ...], kernel_dilation: tuple[int, ...]
) -> torch.Tensor:
    coords_cpu = coords.to(dtype=torch.int32, device="cpu")
    offsets = kernel_offsets_from_size(kernel_size, kernel_dilation, device="cpu")
    seen = {tuple(int(v) for v in coord.tolist()) for coord in coords_cpu}
    for offset in offsets:
        offset_vals = offset.tolist()
        for coord in coords_cpu:
            candidate = tuple(
                int(coord[i].item()) + int(offset_vals[i]) for i in range(len(offset_vals))
            )
            seen.add(candidate)
    if not seen:
        return torch.empty((0, coords_cpu.shape[1]), dtype=torch.int32)
    return torch.tensor(sorted(seen), dtype=torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for expand_coords")
def test_expand_coords_matches_cpu_reference():
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
        ],
        dtype=torch.int32,
        device=torch.device("cuda"),
    )
    kernel_size = (3, 3, 3)
    kernel_dilation = (1, 1, 1)

    gpu_coords, _ = expand_coords(coords, kernel_size=kernel_size, kernel_dilation=kernel_dilation)
    cpu_coords = _cpu_expand(coords.cpu(), kernel_size, kernel_dilation).to(coords.device)

    gpu_sorted = sorted(map(tuple, gpu_coords.cpu().tolist()))
    cpu_sorted = sorted(map(tuple, cpu_coords.cpu().tolist()))
    assert gpu_sorted == cpu_sorted
