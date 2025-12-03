# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.coords.ops.batch_index import (
    batch_index_from_offset,
    offsets_from_batch_index,
)
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING, encode
from warpconvnet.geometry.coords.ops.expand import expand_coords
from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.geometry.types.voxels import Voxels

from warpconvnet.nn.functional.sparse_conv.helper import (
    STRIDED_CONV_MODE,
    _apply_generative_policy,
    generate_output_coords_and_kernel_map,
)
from warpconvnet.nn.functional.sparse_ops import prune_spatially_sparse_tensor
from warpconvnet.nn.modules.prune import SparsePrune
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv


def _sorted_coords_by_morton(coords: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Sort coordinates within each batch segment for deterministic comparison."""
    sort_result = encode(
        coords,
        batch_offsets=offsets.to(coords.device),
        order=POINT_ORDERING.MORTON_XYZ,
        return_perm=True,
    )
    return coords[sort_result.perm]


def _sorted_batch_indexed(coords: torch.Tensor) -> torch.Tensor:
    """Sort batch-indexed coordinates via Morton order within each batch."""
    batch_index = coords[:, 0]
    offsets = offsets_from_batch_index(batch_index.cpu()).to(coords.device)
    sort_result = encode(
        coords[:, 1:],
        batch_offsets=offsets,
        order=POINT_ORDERING.MORTON_XYZ,
        return_perm=True,
    )
    return coords[sort_result.perm]


def test_generative_matches_intcoords_expand(toy_voxels):
    """Ensure generative convolution reuses IntCoords.expand semantics."""
    torch.manual_seed(0)
    voxels = toy_voxels
    C_in, C_out = voxels.num_channels, voxels.num_channels + 5

    conv = SpatiallySparseConv(
        C_in,
        C_out,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        generative=True,
    ).to(voxels.device)

    out = conv(voxels)
    expected_coords = voxels.batched_coordinates.expand(kernel_size=(3, 3, 3), dilation=(1, 1, 1))

    assert out.num_channels == C_out
    actual_sorted = _sorted_coords_by_morton(out.coordinate_tensor, out.offsets)
    expected_sorted = _sorted_coords_by_morton(
        expected_coords.batched_tensor,
        expected_coords.offsets,
    )
    assert torch.equal(actual_sorted, expected_sorted)
    assert torch.equal(out.offsets.cpu(), expected_coords.offsets)


def test_generative_stride_only_matches_expand_pipeline(toy_voxels):
    """Generative + stride-only should match striding then IntCoords expansion."""
    torch.manual_seed(0)
    voxels = toy_voxels
    C_in, C_out = voxels.num_channels, voxels.num_channels + 3
    stride = (2, 2, 2)

    conv = SpatiallySparseConv(
        C_in,
        C_out,
        kernel_size=(3, 3, 3),
        stride=stride,
        generative=True,
        stride_mode=STRIDED_CONV_MODE.STRIDE_ONLY,
    ).to(voxels.device)

    out = conv(voxels)

    strided_coords, strided_offsets = stride_coords(
        voxels.batch_indexed_coordinates,
        stride=stride,
    )
    strided_intcoords = IntCoords(strided_coords[:, 1:], offsets=strided_offsets)
    expected_coords = strided_intcoords.expand(kernel_size=(3, 3, 3), dilation=(1, 1, 1))

    assert out.num_channels == C_out
    actual_sorted = _sorted_coords_by_morton(out.coordinate_tensor, out.offsets)
    expected_sorted = _sorted_coords_by_morton(
        expected_coords.batched_tensor,
        expected_coords.offsets,
    )
    assert torch.equal(actual_sorted, expected_sorted)
    assert torch.equal(out.offsets.cpu(), expected_coords.offsets)


def test_intcoords_prune_preserves_batch_offsets(toy_voxels):
    """Pruning via IntCoords maintains per-batch counts and metadata."""
    torch.manual_seed(0)
    voxels = toy_voxels
    coords = voxels.batched_coordinates

    mask = torch.zeros(
        coords.batched_tensor.shape[0], dtype=torch.bool, device=coords.batched_tensor.device
    )
    mask[::2] = True  # Keep every other coordinate to ensure multiple batches remain populated

    pruned = coords.prune(mask)

    # Expect tensor contents to match the masked coordinates
    assert torch.equal(pruned.batched_tensor.cpu(), coords.batched_tensor[mask].cpu())

    batch_indices = batch_index_from_offset(coords.offsets).to(mask.device)
    expected_counts = torch.bincount(
        batch_indices[mask].long(),
        minlength=voxels.batch_size,
    )
    pruned_counts = pruned.offsets[1:] - pruned.offsets[:-1]

    assert torch.equal(pruned_counts.cpu(), expected_counts.cpu())
    assert pruned.tensor_stride == coords.tensor_stride


def test_prune_functional_matches_intcoords(toy_voxels):
    """Functional pruning should mirror IntCoords.prune and mask features."""
    voxels = toy_voxels
    mask = torch.zeros(voxels.coordinate_tensor.shape[0], dtype=torch.bool, device=voxels.device)
    mask[1::3] = True

    functional_pruned = prune_spatially_sparse_tensor(voxels, mask)
    coords_pruned = voxels.batched_coordinates.prune(mask)
    feats_pruned = voxels.feature_tensor[mask]

    assert isinstance(functional_pruned, voxels.__class__)
    assert torch.equal(functional_pruned.coordinate_tensor, coords_pruned.batched_tensor)
    assert torch.equal(functional_pruned.offsets, coords_pruned.offsets)
    assert torch.equal(functional_pruned.feature_tensor, feats_pruned)


def test_prune_functional_rejects_shape_mismatch(toy_voxels):
    voxels = toy_voxels
    mask = torch.ones(
        voxels.coordinate_tensor.shape[0] - 1, dtype=torch.bool, device=voxels.device
    )
    with pytest.raises(ValueError):
        prune_spatially_sparse_tensor(voxels, mask)


def test_sparse_prune_module(toy_voxels):
    voxels = toy_voxels
    mask = torch.zeros_like(voxels.coordinate_tensor[:, 0], dtype=torch.bool, device=voxels.device)
    mask[::4] = True

    module = SparsePrune().to(voxels.device)
    out = module(voxels, mask)

    expected = prune_spatially_sparse_tensor(voxels, mask)
    assert torch.equal(out.coordinate_tensor, expected.coordinate_tensor)
    assert torch.equal(out.feature_tensor, expected.feature_tensor)
    assert torch.equal(out.offsets, expected.offsets)


def _manual_expand_after_stride(voxels, stride):
    strided_coords, strided_offsets = stride_coords(
        voxels.batch_indexed_coordinates,
        stride=stride,
    )
    strided_intcoords = IntCoords(strided_coords[:, 1:], offsets=strided_offsets)
    return strided_intcoords.expand(kernel_size=(3, 3, 3), dilation=(1, 1, 1)), strided_coords


def test_apply_generative_policy_stride_only_kernel_inputs(toy_voxels):
    """Kernel-map inputs remain original coords for stride-only generative expansion."""
    voxels = toy_voxels
    stride = (2, 2, 2)
    (
        out_coords,
        out_offsets,
        kernel_in_coords,
    ) = _apply_generative_policy(
        voxels,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        stride=stride,
        stride_mode=STRIDED_CONV_MODE.STRIDE_ONLY,
        transposed=False,
    )

    manual_expanded, _ = _manual_expand_after_stride(voxels, stride)

    assert torch.equal(
        _sorted_batch_indexed(kernel_in_coords),
        _sorted_batch_indexed(voxels.batch_indexed_coordinates),
    )
    assert torch.equal(out_offsets, manual_expanded.offsets)
    assert torch.equal(
        _sorted_coords_by_morton(out_coords[:, 1:], out_offsets),
        _sorted_coords_by_morton(manual_expanded.batched_tensor, manual_expanded.offsets),
    )


def test_apply_generative_policy_reduce_stride_kernel_inputs(toy_voxels):
    """Kernel-map inputs switch to strided coords under reduce-and-stride generative mode."""
    voxels = toy_voxels
    stride = (2, 2, 2)
    manual_expanded, strided_coords = _manual_expand_after_stride(voxels, stride)
    (
        out_coords,
        out_offsets,
        kernel_in_coords,
    ) = _apply_generative_policy(
        voxels,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        stride=stride,
        stride_mode=STRIDED_CONV_MODE.REDUCE_AND_STRIDE,
        transposed=False,
    )

    assert torch.equal(
        _sorted_batch_indexed(kernel_in_coords),
        _sorted_batch_indexed(strided_coords),
    )
    assert torch.equal(out_offsets, manual_expanded.offsets)
    assert torch.equal(
        _sorted_coords_by_morton(out_coords[:, 1:], out_offsets),
        _sorted_coords_by_morton(manual_expanded.batched_tensor, manual_expanded.offsets),
    )


@pytest.mark.parametrize(
    "stride_mode",
    [STRIDED_CONV_MODE.STRIDE_ONLY, STRIDED_CONV_MODE.REDUCE_AND_STRIDE],
)
def test_generate_output_coords_matches_helper(toy_voxels, stride_mode):
    """generate_output_coords_and_kernel_map should mirror helper outputs exactly."""
    voxels = toy_voxels
    stride = (2, 2, 2)
    (
        helper_coords,
        helper_offsets,
        _,
    ) = _apply_generative_policy(
        voxels,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        stride=stride,
        stride_mode=stride_mode,
        transposed=False,
    )
    (
        gen_coords,
        gen_offsets,
        _,
    ) = generate_output_coords_and_kernel_map(
        input_sparse_tensor=voxels,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        stride=stride,
        generative=True,
        transposed=False,
        stride_mode=stride_mode,
    )

    assert torch.equal(helper_offsets, gen_offsets)
    assert torch.equal(
        _sorted_batch_indexed(helper_coords),
        _sorted_batch_indexed(gen_coords),
    )


def test_generate_output_coords_transposed_generative(toy_voxels):
    """Transposed generative convolution should expand coordinates like non-transposed generative."""
    voxels = toy_voxels
    batch_indexed_out_coords, out_offsets, kernel_map = generate_output_coords_and_kernel_map(
        input_sparse_tensor=voxels,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        stride=(1, 1, 1),
        generative=True,
        transposed=True,
    )

    # The output coordinates should match IntCoords.expand semantics
    expected_coords = voxels.batched_coordinates.expand(kernel_size=(3, 3, 3), dilation=(1, 1, 1))
    actual_sorted = _sorted_coords_by_morton(batch_indexed_out_coords[:, 1:], out_offsets)
    expected_sorted = _sorted_coords_by_morton(
        expected_coords.batched_tensor,
        expected_coords.offsets,
    )
    assert torch.equal(actual_sorted, expected_sorted)
    assert torch.equal(out_offsets, expected_coords.offsets)

    # Kernel map should have valid structure
    assert kernel_map is not None
    assert len(kernel_map) > 0


def test_large_scale_expand_uniqueness():
    """Ensure expand_coords does not produce duplicates on large input."""
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")

    # Generate large random coordinates
    N = 1000000
    coords_range = 60

    num_batches = 4
    coords_list = []
    for i in range(num_batches):
        # Generate random coordinates
        c = torch.randint(
            -coords_range, coords_range, (N // num_batches, 3), device=device, dtype=torch.int32
        )
        b = torch.full((N // num_batches, 1), i, device=device, dtype=torch.int32)
        coords_list.append(torch.cat([b, c], dim=1))

    batch_indexed_coords = torch.cat(coords_list, dim=0)
    # Ensure input uniqueness per batch
    batch_indexed_coords = torch.unique(batch_indexed_coords, dim=0)

    kernel_size = (3, 3, 3)
    dilation = (1, 1, 1)

    out_coords, out_offsets = expand_coords(
        batch_indexed_coords, kernel_size=kernel_size, kernel_dilation=dilation
    )

    # Check for duplicates
    unique_out, counts = torch.unique(out_coords, dim=0, return_counts=True)
    if unique_out.shape[0] != out_coords.shape[0]:
        num_duplicates = out_coords.shape[0] - unique_out.shape[0]
        duplicate_mask = counts > 1
        duplicate_examples = unique_out[duplicate_mask]
        # Find frequencies of duplicates
        max_dups = counts.max().item()
        pytest.fail(
            f"Found {num_duplicates} duplicate coordinates in expanded output. "
            f"Total: {out_coords.shape[0]}, Unique: {unique_out.shape[0]}. "
            f"Max duplicates for a single coord: {max_dups}. "
            f"Example duplicates: {duplicate_examples[:5]}"
        )


def test_large_scale_transposed_generative_duplicates():
    """
    Reproduce duplicate coordinates issue with SpatiallySparseConv
    configured as transposed=True, generative=True, stride=(2,2,2).
    """
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")

    # Generate large random coordinates
    # N needs to be large enough to cause hash collisions or stress the table resizing
    N = 1000000
    coords_range = 50  # Very dense

    num_batches = 1  # Single batch to focus on collisions within one set
    coords_list = []
    for i in range(num_batches):
        c = torch.randint(
            -coords_range, coords_range, (N // num_batches, 3), device=device, dtype=torch.int32
        )
        # Ensure uniqueness within batch for valid input
        c = torch.unique(c, dim=0)
        b = torch.full((c.shape[0], 1), i, device=device, dtype=torch.int32)
        coords_list.append(torch.cat([b, c], dim=1))

    batch_indexed_coords = torch.cat(coords_list, dim=0)

    # Construct Voxels object
    # We need features for the forward pass, even though we only care about coords
    features = torch.randn(batch_indexed_coords.shape[0], 16, device=device)

    # Re-split by batch for Voxels constructor
    coords_per_batch = []
    feats_per_batch = []
    for i in range(num_batches):
        mask = batch_indexed_coords[:, 0] == i
        coords_per_batch.append(batch_indexed_coords[mask, 1:])
        feats_per_batch.append(features[mask])

    voxels = Voxels(
        batched_coordinates=coords_per_batch, batched_features=feats_per_batch, device=device
    )

    # Configure the problematic layer
    in_channels = 16
    out_channels = 16
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)

    conv = SpatiallySparseConv(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        generative=True,
        transposed=True,
    ).to(device)

    # Run forward pass
    out_voxels = conv(voxels)

    # Check for duplicates in output coordinates
    out_coords = out_voxels.batch_indexed_coordinates
    unique_out, counts = torch.unique(out_coords, dim=0, return_counts=True)

    if unique_out.shape[0] != out_coords.shape[0]:
        num_duplicates = out_coords.shape[0] - unique_out.shape[0]
        max_dups = counts.max().item()
        duplicate_examples = unique_out[counts > 1][:5]

        pytest.fail(
            f"Found {num_duplicates} duplicate coordinates in output.\n"
            f"Total output coords: {out_coords.shape[0]}\n"
            f"Unique output coords: {unique_out.shape[0]}\n"
            f"Max duplicates for a single coord: {max_dups}\n"
            f"Examples: {duplicate_examples}"
        )
