# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.coords.ops.serialization import (
    encode,
    morton_code,
    POINT_ORDERING,
)


@pytest.fixture
def setup_coordinates():
    """Setup test coordinates for serialization testing."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample 3D coordinates
    batch_size = 3
    num_points = 100

    # Generate random coordinates in a reasonable range for Morton coding
    coords = torch.randint(0, 100, (num_points, 3), device=device, dtype=torch.int32)

    # Create batch indices
    batch_indices = torch.randint(0, batch_size, (num_points,), device=device)

    # Create offsets for batched processing
    offsets = torch.tensor([0, 30, 70, num_points], device=device)

    return coords, batch_indices, offsets, batch_size, device


@pytest.fixture
def setup_grid_coordinates():
    """Setup regular grid coordinates for deterministic testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a small regular grid
    x = torch.arange(4, device=device)
    y = torch.arange(4, device=device)
    z = torch.arange(4, device=device)

    # Create meshgrid and flatten
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    return coords, device


def test_encode_with_different_orders(setup_grid_coordinates):
    """Test encoding with different coordinate orderings."""
    coords, device = setup_grid_coordinates

    # Test all supported orderings
    codes_by_order = {}
    for order in POINT_ORDERING:
        codes = encode(coords, order=order)
        codes_by_order[order] = codes

        assert codes.shape == (coords.shape[0],), f"Codes shape mismatch for order {order}"
        assert codes.dtype == torch.int64, f"Codes dtype mismatch for order {order}"


def test_morton_code_single_batch(setup_coordinates):
    """Test morton_code function with single batch."""
    coords, _, _, _, device = setup_coordinates

    # Test without offsets (single batch)
    codes = morton_code(coords, order=POINT_ORDERING.MORTON_XYZ)

    assert codes.shape == (coords.shape[0],), "Codes shape mismatch"
    assert codes.dtype == torch.int64, "Codes should be int64"

    # Test that we can get ordering from encode
    result = encode(coords, order=POINT_ORDERING.MORTON_XYZ, return_perm=True)
    ordering = result.perm

    assert ordering.shape == (coords.shape[0],), "Ordering shape mismatch"
    assert ordering.dtype == torch.int64, "Ordering should be int64"

    # Test that ordering actually sorts the codes
    sorted_codes = codes[ordering]
    assert torch.all(sorted_codes[1:] >= sorted_codes[:-1]), "Codes should be sorted by ordering"


def test_morton_code_multi_batch(setup_coordinates):
    """Test morton_code function with multiple batches."""
    coords, batch_indices, offsets, batch_size, device = setup_coordinates

    # Test with offsets (multi-batch) — use encode() since morton_code() doesn't support batch_offsets
    codes = encode(coords.float(), batch_offsets=offsets, order=POINT_ORDERING.MORTON_XYZ)

    assert codes.shape == (coords.shape[0],), "Codes shape mismatch for multi-batch"

    # Test that we can get ordering from encode
    result = encode(
        coords.float(), batch_offsets=offsets, order=POINT_ORDERING.MORTON_XYZ, return_perm=True
    )
    ordering = result.perm

    # Test that ordering sorts within each batch
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        batch_codes = codes[start:end]
        batch_ordering = ordering[start:end] - start  # Adjust for batch offset

        sorted_batch_codes = batch_codes[batch_ordering]
        assert torch.all(
            sorted_batch_codes[1:] >= sorted_batch_codes[:-1]
        ), f"Batch {i} codes should be sorted"


def test_coordinate_permutation_effects(setup_grid_coordinates):
    """Test that coordinate permutations have the expected effects."""
    coords, device = setup_grid_coordinates

    # Test specific coordinate transformations
    # Original point: [1, 2, 3]
    test_coord = torch.tensor([[1, 2, 3]], device=device, dtype=torch.float32)

    # Test xyz vs yxz (swap x and y)
    code_xyz = encode(test_coord, order=POINT_ORDERING.MORTON_XYZ)

    # Manually swap coordinates and encode with xyz - should match yxz encoding
    swapped_coord = test_coord[:, [1, 0, 2]]  # [2, 1, 3]
    code_swapped = encode(swapped_coord, order=POINT_ORDERING.MORTON_XYZ)
    code_yxz = encode(test_coord, order=POINT_ORDERING.MORTON_YXZ)

    assert (
        code_swapped.item() == code_yxz.item()
    ), "yxz ordering should match manually swapped coordinates"


def test_ordering_deterministic():
    """Test that ordering is deterministic for the same input."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coords = torch.randint(0, 50, (20, 3), device=device, dtype=torch.float32)

    # Generate codes multiple times
    codes1 = encode(coords, order=POINT_ORDERING.MORTON_XYZ)
    codes2 = encode(coords, order=POINT_ORDERING.MORTON_XYZ)
    codes3 = encode(coords, order=POINT_ORDERING.MORTON_YXZ)
    codes4 = encode(coords, order=POINT_ORDERING.MORTON_YXZ)

    # Same order should give same results
    assert torch.equal(codes1, codes2), "xyz encoding should be deterministic"
    assert torch.equal(codes3, codes4), "yxz encoding should be deterministic"


def test_empty_coordinates():
    """Test handling of empty coordinate arrays."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    empty_coords = torch.empty((0, 3), device=device, dtype=torch.float32)

    # Test that empty coordinates don't crash
    codes = encode(empty_coords, order=POINT_ORDERING.MORTON_XYZ)
    assert codes.shape == (0,), "Empty coordinates should produce empty codes"

    # Test morton_code with empty coordinates
    codes = morton_code(empty_coords, order=POINT_ORDERING.MORTON_XYZ)
    assert codes.shape == (0,), "Empty coordinates should produce empty codes in morton_code"


def test_large_coordinates():
    """Test with coordinates at the edge of the supported range."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test coordinates near the maximum for 16-bit path
    large_coords = torch.tensor(
        [
            [0, 0, 0],
            [65535, 65535, 65535],  # Max for 16-bit
            [1000, 2000, 3000],
        ],
        device=device,
        dtype=torch.int32,
    )

    # Should not crash and should produce valid codes
    codes = encode(large_coords, order=POINT_ORDERING.MORTON_XYZ)
    assert codes.shape == (3,), "Large coordinates should be handled properly"

    # All codes should be different
    assert len(torch.unique(codes)) == 3, "Large coordinates should produce unique codes"


def test_encode_with_permutations_and_inverse():
    """Test the encode function with permutation and inverse permutation functionality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test coordinates
    coords = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.int32,
        device=device,
    )

    # Test the function with both permutation and inverse
    result = encode(coords, order=POINT_ORDERING.MORTON_XYZ, return_perm=True, return_inverse=True)

    # Check that result is a SerializationResult with all fields
    assert hasattr(result, "codes"), "Result should have codes field"
    assert hasattr(result, "perm"), "Result should have perm field"
    assert hasattr(result, "inverse_perm"), "Result should have inverse_perm field"
    assert result.perm is not None, "Perm should not be None when requested"
    assert result.inverse_perm is not None, "Inverse perm should not be None when requested"

    # Check shapes
    assert result.codes.shape == (coords.shape[0],), "Codes shape mismatch"
    assert result.perm.shape == (coords.shape[0],), "Perm shape mismatch"
    assert result.inverse_perm.shape == (coords.shape[0],), "Inverse perm shape mismatch"

    # Sort coordinates using permutation
    sorted_coords = coords[result.perm]

    # Restore original order using inverse permutation
    restored_coords = sorted_coords[result.inverse_perm]

    # Verify they match the original
    assert torch.allclose(restored_coords.float(), coords.float()), "Restoration failed!"

    # Test that inverse_perm[perm[i]] = i (inverse permutation property)
    identity_check = result.inverse_perm[result.perm]
    expected_identity = torch.arange(len(coords), device=device)
    assert torch.equal(identity_check, expected_identity), "Inverse permutation property failed!"


def test_encode_return_combinations():
    """Test different return value combinations of the encode function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coords = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ],
        dtype=torch.int32,
        device=device,
    )

    # Just codes (backward compatibility)
    codes_only = encode(coords, order=POINT_ORDERING.MORTON_XYZ)
    assert isinstance(
        codes_only, torch.Tensor
    ), "Should return tensor when no permutations requested"
    assert codes_only.shape == (coords.shape[0],), "Codes only shape mismatch"

    # Codes and permutation
    result_perm = encode(coords, order=POINT_ORDERING.MORTON_XYZ, return_perm=True)
    assert hasattr(result_perm, "codes"), "Result should have codes field"
    assert hasattr(result_perm, "perm"), "Result should have perm field"
    assert result_perm.perm is not None, "Perm should not be None when requested"
    assert result_perm.inverse_perm is None, "Inverse perm should be None when not requested"
    assert torch.equal(result_perm.codes, codes_only), "Codes should match"

    # Codes and inverse permutation
    result_inverse = encode(coords, order=POINT_ORDERING.MORTON_XYZ, return_inverse=True)
    assert hasattr(result_inverse, "codes"), "Result should have codes field"
    assert hasattr(result_inverse, "inverse_perm"), "Result should have inverse_perm field"
    assert (
        result_inverse.inverse_perm is not None
    ), "Inverse perm should not be None when requested"
    assert torch.equal(result_inverse.codes, codes_only), "Codes should match"

    # All three (codes, perm, inverse)
    result_all = encode(
        coords, order=POINT_ORDERING.MORTON_XYZ, return_perm=True, return_inverse=True
    )
    assert result_all.perm is not None, "Perm should not be None when requested"
    assert result_all.inverse_perm is not None, "Inverse perm should not be None when requested"
    assert torch.equal(result_all.codes, codes_only), "Codes should match"
    assert torch.equal(result_all.perm, result_perm.perm), "Perm should match"
    assert torch.equal(
        result_all.inverse_perm, result_inverse.inverse_perm
    ), "Inverse perm should match"


def test_encode_empty_coordinates_with_permutations():
    """Test handling of empty coordinates with permutation requests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    empty_coords = torch.empty(0, 3, dtype=torch.int32, device=device)

    # Test with permutations requested
    result = encode(empty_coords, return_perm=True, return_inverse=True)
    assert hasattr(result, "codes"), "Result should have codes field"
    assert hasattr(result, "perm"), "Result should have perm field"
    assert hasattr(result, "inverse_perm"), "Result should have inverse_perm field"
    assert result.codes.shape[0] == 0, "Empty codes test failed!"
    assert result.perm.shape[0] == 0, "Empty perm test failed!"
    assert result.inverse_perm.shape[0] == 0, "Empty inverse test failed!"
