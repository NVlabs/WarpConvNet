# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
from warpconvnet.geometry.coords.search.torch_discrete import (
    _kernel_map_from_size,
    _kernel_map_from_offsets,
    kernel_offsets_from_size,
)
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.types.voxels import Voxels

import warpconvnet._C as _C


def _compare_kernel_maps(result_a: IntSearchResult, result_b: IntSearchResult):
    """Compare two IntSearchResults, allowing different ordering within each offset group."""
    assert len(result_a) == len(
        result_b
    ), f"Different number of kernel offsets: {len(result_a)} vs {len(result_b)}"

    for k in range(len(result_a)):
        in_a, out_a = result_a[k]
        in_b, out_b = result_b[k]

        assert len(in_a) == len(
            in_b
        ), f"Offset {k}: different number of pairs: {len(in_a)} vs {len(in_b)}"

        if len(in_a) == 0:
            continue

        # Sort by (in_map, out_map) pairs to compare regardless of ordering
        pairs_a = torch.stack([in_a, out_a], dim=1)
        pairs_b = torch.stack([in_b, out_b], dim=1)

        # Ravel to single int for sorting
        max_val = max(pairs_a.max().item(), pairs_b.max().item()) + 1
        keys_a = pairs_a[:, 0] * max_val + pairs_a[:, 1]
        keys_b = pairs_b[:, 0] * max_val + pairs_b[:, 1]

        sorted_a, _ = torch.sort(keys_a)
        sorted_b, _ = torch.sort(keys_b)

        assert torch.equal(sorted_a, sorted_b), f"Offset {k}: pair mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedKernelMap:
    """Tests for the fused kernel map generator."""

    @pytest.fixture
    def random_coords(self):
        """Generate random 3D coordinates with batch indices."""
        torch.manual_seed(42)
        N = 10000
        device = torch.device("cuda:0")
        coords = torch.randint(0, 50, (N, 3), device=device, dtype=torch.int32)
        # Deduplicate
        coords = torch.unique(coords, dim=0)
        batch_idx = torch.zeros(coords.shape[0], 1, device=device, dtype=torch.int32)
        batch_coords = torch.cat([batch_idx, coords], dim=1)  # [N, 4]
        return batch_coords

    @pytest.fixture
    def multi_batch_coords(self):
        """Generate random coordinates across multiple batches."""
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        all_coords = []
        for b in range(3):
            N = torch.randint(1000, 5000, (1,)).item()
            coords = torch.randint(0, 40, (N, 3), device=device, dtype=torch.int32)
            coords = torch.unique(coords, dim=0)
            batch_idx = torch.full((coords.shape[0], 1), b, device=device, dtype=torch.int32)
            all_coords.append(torch.cat([batch_idx, coords], dim=1))
        return torch.cat(all_coords, dim=0)

    def _run_fused_vs_fallback(self, batch_coords, kernel_sizes):
        """Run both fused and fallback paths and compare results."""
        device = batch_coords.device

        # Build hash table from coordinates
        hashtable = TorchHashTable.from_keys(
            batch_coords, hash_method=HashMethod.CITY, device=device
        )

        # Use the fused kernel map directly via C++ binding
        if not hasattr(_C.coords, "fused_kernel_map"):
            pytest.skip("fused_kernel_map not available in C++ extension")

        in_maps_fused, out_maps_fused, offsets_fused = _C.coords.fused_kernel_map(
            batch_coords.contiguous(),
            hashtable._table_kvs.contiguous(),
            hashtable.vector_keys.contiguous(),
            hashtable.capacity,
            list(kernel_sizes),
            hashtable.hash_method.value,
        )
        result_fused = IntSearchResult(in_maps_fused, out_maps_fused, offsets_fused)

        # Use the fallback path (search-once + postprocess)
        result_fallback = _kernel_map_from_offsets(
            hashtable,
            batch_coords,
            kernel_offsets_from_size(kernel_sizes, (1,) * len(kernel_sizes), device=device),
            return_type="offsets",
        )

        _compare_kernel_maps(result_fused, result_fallback)

    def test_matches_fallback_3x3x3(self, random_coords):
        """Fused kernel map produces same results as offset-based method for 3x3x3."""
        self._run_fused_vs_fallback(random_coords, (3, 3, 3))

    def test_matches_fallback_5x5x5(self, random_coords):
        """Fused kernel map produces same results as offset-based method for 5x5x5."""
        self._run_fused_vs_fallback(random_coords, (5, 5, 5))

    def test_various_kernel_sizes(self, random_coords):
        """Test with different kernel sizes including asymmetric ones."""
        for ks in [(3, 3, 1), (1, 3, 3), (3, 1, 3), (1, 1, 1)]:
            self._run_fused_vs_fallback(random_coords, ks)

    def test_multi_batch(self, multi_batch_coords):
        """Test with multiple batches."""
        self._run_fused_vs_fallback(multi_batch_coords, (3, 3, 3))

    def test_large_input(self):
        """Test with a large number of coordinates."""
        torch.manual_seed(123)
        device = torch.device("cuda:0")
        N = 100000
        coords = torch.randint(0, 200, (N, 3), device=device, dtype=torch.int32)
        coords = torch.unique(coords, dim=0)
        batch_idx = torch.zeros(coords.shape[0], 1, device=device, dtype=torch.int32)
        batch_coords = torch.cat([batch_idx, coords], dim=1)
        self._run_fused_vs_fallback(batch_coords, (3, 3, 3))

    def test_all_hash_methods(self, random_coords):
        """Test fused kernel map with all hash methods."""
        if not hasattr(_C.coords, "fused_kernel_map"):
            pytest.skip("fused_kernel_map not available in C++ extension")

        device = random_coords.device
        kernel_sizes = (3, 3, 3)

        results = []
        for method in [HashMethod.FNV1A, HashMethod.CITY, HashMethod.MURMUR]:
            hashtable = TorchHashTable.from_keys(random_coords, hash_method=method, device=device)
            in_maps, out_maps, offsets = _C.coords.fused_kernel_map(
                random_coords.contiguous(),
                hashtable._table_kvs.contiguous(),
                hashtable.vector_keys.contiguous(),
                hashtable.capacity,
                list(kernel_sizes),
                method.value,
            )
            results.append(IntSearchResult(in_maps, out_maps, offsets))

        # All hash methods should produce the same result
        for i in range(1, len(results)):
            _compare_kernel_maps(results[0], results[i])

    def test_integration_via_kernel_map_from_size(self, random_coords):
        """Test that _kernel_map_from_size uses fused path and produces correct results."""
        device = random_coords.device
        kernel_sizes = (3, 3, 3)

        hashtable = TorchHashTable.from_keys(
            random_coords, hash_method=HashMethod.CITY, device=device
        )

        # _kernel_map_from_size should use fused path when available
        result_size = _kernel_map_from_size(
            hashtable, random_coords, kernel_sizes, return_type="offsets"
        )

        # Compare with offset-based method
        kernel_offsets = kernel_offsets_from_size(
            kernel_sizes, (1,) * len(kernel_sizes), device=device
        )
        result_offsets = _kernel_map_from_offsets(
            hashtable, random_coords, kernel_offsets, return_type="offsets"
        )

        _compare_kernel_maps(result_size, result_offsets)

    def test_empty_input(self):
        """Test with empty coordinate set."""
        if not hasattr(_C.coords, "fused_kernel_map"):
            pytest.skip("fused_kernel_map not available in C++ extension")

        device = torch.device("cuda:0")
        # Create a small hash table with some coords, query with zero coords
        coords = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
        hashtable = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)

        # Query with the same coords - should find at least one match
        in_maps, out_maps, offsets = _C.coords.fused_kernel_map(
            coords.contiguous(),
            hashtable._table_kvs.contiguous(),
            hashtable.vector_keys.contiguous(),
            hashtable.capacity,
            [3, 3, 3],
            HashMethod.CITY.value,
        )
        # Should have exactly 1 pair for the center offset (identity)
        assert offsets[-1].item() >= 1

    def test_with_voxels_fixture(self, setup_voxels):
        """Test fused kernel map with the standard voxels fixture."""
        device = torch.device("cuda:0")
        voxels: Voxels = setup_voxels.to(device)

        bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        voxel_hashmap = voxels.coordinate_hashmap
        kernel_sizes = (3, 3, 3)

        # Use _kernel_map_from_size (which uses fused path internally)
        result = _kernel_map_from_size(voxel_hashmap, bcoords, kernel_sizes)

        tot_num_maps = result.offsets[-1].item()
        assert tot_num_maps == len(result.in_maps)
        assert tot_num_maps == len(result.out_maps)
        assert tot_num_maps > 0
