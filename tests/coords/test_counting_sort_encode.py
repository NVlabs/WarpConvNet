"""Tests for counting_sort encoding method in voxel_encode.

Usage:
    pytest tests/coords/test_counting_sort_encode.py -v
"""
import pytest
import random
import torch

from warpconvnet.models.internal.backbones.components.voxel_encode import voxel_encode
from warpconvnet.geometry.coords.ops.serialization import SerializationResult


def _make_batched_coords(Ns, coord_range=64, seed=42):
    """Create batched grid coordinates on CUDA."""
    torch.manual_seed(seed)
    grid_coord = torch.randint(0, coord_range, (sum(Ns), 3), device="cuda")
    batch_offsets = torch.zeros(len(Ns) + 1, dtype=torch.int32)
    batch_offsets[1:] = torch.tensor(Ns, dtype=torch.int32).cumsum(0)
    return grid_coord, batch_offsets


def _check_grouping_equivalent(result_ref, result_test, grid_coord, window_size):
    """Check that two SerializationResults produce equivalent groupings.

    Two groupings are equivalent if they group exactly the same voxels together
    (i.e., the sorted counts match and within each group, voxels have the same
    window code).
    """
    # Counts must match (sorted, since window ordering may differ)
    counts_ref = sorted(result_ref.counts.cpu().tolist())
    counts_test = sorted(result_test.counts.cpu().tolist())
    assert counts_ref == counts_test, f"Counts mismatch: {counts_ref} vs {counts_test}"

    # Check perm/inverse_perm consistency
    N = grid_coord.shape[0]
    if result_test.perm is not None and result_test.inverse_perm is not None:
        # perm[inverse_perm[i]] == i for all i
        roundtrip = result_test.perm[result_test.inverse_perm]
        assert torch.equal(roundtrip, torch.arange(N, device=grid_coord.device)), \
            "perm[inverse_perm] != identity"

    # Check that within each group (contiguous in perm), all voxels
    # have the same window code
    if result_test.perm is not None and result_test.counts is not None:
        ws = torch.tensor(window_size, device=grid_coord.device, dtype=torch.int32)
        min_c = grid_coord.min(dim=0).values.int()
        voxel_coords = (grid_coord.int() - min_c) // ws
        sorted_voxel_coords = voxel_coords[result_test.perm]
        offset = 0
        for count in result_test.counts.cpu().tolist():
            group = sorted_voxel_coords[offset:offset + count]
            # All voxels in group should have same voxel_coord
            assert (group == group[0]).all(), \
                f"Group not homogeneous at offset {offset}: {group}"
            offset += count


class TestCountingSortEncode:
    """Tests for encoding_method='counting_sort'."""

    def test_basic_correctness(self):
        """counting_sort produces same grouping as ravel."""
        Ns = [500, 300, 200]
        grid_coord, batch_offsets = _make_batched_coords(Ns)
        ws = (8, 8, 8)

        ref = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                           return_perm=True, return_inverse=True, return_counts=True,
                           encoding_method="ravel")
        test = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                            return_perm=True, return_inverse=True, return_counts=True,
                            encoding_method="counting_sort")

        _check_grouping_equivalent(ref, test, grid_coord, ws)

    def test_matches_ravel_fast(self):
        """counting_sort produces same grouping as ravel_fast."""
        Ns = [400, 600]
        grid_coord, batch_offsets = _make_batched_coords(Ns, coord_range=128)
        ws = (16, 16, 16)

        ref = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                           return_perm=True, return_inverse=True, return_counts=True,
                           encoding_method="ravel_fast")
        test = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                            return_perm=True, return_inverse=True, return_counts=True,
                            encoding_method="counting_sort")

        _check_grouping_equivalent(ref, test, grid_coord, ws)

    def test_perm_inverse_consistency(self):
        """perm and inverse_perm are consistent inverses."""
        Ns = [1000]
        grid_coord, batch_offsets = _make_batched_coords(Ns)
        ws = (4, 4, 4)

        result = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                              return_perm=True, return_inverse=True, return_counts=True,
                              encoding_method="counting_sort")

        N = grid_coord.shape[0]
        device = grid_coord.device
        # perm[inverse_perm[i]] == i
        assert torch.equal(result.perm[result.inverse_perm],
                           torch.arange(N, device=device))
        # inverse_perm[perm[i]] == i
        assert torch.equal(result.inverse_perm[result.perm],
                           torch.arange(N, device=device))

    def test_counts_sum_to_n(self):
        """Sum of counts must equal total number of voxels."""
        Ns = [300, 400, 300]
        grid_coord, batch_offsets = _make_batched_coords(Ns)
        ws = (8, 8, 8)

        result = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                              return_perm=True, return_inverse=True, return_counts=True,
                              encoding_method="counting_sort")

        assert result.counts.sum().item() == sum(Ns)

    def test_asymmetric_window_size(self):
        """Non-cubic window sizes work correctly."""
        Ns = [500, 500]
        grid_coord, batch_offsets = _make_batched_coords(Ns, coord_range=64)
        ws = (4, 8, 16)

        ref = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                           return_perm=True, return_inverse=True, return_counts=True,
                           encoding_method="ravel_fast")
        test = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                            return_perm=True, return_inverse=True, return_counts=True,
                            encoding_method="counting_sort")

        _check_grouping_equivalent(ref, test, grid_coord, ws)

    def test_integer_window_size(self):
        """Single integer window size is broadcast to all dims."""
        Ns = [300, 200]
        grid_coord, batch_offsets = _make_batched_coords(Ns)

        result = voxel_encode(grid_coord, batch_offsets, window_size=8,
                              return_perm=True, return_inverse=True, return_counts=True,
                              encoding_method="counting_sort")

        assert result.counts.sum().item() == sum(Ns)
        _check_grouping_equivalent(
            voxel_encode(grid_coord, batch_offsets, window_size=8,
                         return_perm=True, return_inverse=True, return_counts=True,
                         encoding_method="ravel"),
            result, grid_coord, (8, 8, 8))

    def test_single_batch(self):
        """Works without batch_offsets (single batch)."""
        torch.manual_seed(42)
        grid_coord = torch.randint(0, 32, (500, 3), device="cuda")
        ws = (8, 8, 8)

        result = voxel_encode(grid_coord, batch_offsets=None, window_size=ws,
                              return_perm=True, return_inverse=True, return_counts=True,
                              encoding_method="counting_sort")

        assert result.counts.sum().item() == 500
        N = grid_coord.shape[0]
        assert torch.equal(result.perm[result.inverse_perm],
                           torch.arange(N, device="cuda"))

    def test_coord_offset(self):
        """Coordinate offset shifts window boundaries."""
        Ns = [500, 500]
        grid_coord, batch_offsets = _make_batched_coords(Ns)
        ws = (8, 8, 8)

        result_zero = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                                   coord_offset=(0.0, 0.0, 0.0),
                                   return_perm=True, return_counts=True,
                                   encoding_method="counting_sort")
        result_half = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                                   coord_offset=(0.5, 0.5, 0.5),
                                   return_perm=True, return_counts=True,
                                   encoding_method="counting_sort")

        # Different offsets generally produce different groupings
        # but both must have counts summing to N
        assert result_zero.counts.sum().item() == 1000
        assert result_half.counts.sum().item() == 1000

    def test_batch_isolation(self):
        """Voxels from different batches never share a window."""
        Ns = [200, 300, 500]
        grid_coord, batch_offsets = _make_batched_coords(Ns, coord_range=16)
        ws = (8, 8, 8)

        result = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                              return_perm=True, return_counts=True,
                              encoding_method="counting_sort")

        # Check that each group only contains voxels from one batch
        offset = 0
        for count in result.counts.cpu().tolist():
            group_indices = result.perm[offset:offset + count]
            # Determine which batch each index belongs to
            batch_ids = set()
            for idx in group_indices.cpu().tolist():
                for b in range(len(Ns)):
                    if batch_offsets[b].item() <= idx < batch_offsets[b + 1].item():
                        batch_ids.add(b)
                        break
            assert len(batch_ids) == 1, \
                f"Group at offset {offset} spans batches {batch_ids}"
            offset += count

    def test_many_batches(self):
        """Works with many batches (B=16)."""
        Ns = [100] * 16
        grid_coord, batch_offsets = _make_batched_coords(Ns, coord_range=32)
        ws = (8, 8, 8)

        ref = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                           return_perm=True, return_inverse=True, return_counts=True,
                           encoding_method="ravel_fast")
        test = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                            return_perm=True, return_inverse=True, return_counts=True,
                            encoding_method="counting_sort")

        _check_grouping_equivalent(ref, test, grid_coord, ws)

    def test_large_voxel_count(self):
        """Stress test with 100k+ voxels."""
        Ns = [50000, 50000]
        grid_coord, batch_offsets = _make_batched_coords(Ns, coord_range=256)
        ws = (8, 8, 8)

        result = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                              return_perm=True, return_inverse=True, return_counts=True,
                              encoding_method="counting_sort")

        assert result.counts.sum().item() == sum(Ns)
        N = grid_coord.shape[0]
        assert torch.equal(result.perm[result.inverse_perm],
                           torch.arange(N, device="cuda"))

    def test_codes_only(self):
        """Returns just codes when no perm/inverse/counts requested."""
        Ns = [200, 300]
        grid_coord, batch_offsets = _make_batched_coords(Ns)

        result = voxel_encode(grid_coord, batch_offsets, window_size=8,
                              encoding_method="counting_sort")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (500,)

    def test_cpu_batch_offsets(self):
        """batch_offsets on CPU still works (should be moved to device)."""
        Ns = [300, 200]
        torch.manual_seed(42)
        grid_coord = torch.randint(0, 32, (500, 3), device="cuda")
        batch_offsets = torch.tensor([0, 300, 500], dtype=torch.int32)  # CPU

        result = voxel_encode(grid_coord, batch_offsets, window_size=8,
                              return_perm=True, return_inverse=True, return_counts=True,
                              encoding_method="counting_sort")

        assert result.counts.sum().item() == 500

    def test_negative_coordinates(self):
        """Handles negative grid coordinates correctly."""
        torch.manual_seed(42)
        grid_coord = torch.randint(-50, 50, (500, 3), device="cuda")
        batch_offsets = torch.tensor([0, 500], dtype=torch.int32)
        ws = (8, 8, 8)

        ref = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                           return_perm=True, return_inverse=True, return_counts=True,
                           encoding_method="ravel_fast")
        test = voxel_encode(grid_coord, batch_offsets, window_size=ws,
                            return_perm=True, return_inverse=True, return_counts=True,
                            encoding_method="counting_sort")

        _check_grouping_equivalent(ref, test, grid_coord, ws)


class TestCountingSortBenchmark:
    """Benchmarks comparing counting_sort vs ravel_fast."""

    @pytest.mark.parametrize("N,coord_range,ws", [
        (10000, 64, 8),
        (50000, 128, 8),
        (100000, 256, 8),
    ])
    def test_benchmark(self, N, coord_range, ws):
        """Benchmark counting_sort vs ravel_fast (prints timing, always passes)."""
        torch.manual_seed(42)
        B = 4
        Ns = [N // B] * B
        grid_coord, batch_offsets = _make_batched_coords(Ns, coord_range=coord_range)

        # Warmup
        for method in ["ravel_fast", "counting_sort"]:
            for _ in range(3):
                voxel_encode(grid_coord, batch_offsets, window_size=ws,
                             return_perm=True, return_inverse=True, return_counts=True,
                             encoding_method=method)
        torch.cuda.synchronize()

        import time

        results = {}
        for method in ["ravel_fast", "counting_sort"]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            iters = 20
            for _ in range(iters):
                voxel_encode(grid_coord, batch_offsets, window_size=ws,
                             return_perm=True, return_inverse=True, return_counts=True,
                             encoding_method=method)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / iters * 1000
            results[method] = elapsed

        speedup = results["ravel_fast"] / results["counting_sort"]
        print(f"\n  N={N}, range={coord_range}, ws={ws}: "
              f"ravel_fast={results['ravel_fast']:.2f}ms, "
              f"counting_sort={results['counting_sort']:.2f}ms, "
              f"speedup={speedup:.2f}x")
