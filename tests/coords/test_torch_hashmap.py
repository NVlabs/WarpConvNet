# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod


@pytest.fixture
def device():
    """Fixture for device configuration."""
    return "cuda:0"


@pytest.fixture
def sample_keys_torch(device):
    """Fixture providing sample vector keys as Torch tensors."""
    N = 1 << 16  # 65536
    return torch.randint(0, 10000, (N, 4), device=device, dtype=torch.int32)


@pytest.fixture
def sample_keys_torch_large(device):
    """Fixture providing larger sample vector keys for benchmark."""
    N = 1 << 22  # 4194304
    return torch.randint(0, 100000, (N, 4), device=device, dtype=torch.int32)


# --- TorchHashTable Tests ---


@pytest.mark.parametrize("hash_method", list(HashMethod))
def test_torch_hash_table_creation_and_search(device, sample_keys_torch, hash_method):
    """Test TorchHashTable creation, insertion, and search with various hash methods."""
    table = TorchHashTable.from_keys(sample_keys_torch, hash_method=hash_method, device=device)
    assert table.key_dim == sample_keys_torch.shape[1]

    # Search for existing keys
    results = table.search(sample_keys_torch)
    assert results.device == torch.device(device)
    assert (
        results.cpu().numpy() != -1
    ).all(), f"All existing keys should be found with {hash_method}"

    # Verify unique keys retrieval
    unique_indices = table.unique_index
    unique_keys_retrieved = table.unique_vector_keys

    # Basic check: number of unique keys should be <= total keys
    assert unique_keys_retrieved.shape[0] <= sample_keys_torch.shape[0]
    if unique_keys_retrieved.shape[0] > 0:
        assert unique_keys_retrieved.shape[1] == sample_keys_torch.shape[1]

    # Search for a subset of keys
    subset_keys = sample_keys_torch[: sample_keys_torch.shape[0] // 2]
    results_subset = table.search(subset_keys)
    assert (results_subset.cpu().numpy() != -1).all(), f"Subset search failed with {hash_method}"

    # Search for non-existent keys
    non_existent_keys = torch.randint(
        10001, 20000, sample_keys_torch.shape, device=device, dtype=torch.int32
    )
    results_non_existent = table.search(non_existent_keys)
    assert (
        results_non_existent.cpu().numpy() == -1
    ).all(), f"Non-existent key search failed with {hash_method}"


def test_torch_hash_table_serialization(device, sample_keys_torch):
    """Test TorchHashTable serialization and deserialization."""
    original_table = TorchHashTable.from_keys(
        sample_keys_torch, hash_method=HashMethod.CITY, device=device
    )
    data_dict = original_table.to_dict()

    new_table = TorchHashTable(capacity=1, device=device)  # Dummy capacity, will be overwritten
    new_table.from_dict(data_dict)

    assert new_table.capacity == original_table.capacity
    assert new_table.hash_method == original_table.hash_method
    assert new_table.key_dim == original_table.key_dim
    assert new_table.device == original_table.device

    # Verify search results match
    original_results = original_table.search(sample_keys_torch)
    loaded_results = new_table.search(sample_keys_torch)
    torch.testing.assert_close(
        original_results, loaded_results, msg="Search results should match after serialization"
    )


# --- Benchmark Tests for TorchHashTable ---


@pytest.mark.parametrize("hash_method", list(HashMethod))
def test_benchmark_torch_insert(benchmark, device, sample_keys_torch_large, hash_method):
    """Benchmark TorchHashTable insert operation (standard pytest-benchmark)."""
    keys = sample_keys_torch_large
    # Calculate capacity based on keys for a fair benchmark setup if from_keys isn't used directly
    capacity = max(16, int(keys.shape[0] * 2))
    # Setup happens once per round in standard benchmark
    table = TorchHashTable(capacity=capacity, hash_method=hash_method, device=device)

    # Benchmark the insert method directly.
    # NOTE: This benchmarks the *first* insert. Subsequent calls in other rounds
    # might hit already allocated memory if insert reuses internal buffers without reallocating.
    # For a true insert benchmark, setup might need table re-creation inside benchmark call,
    # which pytest-benchmark's pedantic mode handles better.
    # Example using setup: benchmark.pedantic(table.insert, args=(keys,), iterations=5, rounds=10)
    benchmark(table.insert, keys)


@pytest.mark.parametrize("hash_method", list(HashMethod))
def test_benchmark_torch_search_existing_min_of_k(
    benchmark, device, sample_keys_torch_large, hash_method
):
    """Benchmark TorchHashTable search (existing), reporting stats on the minimum of K=10 runs per round."""
    keys = sample_keys_torch_large
    table = TorchHashTable.from_keys(keys, hash_method=hash_method, device=device)
    k = 10  # Number of inner iterations to find the minimum from

    # Ensure setup (like JIT compilation of kernels) happens before timing loop
    _ = table.search(keys[:1])  # Warm-up run
    torch.cuda.synchronize(device)

    def run_search_k_times_and_return_min_time():
        times_sec = []
        for _ in range(k):
            # Use torch events for accurate GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            # --- The actual operation ---
            _result = table.search(keys)
            # --------------------------
            end_event.record()

            # Waits for GPU operations to complete to get accurate timing
            torch.cuda.synchronize(device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            times_sec.append(elapsed_time_ms / 1000.0)  # Convert ms to seconds

        # Return the minimum time observed in this batch of k runs
        # pytest-benchmark expects the return value to be the time in seconds
        return min(times_sec)

    # Benchmark the helper function. pytest-benchmark will run this helper
    # multiple times (rounds) and report statistics (min, max, mean, median...)
    # based on the *minimum times* returned by the helper in each round.
    benchmark(run_search_k_times_and_return_min_time)


@pytest.mark.parametrize("hash_method", list(HashMethod))
def test_benchmark_torch_search_non_existent_min_of_k(
    benchmark, device, sample_keys_torch_large, hash_method
):
    """Benchmark TorchHashTable search (non-existent), reporting stats on the minimum of K=10 runs per round."""
    keys = sample_keys_torch_large
    non_existent_keys = torch.randint(100001, 200000, keys.shape, device=device, dtype=torch.int32)
    table = TorchHashTable.from_keys(keys, hash_method=hash_method, device=device)
    k = 10  # Number of inner iterations

    # Warm-up
    _ = table.search(non_existent_keys[:1])
    torch.cuda.synchronize(device)

    def run_search_k_times_and_return_min_time():
        times_sec = []
        for _ in range(k):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            # --- The actual operation ---
            _result = table.search(non_existent_keys)
            # --------------------------
            end_event.record()

            torch.cuda.synchronize(device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            times_sec.append(elapsed_time_ms / 1000.0)

        return min(times_sec)

    benchmark(run_search_k_times_and_return_min_time)


# --- Warp-Cooperative Search Tests ---


@pytest.mark.parametrize("hash_method", list(HashMethod))
@pytest.mark.parametrize(
    "num_keys,key_dim",
    [
        (100, 4),  # Small table, 4D keys
        (100, 3),  # Small table, 3D keys
        (10000, 4),  # Medium table, 4D keys
        (10000, 3),  # Medium table, 3D keys
        (100000, 4),  # Large table, 4D keys
        (100000, 3),  # Large table, 3D keys
    ],
)
def test_warp_cooperative_search_matches_standard(device, hash_method, num_keys, key_dim):
    """Verify warp-cooperative search produces identical results to standard search."""
    keys = torch.randint(0, 50000, (num_keys, key_dim), device=device, dtype=torch.int32)
    table = TorchHashTable.from_keys(keys, hash_method=hash_method, device=device)

    # Search for existing keys with both methods
    results_standard = table.search(keys, cooperative=False)
    results_cooperative = table.search(keys, cooperative=True)
    torch.testing.assert_close(
        results_standard,
        results_cooperative,
        msg=f"Warp-cooperative search must match standard search for existing keys "
        f"(hash={hash_method}, N={num_keys}, dim={key_dim})",
    )

    # All existing keys should be found
    assert (results_cooperative.cpu() != -1).all(), (
        f"Warp-cooperative search failed to find existing keys "
        f"(hash={hash_method}, N={num_keys}, dim={key_dim})"
    )

    # Search for non-existent keys with both methods
    non_existent = torch.randint(
        50001, 100000, (num_keys, key_dim), device=device, dtype=torch.int32
    )
    results_standard_ne = table.search(non_existent, cooperative=False)
    results_cooperative_ne = table.search(non_existent, cooperative=True)
    torch.testing.assert_close(
        results_standard_ne,
        results_cooperative_ne,
        msg=f"Warp-cooperative search must match standard search for non-existent keys "
        f"(hash={hash_method}, N={num_keys}, dim={key_dim})",
    )

    # All non-existent keys should return -1
    assert (results_cooperative_ne.cpu() == -1).all(), (
        f"Warp-cooperative search returned false positives for non-existent keys "
        f"(hash={hash_method}, N={num_keys}, dim={key_dim})"
    )


@pytest.mark.parametrize("hash_method", list(HashMethod))
@pytest.mark.parametrize("load_factor", [0.25, 0.45, 0.50])
def test_warp_cooperative_search_load_factors(device, hash_method, load_factor):
    """Test warp-cooperative search at different hash table load factors."""
    # Use a fixed capacity and vary the number of keys to achieve the target load factor.
    capacity = 1 << 14  # 16384
    num_keys = int(capacity * load_factor)
    key_dim = 4
    keys = torch.randint(0, 50000, (num_keys, key_dim), device=device, dtype=torch.int32)
    table = TorchHashTable.from_keys(
        keys, hash_method=hash_method, device=device, capacity=capacity
    )

    results_standard = table.search(keys, cooperative=False)
    results_cooperative = table.search(keys, cooperative=True)
    torch.testing.assert_close(
        results_standard,
        results_cooperative,
        msg=f"Mismatch at load_factor={load_factor} with {hash_method}",
    )
    assert (
        results_cooperative.cpu() != -1
    ).all(), f"Warp-cooperative search missed keys at load_factor={load_factor} with {hash_method}"


@pytest.mark.parametrize("hash_method", list(HashMethod))
def test_warp_cooperative_search_mixed_keys(device, hash_method):
    """Test warp-cooperative search with a mix of existing and non-existing keys."""
    num_keys = 10000
    key_dim = 4
    keys = torch.randint(0, 50000, (num_keys, key_dim), device=device, dtype=torch.int32)
    table = TorchHashTable.from_keys(keys, hash_method=hash_method, device=device)

    # Create mixed search keys: half existing, half non-existing
    existing = keys[: num_keys // 2]
    non_existing = torch.randint(
        50001, 100000, (num_keys // 2, key_dim), device=device, dtype=torch.int32
    )
    mixed = torch.cat([existing, non_existing], dim=0)

    results_standard = table.search(mixed, cooperative=False)
    results_cooperative = table.search(mixed, cooperative=True)
    torch.testing.assert_close(
        results_standard,
        results_cooperative,
        msg=f"Warp-cooperative search must match standard for mixed keys with {hash_method}",
    )


if __name__ == "__main__":
    pytest.main([__file__])
