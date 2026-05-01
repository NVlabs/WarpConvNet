# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for PackedHashTable128 (D=7, CoordBits=17)."""
import pytest
import torch

from warpconvnet.geometry.coords.search.packed128_hashmap import PackedHashTable128


@pytest.fixture(autouse=True)
def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def _enable_debug_hash(monkeypatch):
    monkeypatch.setenv("WARPCONVNET_DEBUG_HASH", "1")


def _rand_unique_coords(
    n: int, dim: int = 7, lo: int = -1000, hi: int = 1000, seed: int = 0
) -> torch.Tensor:
    """Random coords, deduplicated. Returns possibly fewer than n rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    coords = torch.randint(
        lo,
        hi,
        (n, dim),
        dtype=torch.int32,
        device="cuda",
        generator=g,
    )
    return torch.unique(coords, dim=0)


def test_insert_search_roundtrip_small():
    coords = _rand_unique_coords(n=100)
    ht = PackedHashTable128.from_keys(coords)
    results = ht.search(coords)
    expected = torch.arange(coords.shape[0], dtype=torch.int32, device="cuda")
    assert torch.equal(results, expected)


def test_insert_search_roundtrip_large():
    coords = _rand_unique_coords(n=100_000, lo=-10000, hi=10000, seed=1)
    ht = PackedHashTable128.from_keys(coords)
    results = ht.search(coords)
    expected = torch.arange(coords.shape[0], dtype=torch.int32, device="cuda")
    assert torch.equal(results, expected)


def test_search_miss_returns_minus_one():
    coords = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32, device="cuda")
    ht = PackedHashTable128.from_keys(coords)
    miss = torch.tensor([[10, 20, 30, 40, 50, 60, 70]], dtype=torch.int32, device="cuda")
    r = ht.search(miss)
    assert r.item() == -1


def test_partial_hits_and_misses():
    coords = _rand_unique_coords(n=1000, seed=2)
    ht = PackedHashTable128.from_keys(coords)
    half = coords.shape[0] // 2
    miss = torch.full(
        (half, 7),
        -60000,
        dtype=torch.int32,
        device="cuda",
    )
    miss[:, 0] = miss[:, 0] + torch.arange(half, dtype=torch.int32, device="cuda")

    queries = torch.cat([coords[:half], miss], dim=0)
    r = ht.search(queries)
    assert torch.equal(
        r[:half],
        torch.arange(half, dtype=torch.int32, device="cuda"),
    )
    assert (r[half:] == -1).all()


def test_overflow_raises():
    n = 4
    coords = torch.zeros((n, 7), dtype=torch.int32, device="cuda")
    coords[:, 0] = torch.arange(n, dtype=torch.int32)
    ht = PackedHashTable128(capacity=4)
    with pytest.raises(AssertionError):
        ht.insert(coords)


def test_insert_within_capacity_succeeds():
    coords = torch.zeros((3, 7), dtype=torch.int32, device="cuda")
    coords[:, 0] = torch.arange(3, dtype=torch.int32)
    ht = PackedHashTable128(capacity=8)
    ht.insert(coords)
    r = ht.search(coords)
    assert (r == torch.arange(3, dtype=torch.int32, device="cuda")).all()


def test_range_check_low(_enable_debug_hash):
    bad = torch.tensor([[-100000, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        PackedHashTable128.from_keys(bad)


def test_range_check_high(_enable_debug_hash):
    bad = torch.tensor([[100000, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        PackedHashTable128.from_keys(bad)


def test_range_check_disabled_by_default():
    """Out-of-range coords silently truncate when WARPCONVNET_DEBUG_HASH unset."""
    # Within int32, beyond CoordBits range; pack will truncate, search should
    # round-trip the truncated form. This documents the current behavior.
    bad = torch.tensor([[100000, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
    ht = PackedHashTable128.from_keys(bad)  # no raise
    r = ht.search(bad)
    assert r.item() == 0


def test_range_check_exact_bounds():
    coords = torch.tensor(
        [
            [PackedHashTable128.COORD_MIN, 0, 0, 0, 0, 0, 0],
            [PackedHashTable128.COORD_MAX, 0, 0, 0, 0, 0, 0],
            [0, PackedHashTable128.COORD_MIN, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, PackedHashTable128.COORD_MAX],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    ht = PackedHashTable128.from_keys(coords)
    r = ht.search(coords)
    assert (r == torch.arange(4, dtype=torch.int32, device="cuda")).all()


def test_negative_and_positive_coexist():
    coords = torch.tensor(
        [
            [-1, -2, -3, -4, -5, -6, -7],
            [1, 2, 3, 4, 5, 6, 7],
            [-1, 2, -3, 4, -5, 6, -7],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    ht = PackedHashTable128.from_keys(coords)
    r = ht.search(coords)
    assert (r == torch.arange(4, dtype=torch.int32, device="cuda")).all()
    assert r[0].item() != r[1].item()
    assert r[2].item() != r[3].item()


def test_empty_input():
    coords = torch.zeros((0, 7), dtype=torch.int32, device="cuda")
    ht = PackedHashTable128.from_keys(coords)
    queries = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32, device="cuda")
    r = ht.search(queries)
    assert r.item() == -1


# ============================================================================
# batched_search tests
# ============================================================================


def test_batched_search_matches_loop():
    """Batched search must be bit-exact with K sequential search() calls."""
    g = torch.Generator(device="cuda").manual_seed(7)
    coords = torch.randint(
        -500,
        500,
        (5000, 7),
        dtype=torch.int32,
        device="cuda",
        generator=g,
    )
    coords = torch.unique(coords, dim=0)
    ht = PackedHashTable128.from_keys(coords)

    queries = coords[:1000].contiguous()
    offsets = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 5000, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    M = queries.shape[0]
    K = offsets.shape[0]

    batched = ht.batched_search(queries, offsets)
    assert batched.shape == (K, M)
    assert batched.dtype == torch.int32

    expected = torch.empty((K, M), dtype=torch.int32, device="cuda")
    for k in range(K):
        expected[k] = ht.search(queries + offsets[k])

    assert torch.equal(batched, expected)


def test_batched_search_perm_d6_blur_pattern():
    """K=14 axis-aligned offsets, the permutohedral d=6 blur shape.

    Random 7-D coords are too sparse for unit-step neighbours to land on
    other existing coords; we don't assert any hits. Correctness is the
    bit-exact-vs-loop check.
    """
    g = torch.Generator(device="cuda").manual_seed(11)
    coords = torch.randint(
        -200,
        200,
        (50_000, 7),
        dtype=torch.int32,
        device="cuda",
        generator=g,
    )
    coords = torch.unique(coords, dim=0)
    M = coords.shape[0]
    ht = PackedHashTable128.from_keys(coords)

    eye = torch.eye(7, dtype=torch.int32, device="cuda")
    offsets = torch.cat([eye, -eye], dim=0)

    batched = ht.batched_search(coords, offsets)
    assert batched.shape == (14, M)

    for k in range(14):
        assert torch.equal(batched[k], ht.search(coords + offsets[k]))


def test_batched_search_all_misses():
    coords = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32, device="cuda")
    ht = PackedHashTable128.from_keys(coords)
    queries = torch.tensor([[10000, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device="cuda")
    offsets = torch.tensor(
        [[0, 1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    r = ht.batched_search(queries, offsets)
    assert r.shape == (2, 1)
    assert (r == -1).all()


def test_batched_search_empty_queries():
    coords = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32, device="cuda")
    ht = PackedHashTable128.from_keys(coords)
    queries = torch.zeros((0, 7), dtype=torch.int32, device="cuda")
    offsets = torch.zeros((3, 7), dtype=torch.int32, device="cuda")
    r = ht.batched_search(queries, offsets)
    assert r.shape == (3, 0)


def test_batched_search_K_out_of_range():
    coords = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32, device="cuda")
    ht = PackedHashTable128.from_keys(coords)
    queries = coords
    bad_offsets = torch.zeros((33, 7), dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError):
        ht.batched_search(queries, bad_offsets)


def test_batched_search_K1():
    """Edge case: K=1 reduces to a single-pass search."""
    coords = torch.randint(
        -50,
        50,
        (200, 7),
        dtype=torch.int32,
        device="cuda",
    )
    coords = torch.unique(coords, dim=0)
    ht = PackedHashTable128.from_keys(coords)
    offsets = torch.zeros((1, 7), dtype=torch.int32, device="cuda")
    r = ht.batched_search(coords, offsets)
    assert r.shape == (1, coords.shape[0])
    assert torch.equal(
        r[0],
        torch.arange(coords.shape[0], dtype=torch.int32, device="cuda"),
    )
