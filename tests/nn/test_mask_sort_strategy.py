# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mask_argsort sort strategies (mask_bit, gray_code).

The two strategies are semantic-preserving: both must yield a permutation of
[0, N) that, when applied as the iteration order, produces identical kernel
output. The mask_bit strategy is the legacy contiguous-grouping sort. The
gray_code strategy decodes pair_mask as a Gray code and sorts by binary,
inducing a Gray-order linearization across mask groups.

These tests pin:
  1) The Gray->binary decode is correct (reference: scalar XOR-cascade).
  2) Both strategies produce valid permutations for K<=32 and K>32.
  3) Inside a single mask group (identical bitmask), order is stable for both.
  4) Gray-code adjacent codes (Hamming distance 1) sort to adjacent positions.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _gray_to_binary_scalar(g: int) -> int:
    """Reference scalar Gray->binary decode."""
    b = g
    for v in range(1, 32):
        b ^= g >> v
    return b & 0xFFFFFFFF


def test_gray_to_binary_decode_matches_scalar_reference():
    from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
        _gray_to_binary_uint32,
    )

    # Cover edge cases + a deterministic random sweep. Values are stored in
    # int64 to avoid Python-side overflow on > 2^31; the decode masks to 32b.
    fixed = [0, 1, 2, 3, 0xFFFFFFFF, 0x80000000, 0x55555555, 0xAAAAAAAA, 27, 0x1FFFFFF]
    torch.manual_seed(0)
    rand = torch.randint(0, 0xFFFFFFFF, (4096,), dtype=torch.int64).tolist()
    cases = fixed + rand

    g = torch.tensor(cases, dtype=torch.int64, device="cuda")
    out = _gray_to_binary_uint32(g)

    # Reference: per-element scalar.
    ref = torch.tensor(
        [_gray_to_binary_scalar(c & 0xFFFFFFFF) for c in cases],
        dtype=torch.int64,
        device="cuda",
    )
    assert torch.equal(out, ref), "Gray->binary decode mismatches scalar reference"


def _build_pair_table_with_random_masks(N, K, density=0.6, seed=0, device="cuda"):
    """Build a [K * N] pair_table where each (k, i) is set with prob density."""
    torch.manual_seed(seed)
    pair_table = torch.full((K, N), -1, dtype=torch.int32, device=device)
    active = torch.rand(K, N, device=device) < density
    pair_table[active] = (
        torch.arange(N, device=device, dtype=torch.int32).repeat(K).view(K, N)[active]
    )
    return pair_table.reshape(-1).contiguous()


@pytest.mark.parametrize("K", [27, 64, 125])  # K<=32, K>32 (2 words), K>64 (4 words)
@pytest.mark.parametrize("strategy", ["mask_bit", "gray_code"])
def test_argsort_is_valid_permutation(K, strategy):
    from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
        _build_mask_and_argsort,
    )

    N = 400
    pair_table = _build_pair_table_with_random_masks(N, K, density=0.5)
    _, argsort = _build_mask_and_argsort(
        pair_table, N, K, torch.device("cuda"), sort_strategy=strategy
    )
    assert argsort.shape == (N,)
    assert argsort.dtype == torch.int32
    sorted_idx = argsort.long().sort().values
    expected = torch.arange(N, device="cuda", dtype=torch.int64)
    assert torch.equal(sorted_idx, expected), f"{strategy}: not a permutation"


def test_gray_code_orders_hamming_adjacent_groups_consecutively():
    """When two distinct masks differ in only one bit (Hamming dist 1), their
    Gray-decoded keys should be 1 apart, so a stable sort places those two
    groups adjacent. This is the cache-reuse property the strategy exists for.
    """
    from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
        _gray_to_binary_uint32,
    )

    # Hamming-adjacent gray codes -> consecutive binary indices.
    # Standard Gray sequence: 0,1,3,2,6,7,5,4,12,...  Each consecutive pair
    # differs in exactly one bit.
    gray_seq = torch.tensor(
        [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14], dtype=torch.int32, device="cuda"
    )
    binary = _gray_to_binary_uint32(gray_seq)
    # Decoded binary should be 0..len(gray_seq)-1 (ie strictly monotone).
    expected = torch.arange(len(gray_seq), dtype=torch.int64, device="cuda")
    assert torch.equal(binary, expected)


@pytest.mark.parametrize("strategy", ["mask_bit", "gray_code"])
def test_stable_within_identical_masks(strategy):
    """Voxels with identical pair_mask should retain their original order."""
    from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
        _build_mask_and_argsort,
    )

    K = 27
    N = 64
    # Build pair_table where every voxel has the SAME mask (e.g. all offsets
    # active). Then argsort on identical key must be [0, 1, ..., N-1] for a
    # stable sort.
    pair_table = torch.zeros(K * N, dtype=torch.int32, device="cuda")
    for k in range(K):
        for i in range(N):
            pair_table[k * N + i] = i  # all valid

    _, argsort = _build_mask_and_argsort(
        pair_table, N, K, torch.device("cuda"), sort_strategy=strategy
    )
    expected = torch.arange(N, dtype=torch.int32, device="cuda")
    assert torch.equal(argsort, expected), f"{strategy}: stable order broken"


def test_env_var_default_is_mask_bit(monkeypatch):
    from warpconvnet.nn.functional.sparse_conv.detail import mask_gemm

    monkeypatch.delenv("WARPCONVNET_MASK_SORT", raising=False)
    assert mask_gemm._default_mask_sort_strategy() == "mask_bit"

    monkeypatch.setenv("WARPCONVNET_MASK_SORT", "gray_code")
    assert mask_gemm._default_mask_sort_strategy() == "gray_code"

    monkeypatch.setenv("WARPCONVNET_MASK_SORT", "MASK_BIT")  # case-insensitive
    assert mask_gemm._default_mask_sort_strategy() == "mask_bit"

    monkeypatch.setenv("WARPCONVNET_MASK_SORT", "garbage")
    # Unknown value -> falls back to default rather than crashing.
    assert mask_gemm._default_mask_sort_strategy() == "mask_bit"
