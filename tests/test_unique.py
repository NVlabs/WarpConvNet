# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.utils.argsort import argsort
from warpconvnet.utils.unique import ToUnique, unique_torch


@pytest.fixture
def setup_data():
    """Setup test data."""
    torch.manual_seed(0)
    return None


def test_to_unique():
    """Test ToUnique functionality."""
    x = torch.randint(0, 5, (10,))
    to_unique = ToUnique()

    unique, to_orig_indices, to_csr_indices, to_csr_offsets, _ = unique_torch(x)

    assert torch.allclose(x, unique[to_orig_indices])
    assert torch.allclose(torch.sort(x[to_csr_indices]).values, x[to_csr_indices])

    unique = to_unique.to_unique(x)
    orig_x = to_unique.to_original(unique)

    assert torch.allclose(x, orig_x)


@pytest.mark.benchmark(group="unique")
@pytest.mark.parametrize("backend", ["torch", "ravel", "morton"])
def test_unique(setup_data, benchmark, backend):
    """Test unique with different backends."""
    x = torch.randint(0, 1000, (1000000, 3), device="cuda")
    to_unique = ToUnique(unique_method=backend)

    result = benchmark.pedantic(
        lambda: to_unique.to_unique(x),
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )
