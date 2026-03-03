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


@pytest.mark.benchmark(group="argsort")
def test_argsort(setup_data, benchmark):
    """Benchmark argsort with torch backend."""
    device = "cuda:0"
    N = 1000000
    rand_perm = torch.randperm(N, device=device).int()

    result = benchmark.pedantic(
        lambda: argsort(rand_perm),
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )
