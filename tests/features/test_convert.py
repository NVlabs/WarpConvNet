# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from warpconvnet.geometry.features.ops.convert import cat_to_pad_tensor, cat_to_pad, pad_to_cat
from warpconvnet.geometry.features.pad import PadFeatures
from warpconvnet.geometry.features.cat import CatFeatures


@pytest.fixture
def setup_batch_data():
    """Setup batch data with random features."""
    torch.manual_seed(0)
    device = "cuda:0"
    return device


@pytest.mark.benchmark(group="batch_copy")
@pytest.mark.parametrize(
    "batch_config",
    [
        {"B": 48, "min_N": 1_000, "max_N": 2_000, "name": "small"},
        {"B": 12, "min_N": 10_000, "max_N": 20_000, "name": "medium"},
        {"B": 4, "min_N": 100_000, "max_N": 1_000_000, "name": "large"},
    ],
)
@pytest.mark.parametrize("num_channels", [16, 32, 64, 128, 256])
class TestBatchCopy:
    def test_torch_backend(
        self, benchmark: BenchmarkFixture, setup_batch_data, batch_config, num_channels
    ):
        """Test torch backend performance."""
        device = setup_batch_data
        B, min_N, max_N = batch_config["B"], batch_config["min_N"], batch_config["max_N"]

        # Setup data
        Ns = torch.randint(min_N, max_N, (B,))
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(Ns), dim=0)],
            dim=0,
        ).int()
        features = torch.cat([torch.rand((N, num_channels)) for N in Ns], dim=0).to(device)

        def run_benchmark():
            out_features = cat_to_pad_tensor(features, offsets)
            torch.cuda.synchronize()
            return out_features

        benchmark.pedantic(
            run_benchmark,
            iterations=10,
            rounds=3,
            warmup_rounds=1,
        )


def test_cat_to_pad(setup_batch_data):
    """Test correctness of cat to pad conversion."""
    device = setup_batch_data
    B, min_N, max_N, C = 4, 1000, 2000, 16

    # Setup data
    Ns = torch.randint(min_N, max_N, (B,))
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(Ns), dim=0)], dim=0
    ).int()
    features = torch.cat([torch.rand((N, C)) for N in Ns], dim=0).to(device)

    # Test torch backend
    torch_out = cat_to_pad_tensor(features, offsets)

    # Verify output shape
    max_N_actual = Ns.max().item()
    assert torch_out.shape == (B, max_N_actual, C)

    # Verify values for each batch
    for b in range(B):
        assert torch.allclose(
            torch_out[b, : Ns[b]],
            features[offsets[b] : offsets[b + 1]],
            rtol=1e-5,
            atol=1e-5,
        )


def test_pad_to_cat(setup_batch_data):
    """Test correctness of pad to cat conversion."""
    device = setup_batch_data
    B, N, C = 4, 1000, 16

    # Create padded tensor
    padded = torch.rand(B, N, C).to(device)
    offsets = torch.tensor([0, N // 2, 3 * N // 4, 7 * N // 8, N], dtype=torch.int32)
    diff = offsets.diff()

    # Create PadFeatures object
    pad_features = PadFeatures(padded, offsets)

    # Convert to CatFeatures and back
    cat_features = pad_to_cat(pad_features)
    pad_features_converted = cat_to_pad(cat_features)

    # Check shapes
    assert cat_features.batched_tensor.shape[0] == offsets[-1]
    assert tuple(pad_features_converted.batched_tensor.shape) == (B, diff.max(), C)

    # Check values in valid regions
    for i in range(B):
        start, end = offsets[i], offsets[i + 1]
        assert torch.allclose(
            padded[i, : end - start],
            pad_features_converted.batched_tensor[i, : end - start],
            rtol=1e-5,
            atol=1e-5,
        )
