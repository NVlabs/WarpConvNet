# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor


def random_sample(
    batch_offsets: Int[Tensor, "B+1"],  # noqa: F821
    num_samples_per_batch: int,
) -> Tuple[Int[Tensor, "BS"], Int[Tensor, "B+1"]]:  # noqa: F821
    """
    Randomly sample points from the batched coordinates.

    Args:
        batch_offsets: Int[Tensor, "B+1"],
        num_samples_per_batch: int

    Returns:
        sample_offsets: Int[Tensor, "B+1"],
        sampled_indices: Int[Tensor, "BS"],
    """
    # Sample indices
    num_points = batch_offsets.diff()
    B = len(num_points)
    rand_ratios = torch.rand((B, num_samples_per_batch), device=num_points.device)
    rand_indices = (num_points.view(B, 1) * rand_ratios).floor().int()
    rand_indices = rand_indices + batch_offsets[:-1].view(B, 1)
    indices = rand_indices.view(-1)
    sample_offsets = (
        torch.arange(B + 1, device=num_points.device) * num_samples_per_batch
    )
    return indices, sample_offsets
