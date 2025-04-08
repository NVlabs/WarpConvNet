# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
from jaxtyping import Int

import torch
from torch import Tensor


def random_sample_per_batch(
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    num_samples: int,
) -> Tuple[Int[Tensor, "M"], Int[Tensor, "B+1"]]:  # noqa: F821
    """
    Randomly downsample the coordinates to the specified number of points
    """
    num_points = offsets.diff()
    batch_size = len(num_points)
    # sample sample_points per batch. BxN
    sampled_indices = torch.floor(torch.rand(batch_size, num_samples) * num_points.view(-1, 1)).to(
        torch.int32
    )
    # Add offsets
    sampled_indices = sampled_indices + offsets[:-1].view(-1, 1)
    sampled_indices = sampled_indices.view(-1)
    # Create new offsets
    new_offsets = torch.arange(batch_size + 1) * num_samples
    return sampled_indices, new_offsets
