# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import torch
from jaxtyping import Int


def ravel_multi_index(
    multi_index: Int[torch.Tensor, "* D"],  # noqa: F821
    spatial_shape: Tuple[int, ...],  # noqa: F821
) -> Int[torch.Tensor, "*"]:
    """
    Converts a tuple of index arrays into an array of flat indices.

    Args:
        multi_index: A tensor of coordinate vectors, (*, D).
        dims: The source shape.
    """
    # assert multi index is integer dtype
    assert multi_index.dtype in [torch.int16, torch.int32, torch.int64]
    assert multi_index.shape[-1] == len(spatial_shape)
    # Convert dims to a list of tuples
    if isinstance(spatial_shape, torch.Tensor):
        spatial_shape = tuple(spatial_shape.cpu().tolist())
    strides = torch.tensor(
        [np.prod(spatial_shape[i + 1 :]) for i in range(len(spatial_shape))], dtype=torch.int64
    ).to(multi_index.device)
    return (multi_index * strides).sum(dim=-1)


def ravel_multi_index_auto_shape(
    x: Int[torch.Tensor, "* D"],  # noqa: F821
    dim: int = 0,
) -> Int[torch.Tensor, "*"]:
    min_coords = x.min(dim=dim).values
    shifted_x = x - min_coords
    shape = shifted_x.max(dim=dim).values + 1
    raveled_x = ravel_multi_index(shifted_x, tuple(shape.cpu().tolist()))
    return raveled_x
