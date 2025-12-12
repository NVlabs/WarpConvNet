# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from warpconvnet import _C


def farthest_point_sampling(points: torch.Tensor, offsets: torch.Tensor, K: int) -> torch.Tensor:
    """
    Farthest Point Sampling for packed coordinates.

    Args:
        points: (N, 3) point cloud, packed
        offsets: (B+1) offsets
        K: number of samples to select per batch item

    Returns:
        idxs: (B*K,) global indices of selected points
    """
    if _C is None:
        raise ImportError("warpconvnet C++ extension is not available.")

    N, _ = points.shape
    B = offsets.shape[0] - 1

    # Initialize output tensor
    # We return B*K indices (flattened)
    idxs = torch.empty((B * K,), dtype=torch.int32, device=points.device)

    # Initialize min distances to infinity
    temp = torch.full((N,), float("inf"), dtype=points.dtype, device=points.device)

    offsets = offsets.to(torch.int32)

    _C.sampling.farthest_point_sampling(points, offsets, temp, idxs, K)

    return idxs
