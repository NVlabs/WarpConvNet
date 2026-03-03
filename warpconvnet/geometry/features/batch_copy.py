# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor


def copy_batch_torch(
    in_features: Float[Tensor, "N F"],
    row_splits: Int[Tensor, "B+1"],  # noqa: F821
    pad_multiple: Optional[int] = None,
) -> Float[Tensor, "B M F"]:
    num_points = row_splits.diff()
    device = in_features.device
    out_num_points = (
        num_points.max()
        if pad_multiple is None
        else ((num_points.max() + pad_multiple - 1) // pad_multiple) * pad_multiple
    )
    out_features = torch.zeros(
        (row_splits.shape[0] - 1, out_num_points, in_features.shape[1]),
        dtype=in_features.dtype,
        device=device,
    )
    for batch_idx in range(row_splits.shape[0] - 1):
        out_features[batch_idx, : num_points[batch_idx]] = in_features[
            row_splits[batch_idx] : row_splits[batch_idx + 1]
        ]
    return out_features
