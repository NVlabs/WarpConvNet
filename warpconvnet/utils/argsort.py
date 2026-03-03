# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from jaxtyping import Int
from torch import Tensor


def argsort(
    data: Int[Tensor, "N"],  # noqa: F821
) -> Int[Tensor, "N"]:  # noqa: F821
    """
    Sorts the input data and returns the indices that would sort the data.

    Args:
        data: The input data to be sorted
    """
    return torch.argsort(data)
