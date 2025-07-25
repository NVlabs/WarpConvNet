# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import repeat
from typing import List, Tuple, Union, Any

import torch


def ntuple(x: Union[int, List[int], Tuple[int, ...], torch.Tensor], ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x


def _pad_tuple(x: Any, y: Any, number_of_outputs: int) -> Tuple[Any, ...]:
    """Pad a tuple with None values to the correct length."""
    assert number_of_outputs >= 2
    if number_of_outputs == 2:
        return x, y
    else:
        return_list = [None] * number_of_outputs
        if x is not None:
            return_list[0] = x
        if y is not None:
            return_list[1] = y
        return tuple(return_list)
