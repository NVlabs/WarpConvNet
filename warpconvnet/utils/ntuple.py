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


def _pad_values(number_of_outputs: int, *values: Any) -> Tuple[Any, ...]:
    """Pad values with None to reach ``number_of_outputs`` length.

    The provided values are placed in order starting from index 0. Positions
    corresponding to missing inputs remain None.
    """
    assert number_of_outputs >= 1
    assert number_of_outputs >= len(
        values
    ), f"number_of_outputs ({number_of_outputs}) must be >= number of inputs ({len(values)})"

    if len(values) == number_of_outputs:
        return tuple(values)

    return_list = [None] * number_of_outputs
    for idx, value in enumerate(values):
        if value is not None:
            return_list[idx] = value
    return tuple(return_list)


def _pad_tuple(x: Any, y: Any, number_of_outputs: int) -> Tuple[Any, ...]:
    """Pad a tuple with None values to the correct length.

    Backwards-compatible helper that delegates to ``_pad_values``. Accepts two
    inputs (``x`` and ``y``). For vararg support, use ``_pad_values`` directly.
    """
    return _pad_values(number_of_outputs, x, y)
