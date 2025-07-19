# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def get_freqs(
    num_freqs: int, data_range: float = 2.0, device: Optional[torch.device] = None
) -> Float[Tensor, "num_freqs"]:  # noqa: F821
    """Generate logarithmically spaced frequencies used in positional encoding.

    Args:
        num_freqs: Number of frequency bands to generate.
        data_range: Range of the input data that the encoding will span.
        device: Device to create the frequency tensor on.

    Returns:
        1‑D tensor containing ``num_freqs`` frequencies.
    """
    if device is None:
        device = torch.device("cpu")
    freqs = 2 ** torch.arange(start=0, end=num_freqs, device=device)
    freqs = (2 * np.pi / data_range) * freqs
    return freqs


def sinusoidal_encoding(
    x: Float[Tensor, "... D"],
    num_channels: Optional[int] = None,
    data_range: Optional[float] = None,
    encoding_axis: int = -1,
    freqs: Optional[Float[Tensor, "num_channels"]] = None,  # noqa: F821
    concat_input: bool = False,
) -> Float[Tensor, "... D*num_channels"]:
    """
    Apply sinusoidal encoding to the input tensor.

    Args:
        x: Input tensor of any shape.
        num_channels: Number of channels in the output per input channel.
        data_range: Range of the input data (max - min).
        encoding_axis: Axis to apply the encoding to. If None, the encoding is applied to the last axis.
        freqs: Frequencies to use for the sinusoidal encoding. If None, the frequencies are calculated from the data range and num_channels.
        concat_input: Whether to concatenate the input to the output.

    Returns:
        Tensor with sinusoidal encoding applied.
        For input shape [..., C], the output shape is [..., C*num_channels].
        If concat_input is True, the input is concatenated to the output.
    """
    assert encoding_axis == -1, "Only encoding_axis=-1 is supported at the moment"

    x = x.unsqueeze(encoding_axis)
    if freqs is None:
        assert (
            num_channels is not None and data_range is not None
        ), "num_channels and data_range must be provided if freqs are not given"
        assert (
            num_channels % 2 == 0
        ), f"num_channels must be even for sin/cos, got {num_channels}"
        freqs = get_freqs(num_channels // 2, data_range, device=x.device)
    freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)
    freqed_x = x * freqs
    if concat_input:
        return torch.cat(
            [freqed_x.cos(), freqed_x.sin(), x], dim=encoding_axis
        ).flatten(start_dim=-2)
    else:
        return torch.cat([freqed_x.cos(), freqed_x.sin()], dim=encoding_axis).flatten(
            start_dim=-2
        )
