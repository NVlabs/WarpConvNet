# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from warpconvnet.nn.functional.encodings import get_freqs, sinusoidal_encoding


def normalize_coordinates(
    xyz: Float[Tensor, "N 3"],
    min_coord: Float[Tensor, "3"],
    max_coord: Float[Tensor, "3"],
) -> Float[Tensor, "N 3"]:
    """
    Normalize coordinates with min to 0 and max to 1

    xyz: N x 3
    min_max: [[3], [3]] - min and max XYZ coords
    """
    assert min_coord.shape[-1] == xyz.shape[-1]
    assert min_coord.shape == max_coord.shape

    diff = max_coord - min_coord
    normalized_xyz = (xyz - min_coord) / diff
    return normalized_xyz


class SinusoidalEncoding(nn.Module):
    def __init__(
        self, num_channels: int, data_range: float = 2.0, concat_input: bool = True
    ):
        """
        Initialize a sinusoidal encoding layer.

        Args:
            num_channels: Number of channels to encode. Must be even.
            data_range: The range of the data. For example, if the data is in the range [0, 1], then data_range=1.
            concat_input: Whether to concatenate the input to the output.
        """
        super().__init__()
        assert (
            num_channels % 2 == 0
        ), f"num_channels must be even for sin/cos, got {num_channels}"
        self.num_channels = num_channels
        self.concat_input = concat_input
        self.register_buffer("freqs", get_freqs(num_channels // 2, data_range))

    def num_output_channels(self, num_input_channels: int) -> int:
        if self.concat_input:
            return (num_input_channels + 1) * self.num_channels
        else:
            return num_input_channels * self.num_channels

    def forward(
        self, x: Float[Tensor, "* C"]
    ) -> Float[Tensor, "* num_channels*C"]:  # noqa: F821
        return sinusoidal_encoding(x, freqs=self.freqs, concat_input=self.concat_input)


class FourierEncoding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        std: float = 1.0,
        input_range: Optional[Tuple[Float[Tensor, "3"], Float[Tensor, "3"]]] = None,
        learnable: bool = False,
    ):
        super().__init__()
        assert out_channels % 2 == 0
        self.to_gaussian = nn.Linear(in_channels, out_channels // 2, bias=False)
        self.to_gaussian.weight.data.normal_(std=2 * np.pi * std)

        if not learnable:
            self.to_gaussian.requires_grad = False

        self.normalize = input_range is not None
        if input_range is not None:
            # Convert the input range to a tensor if it's not None
            if not isinstance(input_range, torch.Tensor):
                input_range = torch.tensor(input_range, dtype=torch.float32)
            self.register_buffer("input_range", input_range)

    def forward(self, xyz: Float[Tensor, "N 3"]):
        if self.normalize:
            xyz = normalize_coordinates(
                xyz, min_coord=self.input_range[0], max_coord=self.input_range[1]
            )

        xyz_proj = self.to_gaussian(xyz)
        if xyz_proj.is_nested:
            return torch.nested.nested_tensor(
                [torch.cat([x.sin(), x.cos()], dim=-1) for x in xyz_proj.unbind()]
            )
        else:
            return torch.cat([xyz_proj.sin(), xyz_proj.cos()], dim=-1)
