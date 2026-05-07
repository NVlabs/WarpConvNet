# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from warpconvnet.nn.modules.conv3d_blocks import DownsampleBlock3d, ResBlock3d, UpsampleBlock3d
from warpconvnet.nn.modules.sparse_unet import (
    SparseChannelToSpatialResBlock3d,
    SparseUNetDecoderStages,
)
from warpconvnet.nn.utils import manual_cast, str_to_dtype, zero_module


def test_nn_utils_dtype_and_zero_module():
    assert str_to_dtype("bf16") is torch.bfloat16
    x = torch.ones(2, dtype=torch.float32)
    assert manual_cast(x, torch.float16).dtype is torch.float16

    layer = nn.Linear(4, 4)
    zero_module(layer)
    assert torch.count_nonzero(layer.weight) == 0
    assert torch.count_nonzero(layer.bias) == 0


def test_dense_3d_blocks_shapes():
    x = torch.randn(2, 4, 8, 8, 8)
    res = ResBlock3d(4, 6)
    assert res(x).shape == (2, 6, 8, 8, 8)

    down = DownsampleBlock3d(6, 8)
    y = down(res(x))
    assert y.shape == (2, 8, 4, 4, 4)

    up = UpsampleBlock3d(8, 6)
    assert up(y).shape == (2, 6, 8, 8, 8)


def test_sparse_channel_to_spatial_res_block_state_dict_layout():
    block = SparseChannelToSpatialResBlock3d(channels=64, out_channels=32, pred_subdiv=True)
    keys = set(block.state_dict())
    assert {"norm1.weight", "norm1.bias"} <= keys
    assert {"conv1.weight", "conv2.weight", "to_subdiv.weight"} <= keys
    assert block.conv1.weight.shape == (27, 64, 32 * 8)


class _AddBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pred_subdiv: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(out_channels)))
        self.pred_subdiv = pred_subdiv

    def forward(self, x: torch.Tensor, subdiv: torch.Tensor | None = None):
        if subdiv is not None:
            return x + self.weight + subdiv
        out = x + self.weight
        if self.pred_subdiv:
            return out, self.weight.detach()
        return out


def test_sparse_unet_decoder_stage_assembly_state_keys_and_run():
    stages = SparseUNetDecoderStages(
        model_channels=[4, 8],
        num_blocks=[1, 1],
        block_type=["block", "block"],
        up_block_type=["up"],
        block_args=[{}, {}],
        block_registry={"block": _AddBlock, "up": _UpBlock},
        up_block_kwargs={"pred_subdiv": True},
    )

    assert {"0.0.weight", "0.1.weight", "1.0.weight"} <= set(stages.state_dict())
    out, subs = stages.run(torch.tensor(0.0), return_subs=True)
    assert out.item() == 20.0
    assert len(subs) == 1 and subs[0].item() == 8.0

    early = stages.run(torch.tensor(0.0), stop_before_stage=1)
    assert early.item() == 12.0

    guided = stages.run(torch.tensor(0.0), guide_subs=[torch.tensor(3.0)])
    assert guided.item() == 23.0
