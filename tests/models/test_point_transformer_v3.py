# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.types.points import Points
from warpconvnet.models.point_transformer_v3 import PointTransformerV3
from warpconvnet.nn.modules.sparse_pool import PointToVoxel


@pytest.fixture
def pc(device: torch.device = torch.device("cuda:0")):
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = [N.item() for N in torch.randint(min_N, max_N, (B,))]
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features).to(device)


def test_point_transformer_v3(pc: Points):
    point_transformer = PointToVoxel(
        PointTransformerV3(
            in_channels=pc.feature_tensor.shape[-1],
            enc_depths=(3, 3, 3, 6, 3),
            enc_channels=(48, 96, 192, 384, 512),
            enc_num_head=(3, 6, 12, 24, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(3, 3, 3, 3),
            dec_channels=(48, 96, 192, 384),
            dec_num_head=(4, 6, 12, 24),
            dec_patch_size=(1024, 1024, 1024, 1024),
            shuffle_orders=True,
        ),
        voxel_size=0.02,
        reduction="mean",
        concat_unpooled_pc=False,
    ).to(pc.device)
    out = point_transformer(pc)
    assert isinstance(out, Points)
    assert out.feature_tensor.shape[-1] == 48
    assert len(out) == len(pc)
