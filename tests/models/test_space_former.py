# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.types.conversion.to_voxels import points_to_voxels
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.space_former import SpaCeFormer


@pytest.fixture
def voxels():
    torch.manual_seed(0)
    B, min_N, max_N, C = 3, 1000, 10000, 32
    Ns = [N.item() for N in torch.randint(min_N, max_N, (B,))]
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    pc = Points(coords, features).to("cuda:0")
    return points_to_voxels(pc, voxel_size=0.02), C


@pytest.mark.parametrize(
    "enc_attn_types,dec_attn_types",
    [
        ("sscs", "scs"),
        ("curve", "curve"),
        ("space", "space"),
        ("ccca", "ccs"),  # "all" at deepest enc level
        ("all", "all"),
    ],
)
@pytest.mark.parametrize("block_type", ["pre_norm", "post_norm", "stream_norm"])
@pytest.mark.parametrize("use_rope", [False, True])
def test_space_former_forward(voxels, enc_attn_types, dec_attn_types, block_type, use_rope):
    st, C = voxels
    model = SpaCeFormer(
        in_channels=C,
        enc_depths=(2, 2, 2, 2),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(32, 32, 1024, 1024),
        dec_depths=(2, 2, 2),
        dec_channels=(32, 64, 128),
        dec_num_head=(2, 4, 8),
        dec_patch_size=(32, 32, 1024),
        enc_attn_types=enc_attn_types,
        dec_attn_types=dec_attn_types,
        block_type=block_type,
        use_rope=use_rope,
        out_channels=20,
    ).to("cuda:0")

    out = model(st)
    assert isinstance(out, Voxels)
    assert out.feature_tensor.shape[-1] == 20
