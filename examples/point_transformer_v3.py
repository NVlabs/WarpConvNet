# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PointTransformerV3 model proposed in the paper:
"Point Transformer V3: Simpler, Faster, Stronger"
https://arxiv.org/abs/2312.10035
"""

from typing import Literal, Optional, Tuple

import pytest

import torch
import torch.nn as nn

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.activations import GELU, DropPath
from warpconvnet.nn.modules.attention import PatchAttention
from warpconvnet.nn.modules.base_module import BaseSpatialModel, BaseSpatialModule
from warpconvnet.nn.modules.mlp import Linear
from warpconvnet.nn.modules.normalizations import LayerNorm
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.sparse_pool import PointToSparseWrapper, SparseMaxPool, SparseUnpool


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.mlp = Sequential(
            nn.Linear(in_channels, hidden_channels),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.mlp(x)
        return x


class PatchAttentionBlock(BaseSpatialModule):
    def __init__(
        self,
        in_channels: int,
        attention_channels: int,
        patch_size: int,
        num_heads: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type = LayerNorm,
        act_layer: type = GELU,
        attn_type: Literal["patch"] = "patch",
        order: POINT_ORDERING = POINT_ORDERING.MORTON_XYZ,
    ):
        super().__init__()
        self.order = order
        self.conv = Sequential(
            SparseConv3d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                bias=True,
            ),
            nn.Linear(in_channels, attention_channels),
            norm_layer(attention_channels),
        )
        self.conv_shortcut = (
            nn.Identity()
            if in_channels == attention_channels
            else Linear(in_channels, attention_channels)
        )

        self.norm1 = norm_layer(attention_channels)
        self.attention = PatchAttention(
            attention_channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order=order,
        )
        self.norm2 = norm_layer(attention_channels)
        self.mlp = MLP(
            in_channels=attention_channels,
            hidden_channels=int(attention_channels * mlp_ratio),
            out_channels=attention_channels,
            drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Geometry, order: Optional[POINT_ORDERING | str] = None) -> Geometry:
        x = self.conv(x) + self.conv_shortcut(x)

        # Attention block
        x = self.drop_path(self.attention(self.norm1(x), order)) + x

        # MLP block
        x = self.drop_path(self.mlp(self.norm2(x))) + x
        return x


class SerializedUnpooling(BaseSpatialModule):
    """Unpooling module that adds projected features instead of concatenating them.

    This matches the official Point Transformer V3 implementation's SerializedUnpooling.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        norm_layer: type = nn.BatchNorm1d,
        act_layer: type = GELU,
    ):
        super().__init__()
        self.proj = Sequential(
            nn.Linear(in_channels, out_channels),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            act_layer() if act_layer is not None else nn.Identity(),
        )
        self.proj_skip = Sequential(
            nn.Linear(skip_channels, out_channels),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            act_layer() if act_layer is not None else nn.Identity(),
        )
        self.unpool = SparseUnpool(
            kernel_size=kernel_size,
            stride=stride,
            concat_unpooled_st=False,
        )

    def forward(self, x: Geometry, skip: Geometry) -> Geometry:
        # Project upsampled features
        x = self.proj(x)
        # Unpool to original resolution, using skip for unpooling metadata
        x = self.unpool(x, skip)
        # Project skip connection features
        skip = self.proj_skip(skip)
        # Add them together
        out = x.replace(batched_features=x.batched_features + skip.batched_features)
        return out


class PointTransformerV3(BaseSpatialModel):
    def __init__(
        self,
        in_channels: int = 6,
        enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2),
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32),
        enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Tuple[int, ...] = (2, 2, 2, 2),
        dec_channels: Tuple[int, ...] = (64, 64, 128, 256),
        dec_num_head: Tuple[int, ...] = (4, 4, 8, 16),
        dec_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024),
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.2,
        orders: Tuple[POINT_ORDERING, ...] = tuple(POINT_ORDERING),
        shuffle_orders: bool = True,
        attn_type: Literal["patch"] = "patch",
        **kwargs,
    ):
        super().__init__()

        num_level = len(enc_depths)
        assert num_level == len(enc_channels)
        assert num_level == len(enc_num_head)
        assert num_level == len(enc_patch_size)

        assert num_level - 1 == len(dec_channels)
        assert num_level - 1 == len(dec_depths)
        assert num_level - 1 == len(dec_num_head)
        assert num_level - 1 == len(dec_patch_size)
        self.num_level = num_level
        self.shuffle_orders = shuffle_orders
        self.orders = orders

        self.conv = Sequential(
            SparseConv3d(
                in_channels,
                enc_channels[0],
                kernel_size=5,
                bias=False,
            ),
            nn.BatchNorm1d(enc_channels[0]),
            nn.GELU(),
        )

        encs = nn.ModuleList()
        down_convs = nn.ModuleList()
        for i in range(num_level):
            level_blocks = nn.ModuleList(
                [
                    PatchAttentionBlock(
                        in_channels=enc_channels[i],
                        attention_channels=enc_channels[i],
                        patch_size=enc_patch_size[i],
                        num_heads=enc_num_head[i],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path,
                        order=self.orders[i % len(self.orders)],
                        attn_type=attn_type,
                    )
                    for _ in range(enc_depths[i])
                ]
            )
            encs.append(level_blocks)

            if i < num_level - 1:
                down_convs.append(
                    Sequential(
                        nn.Linear(enc_channels[i], enc_channels[i + 1]),
                        SparseMaxPool(
                            kernel_size=2,
                            stride=2,
                        ),
                        nn.BatchNorm1d(enc_channels[i + 1]),
                        nn.GELU(),
                    )
                )

        decs = nn.ModuleList()
        up_convs = nn.ModuleList()
        dec_channels_list = list(dec_channels) + [enc_channels[-1]]
        for i in reversed(range(num_level - 1)):
            up_convs.append(
                SerializedUnpooling(
                    in_channels=dec_channels_list[i + 1],
                    skip_channels=enc_channels[i],
                    out_channels=dec_channels_list[i],
                    kernel_size=2,
                    stride=2,
                    norm_layer=nn.BatchNorm1d,
                    act_layer=GELU,
                )
            )
            level_blocks = nn.ModuleList(
                [
                    PatchAttentionBlock(
                        in_channels=dec_channels_list[i],
                        attention_channels=dec_channels_list[i],
                        patch_size=dec_patch_size[i],
                        num_heads=dec_num_head[i],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path,
                        order=self.orders[i % len(self.orders)],
                        attn_type=attn_type,
                    )
                    for _ in range(dec_depths[i])
                ]
            )
            decs.append(level_blocks)

        self.encs = encs
        self.down_convs = down_convs
        self.decs = decs
        self.up_convs = up_convs

        out_channels = kwargs.get("out_channels")
        if out_channels is not None:
            self.out_channels = out_channels
            self.final = Linear(dec_channels_list[0], out_channels)
        else:
            self.final = nn.Identity()

    def _select_order(self, blk_idx: int) -> POINT_ORDERING:
        """Selects the point ordering for a block.

        Use `torch.manual_seed` to control randomness.
        """
        if self.shuffle_orders:
            idx = torch.randint(0, len(self.orders), (1,)).item()
            return self.orders[idx]
        return self.orders[blk_idx % len(self.orders)]

    def forward(self, x: Geometry) -> Geometry:
        x = self.conv(x)
        skips = []

        blk_idx = 0
        # Encoder
        for level in range(self.num_level):
            # Process each block individually in this level
            level_blocks = self.encs[level]
            for block in level_blocks.children():
                selected_order = self._select_order(blk_idx)
                x = block(x, selected_order)
                blk_idx += 1

            if level < self.num_level - 1:
                skips.append(x)
                x = self.down_convs[level](x)

        # Decoder
        for level in range(self.num_level - 1):
            x = self.up_convs[level](x, skips[-(level + 1)])

            level_blocks = self.decs[level]
            for block in level_blocks.children():
                selected_order = self._select_order(blk_idx)
                x = block(x, selected_order)
                blk_idx += 1

        return self.final(x)


# Pytests
@pytest.fixture
def pc(device: torch.device = torch.device("cuda:0")):
    # Batch size, min number of points, max number of points, number of features
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = [N.item() for N in torch.randint(min_N, max_N, (B,))]
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    print(
        f"Using total of {sum(Ns)} points with [{', '.join([str(N) for N in Ns])}] points per batch. Using {C} features."
    )
    return Points(coords, features).to(device)


# Usage:
# pytest -v -s examples/point_transformer_v3.py::test_point_transformer_v3
def test_point_transformer_v3(pc: Points):
    point_transformer = PointToSparseWrapper(
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
