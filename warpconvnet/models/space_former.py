# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpaCeFormer: Mixed Spatial Attention U-Net for sparse 3D voxels.

Paper: https://arxiv.org/abs/2604.20395

The name follows the paper's "space-curve" etymology. Each level picks between
two attention flavors:
  - ``curve``: serialized 1D attention along a space-filling curve via
    `warpconvnet.nn.modules.attention.PatchAttention`.
  - ``space``: window-grouped 3D attention via
    `warpconvnet.nn.modules.space_attention.SpaceAttention`.

Both blocks share a pre/post norm sandwich with a residual sparse-conv branch and
an FFN.
"""

from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.nn.functional.voxel_encode import WINDOW_OFFSET_TYPE
from warpconvnet.nn.modules.base_module import BaseSpatialModel
from warpconvnet.nn.modules.mlp import Linear
from warpconvnet.nn.modules.sequential import Sequential, TupleSequential
from warpconvnet.nn.modules.space_attention import block_factory
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.sparse_pool import SparseMaxPool, SparseUnpool
from warpconvnet.utils.logger import get_logger

logger = get_logger(__name__, rank_zero_only=True)


def _parse_attn_type(
    attn_type: Union[
        List[Literal["curve", "space", "all"]], Literal["curve", "space", "all"], str
    ],
    num_level: int,
) -> List[Literal["curve", "space", "all"]]:
    """Parse ``attn_type`` into a per-level list.

    Accepts:
      - a single string (``"curve"`` / ``"space"`` / ``"all"``): broadcast to all levels.
      - a compact code string of length ``num_level`` using
        ``{"c": "curve", "s": "space", "a": "all"}`` (e.g. ``"ccssa"`` for 5 levels).
      - an explicit sequence of valid attention names.
    """
    MAPPING = {"c": "curve", "s": "space", "a": "all"}
    VALID = set(MAPPING.values())

    if isinstance(attn_type, str):
        if attn_type.lower() in VALID:
            return [attn_type] * num_level
        assert (
            len(attn_type) == num_level
        ), f"attn_type compact code {attn_type!r} must have length {num_level}"
        result = []
        for c in attn_type:
            assert c in MAPPING, f"Invalid attention code {c!r} in {attn_type!r}"
            result.append(MAPPING[c])
        return result

    if isinstance(attn_type, Sequence):
        assert (
            len(attn_type) == num_level
        ), f"attn_type sequence length {len(attn_type)} != num_level {num_level}"
        for t in attn_type:
            assert t in VALID, f"Invalid attention type: {t!r}"
        return list(attn_type)

    raise ValueError(f"Invalid attn_type: {attn_type!r}")


class SpaCeFormer(BaseSpatialModel):
    """Mixed spatial attention U-Net for sparse voxels.

    Paper: https://arxiv.org/abs/2604.20395

    The encoder has ``num_level = len(enc_depths)`` stages. Each stage runs a
    stack of attention blocks whose flavor (``curve`` or ``space``) is selected
    by ``enc_attn_types``. Stages are connected with stride-2 downsamples.

    The decoder mirrors the encoder with ``num_level - 1`` stages. Skip features
    from the encoder are concatenated through `SparseUnpool`.

    Args:
        in_channels: input feature dimension.
        enc_depths / enc_channels / enc_num_head / enc_patch_size: per-level
            encoder configuration. All four tuples must have ``num_level``
            entries.
        enc_attn_types: per-level attention type. Either a single string, a
            compact code (``"c"``/``"s"`` per level), or an explicit list.
        dec_*: decoder counterparts. Length must be ``num_level - 1``.
        block_type: norm-residual layout. One of ``pre_norm`` (modern default,
            ``x + sublayer(norm(x))``), ``post_norm`` (``norm(x + sublayer(x))``-ish),
            or ``stream_norm`` (residual stream is normalized, paper's choice).
        kernel_size: sparse conv kernel size in attention residual branches.
        mlp_ratio: FFN expansion ratio.
        qkv_bias / qk_scale / attn_drop / proj_drop / drop_path: standard
            transformer regularization.
        shuffle_orders: when True, randomize curve/window ordering each forward.
        use_rope / rope_base / enc_rope_bases / dec_rope_bases: RoPE controls.
            ``rope_base`` overrides the per-level lists if provided.
        voxel_offsets: list of window offsets used by ``space`` levels.
        patch_orders: list of space-filling-curve orderings used by ``curve`` levels.
        conv_norm_layer: norm applied after stem and pool/unpool conv layers.
        out_channels: optional final linear projection.
    """

    def __init__(
        self,
        in_channels: int = 6,
        enc_depths: Tuple[int, ...] = (2, 2, 2, 6, 2),
        enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_head: Tuple[int, ...] = (2, 4, 8, 16, 32),
        enc_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024, 1024),
        enc_attn_types: Union[Literal["curve", "space", "all"], str] = "curve",
        dec_depths: Tuple[int, ...] = (2, 2, 2, 2),
        dec_channels: Tuple[int, ...] = (64, 64, 128, 256),
        dec_num_head: Tuple[int, ...] = (4, 4, 8, 16),
        dec_patch_size: Tuple[int, ...] = (1024, 1024, 1024, 1024),
        dec_attn_types: Union[Literal["curve", "space", "all"], str] = "curve",
        block_type: Literal["pre_norm", "post_norm", "stream_norm"] = "pre_norm",
        kernel_size: int = 3,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.2,
        shuffle_orders: bool = True,
        use_rope: bool = False,
        rope_base: Optional[int] = None,
        enc_rope_bases: Tuple[int, ...] = (250, 250, 250, 250, 250),
        dec_rope_bases: Tuple[int, ...] = (250, 250, 250, 250),
        voxel_offsets: List[Union[WINDOW_OFFSET_TYPE, str]] = ["zero", "xyz"],
        patch_orders: Tuple[POINT_ORDERING, ...] = tuple(POINT_ORDERING),
        conv_norm_layer: Optional[type] = nn.BatchNorm1d,
        out_channels: Optional[int] = None,
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

        enc_attn_types = _parse_attn_type(enc_attn_types, num_level)
        dec_attn_types = _parse_attn_type(dec_attn_types, num_level - 1)
        self.enc_attn_types = enc_attn_types
        self.dec_attn_types = dec_attn_types

        self.voxel_offsets = voxel_offsets
        self.patch_orders = patch_orders

        self._log_level_configs(
            phase_label="Encoder",
            prefix="enc",
            block_type=block_type,
            attn_types=enc_attn_types,
            depths=enc_depths,
            indices=range(num_level),
        )
        self._log_level_configs(
            phase_label="Decoder",
            prefix="dec",
            block_type=block_type,
            attn_types=dec_attn_types,
            depths=dec_depths,
            indices=list(reversed(range(num_level - 1))),
        )

        if conv_norm_layer is None:
            conv_norm_layer = nn.Identity

        if rope_base is not None:
            enc_rope_bases = tuple([rope_base] * num_level)
            dec_rope_bases = tuple([rope_base] * (num_level - 1))
        logger.info(f"enc_rope_bases: {enc_rope_bases}")
        logger.info(f"dec_rope_bases: {dec_rope_bases}")

        block_cls = block_factory(block_type)

        self.conv = Sequential(
            SparseConv3d(
                in_channels,
                enc_channels[0],
                kernel_size=5,
            ),
            conv_norm_layer(enc_channels[0]),
            nn.GELU(),
        )

        encs = nn.ModuleList()
        down_convs = nn.ModuleList()
        for i in range(num_level):
            level_blocks = nn.ModuleList(
                [
                    block_cls(
                        in_channels=enc_channels[i],
                        attention_channels=enc_channels[i],
                        patch_size=enc_patch_size[i],
                        num_heads=enc_num_head[i],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path,
                        order=self._select_order(i, enc_attn_types[i]),
                        attn_type=enc_attn_types[i],
                        use_rope=use_rope,
                        rope_base=enc_rope_bases[i],
                    )
                    for _ in range(enc_depths[i])
                ]
            )
            encs.append(level_blocks)

            if i < num_level - 1:
                down_convs.append(
                    Sequential(
                        nn.Linear(enc_channels[i], enc_channels[i + 1]),
                        SparseMaxPool(kernel_size=2, stride=2),
                        conv_norm_layer(enc_channels[i + 1]),
                        nn.GELU(),
                    )
                )

        decs = nn.ModuleList()
        up_convs = nn.ModuleList()
        dec_channels_list = list(dec_channels) + [enc_channels[-1]]

        for i in reversed(range(num_level - 1)):
            up_convs.append(
                TupleSequential(
                    nn.Linear(dec_channels_list[i + 1], dec_channels_list[i]),
                    SparseUnpool(
                        kernel_size=2,
                        stride=2,
                        concat_unpooled_st=True,
                    ),
                    nn.Linear(dec_channels_list[i] + enc_channels[i], dec_channels_list[i]),
                    conv_norm_layer(dec_channels_list[i]),
                    nn.GELU(),
                    tuple_layer=1,
                )
            )
            level_blocks = nn.ModuleList(
                [
                    block_cls(
                        in_channels=dec_channels_list[i],
                        attention_channels=dec_channels_list[i],
                        patch_size=dec_patch_size[i],
                        num_heads=dec_num_head[i],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path,
                        order=self._select_order(i, dec_attn_types[i]),
                        attn_type=dec_attn_types[i],
                        use_rope=use_rope,
                        rope_base=dec_rope_bases[i],
                    )
                    for _ in range(dec_depths[i])
                ]
            )
            decs.append(level_blocks)

        self.encs = encs
        self.down_convs = down_convs
        self.decs = decs
        self.up_convs = up_convs

        if out_channels is not None:
            self.out_channels = out_channels
            self.final = Linear(dec_channels_list[0], out_channels)
        else:
            self.final = nn.Identity()

    def _log_level_configs(
        self,
        phase_label: str,
        prefix: str,
        block_type,
        attn_types,
        depths,
        indices,
    ) -> None:
        for i in indices:
            if attn_types[i] in ("space", "all"):
                logger.info(
                    f"[{phase_label} Level {i}] {prefix}_block_type: {block_type}, "
                    f"{prefix}_attn_type: {attn_types[i]}, voxel_offset: "
                    f"{self.voxel_offsets[i % len(self.voxel_offsets)]}, "
                    f"{depths[i]} blocks"
                )
            else:
                logger.info(
                    f"[{phase_label} Level {i}] {prefix}_block_type: {block_type}, "
                    f"{prefix}_attn_type: {attn_types[i]}, patch_order: "
                    f"{self.patch_orders[i % len(self.patch_orders)]}, "
                    f"{depths[i]} blocks"
                )

    def _select_order(self, blk_idx: int, order_type: Literal["curve", "space", "all"]) -> str:
        """Select an ordering for a block.

        For ``space`` / ``all`` blocks, picks a window-offset key from
        ``voxel_offsets`` (the offset is unused by ``all`` but kept for shape
        parity). For ``curve`` blocks, picks a space-filling-curve ordering
        from ``patch_orders``. Use `torch.manual_seed` to control randomness
        when ``shuffle_orders=True``.
        """
        if order_type in ("space", "all"):
            orders = self.voxel_offsets
        else:
            orders = self.patch_orders

        if self.shuffle_orders:
            idx = torch.randint(0, len(orders), (1,)).item()
            return orders[idx]
        return orders[blk_idx % len(orders)]

    def forward(self, x: Geometry) -> Geometry:
        x = self.conv(x)
        skips = []

        block_idx = 0
        for level in range(self.num_level):
            for block in self.encs[level].children():
                selected_order = self._select_order(block_idx, self.enc_attn_types[level])
                x = block(x, selected_order)
                block_idx += 1

            if level < self.num_level - 1:
                skips.append(x)
                x = self.down_convs[level](x)

        for level in range(self.num_level - 1):
            x = self.up_convs[level](x, skips[-(level + 1)])

            for block in self.decs[level].children():
                selected_order = self._select_order(block_idx, self.dec_attn_types[-(level + 1)])
                x = block(x, selected_order)
                block_idx += 1

        return self.final(x)
