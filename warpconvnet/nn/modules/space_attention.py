# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.nn.functional.flash_attn_utils import flash_attn_varlen_qkvpacked
from warpconvnet.nn.functional.voxel_encode import (
    WINDOW_OFFSET_TYPE,
    voxel_encode_cached,
)
from warpconvnet.nn.modules.activations import DropPath
from warpconvnet.nn.modules.attention import FeedForward, PatchAttention
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.modules.mlp import BatchedLinear, Linear
from warpconvnet.nn.modules.normalizations import LayerNorm
from warpconvnet.nn.modules.rope import VoxelRotaryPositionalEmbeddings
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv3d


class SpaceAttention(BaseSpatialModule):
    """Window-grouped voxel attention with optional 3D RoPE.

    Voxels are grouped into 3D windows of size ``window_size`` (an offset can be
    applied to shift the window grid). Attention runs independently within each
    window via ``flash_attn_varlen_qkvpacked``. When ``window_size == "all"``,
    attention is computed across the full batch sequence.

    Used as the spatial attention block in SpaCeFormer.

    Args:
        dim: feature dimension.
        window_size: 3-tuple of window extents per axis. Pass ``"all"`` for
            full-sequence attention.
        num_heads: attention heads.
        qkv_bias: bias on QKV projection.
        qk_scale: override for ``head_dim ** -0.5`` scaling.
        attn_drop / proj_drop: dropout probabilities.
        offset: default coordinate offset, either a string from
            `WINDOW_OFFSET_TYPE` or an explicit 3-tuple of float fractions
            of ``window_size``.
        combine_consecutive_ones: collapse runs of length-1 windows into a single
            attention sequence to reduce kernel launch overhead.
        use_rope: enable `VoxelRotaryPositionalEmbeddings` on Q/K.
        rope_base: RoPE base. See
            `warpconvnet.nn.modules.rope.suggest_voxel_rope_base`.
        use_batched_qkv: use `BatchedLinear` for QKV (Muon-friendly).
        encoding_method: voxel-window encoding backend
            (``counting_sort``/``ravel_fast``/``ravel``/``morton``).
    """

    def __init__(
        self,
        dim: int,
        window_size: Optional[Union[Tuple[int, int, int], int, str]] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        offset: Union[WINDOW_OFFSET_TYPE, Tuple[float, float, float]] = "zero",
        combine_consecutive_ones: bool = False,
        use_rope: bool = True,
        rope_base: int = 250,
        use_batched_qkv: bool = True,
        encoding_method: str = "counting_sort",
    ):
        super().__init__()
        self.encoding_method = encoding_method

        if isinstance(window_size, str):
            assert window_size == "all", f"Invalid window_size: {window_size}"

        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)

        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.use_batched_qkv = use_batched_qkv

        if use_batched_qkv:
            self.qkv = BatchedLinear(dim, dim, num_matrices=3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.offset = offset
        self.attn_drop_p = attn_drop
        self.combine_consecutive_ones = combine_consecutive_ones

        self.use_rope = use_rope
        if use_rope:
            self.rope = VoxelRotaryPositionalEmbeddings(
                dim=dim,
                num_heads=num_heads,
                base=rope_base,
            )

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _attn_offset(self, counts: Int[Tensor, "N"]) -> Int[Tensor, "B"]:  # noqa: F821
        result_middle = torch.cumsum(counts, dim=0)
        result = torch.cat(
            [torch.zeros(1, device=counts.device, dtype=counts.dtype), result_middle]
        ).int()
        return result.contiguous()

    def _attn_offset_combine_consecutive_ones(
        self, counts: Int[Tensor, "N"]  # noqa: F821
    ) -> Int[Tensor, "B"]:  # noqa: F821
        if len(counts) == 0:
            return torch.zeros(1, device=counts.device, dtype=torch.int32)

        is_one = counts == 1
        if not is_one.any():
            result_middle = torch.cumsum(counts, dim=0)
            result = torch.cat(
                [
                    torch.zeros(1, device=counts.device, dtype=counts.dtype),
                    result_middle,
                ]
            ).int()
            return result.contiguous()

        is_not_one = ~is_one
        prev_is_not_one = torch.cat([torch.tensor([True], device=counts.device), is_not_one[:-1]])
        group_starts = is_not_one | prev_is_not_one

        group_ids = torch.cumsum(group_starts.int(), dim=0) - 1

        group_sizes = torch.zeros_like(counts)
        ones_mask = is_one.to(counts.dtype)
        group_sizes.scatter_add_(0, group_ids, ones_mask)

        gathered_sizes = group_sizes[group_ids]
        combined_values = torch.where(is_one, gathered_sizes, counts)
        combined_counts = combined_values[group_starts]

        result_middle = torch.cumsum(combined_counts, dim=0)
        result = torch.cat(
            [torch.zeros(1, device=counts.device, dtype=counts.dtype), result_middle]
        ).int()
        return result.contiguous()

    def forward(
        self,
        x: Geometry,
        coord_offset: Union[Tuple[float, float, float], WINDOW_OFFSET_TYPE] = "zero",
    ) -> Geometry:
        if coord_offset is None:
            coord_offset = self.offset

        assert isinstance(coord_offset, str) or (
            isinstance(coord_offset, tuple) and len(coord_offset) == 3
        ), "coord_offset must be a tuple of 3 floats or a string"

        feats = x.features
        coords = x.coordinate_tensor
        M, C = feats.shape[:2]
        code_result = None

        if self.window_size == "all":
            attn_offsets = x.offsets.to(feats.device)
            max_seqlen = x.offsets.diff().max().item()
        else:
            code_result = voxel_encode_cached(
                x.coordinate_tensor,
                batch_offsets=x.offsets,
                window_size=self.window_size,
                coord_offset=coord_offset,
                encoding_method=self.encoding_method,
            )
            feats = feats[code_result.perm]
            if self.use_rope:
                coords = coords[code_result.perm]

            if self.combine_consecutive_ones:
                attn_offsets = self._attn_offset_combine_consecutive_ones(code_result.counts).to(
                    feats.device
                )
            else:
                attn_offsets = self._attn_offset(code_result.counts).to(feats.device)

            max_seqlen = code_result.counts.max().item()

        qkv_raw = self.qkv(feats)

        if self.use_rope:
            qkv: Float[Tensor, "M 3 H D"] = self.rope(qkv_raw, coords)
        else:
            qkv: Float[Tensor, "M 3 H D"] = qkv_raw.reshape(
                -1, 3, self.num_heads, C // self.num_heads
            )

        if qkv.dtype not in [torch.float16, torch.bfloat16]:
            qkv = qkv.to(torch.float16)

        out_feat: Float[Tensor, "M H D"] = flash_attn_varlen_qkvpacked(
            qkv,
            attn_offsets,
            max_seqlen=max_seqlen,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            softmax_scale=self.scale,
        )
        out_feat: Float[Tensor, "M C"] = out_feat.reshape(M, C).to(feats.dtype)

        out_feat = self.proj(out_feat)
        out_feat = self.proj_drop(out_feat)

        if code_result is not None:
            out_feat = out_feat[code_result.inverse_perm]

        return x.replace(batched_features=out_feat.to(feats.dtype))


# ----------------------------------------------------------------------------
# Reusable attention blocks for space-curve-style transformer backbones.
#
# Each block bundles: (1) a residual sparse-conv shortcut that sets the channel
# count, (2) one of {`PatchAttention`, `SpaceAttention`} as the attention
# sublayer, (3) a `FeedForward` MLP. Concrete subclasses choose the
# norm-residual layout (pre-norm / post-norm / stream-norm). SpaCeFormer is one
# consumer; downstream backbones can build on the same primitives.
# ----------------------------------------------------------------------------


class AllAttention(SpaceAttention):
    """Full-sequence variant of `SpaceAttention`.

    Pins ``window_size="all"`` so attention runs across the entire batched
    sequence at the current level (no window grouping). Useful at the deepest
    decoder/encoder stage where the voxel count is small enough for a single
    flash-attn call. Any ``window_size`` kwarg is accepted and ignored.
    """

    def __init__(
        self,
        dim: int,
        window_size=None,
        num_heads: int = 8,
        **kwargs,
    ):
        super().__init__(dim=dim, window_size="all", num_heads=num_heads, **kwargs)


STR2ATTN = {
    "curve": PatchAttention,
    "space": SpaceAttention,
    "all": AllAttention,
}


class SpaCeFormerBlockBase(BaseSpatialModule):
    """Shared sparse-conv shortcut + attention + FFN parameter holder.

    Subclasses supply the norm-residual layout in their ``forward``. The
    attention sublayer is selected via ``attn_type`` from `STR2ATTN`:

    - ``"curve"`` → `PatchAttention` (1D attention along a space-filling curve).
    - ``"space"`` → `SpaceAttention` (3D window-grouped attention).
    - ``"all"`` → `AllAttention` (full-sequence attention; ignores patch_size).
    """

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
        attn_type: Literal["curve", "space", "all"] = "curve",
        order: POINT_ORDERING = POINT_ORDERING.RANDOM,
        use_rope: bool = False,
        rope_base: int = 250,
    ):
        super().__init__()
        self.order = order
        assert attn_type in STR2ATTN, f"Invalid attention type: {attn_type}"
        attn_block = STR2ATTN[attn_type]

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
        if attn_type == "curve":
            self.attention = attn_block(
                dim=attention_channels,
                patch_size=patch_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                order=order,
                use_rope=use_rope,
                rope_base=rope_base,
            )
        else:  # "space"
            self.attention = attn_block(
                dim=attention_channels,
                window_size=patch_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                offset=order if isinstance(order, str) else "zero",
                use_rope=use_rope,
                rope_base=rope_base,
            )
        self.norm2 = norm_layer(attention_channels)
        self.mlp = FeedForward(
            dim=attention_channels,
            hidden_dim=int(attention_channels * mlp_ratio),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


class PreNormBlock(SpaCeFormerBlockBase):
    """Standard pre-LN: ``x + sublayer(norm(x))``. Modern transformer default."""

    def forward(self, x: Geometry, order: Optional[Union[POINT_ORDERING, str]] = None) -> Geometry:
        x = self.conv(x) + self.conv_shortcut(x)
        x = self.drop_path(self.attention(self.norm1(x), order)) + x
        x = self.drop_path(self.mlp(self.norm2(x))) + x
        return x


class PostNormBlock(SpaCeFormerBlockBase):
    """Post-LN variant: norm applied after each sublayer's residual sum."""

    def forward(self, x: Geometry, order: Optional[Union[POINT_ORDERING, str]] = None) -> Geometry:
        x = self.conv(x) + self.conv_shortcut(x)
        x = self.drop_path(self.attention(x, order)) + x
        x = self.norm1(x)
        x = self.drop_path(self.mlp(x)) + x
        x = self.norm2(x)
        return x


class StreamNormBlock(SpaCeFormerBlockBase):
    """Stream-norm: residual stream stays normalized.

    ``x = norm(x); x = sublayer(x) + x`` — the residual itself is the
    post-norm value, so gradients flow through the norm on the skip path.
    """

    def forward(self, x: Geometry, order: Optional[Union[POINT_ORDERING, str]] = None) -> Geometry:
        x = self.conv(x) + self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.drop_path(self.attention(x, order)) + x
        x = self.norm2(x)
        x = self.drop_path(self.mlp(x)) + x
        return x


BLOCK_REGISTRY = {
    "pre_norm": PreNormBlock,
    "post_norm": PostNormBlock,
    "stream_norm": StreamNormBlock,
}


def block_factory(block_type: Literal["pre_norm", "post_norm", "stream_norm"]) -> type:
    """Look up a block class by name from `BLOCK_REGISTRY`."""
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(
            f"Invalid block type: {block_type!r}. Must be one of {list(BLOCK_REGISTRY)}"
        )
    return BLOCK_REGISTRY[block_type]
