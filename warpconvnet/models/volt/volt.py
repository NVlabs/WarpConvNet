# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Volt: a Volume Transformer for sparse-voxel semantic segmentation.

WarpConvNet-native port of the Volt model (https://github.com/YilmazKadir/Volt).
The scene is partitioned into non-overlapping ``kernel_size**3`` voxel patches, each
linearly embedded into a token; a plain Transformer with axial RoPE and global
per-scene attention processes the tokens, which are then un-embedded back to voxel
resolution.

The architecture is kept faithful to the reference implementation so it can be
verified against the published ScanNet numbers. WarpConvNet primitives are used at
the boundaries: `Voxels` for geometry I/O and
`warpconvnet.nn.functional.flash_attn_utils.flash_attn_varlen_qkvpacked` for
the variable-length flash-attention (which transparently chunks sequences whose
count exceeds the flash-attn limit).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.flash_attn_utils import flash_attn_varlen_qkvpacked
from warpconvnet.nn.modules.sparse_conv import SparseConv3d


# --- Minimal vendored transformer layers (drop-in, output-identical to the timm
# equivalents previously used: DropPath, Mlp, LayerScale). Vendored to drop the timm
# dependency; parameter names (fc1/fc2, gamma) match timm so checkpoints still load.
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Stochastic depth, identical to timm.layers.drop_path."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """Two-layer MLP, matching timm.layers.Mlp with default drop=0 / norm=None."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    """Per-channel learnable scale, matching timm vision_transformer.LayerScale."""

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def _grouping(indices, K):
    """Non-overlapping K**3 patch grouping shared by the tokenizers / detokenizer.

    Returns ``(coarse_indices [T,4], inverse [N], offset_id [N])``.
    """
    coarse_indices_per_voxel = indices // indices.new_tensor([1, K, K, K])
    coarse_indices, inverse = torch.unique(
        coarse_indices_per_voxel, dim=0, sorted=True, return_inverse=True
    )
    offset = indices[:, 1:] % K
    offset_id = offset[:, 0] * K * K + offset[:, 1] * K + offset[:, 2]
    return coarse_indices, inverse, offset_id


class Tokenizer(nn.Module):
    """Group voxels into non-overlapping ``K**3`` patches and linearly embed each."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.proj = nn.Linear(kernel_size**3 * in_channels, out_channels)

    def forward(self, features, indices):
        K = self.kernel_size
        coarse_indices, inverse, offset_id = _grouping(indices, K)

        patches = features.new_zeros(coarse_indices.shape[0], K**3, features.shape[1])
        patches[inverse, offset_id] = features

        coarse_features = self.proj(patches.flatten(1))
        return coarse_features, coarse_indices, inverse, offset_id


class RoPE(nn.Module):
    """Axial rotary positional embedding with anisotropic per-axis frequency split."""

    def __init__(
        self,
        theta: float = 100.0,
        freq_split: tuple = (12, 12, 8),
        max_grid_size: tuple = (1024, 1024, 512),
    ) -> None:
        super().__init__()
        freqs_x = 1.0 / theta ** torch.linspace(0, 1, freq_split[0])
        freqs_y = 1.0 / theta ** torch.linspace(0, 1, freq_split[1])
        freqs_z = 1.0 / theta ** torch.linspace(0, 1, freq_split[2])

        self.register_buffer(
            "cis_cache_x", self._precompute(freqs_x, max_grid_size[0]), persistent=False
        )
        self.register_buffer(
            "cis_cache_y", self._precompute(freqs_y, max_grid_size[1]), persistent=False
        )
        self.register_buffer(
            "cis_cache_z", self._precompute(freqs_z, max_grid_size[2]), persistent=False
        )

    def _precompute(self, freqs, max_pos):
        freqs_pos = torch.outer(torch.arange(max_pos).float(), freqs)
        return torch.polar(torch.ones_like(freqs_pos), freqs_pos)

    def compute_axial_cis_efficient(self, indices):
        cis_x = self.cis_cache_x[indices[:, 0]]
        cis_y = self.cis_cache_y[indices[:, 1]]
        cis_z = self.cis_cache_z[indices[:, 2]]
        return torch.cat([cis_x, cis_y, cis_z], dim=-1).unsqueeze(0)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and variable-length flash attention."""

    def __init__(self, dim: int = 768, num_heads: int = 12, qk_norm: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.h_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.h_dim) if qk_norm else nn.Identity()

    @staticmethod
    def apply_rotary_emb(q, k, freqs_cis):
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_ * freqs_cis).flatten(2)
        k_out = torch.view_as_real(k_ * freqs_cis).flatten(2)
        return q_out.type_as(q), k_out.type_as(k)

    def forward(self, x, freqs_cis, cu_seqlens, max_seqlen):
        N, C = x.shape
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.h_dim).permute(1, 2, 0, 3)
        q, k, v = qkv.unbind(dim=0)

        q, k = self.q_norm(q).to(q.dtype), self.k_norm(k).to(k.dtype)
        q, k = self.apply_rotary_emb(q, k, freqs_cis)
        qkv = torch.stack([q, k, v], dim=0).permute(2, 0, 1, 3)

        qkv_dtype = qkv.dtype
        x = flash_attn_varlen_qkvpacked(qkv.half(), cu_seqlens, max_seqlen=max_seqlen)
        x = x.reshape(-1, C).to(qkv_dtype)
        x = self.proj(x)
        return x


class TokenConv(nn.Module):
    """Stride-1 sparse conv over the coarse token grid (local mixing among patches)."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = SparseConv3d(dim, dim, kernel_size=kernel_size, stride=1)

    def forward(self, feats, coords3, offsets):
        vox = Voxels(batched_coordinates=coords3, batched_features=feats, offsets=offsets)
        return self.conv(vox).features


class SparseResBlock(nn.Module):
    """ResNet-style non-strided sparse-conv block (conv-BN-act-conv-BN + skip)."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()

        def bn(c):
            return nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)

        self.conv1 = SparseConv3d(dim, dim, kernel_size=kernel_size, stride=1)
        self.bn1 = bn(dim)
        self.conv2 = SparseConv3d(dim, dim, kernel_size=kernel_size, stride=1)
        self.bn2 = bn(dim)
        self.act = nn.GELU()

    def forward(self, vox: Voxels) -> Voxels:
        identity = vox.features
        x = self.conv1(vox)
        x = x.replace(batched_features=self.act(self.bn1(x.features)))
        x = self.conv2(x)
        x = x.replace(batched_features=self.bn2(x.features))
        return x.replace(batched_features=self.act(x.features + identity))


class ConvBlockTokenizer(nn.Module):
    """ResNet-style non-strided sparse-conv stem followed by the per-slot linear embed.

    Enriches fine voxel features with a multi-block conv stem (overlapping receptive
    fields) and then applies Volt's original ``flatten(K**3 * C) -> Linear`` per-slot
    embed, so local context is added *alongside* the within-patch structure (rather
    than pooled away).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stem_dim: int = 64,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stem = SparseConv3d(in_channels, stem_dim, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm1d(stem_dim, eps=1e-3, momentum=0.01)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList([SparseResBlock(stem_dim) for _ in range(num_blocks)])
        self.proj = nn.Linear(kernel_size**3 * stem_dim, out_channels)

    def forward(self, voxels: Voxels, indices):
        K = self.kernel_size
        x = self.stem(voxels)
        x = x.replace(batched_features=self.act(self.bn(x.features)))
        for blk in self.blocks:
            x = blk(x)
        fine = x.features  # [N, stem_dim], overlapping local context

        coarse_indices, inverse, offset_id = _grouping(indices, K)
        patches = fine.new_zeros(coarse_indices.shape[0], K**3, fine.shape[1])
        patches[inverse, offset_id] = fine  # per-slot, no pooling
        coarse = self.proj(patches.flatten(1))
        return coarse, coarse_indices, inverse, offset_id


class Block(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        qk_norm: bool = False,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.norm0 = norm_layer(dim)
            self.conv = TokenConv(dim)
            self.ls0 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path0 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(dim=dim, num_heads=num_heads, qk_norm=qk_norm)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, freqs_cis, cu_seqlens, max_seqlen, coords3=None, offsets=None):
        if self.use_conv:
            x = x + self.drop_path0(self.ls0(self.conv(self.norm0(x), coords3, offsets)))
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), freqs_cis, cu_seqlens, max_seqlen))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Detokenizer(nn.Module):
    """Un-embed each token back to its ``K**3`` voxel patch (transposed-conv analogue)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels, kernel_size**3 * out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, coarse_features, inverse, offset_id):
        K = self.kernel_size
        all_offsets = self.proj(coarse_features).view(-1, K**3, self.out_channels)
        fine_features = all_offsets[inverse, offset_id]
        fine_features = fine_features + self.bias
        return fine_features


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        def bn_layer(c):
            return nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)

        act_layer = nn.GELU

        self.pre = nn.Sequential(
            bn_layer(in_channels),
            act_layer(),
            nn.Linear(in_channels, out_channels, bias=False),
            bn_layer(out_channels),
            act_layer(),
        )
        self.unembed = Detokenizer(out_channels, out_channels, kernel_size=kernel_size)
        self.post = nn.Sequential(bn_layer(out_channels), act_layer())

    def forward(self, x, inverse, offset_id):
        x = self.pre(x)
        x = self.unembed(x, inverse, offset_id)
        x = self.post(x)
        return x


class Volt(nn.Module):
    """Volume Transformer. WarpConvNet-native interface: ``Voxels -> Voxels``.

    Also exposes `forward_tensors` for use as a backbone inside external
    training harnesses (e.g. Pointcept) that pass raw ``(feat, grid_coord, batch)``.
    """

    def __init__(
        self,
        in_channels: int = 6,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        init_values: Optional[float] = None,
        qk_norm: bool = False,
        drop_path: float = 0.3,
        stride: int = 5,
        kernel_size: int = 5,
        up_mlp_dim: int = 256,
        increase_drop_path: bool = True,
        tokenizer_type: str = "linear",
        conv_before_attn: bool = False,
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU
        self.conv_before_attn = conv_before_attn

        assert stride == kernel_size, "Volt only supports non-overlapping patches"
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == "convblock":
            self.tokenizer = ConvBlockTokenizer(
                in_channels=in_channels, out_channels=embed_dim, kernel_size=kernel_size
            )
        elif tokenizer_type == "linear":
            self.tokenizer = Tokenizer(
                in_channels=in_channels, out_channels=embed_dim, kernel_size=kernel_size
            )
        else:
            raise ValueError(
                f"unknown tokenizer_type {tokenizer_type!r}; use 'linear' or 'convblock'"
            )

        if increase_drop_path:
            drop_path_list = torch.linspace(0, drop_path, depth).tolist()
        else:
            drop_path_list = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    drop_path=drop_path_list[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=Mlp,
                    use_conv=conv_before_attn,
                )
                for i in range(depth)
            ]
        )

        self.pos_enc = RoPE()
        self.decoder = Decoder(
            in_channels=embed_dim, out_channels=up_mlp_dim, kernel_size=kernel_size
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "init_weights"):
            module.init_weights()

    @staticmethod
    def compute_seqlens(batch_indices):
        points_per_batch = torch.bincount(batch_indices + 1)
        cu_seqlens = torch.cumsum(points_per_batch, dim=0, dtype=torch.int32)
        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = sequence_lengths.max().item()
        return cu_seqlens, max_seqlen

    def forward_tensors(self, feat, grid_coord, batch):
        """Core forward on raw tensors. Returns per-voxel features ``[N, up_mlp_dim]``."""
        indices = torch.cat([batch.unsqueeze(-1).int(), grid_coord.int()], dim=1).contiguous()
        if self.tokenizer_type == "convblock":
            counts = torch.bincount(batch.long())
            offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).cpu()
            vox = Voxels(
                batched_coordinates=grid_coord.int(), batched_features=feat, offsets=offsets
            )
            features, indices, inverse, offset_id = self.tokenizer(vox, indices)
        else:
            features, indices, inverse, offset_id = self.tokenizer(feat, indices)

        cu_seqlens, max_seqlen = self.compute_seqlens(indices[:, 0])
        freqs_cis = self.pos_enc.compute_axial_cis_efficient(indices[:, 1:])

        coords3 = offsets = None
        if self.conv_before_attn:
            coords3 = indices[:, 1:].contiguous()  # coarse token coords [T, 3]
            offsets = cu_seqlens.long().cpu()  # [B+1] token-grid batch offsets

        for blk in self.blocks:
            features = blk(features, freqs_cis, cu_seqlens, max_seqlen, coords3, offsets)

        features = self.decoder(features, inverse, offset_id)
        return features

    def forward(self, voxels: Voxels) -> Voxels:
        """WarpConvNet-native forward: consumes/returns `Voxels`."""
        bcoords = voxels.batch_indexed_coordinates  # [N, 4] = (batch, x, y, z)
        feat = voxels.features
        out = self.forward_tensors(feat, bcoords[:, 1:], bcoords[:, 0])
        return voxels.replace(batched_features=out)
