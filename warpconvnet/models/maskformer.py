# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

import torch
import warp as wp
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from warpconvnet.geometry.base.coords import Coords
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.ops.voxel import voxel_downsample_random_indices
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.ops.convert import cat_to_pad_tensor
from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.attention import (
    ToAttention,
    ToSpatialFeatures,
    zero_out_points,
)
from warpconvnet.nn.modules.base_module import BaseSpatialModel
from warpconvnet.nn.modules.mlp import BatchedLinear, Linear


class MaskInnerProduct(BaseSpatialModel):

    def forward(
        self, queries: Float[Tensor, "B Q C"], scene_feats: Geometry
    ) -> List[Float[Tensor, "Q N"]]:
        # BxQxC @ BxNxC = BxQxN
        return [
            queries[b] @ scene_feats.batched_features[b].T for b in range(scene_feats.batch_size)
        ]


class FFNLayer(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Float[Tensor, "B M C"]) -> Float[Tensor, "B M C"]:
        x = x + self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_flash: bool = False,
        use_batched_kv: bool = True,
    ):
        """
        Attention module with optional batched QKV for Muon optimization.

        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for attention scores
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            enable_flash: Whether to use flash attention
            use_batched_kv: If True, uses separate K and V matrices stacked as [2, dim, dim]
                           for Muon optimization. Muon can orthogonalize the [dim, dim] matrices
                           more effectively than the concatenated [dim, 2*dim] matrix.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.enable_flash = enable_flash
        self.use_batched_kv = use_batched_kv

        assert not enable_flash, "Flash attention is not supported for cross attention"
        self.attn_drop = nn.Dropout(attn_drop)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        if use_batched_kv:
            # Use BatchedLinear for Muon-friendly KV projection
            self.kv = BatchedLinear(dim, dim, num_matrices=2, bias=qkv_bias)
        else:
            # Original single linear layer approach
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: Float[Tensor, "B Q C"],  # noqa: F821
        kv: Float[Tensor, "B N C"],  # noqa: F821
        num_points: Int[Tensor, "B"],  # noqa: F821
        pos_enc: Optional[Float[Tensor, "B N C"]] = None,  # noqa: F821
    ) -> Float[Tensor, "B Q C"]:
        B, Q, _ = q.shape
        _, N, _ = kv.shape

        # Compute Q and KV projections
        q = self.q(q)
        kv = self.kv(kv).reshape(B, N, 2, self.dim)

        q = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads)
        kv = rearrange(kv, "b n two (h d) -> two b h n d", h=self.num_heads)
        k, v = (
            kv[0],
            kv[1],
        )

        # Apply positional encoding to the key (non-flash path)
        if pos_enc is not None:
            k = k + pos_enc.unsqueeze(1)

        # Mask out padded positions beyond num_points
        mask = (torch.arange(N, device=q.device)[None, :] >= num_points[:, None]).view(B, 1, 1, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn.masked_fill_(mask, torch.finfo(attn.dtype).min)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, Q, self.dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = zero_out_points(x, num_points)

        return x


class ToAttentionWithoutMask(ToAttention):
    def forward(self, x: Geometry) -> Tuple[
        Float[Tensor, "B M C"],
        Union[Float[Tensor, "B M C"], None],
        Float[Tensor, "B M M"],
        Int[Tensor, "B"],
    ]:
        if self.out_type == "nested":
            features = x.nested_features
            coordinates = x.nested_coordinates
        else:
            features, offsets, num_points = (
                x.features,
                x.offsets,
                x.offsets.diff(),
            )
            features = cat_to_pad_tensor(features, offsets)
            coordinates = x.coordinate_tensor

        if self.use_encoding:
            pos_enc = self.encoding(coordinates)
            pos_enc = cat_to_pad_tensor(pos_enc, offsets)
        else:
            pos_enc = None
        return features, pos_enc, num_points


class SpatialFeatureCrossAttention(CrossAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_encoding_channels: int = 32,
        encoding_range: float = 1.0,
        use_encoding: bool = True,
        enable_flash: bool = False,
        use_batched_kv: bool = True,
        **kwargs,
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_flash=enable_flash,
            use_batched_kv=use_batched_kv,
        )
        self.to_attn = ToAttentionWithoutMask(
            dim,
            use_encoding=use_encoding,
            num_encoding_channels=num_encoding_channels,
            encoding_range=encoding_range,
            num_heads=num_heads,
            concat_input=True,
            num_spatial_features=3,
        )
        self.from_attn = ToSpatialFeatures()

    def forward(self, q: Float[Tensor, "B Q C"], kv: Geometry) -> Float[Tensor, "B Q C"]:
        features, pos_enc, num_points = self.to_attn(kv)
        num_points = num_points.to(kv.device)
        y = super().forward(q, features, num_points, pos_enc)
        return y


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = SpatialFeatureCrossAttention(d_model, nhead)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ffn = FFNLayer(d_model, dim_feedforward, dropout)

    def forward(
        self, queries: Float[Tensor, "B Q C"], scene_feats: Geometry
    ) -> Float[Tensor, "B Q C"]:
        queries = self.self_attn_norm(queries + self.self_attn(queries, queries, queries)[0])
        queries = self.cross_attn_norm(queries + self.cross_attn(queries, scene_feats))
        queries = self.ffn(queries)
        return queries


class MaskTransformer(BaseSpatialModel):
    """
    Transformer that attend query embeddings and queries x scene features.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    hidden_dim,
                    num_heads,
                    dim_feedforward,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(
        self,
        scene_features: Geometry,
    ) -> Float[Tensor, "B Q C"]:
        queries = torch.zeros(
            (scene_features.batch_size, self.num_queries, self.hidden_dim),
            dtype=scene_features.dtype,
            device=scene_features.device,
        )
        for layer in self.layers:
            queries += self.query_embed.weight.unsqueeze(0)
            queries = layer(queries, scene_features)
        return self.norm(queries)


class MaskFormer(BaseSpatialModel):
    def __init__(
        self,
        backbone: BaseSpatialModel,
        hidden_dim: int,
        num_queries: int,
        num_heads: int,
        num_decoders: int,
        dim_feedforward: int,
        dropout: float,
        num_classes: int,
        **kwargs,
    ):
        super().__init__()

        self.backbone = backbone

        self.mask_features_head = Linear(hidden_dim, hidden_dim)

        self.mask_transformer = MaskTransformer(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_decoders,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background

        self.mask_inner_product = MaskInnerProduct()

    def forward(self, x: Points) -> Tuple[Float[Tensor, "B Q C"], List[Float[Tensor, "Q N"]]]:
        # Implement the forward pass
        # This is a placeholder implementation and should be adapted to your needs
        scene_features = self.backbone(x)

        # Apply transformer
        queries = self.mask_transformer(scene_features)  # BxQxC
        logits = self.class_head(queries)

        # Find final mask based on the queries
        mask_features: Points = self.mask_features_head(scene_features)
        masks: List[Float[Tensor, "Q N"]] = self.mask_inner_product(queries, mask_features)

        return logits, masks

    def _downsampled_queries(self, x: Geometry, query_voxel_size: float) -> Geometry:
        """
        To main tain the same density of queries as the input points,
        we downsample the input points using voxel_downsample.
        N points to M queries with lower density. M << N.
        """
        query_indices, query_offsets = voxel_downsample_random_indices(
            x.coordinate_tensor, x.offsets, query_voxel_size
        )
        query_pos = x.coordinate_tensor[query_indices]
        query_features = self.query_projection(self.pos_enc(query_pos))
        return x.replace(
            batched_coordinates=Coords(query_pos, offsets=query_offsets),
            batched_features=CatFeatures(query_features, offsets=query_offsets),
        )


if __name__ == "__main__":
    from warpconvnet.dataset.scannet import ScanNetInstanceDataset
    from warpconvnet.models.mink_unet import MinkUNet18
    from warpconvnet.nn.modules.sparse_pool import PointToSparseWrapper

    wp.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ScanNetInstanceDataset(
        root="~/datasets/scannet_hf",
        split="train",
        label_set="scannet200",
    )

    B = 2
    samples = [dataset[i] for i in range(B)]
    pc = Points.from_list_of_coordinates(
        [torch.from_numpy(s["coords"]).float() for s in samples],
        features=[torch.from_numpy(s["colors"]).float() / 255.0 for s in samples],
    ).to(device)

    backbone = PointToSparseWrapper(
        inner_module=MinkUNet18(in_channels=3, out_channels=96),
        voxel_size=0.02,
        concat_unpooled_pc=False,
    )
    maskformer = MaskFormer(
        backbone=backbone,
        hidden_dim=96,
        num_queries=100,
        num_heads=8,
        num_decoders=6,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=200,
    ).to(device)

    logits, masks = maskformer(pc)
    print(f"logits: {logits.shape}")
    print(f"num masks: {len(masks)} | first mask: {masks[0].shape}")
