# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpaCeFormerInstSeg: a proposal-free mask-query decoder for sparse point backbones.

A Mask2Former-style query decoder (learned queries + 3D rotary position
embeddings) that runs on top of a WarpConvNet point backbone such as
``warpconvnet.models.spaceformer.SpaCeFormer``. A single forward pass over an
RGB point cloud produces, for a fixed set of learned queries, an objectness
logit, a per-point mask, and a per-query embedding (e.g. a CLIP feature for
open-vocabulary labeling).

Like ``maskformer.MaskFormer``, this module returns **raw** predictions
(``{logit, mask, clip_feat}``). Turning those into labeled, de-duplicated
instances (mask NMS/DBSCAN, CLIP/text-embedding labeling, dataset label sets) is
downstream, task-specific code and is intentionally kept out of this library.

``build_spaceformer`` constructs the released architecture; the decoder layers,
3D RoPE, query MLP and the (checkpoint-compat) positional-encoding buffer are all
inlined below to keep the model self-contained.
"""

import logging
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.types.points import Points
from warpconvnet.nn.modules.sparse_pool import PointToSparseWrapper

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 3D rotary positional embeddings (decoder variant: [N, C] -> [N, C])
# --------------------------------------------------------------------------- #
class VoxelRotaryPositionalEmbeddings(nn.Module):
    """3D RoPE applied to flat ``[N, C]`` features given ``[N, 3]`` coordinates.

    RoPE is applied to the largest per-head multiple of 6 (x/y/z each take a
    third); any remaining head dims pass through unchanged. Carries no learnable
    parameters unless ``learnable_theta=True``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        base: int = 10_000,
        learnable_theta: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_dim = (self.head_dim // 6) * 6
        self.pass_dim = self.head_dim - self.rope_dim
        if self.rope_dim > 0:
            theta = 1.0 / (
                base ** (torch.arange(0, self.rope_dim // 3, 2).float() / (self.rope_dim // 3))
            )
            if learnable_theta:
                self.theta = nn.Parameter(theta)
            else:
                self.register_buffer("theta", theta, persistent=False)
        else:
            self.theta = None

    def forward(
        self,
        x: Float[Tensor, "N C"],  # noqa: F821, F722
        coords: Float[Tensor, "N 3"],  # noqa: F821, F722
        **kwargs: Any,
    ) -> Float[Tensor, "N C"]:  # noqa: F821, F722
        n, c = x.shape
        if c != self.num_heads * self.head_dim:
            raise ValueError(
                f"feature dim {c} != num_heads*head_dim {self.num_heads*self.head_dim}"
            )
        if self.rope_dim == 0:
            return x

        x = x.view(n, self.num_heads, self.head_dim)
        if self.pass_dim > 0:
            x_rope, x_pass = x[..., : self.rope_dim], x[..., self.rope_dim :]
        else:
            x_rope, x_pass = x, None

        float_coords = coords.float() - coords.min(dim=0).values
        x_freqs = torch.outer(float_coords[:, 0] + 1, self.theta)
        y_freqs = torch.outer(float_coords[:, 1] + 1, self.theta)
        z_freqs = torch.outer(float_coords[:, 2] + 1, self.theta)
        freqs = torch.cat([x_freqs, y_freqs, z_freqs], dim=-1)
        cos_f, sin_f = torch.cos(freqs).unsqueeze(1), torch.sin(freqs).unsqueeze(1)

        x_rope_shaped = x_rope.float().reshape(n, self.num_heads, self.rope_dim // 2, 2)
        x_rope_out = torch.stack(
            [
                x_rope_shaped[..., 0] * cos_f - x_rope_shaped[..., 1] * sin_f,
                x_rope_shaped[..., 0] * sin_f + x_rope_shaped[..., 1] * cos_f,
            ],
            dim=-1,
        ).reshape(n, self.num_heads, self.rope_dim)

        out = x_rope_out if x_pass is None else torch.cat([x_rope_out, x_pass], dim=-1)
        return out.reshape(n, c).type_as(x)


# --------------------------------------------------------------------------- #
# Mask2Former-style decoder layers (no MinkowskiEngine dependency)
# --------------------------------------------------------------------------- #
def _activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(
                q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
            )[0]
            return tgt + self.dropout(tgt2)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        return self.norm(tgt + self.dropout(tgt2))


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None
    ):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            return tgt + self.dropout(tgt2)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        return self.norm(tgt + self.dropout(tgt2))


class FFNLayer(nn.Module):
    def __init__(
        self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            return tgt + self.dropout(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm(tgt + self.dropout(tgt2))


class _QueryMLP(nn.Module):
    """Two-layer MLP for the query projection (Linear-ReLU-Linear-ReLU).

    Kept as ``self.layers`` (a Sequential) so its parameter names match the
    released checkpoint (``query_proj.layers.0/2``).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class _PosEncBuffer(nn.Module):
    """Holds the ``gauss_B`` buffer present in the released checkpoint.

    The learned-query decoder does not use coordinate positional encodings, so
    this module is intentionally inert — it exists only so the released
    ``state_dict`` loads with no missing/unexpected keys.
    """

    def __init__(self, d_pos: int, d_in: int = 3):
        super().__init__()
        self.register_buffer("gauss_B", torch.zeros(d_in, d_pos // 2))


# --------------------------------------------------------------------------- #
# SpaCeFormerInstSeg model
# --------------------------------------------------------------------------- #
class SpaCeFormerInstSeg(nn.Module):
    """Proposal-free mask-query decoder (learned queries + RoPE).

    Args:
        backbone: a WarpConvNet point backbone (e.g. ``SpaCeFormer``); wrapped
            internally in ``PointToSparseWrapper`` (voxelization at ``voxel_size``).
        hidden_dim / feedforward_dim / num_heads / decoder_iterations: decoder size.
        clip_dim: per-query embedding dim (== backbone ``out_channels``).
        num_queries: number of learned object queries.
        max_sample_size: max points used as attention keys per step (train cap).
        voxel_size / concat_unpooled_pc: passed to ``PointToSparseWrapper``.
        freeze_backbone: run the backbone under ``no_grad`` and in eval.
        rope_base / rope_learnable_theta: decoder RoPE controls.
        use_iou_head: build a learned per-query IoU head (off by default).

    Forward returns raw predictions:
        ``{"logit": [B,Q,2], "mask": List[[N,Q]], "clip_feat": [B,Q,clip_dim]}``
        (plus ``pred_iou`` if an IoU head is built, and ``backbone_pc`` in eval).
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        feedforward_dim: int,
        clip_dim: int,
        num_queries: int,
        num_heads: int,
        decoder_iterations: int,
        max_sample_size: int = 51200,
        voxel_size: float = 0.02,
        concat_unpooled_pc: bool = False,
        freeze_backbone: bool = False,
        rope_base: int = 2000,
        rope_learnable_theta: bool = False,
        use_iou_head: bool = False,
    ):
        super().__init__()
        self.backbone = PointToSparseWrapper(
            backbone, voxel_size=voxel_size, concat_unpooled_pc=concat_unpooled_pc
        )
        self.hidden_dim = hidden_dim
        self.clip_dim = clip_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.decoder_iterations = decoder_iterations
        self.max_sample_size = max_sample_size

        backbone_out_dim = getattr(backbone, "out_channels", clip_dim)

        self.decoder_proj = nn.Linear(backbone_out_dim, hidden_dim)
        self.query_proj = _QueryMLP(hidden_dim)
        self.pos_enc = _PosEncBuffer(hidden_dim)  # checkpoint-compat buffer; unused in forward
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(backbone_out_dim, hidden_dim)
        self.cross_attention = CrossAttentionLayer(d_model=hidden_dim, nhead=num_heads)
        self.self_attention = SelfAttentionLayer(d_model=hidden_dim, nhead=num_heads)
        self.ffn = FFNLayer(d_model=hidden_dim, dim_feedforward=feedforward_dim)

        self.class_head = nn.Linear(hidden_dim, 2)  # objectness over {fg, bg}
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.clip_head = nn.Linear(hidden_dim, clip_dim)
        if use_iou_head:
            self.iou_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.iou_head = None

        self.learned_content = nn.Embedding(num_queries, hidden_dim)
        self.learned_pos = nn.Embedding(num_queries, hidden_dim)
        self.rope_cross = VoxelRotaryPositionalEmbeddings(
            dim=hidden_dim,
            num_heads=num_heads,
            base=rope_base,
            learnable_theta=rope_learnable_theta,
        )
        self.rope_self = VoxelRotaryPositionalEmbeddings(
            dim=hidden_dim,
            num_heads=num_heads,
            base=rope_base,
            learnable_theta=rope_learnable_theta,
        )

        self.b2q = Rearrange("b q d -> q b d")
        self.q2b = Rearrange("q b d -> b q d")

        self.backbone_frozen = False
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone_frozen = True
            log.info("Backbone frozen")

    def forward(self, data_dict: dict) -> dict:
        (
            out_pc,
            point_features,
            point_coords,
            offsets,
            decomposed_pfeats,
            batch_size,
            device,
            _dtype,
        ) = self._extract_backbone_features(data_dict)

        # Learned queries (modular indexing supports eval-time num_queries != trained).
        num_queries = int(data_dict.get("num_queries", self.num_queries))
        query_ids = torch.arange(num_queries, device=device, dtype=torch.long) % self.num_queries
        queries = self.learned_content(query_ids).unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.learned_pos(query_ids).unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.query_proj(query_pos)
        queries = queries + query_pos  # RoPE fuses position; no separate query_pos in attention

        # Learned queries have no physical location -> zero coords (keys still rotate).
        query_coords = torch.zeros(
            (batch_size * num_queries, 3), device=device, dtype=point_coords.dtype
        )

        for _ in range(self.decoder_iterations):
            attn_mask = self.attention_mask_module(queries, decomposed_pfeats)
            global_idx, pad_mask, sample_size = self._sample_key_indices(offsets, device)
            flat_idx = global_idx.reshape(-1)
            batched_fpn = point_features[flat_idx].reshape(batch_size, sample_size, -1)
            batched_attn = attn_mask[flat_idx].reshape(batch_size, sample_size, -1)
            batched_coords = point_coords[flat_idx].reshape(batch_size, sample_size, -1)

            batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == sample_size] = False
            batched_attn = torch.logical_or(batched_attn, pad_mask[..., None])
            batched_fpn = self.linear(batched_fpn)

            q_rope = self.rope_cross(rearrange(queries, "b q d -> (b q) d"), query_coords)
            k_rope = self.rope_cross(
                rearrange(batched_fpn, "b n d -> (b n) d"),
                rearrange(batched_coords, "b n c -> (b n) c"),
            )
            q_rope = rearrange(q_rope, "(b q) d -> q b d", b=batch_size, q=num_queries)
            k_rope = rearrange(k_rope, "(b n) d -> n b d", b=batch_size, n=sample_size)

            batched_attn = repeat(batched_attn, "b q n -> (b h) n q", h=self.num_heads)
            queries = self.cross_attention(
                q_rope,
                k_rope,
                memory_mask=batched_attn,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None,
            )
            qs_rope = self.rope_self(rearrange(queries, "q b d -> (b q) d"), query_coords)
            qs_rope = rearrange(qs_rope, "(b q) d -> q b d", b=batch_size, q=num_queries)
            queries = self.self_attention(qs_rope, query_pos=None)
            queries = self.ffn(queries)
            queries = self.q2b(queries)

        pred_classes, pred_masks, pred_clip_feats = self.mask_module(queries, decomposed_pfeats)
        result = dict(logit=pred_classes, mask=pred_masks, clip_feat=pred_clip_feats)
        if self.iou_head is not None:
            result["pred_iou"] = self.iou_head(self.decoder_norm(queries)).squeeze(-1)
        if not self.training:
            # Eval only: keeping Points alive during training leaks the geometry cache.
            result["backbone_pc"] = out_pc
        return result

    # ------------------------------------------------------------------ helpers
    def _extract_backbone_features(self, data_dict: dict):
        if "pc" not in data_dict:
            pc = Points(
                batched_coordinates=data_dict["coord"],
                batched_features=data_dict["feat"],
                offsets=data_dict["offset"],
            )
        else:
            pc = data_dict["pc"]

        with torch.no_grad() if self.backbone_frozen else nullcontext():
            if self.backbone_frozen and self.backbone.training:
                self.backbone.eval()
            out_pc = self.backbone(pc)

        point_features = out_pc.features
        point_coords = out_pc.coordinates
        offsets = out_pc.offsets
        batch_size = len(offsets) - 1
        device, dtype = point_features.device, point_features.dtype

        point_features_proj = self.decoder_proj(point_features)
        decomposed_pfeats = [
            point_features_proj[offsets[i] : offsets[i + 1]] for i in range(batch_size)
        ]
        return (
            out_pc,
            point_features,
            point_coords,
            offsets,
            decomposed_pfeats,
            batch_size,
            device,
            dtype,
        )

    def _sample_key_indices(self, offsets: torch.Tensor, device: torch.device):
        counts = (offsets[1:] - offsets[:-1]).to(device)
        if torch.min(counts).item() == 1:
            raise RuntimeError("only a single point gives nans in cross-attention")
        sample_size = int(torch.max(counts).item())
        if self.training:
            sample_size = min(sample_size, self.max_sample_size)
        max_count = int(torch.max(counts).item())
        batch_size = len(offsets) - 1

        base_local = torch.arange(max_count, device=device).unsqueeze(0).expand(batch_size, -1)
        valid_local = base_local < counts[:, None]
        rand_scores = torch.rand(batch_size, max_count, device=device)
        rand_scores = torch.where(
            valid_local, rand_scores, torch.full_like(rand_scores, float("-inf"))
        )
        topk_local = torch.topk(rand_scores, k=sample_size, dim=1).indices
        pad_mask = topk_local >= counts[:, None]
        topk_local = torch.where(pad_mask, torch.zeros_like(topk_local), topk_local)
        global_idx = topk_local + offsets[:-1].to(device)[:, None]
        return global_idx, pad_mask, sample_size

    def attention_mask_module(self, queries, decomposed_point_feats: List[torch.Tensor]):
        queries = self.decoder_norm(queries)
        mask_embeds = self.mask_head(queries)
        pred_masks = [pf @ me.T for pf, me in zip(decomposed_point_feats, mask_embeds)]
        return (torch.vstack(pred_masks) < 0).detach()

    def mask_module(self, queries, decomposed_point_feats: List[torch.Tensor]):
        queries = self.decoder_norm(queries)
        pred_classes = self.class_head(queries)
        mask_embeds = self.mask_head(queries)
        pred_masks = [pf @ me.T for pf, me in zip(decomposed_point_feats, mask_embeds)]
        pred_clip_feats = self.clip_head(queries)
        return pred_classes, pred_masks, pred_clip_feats


# --------------------------------------------------------------------------- #
# Builders for the released open-vocabulary checkpoint
# --------------------------------------------------------------------------- #
GRID_SIZE = 0.02
CLIP_TEXT_DIM = 1152  # SigLIP2 so400m text dim == clip-head output dim
NUM_QUERIES = 200
HIDDEN_DIM = 512


def build_backbone(out_channels: int = CLIP_TEXT_DIM) -> nn.Module:
    """Build the ``SpaCeFormer`` point backbone used by the released checkpoint."""
    from .space_former import SpaCeFormer

    return SpaCeFormer(
        in_channels=3,
        out_channels=out_channels,
        block_type="stream_norm",
        enc_channels=[32, 64, 128, 256, 512],
        enc_num_head=[2, 4, 8, 16, 32],
        enc_patch_size=[64, 48, 1024, 1024, 1024],
        enc_depths=[2, 2, 2, 6, 2],
        enc_attn_types="ssccc",
        dec_channels=[512, 384, 256, 256],
        dec_num_head=[8, 8, 8, 16],
        dec_patch_size=[64, 48, 1024, 1024],
        dec_depths=[2, 2, 2, 2],
        dec_attn_types="sscc",
        kernel_size=3,
        voxel_offsets=["zero", "xy", "xz", "yz", "xyz"],
        patch_orders=[
            "random",
            "morton_xyz",
            "morton_xzy",
            "morton_yxz",
            "morton_yzx",
            "morton_zxy",
            "morton_zyx",
        ],
        mlp_ratio=2.0,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.2,
        shuffle_orders=True,
        use_rope=True,
        rope_base=2000,
        enc_rope_bases=[250, 250, 250, 250, 250],
        dec_rope_bases=[250, 250, 250, 250],
    )


def build_spaceformer(
    use_iou_head: bool = False,
    device: Optional[torch.device] = None,
) -> SpaCeFormerInstSeg:
    """Build the released proposal-free SpaCeFormerInstSeg (learned queries + RoPE)."""
    net = SpaCeFormerInstSeg(
        backbone=build_backbone(),
        hidden_dim=HIDDEN_DIM,
        feedforward_dim=1024,
        clip_dim=CLIP_TEXT_DIM,
        num_queries=NUM_QUERIES,
        num_heads=8,
        decoder_iterations=3,
        max_sample_size=51200,
        voxel_size=GRID_SIZE,
        concat_unpooled_pc=False,
        freeze_backbone=False,
        rope_base=2000,
        rope_learnable_theta=False,
        use_iou_head=use_iou_head,
    )
    net.eval()
    if device is not None:
        net = net.to(device)
    return net


def load_spaceformer_checkpoint(
    net: nn.Module,
    ckpt_path: str,
    strict: bool = False,
) -> Tuple[list, list]:
    """Load a released weights-only checkpoint onto ``net``.

    The checkpoint is a Lightning ``state_dict`` whose keys are prefixed with
    ``net.`` (plus an unrelated ``caption_loss.logit_scale``); we strip the prefix
    and load with ``strict=False``. Returns ``(missing_keys, unexpected_keys)``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    net_state = {k[len("net.") :]: v for k, v in state_dict.items() if k.startswith("net.")}
    missing, unexpected = net.load_state_dict(net_state, strict=strict)
    log.info(
        "loaded %s: %d tensors -> %d missing, %d unexpected",
        ckpt_path,
        len(net_state),
        len(missing),
        len(unexpected),
    )
    return missing, unexpected
