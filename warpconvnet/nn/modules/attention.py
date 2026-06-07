# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

try:
    import flash_attn
except ImportError:
    flash_attn = None

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING, encode
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.encodings import SinusoidalEncoding
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.patch import CatPatchFeatures
from warpconvnet.geometry.features.ops.convert import (
    cat_to_pad_tensor,
    pad_to_cat_tensor,
)
from warpconvnet.nn.modules.normalizations import LayerNorm
from warpconvnet.nn.modules.mlp import BatchedLinear
from warpconvnet.nn.modules.rope import VoxelRotaryPositionalEmbeddings


def zero_out_points(
    x: Float[Tensor, "B N C"], num_points: Int[Tensor, "B"]  # noqa: F821
) -> Float[Tensor, "B N C"]:  # noqa: F821
    """
    Zero out the points in the batch.
    """
    for b in range(num_points.shape[0]):
        x[b, num_points[b] :] = 0
    return x


class ZeroOutPoints(nn.Module):
    def forward(
        self, x: Float[Tensor, "B N C"], num_points: Int[Tensor, "B"]  # noqa: F821
    ) -> Float[Tensor, "B N C"]:  # noqa: F821
        return zero_out_points(x, num_points)


def offset_to_mask(
    x: Float[Tensor, "B M C"],  # noqa: F821
    offsets: Float[Tensor, "B+1"],  # noqa: F821
    max_num_points: int,  # noqa: F821
    dtype: torch.dtype = torch.bool,
) -> Float[Tensor, "B 1 M M"]:  # noqa: F821
    """
    Create a mask for the points in the batch.
    """
    B = x.shape[0]
    assert B == offsets.shape[0] - 1
    mask = torch.zeros(
        (B, 1, max_num_points, max_num_points),
        dtype=dtype,
        device=x.device,
    )
    num_points = offsets.diff()
    if dtype == torch.bool:
        for b in range(B):
            # mask[b, :, : num_points[b], : num_points[b]] = True
            mask[b, :, :, : num_points[b]] = True
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mask


class GeometryToPaddedBatch(BaseSpatialModule):
    def __init__(
        self,
        out_channels: int,
        use_encoding: bool = False,
        num_encoding_channels: Optional[int] = None,
        encoding_range: Optional[float] = None,
        num_heads: int = 1,
        concat_input: bool = True,
        num_spatial_features: int = 3,
        out_type: Literal["nested", "cat"] = "cat",
    ):
        super().__init__()
        self.out_type = out_type
        self.use_encoding = use_encoding
        if use_encoding:
            assert num_encoding_channels is not None, "num_encoding_channels must be provided"
            assert encoding_range is not None, "encoding_range must be provided"
            self.encoding = nn.Sequential(
                SinusoidalEncoding(
                    num_channels=num_encoding_channels,
                    data_range=encoding_range,
                    concat_input=concat_input,
                ),
                nn.Linear(
                    num_encoding_channels * num_spatial_features
                    + (num_spatial_features if concat_input else 0),
                    out_channels // num_heads,
                ),
            )

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
        mask = offset_to_mask(features, offsets, features.shape[1])
        return features, pos_enc, mask, num_points


class PaddedBatchToGeometry(nn.Module):
    def forward(self, x: Float[Tensor, "B N C"], target: Geometry) -> Geometry:
        feats = pad_to_cat_tensor(x, target.offsets)
        return target.replace(batched_features=feats)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_flash: bool = True,
        use_batched_qkv: bool = True,
    ):
        """
        Attention module with optional batched QKV for Muon optimization.

        Args:
            dim: Input feature dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for attention scores
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            enable_flash: Whether to use flash attention
            use_batched_qkv: If True, uses separate Q, K, V matrices stacked as [3, dim, dim]
                           for Muon optimization. Muon can orthogonalize the [dim, dim] matrices
                           more effectively than the concatenated [dim, 3*dim] matrix.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.enable_flash = enable_flash
        self.use_batched_qkv = use_batched_qkv

        if enable_flash:
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.attn_drop_p = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)

        if use_batched_qkv:
            # Use BatchedLinear for Muon-friendly QKV projection
            self.qkv = BatchedLinear(dim, dim, num_matrices=3, bias=qkv_bias)
        else:
            # Original single linear layer approach
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Float[Tensor, "B N C"],  # noqa: F821
        pos_enc: Optional[Float[Tensor, "B N C"]] = None,  # noqa: F821
        mask: Optional[Float[Tensor, "B N N"]] = None,  # noqa: F821
        num_points: Optional[Int[Tensor, "B"]] = None,  # noqa: F821
    ) -> Float[Tensor, "B N C"]:
        B, N, C = x.shape

        # Compute QKV with unified approach
        if pos_enc is not None and self.enable_flash:
            # Add positional encoding to input before QKV projection for flash attention
            qkv = self.qkv(x + pos_enc).reshape(B, N, 3, C)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, C)

        # Reshape to [B, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if not self.enable_flash:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

            # Apply positional encoding to the query and key (non-flash path)
            if pos_enc is not None:
                q = q + pos_enc.unsqueeze(1)
                k = k + pos_enc.unsqueeze(1)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                # mask is a bool key-validity mask [B, 1, M, M] (True = valid
                # key). Padded keys must be removed from the softmax, so set
                # their scores to -inf instead of adding the mask (which only
                # shifted valid scores by +1 and left padded keys participating).
                attn = attn.masked_fill(~mask, float("-inf"))
                # An empty batch element (num_points==0) leaves a fully-masked
                # row whose softmax is NaN; those padded-query rows are dropped
                # by zero_out_points anyway, so map NaN->0 to keep grads finite.
                # Only reachable when masking set a row fully to -inf, so the
                # scan is skipped on the unmasked hot path.
                attn = torch.nan_to_num(attn.softmax(dim=-1))
            else:
                attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # Flash attention path
            # Flash attention - preserve original dtype if possible
            original_dtype = qkv.dtype
            if qkv.dtype not in [torch.float16, torch.bfloat16]:
                # Convert to half precision for flash attention
                qkv_flash = qkv.half()
            else:
                qkv_flash = qkv

            if num_points is not None:
                # Padded batch: unpad to a variable-length packed layout so
                # padded tokens never enter the softmax. flash_attn_qkvpacked_func
                # has no mask argument, so masking is impossible on the dense
                # [B, N, ...] path; varlen is the only correct option (same
                # approach as PatchAttention).
                num_points_dev = num_points.to(qkv_flash.device)
                valid = (
                    torch.arange(N, device=qkv_flash.device)[None, :] < num_points_dev[:, None]
                )  # [B, N] bool
                qkv_unpad = qkv_flash[valid]  # [total, 3, num_heads, head_dim]
                # cumsum upcasts int32->int64, and flash requires int32 cu_seqlens,
                # so cast again after the pad.
                cu_seqlens = F.pad(num_points_dev.cumsum(0), (1, 0)).to(torch.int32)
                # N is the pad width (== max num_points), a valid upper bound
                # for max_seqlen. Using it avoids a per-forward device->host
                # sync from num_points.max().
                max_seqlen = N
                out_unpad = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv_unpad,
                    cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.attn_drop_p if self.training else 0.0,
                    softmax_scale=self.scale,
                ).reshape(-1, C)
                x = torch.zeros(B, N, C, dtype=out_unpad.dtype, device=out_unpad.device)
                x[valid] = out_unpad
            else:
                x = flash_attn.flash_attn_qkvpacked_func(
                    qkv_flash,
                    dropout_p=self.attn_drop_p if self.training else 0.0,
                    softmax_scale=self.scale,
                ).reshape(B, N, C)

            # Convert back to original dtype if necessary
            if x.dtype != original_dtype:
                x = x.to(original_dtype)

        x = self.proj(x)
        x = self.proj_drop(x)

        if num_points is not None:
            x = zero_out_points(x, num_points)
        return x


class SpatialFeatureAttention(Attention):
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
        use_encoding: bool = False,
        enable_flash: bool = True,
        use_batched_qkv: bool = True,
        **kwargs,
    ):
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            enable_flash,
            use_batched_qkv,
        )
        self.to_attn = GeometryToPaddedBatch(
            dim,
            use_encoding=use_encoding,
            num_encoding_channels=num_encoding_channels,
            encoding_range=encoding_range,
            num_heads=num_heads,
            concat_input=True,
            num_spatial_features=3,
        )
        self.from_attn = PaddedBatchToGeometry()

    def forward(self, x: Geometry) -> Geometry:
        features, pos_enc, mask, num_points = self.to_attn(x)
        y = super().forward(features, pos_enc, mask, num_points)
        y = self.from_attn(y, x)
        return y


class PatchAttention(BaseSpatialModule):
    def __init__(
        self,
        dim: int,
        patch_size: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        order: POINT_ORDERING = POINT_ORDERING.MORTON_XYZ,
        use_batched_qkv: bool = True,
        use_rope: bool = False,
        rope_base: int = 10_000,
    ):
        """
        Patch attention module with optional batched QKV for Muon optimization.

        Args:
            dim: Input feature dimension
            patch_size: Size of patches for attention computation
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for attention scores
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
            order: Point ordering for patch generation
            use_batched_qkv: If True, uses separate Q, K, V matrices stacked as [3, dim, dim]
                           for Muon optimization. Muon can orthogonalize the [dim, dim] matrices
                           more effectively than the concatenated [dim, 3*dim] matrix.
            use_rope: If True, apply 3D RoPE to Q and K via the fused CUDA
                kernel. Uses point-cloud coordinates for the rotation phase.
            rope_base: RoPE base. Use
                `warpconvnet.nn.modules.rope.suggest_voxel_rope_base` for a
                window-aware default.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.use_batched_qkv = use_batched_qkv

        if use_batched_qkv:
            # Use BatchedLinear for Muon-friendly QKV projection
            self.qkv = BatchedLinear(dim, dim, num_matrices=3, bias=qkv_bias)
        else:
            # Original single linear layer approach
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.order = order
        assert flash_attn is not None, "Make sure flash_attn is installed."
        self.attn_drop_p = attn_drop

        self.use_rope = use_rope
        if use_rope:
            self.rope = VoxelRotaryPositionalEmbeddings(
                dim=dim,
                num_heads=num_heads,
                base=rope_base,
            )

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _offset_to_attn_offset(
        self, offsets: Int[Tensor, "B+1"], patch_size: Optional[int] = None
    ) -> Int[Tensor, "B"]:
        """
        Convert offsets to cumulative attention offsets required for flash attention.
        If the patch size is 8 and the offsets are [0, 3, 11, 40] (3 batches),
        the cumulative attention offsets are [0, 3, 3 + 8 = 11, 11 + 8, 11 + 8 + 8, 11 + 8 + 8 + 8, 40].

        Args:
            offsets: (B+1)
            patch_size: Optional[int]
        Returns:
            cum_seqlens: M
        """
        patch_size = patch_size or self.patch_size
        counts = torch.diff(offsets)

        # Calculate number of patches per batch using ceil division
        num_patches_per_batch = (counts + patch_size - 1) // patch_size

        # Fast path: if no patches, return original offsets
        if num_patches_per_batch.sum() == 0:
            return offsets

        # Calculate how many elements each batch contributes (num_patches)
        # We generate the start indices for each patch. The final end point is added later.
        elements_per_batch = num_patches_per_batch

        # Create indices for which batch each element belongs to
        batch_indices = torch.repeat_interleave(
            torch.arange(len(offsets) - 1, device=offsets.device), elements_per_batch
        )

        # Create indices for position within each batch's sequence (0, 1, 2, ...)
        within_batch_indices = torch.cat(
            [
                torch.arange(n, device=offsets.device, dtype=offsets.dtype)
                for n in num_patches_per_batch
            ]
        )

        # Calculate the actual offsets: start_offset + patch_index * patch_size
        start_offsets = offsets[:-1][batch_indices]
        patch_contributions = within_batch_indices * patch_size
        result_middle = start_offsets + patch_contributions

        # Add the final offset
        result = torch.cat([result_middle, offsets[-1].unsqueeze(0)])

        return result.contiguous()

    def forward(self, x: Geometry, order: Optional[POINT_ORDERING] = None) -> Geometry:
        # Assert that x is serialized
        K = self.patch_size

        feats = x.features
        coords = x.coordinate_tensor
        M, C = feats.shape[:2]
        inverse_perm = None
        order = order or self.order
        if not hasattr(x, "order") or (order != x.order):
            # Generate new ordering and inverse permutation
            code_result = encode(
                x.coordinate_tensor,
                batch_offsets=x.offsets,
                order=order,
                return_perm=True,
                return_inverse=True,
            )
            feats = feats[code_result.perm]
            if self.use_rope:
                coords = coords[code_result.perm]
            inverse_perm = code_result.inverse_perm

        if self.use_rope:
            qkv = self.rope(self.qkv(feats), coords)
        else:
            qkv = self.qkv(feats).reshape(M, 3, self.num_heads, C // self.num_heads)
        if qkv.dtype not in [torch.float16, torch.bfloat16]:
            qkv = qkv.to(torch.float16)

        attn_offsets = self._offset_to_attn_offset(x.offsets, K).to(
            device=qkv.device, dtype=torch.int32
        )
        # Warning: When the loss is NaN, this module will fail during backward with
        # index out of bounds error.
        # e.g. /pytorch/aten/src/ATen/native/cuda/ScatterGatherKernel.cu:144: operator(): block: [192,0,0], thread: [32,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "
        # https://discuss.pytorch.org/t/scattergatherkernel-cu-assertion-idx-dim-0-idx-dim-index-size-index-out-of-bounds/195356
        out_feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv,
            attn_offsets,
            max_seqlen=K,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            softmax_scale=self.scale,
        )
        out_feat = out_feat.reshape(M, C).to(feats.dtype)

        out_feat = self.proj(out_feat)
        out_feat = self.proj_drop(out_feat)

        if inverse_perm is not None:
            out_feat = out_feat[inverse_perm]

        return x.replace(batched_features=out_feat.to(feats.dtype))


class FeedForward(BaseSpatialModule):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(
        self, x: Union[Float[Tensor, "B N D"], Geometry]
    ) -> Union[Float[Tensor, "B N D"], Geometry]:
        feat = x.features if isinstance(x, Geometry) else x
        # Apply feed forward
        feat = self.w2(F.silu(self.w1(feat)) * self.w3(feat))
        # Return based on the type of x
        return x.replace(batched_features=feat) if isinstance(x, Geometry) else feat


class TransformerBlock(BaseSpatialModule):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_multiplier: float = 4.0,
        ffn_multiple_of: int = 32,
        norm_eps: float = 1e-5,
        attn_fn: Optional[Callable[..., nn.Module]] = None,
        norm_fn: Optional[Callable[..., nn.Module]] = LayerNorm,
        use_batched_qkv: bool = True,
    ):
        super().__init__()
        if attn_fn is None:
            attn_fn = functools.partial(PatchAttention, patch_size=1024)
        self.dim = dim
        self.attention = attn_fn(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_batched_qkv=use_batched_qkv,
        )
        # Even hidden dimension
        hidden_dim = int(
            (ffn_multiplier * dim + ffn_multiple_of - 1) // ffn_multiple_of * ffn_multiple_of
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
        )
        self.attention_norm = norm_fn(dim, eps=norm_eps)
        self.ffn_norm = norm_fn(dim, eps=norm_eps)

    def forward(
        self,
        x: Geometry,
        *args: Any,
        **kwargs: Any,
    ) -> Geometry:
        h = x + self.attention(self.attention_norm(x), *args, **kwargs)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
