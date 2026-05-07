# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 sparse U-Net VAE decoder + FlexiDualGrid mesh-attr head.

Decoder side only (encoder skipped for inference). Module names mirror
`trellis2.models.sc_vaes.{sparse_unet_vae,fdg_vae}`, while sparse-conv
checkpoint tensors are converted to WarpConvNet's native weight layout by
``load_trellis2_state_dict``.

Mesh extraction (`flexible_dual_grid_to_mesh`) is intentionally deferred to
Phase 10 since it needs the o-voxel CUDA extension. `FlexiDualGridVaeDecoder`
returns the raw 7-channel attrs (vertices, intersected, quad-lerp) and the
caller can pipe them to o-voxel later.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_unet import (
    SparseChannelToSpatialResBlock3d,
    SparseUNetDecoderStages,
)
from warpconvnet.nn.utils import DEFAULT_MIXED_PRECISION_MODULES, convert_module_parameters_to

from .sparse_conv_blocks import SparseConv3d, SparseConvNeXtBlock3d


__all__ = [
    "FlexiDualGridVaeDecoder",
    "SparseResBlockC2S3d",
    "SparseUnetVaeDecoder",
    "convert_trellis2_shape_vae_state_dict",
]


_MIX_PRECISION_MODULES = DEFAULT_MIXED_PRECISION_MODULES + (SparseConv3d,)


def _convert_module_to_f16(m: nn.Module) -> None:
    convert_module_parameters_to(m, torch.float16, module_types=_MIX_PRECISION_MODULES)


def _convert_module_to_f32(m: nn.Module) -> None:
    convert_module_parameters_to(m, torch.float32, module_types=_MIX_PRECISION_MODULES)


def _convert_sparse_conv_weight_to_warpconvnet(weight: torch.Tensor) -> torch.Tensor:
    """Convert upstream shape-decoder sparse-conv weights to WarpConvNet layout.

    Upstream shape: ``(Cout, Kd, Kh, Kw, Cin)``.
    WarpConvNet shape: ``(Kd * Kh * Kw, Cin, Cout)``.
    """
    if weight.ndim != 5:
        return weight
    Cout, Kd, Kh, Kw, Cin = weight.shape
    return weight.permute(1, 2, 3, 4, 0).reshape(Kd * Kh * Kw, Cin, Cout).contiguous()


def convert_trellis2_shape_vae_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a published TRELLIS.2 shape-decoder state dict for this model.

    TRELLIS.2 stores sparse-convolution weights as ``(Cout, Kd, Kh, Kw, Cin)``.
    The WarpConvNet modules in this port use native ``SparseConv3d`` weights,
    ``(K^3, Cin, Cout)``. This function rewrites only keys whose target module
    expects a 3D sparse-conv weight and whose source tensor is 5D.
    """
    expected = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        target = expected.get(key)
        if key.endswith(".weight") and target is not None and target.ndim == 3 and value.ndim == 5:
            converted[key] = _convert_sparse_conv_weight_to_warpconvnet(value)
        else:
            converted[key] = value
    return converted


# -----------------------------------------------------------------------------
# Up-block: SparseResBlockC2S3d
# -----------------------------------------------------------------------------
class SparseResBlockC2S3d(SparseChannelToSpatialResBlock3d):
    """Channel-to-spatial residual upsample block.

    Layout (per upstream
    `trellis2.models.sc_vaes.sparse_unet_vae.SparseResBlockC2S3d`):
        ``conv1`` :  ``channels → out_channels * 8``  (still pre-C2S)
        ``updown``: SparseChannel2Spatial(2)         (factor=2 in 3D ⇒ ×8 voxels)
        ``conv2`` : ``out_channels → out_channels`` (zero-init)
        skip     : repeat_interleave broadcast on the C2S-spread x
        ``to_subdiv`` (optional): predicts the 8-bit subdivision mask used by
            C2S to materialise child voxels.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__(
            channels=channels,
            out_channels=out_channels,
            factor=2,
            use_checkpoint=use_checkpoint,
            pred_subdiv=pred_subdiv,
            conv_cls=SparseConv3d,
        )


# -----------------------------------------------------------------------------
# SparseUnetVaeDecoder
# -----------------------------------------------------------------------------
_BLOCK_REGISTRY = {
    "SparseConvNeXtBlock3d": SparseConvNeXtBlock3d,
    "SparseResBlockC2S3d": SparseResBlockC2S3d,
}


class SparseUnetVaeDecoder(nn.Module):
    """Sparse U-Net decoder used by the TRELLIS.2 shape VAE.

    Channels go from ``model_channels[0]`` (latent-side) up to
    ``model_channels[-1]`` (output-side); each resolution block applies
    ``num_blocks[i]`` ``block_type[i]`` modules followed by an
    ``up_block_type[i]`` (between resolutions).
    """

    def __init__(
        self,
        out_channels: int,
        model_channels: list[int],
        latent_channels: int,
        num_blocks: list[int],
        block_type: list[str],
        up_block_type: list[str],
        block_args: list[dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.from_latent = nn.Linear(latent_channels, model_channels[0])
        self.output_layer = nn.Linear(model_channels[-1], out_channels)

        self.blocks = SparseUNetDecoderStages(
            model_channels=model_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            block_args=block_args,
            block_registry=_BLOCK_REGISTRY,
            up_block_kwargs={"pred_subdiv": pred_subdiv},
        )

        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(_convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(_convert_module_to_f32)

    def load_trellis2_state_dict(self, state_dict: dict[str, torch.Tensor], *args, **kwargs):
        """Load a published TRELLIS.2 shape-decoder checkpoint.

        Sparse-conv weights are converted to native WarpConvNet layout before
        delegating to ``load_state_dict``.
        """
        converted = convert_trellis2_shape_vae_state_dict(state_dict, self)
        return self.load_state_dict(converted, *args, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def upsample(self, x: Voxels, upsample_times: int) -> torch.Tensor:
        """Run only the first ``upsample_times`` C2S stages, return coords.

        Used by `sample_shape_slat_cascade` to derive a high-resolution voxel
        grid from a low-resolution latent without paying for the rest of the
        decoder. Output is the upstream-style ``(N, 1+DIM)`` batch-indexed
        coord tensor of the up-sampled voxels.
        """
        assert self.pred_subdiv, "upsample() requires pred_subdiv=True"
        h = x.replace_features(self.from_latent(x.feats))
        h = h.replace_features(h.feats.to(self.dtype))
        h = self.blocks.run(h, stop_before_stage=upsample_times)
        return h.coords

    def forward(
        self,
        x: Voxels,
        guide_subs: list[Voxels] | None = None,
        return_subs: bool = False,
    ):
        if guide_subs is not None:
            assert not self.pred_subdiv, "guide_subs only valid when pred_subdiv=False"
        if return_subs:
            assert self.pred_subdiv, "return_subs only valid when pred_subdiv=True"

        h = x.replace_features(self.from_latent(x.feats))
        h = h.replace_features(h.feats.to(self.dtype))

        out = self.blocks.run(h, guide_subs=guide_subs, return_subs=return_subs)
        if return_subs:
            h, subs = out
        else:
            h = out

        h = h.replace_features(h.feats.to(x.dtype))
        h = h.replace_features(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = h.replace_features(self.output_layer(h.feats))
        if return_subs:
            return h, subs
        return h


# -----------------------------------------------------------------------------
# FlexiDualGridVaeDecoder
# -----------------------------------------------------------------------------
class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    """Shape VAE decoder producing 7-channel mesh attributes per voxel.

    Channel layout (per upstream `trellis2.models.sc_vaes.fdg_vae`):
        - ``[..., 0:3]`` : sigmoid-decoded vertex offsets in
          ``[-voxel_margin, 1+voxel_margin]``.
        - ``[..., 3:6]`` : per-edge intersection logits (binarised at >0
          for inference).
        - ``[..., 6:7]`` : softplus-decoded quad-lerp weight.

    The actual mesh extraction step
    (``o_voxel.convert.flexible_dual_grid_to_mesh``) is intentionally not
    invoked here so this module stays free of the o-voxel CUDA dependency.
    Returns the three attribute Voxels; downstream callers pipe them into
    the mesh extractor (Phase 10).
    """

    def __init__(
        self,
        resolution: int,
        model_channels: list[int],
        latent_channels: int,
        num_blocks: list[int],
        block_type: list[str],
        up_block_type: list[str],
        block_args: list[dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        super().__init__(
            out_channels=7,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            block_args=block_args,
            use_fp16=use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def decode_attrs(self, h: Voxels) -> tuple[Voxels, Voxels, Voxels]:
        """Split the 7-channel decoder output into mesh attribute Voxels."""
        margin = self.voxel_margin
        feats = h.feats
        vertices = h.replace_features((1 + 2 * margin) * F.sigmoid(feats[..., 0:3]) - margin)
        intersected = h.replace_features(feats[..., 3:6] > 0)
        quad_lerp = h.replace_features(F.softplus(feats[..., 6:7]))
        return vertices, intersected, quad_lerp

    def forward(self, x: Voxels, **kwargs) -> tuple[Voxels, Voxels, Voxels]:
        decoded = super().forward(x, **kwargs)
        h = decoded[0] if isinstance(decoded, tuple) else decoded
        return self.decode_attrs(h)
