# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 image-to-mesh inference, ported to warpconvnet.

Source: https://github.com/microsoft/TRELLIS.2 (microsoft/TRELLIS.2-4B on HF Hub).
Texture flow + texture VAE are not ported; this module provides shape-only image→mesh.
"""

from .blocks_dense import (
    AbsolutePositionEmbedder,
    FeedForwardNet,
    LayerNorm32,
    ModulatedTransformerCrossBlock,
    MultiHeadAttention,
    MultiHeadRMSNorm,
    RotaryPositionEmbedder,
    TimestepEmbedder,
)
from .image_cond import DinoV3FeatureExtractor
from .samplers import (
    FlowEulerCfgSampler,
    FlowEulerGuidanceIntervalSampler,
    FlowEulerSampler,
    Sampler,
)
from .sparse_spatial import (
    SparseChannel2Spatial,
    SparseDownsample,
    SparseSpatial2Channel,
    SparseSubdivide,
    SparseUpsample,
)
from .sparse_structure_flow import SparseStructureFlowModel
from .sparse_structure_vae import (
    DownsampleBlock3d,
    ResBlock3d,
    SparseStructureDecoder,
    SparseStructureEncoder,
    UpsampleBlock3d,
)
from .sparse_ops import (
    from_feats_coords,
    get_scale,
    set_scale,
    sparse_cat,
    sparse_unbind,
)

__all__ = [
    "AbsolutePositionEmbedder",
    "DinoV3FeatureExtractor",
    "DownsampleBlock3d",
    "FeedForwardNet",
    "FlowEulerCfgSampler",
    "FlowEulerGuidanceIntervalSampler",
    "FlowEulerSampler",
    "LayerNorm32",
    "ModulatedTransformerCrossBlock",
    "MultiHeadAttention",
    "MultiHeadRMSNorm",
    "ResBlock3d",
    "RotaryPositionEmbedder",
    "Sampler",
    "SparseChannel2Spatial",
    "SparseDownsample",
    "SparseSpatial2Channel",
    "SparseStructureDecoder",
    "SparseStructureEncoder",
    "SparseStructureFlowModel",
    "SparseSubdivide",
    "SparseUpsample",
    "TimestepEmbedder",
    "UpsampleBlock3d",
    "from_feats_coords",
    "get_scale",
    "set_scale",
    "sparse_cat",
    "sparse_unbind",
]
