# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Neural network building blocks.

Defining this module explicitly keeps documentation tooling from relying on
implicit namespace packages, which mkdocstrings cannot traverse.
"""

from warpconvnet.nn.modules.conv3d_blocks import (
    DownsampleBlock3d,
    ResBlock3d,
    UpsampleBlock3d,
    norm_layer_3d,
)
from warpconvnet.nn.modules.dit import (
    FeedForwardNet,
    ModulatedTransformerBlock,
    ModulatedTransformerCrossBlock,
    MultiHeadAttention,
)
from warpconvnet.nn.modules.embeddings import (
    RotaryPositionEmbedder,
    SinusoidalPositionEmbedder,
    TimestepEmbedder,
)
from warpconvnet.nn.modules.sparse_convnext import SparseConvNeXtBlock3d
from warpconvnet.nn.modules.sparse_dit import (
    ModulatedSparseTransformerBlock,
    ModulatedSparseTransformerCrossBlock,
    SparseFeedForwardNet,
)
from warpconvnet.nn.modules.sparse_dit_attention import (
    SparseMultiHeadAttention,
    SparseRotaryPositionEmbedder,
    sparse_scaled_dot_product_attention,
)
from warpconvnet.nn.modules.sparse_resample import (
    SparseChannel2Spatial,
    SparseDownsample,
    SparseSpatial2Channel,
    SparseSubdivide,
    SparseUpsample,
)
from warpconvnet.nn.modules.sparse_unet import SparseChannelToSpatialResBlock3d
from warpconvnet.nn.modules.sparse_unet import SparseUNetDecoderStages

__all__ = [
    "DownsampleBlock3d",
    "FeedForwardNet",
    "ModulatedSparseTransformerBlock",
    "ModulatedSparseTransformerCrossBlock",
    "ModulatedTransformerBlock",
    "ModulatedTransformerCrossBlock",
    "MultiHeadAttention",
    "ResBlock3d",
    "RotaryPositionEmbedder",
    "SinusoidalPositionEmbedder",
    "SparseChannel2Spatial",
    "SparseChannelToSpatialResBlock3d",
    "SparseConvNeXtBlock3d",
    "SparseDownsample",
    "SparseFeedForwardNet",
    "SparseMultiHeadAttention",
    "SparseRotaryPositionEmbedder",
    "SparseSpatial2Channel",
    "SparseSubdivide",
    "SparseUNetDecoderStages",
    "SparseUpsample",
    "TimestepEmbedder",
    "UpsampleBlock3d",
    "norm_layer_3d",
    "sparse_scaled_dot_product_attention",
]
