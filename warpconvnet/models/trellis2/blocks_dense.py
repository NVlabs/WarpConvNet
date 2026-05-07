# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim for TRELLIS.2 dense transformer blocks.

The actual implementations were promoted into the broader warpconvnet library
so other diffusion / flow models can reuse them. This module is kept as a
backwards-compatible alias surface for the rest of the trellis2 port.
"""
from __future__ import annotations

import torch

from warpconvnet.nn.modules.dit import (
    FeedForwardNet,
    ModulatedTransformerCrossBlock,
    MultiHeadAttention,
)
from warpconvnet.nn.modules.embeddings import (
    RotaryPositionEmbedder,
    SinusoidalPositionEmbedder as AbsolutePositionEmbedder,
    TimestepEmbedder,
)
from warpconvnet.nn.modules.normalizations import LayerNorm32, MultiHeadRMSNorm
from warpconvnet.nn.utils import manual_cast, str_to_dtype


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


__all__ = [
    "AbsolutePositionEmbedder",
    "FeedForwardNet",
    "LayerNorm32",
    "ModulatedTransformerCrossBlock",
    "MultiHeadAttention",
    "MultiHeadRMSNorm",
    "RotaryPositionEmbedder",
    "TimestepEmbedder",
    "manual_cast",
    "modulate",
    "str_to_dtype",
]
