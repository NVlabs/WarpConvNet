# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim — sparse attention promoted to ``warpconvnet.nn.modules``."""

from warpconvnet.nn.modules.sparse_dit_attention import (
    SparseMultiHeadAttention,
    SparseRotaryPositionEmbedder,
    sparse_scaled_dot_product_attention,
)

__all__ = [
    "SparseMultiHeadAttention",
    "SparseRotaryPositionEmbedder",
    "sparse_scaled_dot_product_attention",
]
