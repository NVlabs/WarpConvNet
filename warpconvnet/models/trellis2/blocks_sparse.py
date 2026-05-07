# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim — sparse DiT blocks promoted to ``warpconvnet.nn.modules``."""

from warpconvnet.nn.modules.sparse_dit import (
    ModulatedSparseTransformerBlock,
    ModulatedSparseTransformerCrossBlock,
    SparseFeedForwardNet,
)

__all__ = [
    "ModulatedSparseTransformerBlock",
    "ModulatedSparseTransformerCrossBlock",
    "SparseFeedForwardNet",
]
