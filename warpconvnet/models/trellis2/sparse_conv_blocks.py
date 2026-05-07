# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse 3D conv + ConvNeXt building blocks for the TRELLIS.2 SLAT VAE."""
from __future__ import annotations

from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.sparse_convnext import SparseConvNeXtBlock3d as _SparseConvNeXtBlock3d


__all__ = ["SparseConv3d", "SparseConvNeXtBlock3d"]


class SparseConvNeXtBlock3d(_SparseConvNeXtBlock3d):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__(
            channels=channels,
            mlp_ratio=mlp_ratio,
            kernel_size=3,
            use_checkpoint=use_checkpoint,
            conv_cls=SparseConv3d,
        )
