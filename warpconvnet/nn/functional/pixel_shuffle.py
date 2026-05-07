# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D pixel shuffle / unshuffle (channel↔spatial rearrangement).

PyTorch ships 1D/2D pixel_shuffle but no 3D variant. The 3D form is needed
for upsampling 3D Conv VAEs (e.g. TRELLIS sparse-structure decoder).
"""
from __future__ import annotations

import torch


__all__ = ["pixel_shuffle_3d", "pixel_unshuffle_3d"]


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """``(B, C*r^3, H, W, D)`` → ``(B, C, H*r, W*r, D*r)``."""
    B, C, H, W, D = x.shape
    s = scale_factor
    C_ = C // s**3
    x = x.reshape(B, C_, s, s, s, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(B, C_, H * s, W * s, D * s)


def pixel_unshuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """``(B, C, H*r, W*r, D*r)`` → ``(B, C*r^3, H, W, D)``. Inverse of `pixel_shuffle_3d`."""
    B, C, H, W, D = x.shape
    s = scale_factor
    assert H % s == 0 and W % s == 0 and D % s == 0
    x = x.reshape(B, C, H // s, s, W // s, s, D // s, s)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(B, C * s**3, H // s, W // s, D // s)
