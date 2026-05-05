# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from warpconvnet.nn.functional.fused_rope import fused_rope_qkv


def suggest_voxel_rope_base(
    num_heads: int,
    channel_size: int,
    max_coordinate: int,
    *,
    strategy: str = "scaled_window",
    scale: float = 4.0,
    min_base: int = 8,
    max_base: int = 4096,
    prefer_power_of_two: bool = True,
) -> int:
    """Suggest a RoPE base value for 3D voxel attention.

    Strategies:
      - ``scaled_window`` (default): ``base ≈ scale × max_coordinate``, clamped
        to ``[min_base, max_base]``.
      - ``half_wave``: choose ``base`` so the slowest band spans a half sinusoid
        across window length ``L``: ``θ_min × L ≈ π``.
    """
    window = max(1, int(max_coordinate))
    if strategy == "half_wave":
        head_dim = channel_size // max(1, num_heads)
        rope_dim = (head_dim // 6) * 6
        M = rope_dim // 3
        if M <= 0:
            raw = window / math.pi
        else:
            is_M_even = M % 2 == 0
            decrement = 2 if is_M_even else 1
            alpha = 1.0 - (decrement / float(M))
            if alpha <= 1e-6:
                raw = window / math.pi
            else:
                raw = (window / math.pi) ** (1.0 / alpha)
        proposed = max(2, int(round(raw)))
        proposed = max(min_base, min(proposed, max_base))
    else:
        proposed = max(min_base, min(int(round(scale * window)), max_base))

    if prefer_power_of_two:
        log2_val = math.log2(proposed)
        lower = 1 << int(math.floor(log2_val))
        upper = 1 << int(math.ceil(log2_val))
        proposed = lower if (proposed - lower) <= (upper - proposed) else upper

    return int(proposed)


class VisionRotaryPositionalEmbeddings(nn.Module):
    """2D Rotary Positional Embeddings for image patches (axial frequency 2D RoPE,
    https://arxiv.org/pdf/2403.13298). Applies x/y position rotations independently
    to each tile.
    """

    def __init__(
        self,
        patch_size: int,
        tile_size: int,
        max_num_tiles: int,
        dim: int,
        base: int = 10_000,
        append_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.patch_grid_size = tile_size // patch_size
        self.max_num_tiles = max_num_tiles
        self.dim = dim
        self.base = base
        self.append_cls_token = append_cls_token
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self) -> None:
        patches_per_tile = self.patch_grid_size**2
        patch_idx = torch.arange(
            patches_per_tile, dtype=self.theta.dtype, device=self.theta.device
        )
        if self.append_cls_token:
            patch_idx = torch.cat(
                [patch_idx, -1 * torch.ones(1, dtype=patch_idx.dtype, device=patch_idx.device)]
            )
        else:
            patch_idx = torch.cat(
                [-1 * torch.ones(1, dtype=patch_idx.dtype, device=patch_idx.device), patch_idx]
            )
        patch_x_pos = patch_idx % self.patch_grid_size
        patch_y_pos = patch_idx // self.patch_grid_size

        x_theta = torch.einsum("i, j -> ij", patch_x_pos + 1, self.theta).float()
        y_theta = torch.einsum("i, j -> ij", patch_y_pos + 1, self.theta).float()

        freqs = torch.cat([x_theta, y_theta], dim=-1)
        freqs = freqs.masked_fill(patch_idx.unsqueeze(-1) < 0, 0)

        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        bsz, _, n_h, h_d = x.shape
        xshaped = x.float().reshape(bsz, self.max_num_tiles, -1, n_h, h_d // 2, 2)
        seq_len = xshaped.size(2)

        if seq_len != self.cache.shape[0]:
            raise ValueError(
                f"Input sequence length {seq_len} does not match 2D RoPE cache sequence length {self.cache.shape[0]}."
            )

        rope_cache = self.cache.view(1, 1, seq_len, 1, h_d // 2, 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.reshape(bsz, self.max_num_tiles * seq_len, n_h, h_d)
        return x_out.type_as(x)


class VoxelRotaryPositionalEmbeddings(nn.Module):
    """3D Rotary Positional Embeddings for sparse voxel data.

    Applies RoPE to Q and K of a packed QKV tensor via the fused CUDA kernel
    in `warpconvnet.nn.functional.fused_rope.fused_rope_qkv`. Rotation is
    applied to the largest multiple of 6 dimensions per head; the remaining
    head dims are passed through unchanged.

    Args:
        dim: total embedding dimension (``num_heads * head_dim``).
        num_heads: number of attention heads.
        base: rotation base. Use `suggest_voxel_rope_base` for a window-aware default.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"Dimension {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.base = base

        self.rope_dim = (self.head_dim // 6) * 6
        self.pass_dim = self.head_dim - self.rope_dim

        if self.rope_dim > 0:
            theta = 1.0 / (
                self.base
                ** (torch.arange(0, self.rope_dim // 3, 2).float() / (self.rope_dim // 3))
            )
            self.register_buffer("theta", theta, persistent=False)
        else:
            self.theta = None

    def forward(
        self,
        qkv: Float[Tensor, "M 3 C"],  # noqa: F821
        coords: Float[Tensor, "M 3"],  # noqa: F821
        **kwargs: Any,
    ) -> Float[Tensor, "M 3 H D"]:  # noqa: F821
        """Apply RoPE to Q and K, reshape into ``[M, 3, num_heads, head_dim]``.

        Accepts either ``[M, 3, C]`` or ``[M, 3*C]``. When ``rope_dim == 0``
        (``head_dim < 6``), no rotation is needed — just reshape.
        """
        M = qkv.shape[0]
        if qkv.dim() == 2:
            qkv = qkv.view(M, 3, self.dim)

        if self.rope_dim == 0:
            return qkv.reshape(M, 3, self.num_heads, self.head_dim)

        return fused_rope_qkv(qkv, coords, self.theta, self.num_heads, self.rope_dim)
