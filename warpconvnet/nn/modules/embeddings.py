# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Position / time embeddings for diffusion and flow-matching models.

`TimestepEmbedder`        — sinusoidal scalar→vector for diffusion timestep.
`SinusoidalPositionEmbedder` — multi-axis sin/cos absolute position embedding.
`RotaryPositionEmbedder`  — RoPE phases for arbitrary-dim integer coords (dense
                            tokens). For sparse-voxel RoPE see
                            `warpconvnet.nn.modules.rope.VoxelRotaryPositionalEmbeddings`.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


__all__ = [
    "RotaryPositionEmbedder",
    "SinusoidalPositionEmbedder",
    "TimestepEmbedder",
]


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → 2-layer MLP (Linear-SiLU-Linear)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class SinusoidalPositionEmbedder(nn.Module):
    """Multi-axis sin/cos absolute position embedding for integer coordinates.

    For ``in_channels=D`` input coords ``(N, D)``, produces ``(N, channels)``
    by concatenating per-axis sin/cos embeddings (zero-padded if `channels`
    isn't an exact multiple of ``2*D*freq_dim``).
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**freqs)

    def _sin_cos(self, x: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        assert D == self.in_channels
        embed = self._sin_cos(x.reshape(-1)).reshape(N, -1)
        if embed.shape[1] < self.channels:
            pad = torch.zeros(N, self.channels - embed.shape[1], device=embed.device)
            embed = torch.cat([embed, pad], dim=-1)
        return embed


class RotaryPositionEmbedder(nn.Module):
    """RoPE phases for any-D integer coordinates (dense token version).

    ``head_dim`` must be even. ``dim`` is the number of coordinate axes (3 for
    voxels, 2 for image patches). Use the returned phases inside an attention
    module via ``apply_rotary_embedding``.
    """

    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        return torch.polar(torch.ones_like(phases), phases)

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rot = x_complex * phases.unsqueeze(-2)
        return torch.view_as_real(x_rot).reshape(*x_rot.shape[:-1], -1).to(x.dtype)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert indices.shape[-1] == self.dim
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
            pad_n = self.head_dim // 2 - phases.shape[-1]
            ones = torch.ones(*phases.shape[:-1], pad_n, device=phases.device)
            zeros = torch.zeros(*phases.shape[:-1], pad_n, device=phases.device)
            phases = torch.cat([phases, torch.polar(ones, zeros)], dim=-1)
        return phases
