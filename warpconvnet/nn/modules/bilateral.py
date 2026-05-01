# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nn.Module wrappers around bilateral filter / bilateral grid.

Stateless modules; the underlying functional ops build per-call hash tables
because the lattice is data-dependent. The module exists so callers can
register sigmas as buffers and slot the filter into an nn.Sequential.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from warpconvnet.nn.functional.bilateral import bilateral_filter
from warpconvnet.nn.functional.bilateral_grid import (
    BilateralGrid,
    bilateral_filter_grid,
    fast_bilateral_solver,
)


class BilateralFilter(nn.Module):
    """KNN/radius bilateral filter (Gaussian on xyz + feat)."""

    def __init__(
        self,
        sigma_xyz: float = 0.05,
        sigma_feat: float = 20.0,
        k: int = 16,
        mode: str = "knn",
        radius_mult: float = 3.0,
        chunk_size: int = 32768,
    ):
        super().__init__()
        self.sigma_xyz = sigma_xyz
        self.sigma_feat = sigma_feat
        self.k = k
        self.mode = mode
        self.radius_mult = radius_mult
        self.chunk_size = chunk_size

    def forward(
        self,
        src_xyz: Tensor,
        src_feat: Tensor,
        src_value: Tensor,
        query_xyz: Tensor | None = None,
        query_feat: Tensor | None = None,
    ) -> Tensor:
        return bilateral_filter(
            src_xyz=src_xyz,
            src_feat=src_feat,
            src_value=src_value,
            query_xyz=query_xyz,
            query_feat=query_feat,
            sigma_xyz=self.sigma_xyz,
            sigma_feat=self.sigma_feat,
            k=self.k,
            mode=self.mode,
            radius_mult=self.radius_mult,
            chunk_size=self.chunk_size,
        )


class BilateralFilterGrid(nn.Module):
    """Splat-blur-slice bilateral filter on a sparse d-cube grid (Barron-style)."""

    def __init__(self, sigma_xyz: float = 0.05, sigma_feat: float = 20.0):
        super().__init__()
        self.sigma_xyz = sigma_xyz
        self.sigma_feat = sigma_feat

    def forward(self, src_xyz: Tensor, src_feat: Tensor, src_value: Tensor) -> Tensor:
        return bilateral_filter_grid(
            src_xyz,
            src_feat,
            src_value,
            sigma_xyz=self.sigma_xyz,
            sigma_feat=self.sigma_feat,
        )


class FastBilateralSolver(nn.Module):
    """Confidence-weighted bilateral smoothing via PCG (Barron & Poole 2015)."""

    def __init__(
        self,
        sigma_xyz: float = 0.05,
        sigma_feat: float = 20.0,
        lam: float = 128.0,
        max_iters: int = 25,
        tol: float = 1e-5,
    ):
        super().__init__()
        self.sigma_xyz = sigma_xyz
        self.sigma_feat = sigma_feat
        self.lam = lam
        self.max_iters = max_iters
        self.tol = tol

    def forward(
        self,
        src_xyz: Tensor,
        src_feat: Tensor,
        target: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        return fast_bilateral_solver(
            src_xyz=src_xyz,
            src_feat=src_feat,
            target=target,
            confidence=confidence,
            sigma_xyz=self.sigma_xyz,
            sigma_feat=self.sigma_feat,
            lam=self.lam,
            max_iters=self.max_iters,
            tol=self.tol,
        )


class BilateralFilterGridCached(nn.Module):
    """Build-once / filter-many sparse d-cube bilateral grid.

    Positions fixed across calls (e.g., per-frame in video), only features
    differ. Call ``build_grid`` once, then ``forward`` repeatedly.
    """

    def __init__(self, sigma_xyz: float = 0.05, sigma_feat: float = 20.0):
        super().__init__()
        self.sigma_xyz = sigma_xyz
        self.sigma_feat = sigma_feat
        self._grid: BilateralGrid | None = None

    def build_grid(self, src_xyz: Tensor, src_feat: Tensor) -> BilateralFilterGridCached:
        pos = torch.cat([src_xyz / self.sigma_xyz, src_feat / self.sigma_feat], dim=-1)
        self._grid = BilateralGrid.build(pos)
        return self

    def forward(self, src_value: Tensor) -> Tensor:
        if self._grid is None:
            raise RuntimeError("Call build_grid(src_xyz, src_feat) before forward().")
        return self._grid.filter(src_value, normalize=True)

    @property
    def num_vertices(self) -> int:
        if self._grid is None:
            return 0
        return self._grid.num_vertices
