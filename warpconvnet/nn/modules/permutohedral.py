# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nn.Module wrapper around permutohedral lattice filter."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn

from warpconvnet.nn.functional.permutohedral import (
    PermutohedralLattice,
    bilateral_permutohedral_filter,
    permutohedral_filter,
)


class PermutohedralFilter(nn.Module):
    """Gaussian filter via permutohedral lattice (Adams, Baek, Davis 2010).

    Pre-scales positions by ``sigmas`` (per-axis) or ``sigma`` (scalar) and
    runs splat -> blur -> slice. Lattice coords have d+1 axes so the input
    feature dim is bounded to d <= 6 by the underlying PackedHashTable128.
    """

    def __init__(
        self,
        sigma: float | None = None,
        sigmas: Sequence[float] | None = None,
    ):
        super().__init__()
        if (sigma is None) == (sigmas is None):
            raise ValueError("Pass exactly one of sigma (scalar) or sigmas (per-axis).")
        self.sigma = sigma
        if sigmas is not None:
            self.register_buffer(
                "sigmas",
                torch.as_tensor(list(sigmas), dtype=torch.float32),
            )
        else:
            self.sigmas = None

    def forward(
        self,
        positions: Tensor,
        features: Tensor,
        query_positions: Tensor | None = None,
    ) -> Tensor:
        sigmas = self.sigmas
        if sigmas is not None:
            sigmas = sigmas.to(device=positions.device, dtype=positions.dtype)
        return permutohedral_filter(
            positions=positions,
            features=features,
            sigmas=sigmas,
            sigma=self.sigma,
            query_positions=query_positions,
        )


class PermutohedralFilterCached(nn.Module):
    """Build-once / filter-many permutohedral lattice.

    For pipelines where positions are fixed (video frame sequence, iterative
    bilateral solving) and only features change. Call ``build_lattice`` once,
    then ``forward`` repeatedly with different feature tensors.
    """

    def __init__(
        self,
        sigma: float | None = None,
        sigmas: Sequence[float] | None = None,
    ):
        super().__init__()
        if (sigma is None) == (sigmas is None):
            raise ValueError("Pass exactly one of sigma (scalar) or sigmas (per-axis).")
        self.sigma = sigma
        if sigmas is not None:
            self.register_buffer(
                "sigmas",
                torch.as_tensor(list(sigmas), dtype=torch.float32),
            )
        else:
            self.sigmas = None
        self._lattice: PermutohedralLattice | None = None

    def build_lattice(self, positions: Tensor) -> PermutohedralFilterCached:
        sigmas = self.sigmas
        if sigmas is not None:
            sigmas = sigmas.to(device=positions.device, dtype=positions.dtype)
            scaled = positions / sigmas
        else:
            scaled = positions / self.sigma
        self._lattice = PermutohedralLattice.build(scaled)
        return self

    def forward(
        self,
        features: Tensor,
        query_positions: Tensor | None = None,
    ) -> Tensor:
        if self._lattice is None:
            raise RuntimeError("Call build_lattice(positions) before forward().")
        if query_positions is not None:
            sigmas = self.sigmas
            if sigmas is not None:
                sigmas = sigmas.to(device=query_positions.device, dtype=query_positions.dtype)
                query_positions = query_positions / sigmas
            else:
                query_positions = query_positions / self.sigma
        return self._lattice.filter(features, query_positions=query_positions)

    @property
    def num_vertices(self) -> int:
        if self._lattice is None:
            return 0
        return int(self._lattice.unique_keys.shape[0])


class BilateralPermutohedralFilter(nn.Module):
    """Bilateral (xyz + color) permutohedral filter.

    Lattice coords = concat(xyz / sigma_xyz, feat / sigma_feat). xyz alone is
    just a Gaussian blur; feat (e.g. RGB) is what makes it edge-preserving.
    Constraint: D_xyz + D_feat <= 6 (lattice axes capped at 7).
    """

    def __init__(self, sigma_xyz: float = 0.05, sigma_feat: float = 20.0):
        super().__init__()
        self.sigma_xyz = sigma_xyz
        self.sigma_feat = sigma_feat

    def forward(
        self,
        src_xyz: Tensor,
        src_feat: Tensor,
        src_value: Tensor,
        query_xyz: Tensor | None = None,
        query_feat: Tensor | None = None,
        *,
        normalize: bool = True,
    ) -> Tensor:
        return bilateral_permutohedral_filter(
            src_xyz=src_xyz,
            src_feat=src_feat,
            src_value=src_value,
            sigma_xyz=self.sigma_xyz,
            sigma_feat=self.sigma_feat,
            query_xyz=query_xyz,
            query_feat=query_feat,
            normalize=normalize,
        )


class BilateralPermutohedralFilterCached(nn.Module):
    """Build-once / filter-many bilateral permutohedral.

    For iterative bilateral solving on fixed (xyz, feat). Call ``build_lattice``
    with the source xyz + feat once, then call forward repeatedly with
    different value tensors.
    """

    def __init__(self, sigma_xyz: float = 0.05, sigma_feat: float = 20.0):
        super().__init__()
        self.sigma_xyz = sigma_xyz
        self.sigma_feat = sigma_feat
        self._lattice: PermutohedralLattice | None = None

    def build_lattice(
        self, src_xyz: Tensor, src_feat: Tensor
    ) -> BilateralPermutohedralFilterCached:
        d_xyz = src_xyz.shape[1]
        d_feat = src_feat.shape[1]
        if d_xyz + d_feat > 6:
            raise ValueError(f"D_xyz + D_feat = {d_xyz + d_feat} > 6; lattice axes capped at 7.")
        positions = torch.cat(
            [src_xyz / self.sigma_xyz, src_feat / self.sigma_feat],
            dim=-1,
        )
        self._lattice = PermutohedralLattice.build(positions)
        return self

    def forward(
        self,
        src_value: Tensor,
        query_xyz: Tensor | None = None,
        query_feat: Tensor | None = None,
        *,
        normalize: bool = True,
    ) -> Tensor:
        if self._lattice is None:
            raise RuntimeError("Call build_lattice(src_xyz, src_feat) before forward().")
        if query_xyz is None and query_feat is None:
            qp = None
        else:
            if query_xyz is None or query_feat is None:
                raise ValueError("Pass both query_xyz and query_feat, or neither.")
            qp = torch.cat(
                [query_xyz / self.sigma_xyz, query_feat / self.sigma_feat],
                dim=-1,
            )
        return self._lattice.filter(src_value, query_positions=qp, normalize=normalize)

    @property
    def num_vertices(self) -> int:
        if self._lattice is None:
            return 0
        return int(self._lattice.unique_keys.shape[0])
