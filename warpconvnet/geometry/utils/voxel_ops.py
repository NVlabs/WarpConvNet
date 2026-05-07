# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for building / concatenating / unbinding `Voxels`.

These cover the upstream-style sparse-tensor surface used by other 3D
frameworks (torchsparse, spconv, TRELLIS): a single ``(N, 1+DIM)``
batch-indexed coordinate tensor where column 0 is the batch index.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index
from warpconvnet.geometry.types.voxels import Voxels


__all__ = ["from_feats_coords", "sparse_cat", "sparse_unbind"]


def from_feats_coords(
    feats: torch.Tensor,
    coords: torch.Tensor,
    **kwargs,
) -> Voxels:
    """Build a `Voxels` from upstream-style ``(N, 1+DIM)`` batch-indexed coords.

    Column 0 of ``coords`` is the batch index; remaining columns are integer
    spatial coordinates. ``feats`` is ``(N, C)``. Any extra kwargs flow into
    `Voxels.__init__` (e.g. ``trellis_scale`` for TRELLIS-port models).
    """
    assert coords.ndim == 2 and coords.shape[1] >= 2
    batch_col = coords[:, 0].long()
    spatial = coords[:, 1:].contiguous()
    if "offsets" in kwargs:
        offsets = kwargs.pop("offsets")
    else:
        n_batches = int(batch_col.max().item()) + 1 if coords.shape[0] > 0 else 1
        offsets = offsets_from_batch_index(batch_col.int(), num_batches=n_batches)
    return Voxels(
        batched_coordinates=spatial,
        batched_features=feats,
        offsets=offsets,
        **kwargs,
    )


def sparse_cat(inputs: list[Voxels], dim: int = 0) -> Voxels:
    """Concatenate `Voxels` along the batch dim (`dim=0`) or feature dim."""
    if dim == 0:
        coords_concat: list[torch.Tensor] = []
        start = 0
        for v in inputs:
            c = v.coords.clone()
            c[:, 0] += start
            coords_concat.append(c)
            start += v.batch_size
        return from_feats_coords(
            torch.cat([v.feats for v in inputs], dim=0),
            torch.cat(coords_concat, dim=0),
        )
    return inputs[0].replace_features(torch.cat([v.feats for v in inputs], dim=dim))


def sparse_unbind(v: Voxels, dim: int = 0) -> list[Voxels]:
    """Split `Voxels` along the batch dim (`dim=0`) or feature dim."""
    if dim == 0:
        out: list[Voxels] = []
        for i in range(v.batch_size):
            sl = slice(int(v.offsets[i]), int(v.offsets[i + 1]))
            new_feats = v.feats[sl]
            new_coords = v.coords[sl].clone()
            new_coords[:, 0] = 0
            out.append(from_feats_coords(new_feats, new_coords))
        return out
    return [v.replace_features(f) for f in v.feats.unbind(dim)]
