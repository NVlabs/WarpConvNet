# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse spatial resampling: Down/Up/Spatial2Channel/Channel2Spatial/Subdivide.

Pure-tensor index arithmetic, operates directly on
``warpconvnet.geometry.types.voxels.Voxels``. Each block writes a coord/index
cache onto ``voxels.spatial_cache`` so the paired inverse op can avoid
recomputing the gather table.

Scale tracking is intentionally *not* baked into these classes — `Voxels`
already carries a `tensor_stride`, and TRELLIS-style fractional `_scale`
tracking lives in the model adapter that uses these (see
`warpconvnet.models.trellis2.sparse_spatial`).
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.utils.voxel_ops import from_feats_coords


__all__ = [
    "SparseChannel2Spatial",
    "SparseDownsample",
    "SparseSpatial2Channel",
    "SparseSubdivide",
    "SparseUpsample",
]


def _spatial_shape_of(x: Voxels, cache: dict) -> torch.Size:
    s = cache.get("shape")
    if s is None:
        s = torch.Size((x.coords[:, 1:].max(0).values + 1).tolist())
        cache["shape"] = s
    return s


class SparseDownsample(nn.Module):
    """Stride-`factor` average / max pool over coordinates."""

    def __init__(self, factor: int, mode: Literal["mean", "max"] = "mean"):
        super().__init__()
        assert mode in ("mean", "max")
        self.factor = factor
        self.mode = mode

    def forward(self, x: Voxels) -> Voxels:
        f = self.factor
        cache = x.spatial_cache
        ck = f"downsample_{f}"
        entry = cache.get(ck)
        if entry is None:
            DIM = x.coords.shape[-1] - 1
            coord = list(x.coords.unbind(dim=-1))
            for i in range(DIM):
                coord[i + 1] = coord[i + 1] // f
            MAX = [(s + f - 1) // f for s in _spatial_shape_of(x, cache)]
            OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
            code = sum(c * o for c, o in zip(coord, OFFSET))
            code, idx = code.unique(return_inverse=True)
            new_coords = torch.stack(
                [code // OFFSET[0]] + [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
                dim=-1,
            )
        else:
            new_coords, idx = entry
        new_feats = torch.scatter_reduce(
            torch.zeros(
                new_coords.shape[0],
                x.feats.shape[1],
                device=x.feats.device,
                dtype=x.feats.dtype,
            ),
            dim=0,
            index=idx.unsqueeze(1).expand(-1, x.feats.shape[1]),
            src=x.feats,
            reduce=self.mode,
            include_self=False,
        )
        out = from_feats_coords(new_feats, new_coords.int())
        # Share cache dict so paired Up can reuse the inverse map.
        out._extra_attributes["_spatial_cache"] = cache
        if entry is None:
            cache[ck] = (new_coords, idx)
            cache[f"upsample_{f}"] = (x.coords, idx)
        return out


class SparseUpsample(nn.Module):
    """Inverse of `SparseDownsample`. Requires a paired downsample cache or an
    explicit `subdivision` mask of shape ``(N, factor**DIM)``."""

    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: Voxels, subdivision: Voxels | None = None) -> Voxels:
        f = self.factor
        DIM = x.coords.shape[-1] - 1
        cache = x.spatial_cache
        entry = cache.get(f"upsample_{f}")
        if entry is None:
            if subdivision is None:
                raise ValueError(
                    "SparseUpsample needs either a cached downsample or a subdivision tensor"
                )
            sub = subdivision.feats
            n_leaf = sub.sum(dim=-1)
            subidx = sub.nonzero()[:, -1]
            new_coords = x.coords.clone().detach()
            new_coords[:, 1:] *= f
            new_coords = torch.repeat_interleave(
                new_coords, n_leaf, dim=0, output_size=subidx.shape[0]
            )
            for i in range(DIM):
                new_coords[:, i + 1] += subidx // f**i % f
            idx = torch.repeat_interleave(
                torch.arange(x.coords.shape[0], device=x.device),
                n_leaf,
                dim=0,
                output_size=subidx.shape[0],
            )
        else:
            new_coords, idx = entry
        new_feats = x.feats[idx]
        return from_feats_coords(new_feats, new_coords.int())


class SparseSubdivide(nn.Module):
    """Repeat each voxel `factor**DIM` times along the spatial dims (no
    pooling, no scatter)."""

    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: Voxels) -> Voxels:
        f = self.factor
        DIM = x.coords.shape[-1] - 1
        n_per = f**DIM
        ranges = [torch.arange(f, device=x.device) for _ in range(DIM)]
        grid = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1).reshape(n_per, DIM)
        new_coords = x.coords.repeat_interleave(n_per, dim=0)
        new_coords[:, 1:] = new_coords[:, 1:] * f + grid.repeat(x.coords.shape[0], 1)
        new_feats = x.feats.repeat_interleave(n_per, dim=0)
        return from_feats_coords(new_feats, new_coords.int())


class SparseSpatial2Channel(nn.Module):
    """Pack `factor**DIM` neighbouring voxels into the channel dim. Output has
    `factor`-coarser spatial coords and `factor**DIM`-times more channels.
    Missing children are zero-padded.

    Cache design (read carefully before changing):
      * Forward-direction reuse (the only path that fires in production):
        the (new_coords, idx, subidx) table is cached on the INPUT Voxels'
        ``spatial_cache`` keyed by ``spatial2channel_{f}``. This makes the
        encoder pattern ``h = self.updown(h); x = self.updown(x)`` reuse the
        table on the second call because ``h`` and ``x`` share the cache
        dict via ``replace_features``.
      * The input's cache is NOT propagated to the output Voxels. Earlier
        versions did, which caused two distinct downstream bugs:
          (a) chained S2C across stages saw the previous level's entry under
              the same key and crashed with a shape-mismatch in the scatter.
          (b) keying by ``coords.data_ptr()`` could not save us because the
              caching allocator recycles freed coord buffers, so a new
              coords tensor lands at the address of a stale entry.
        Giving the output a fresh, empty spatial cache fixes both: the
        downstream stage starts clean.
      * Inverse-direction reuse: the ``channel2spatial_{f}`` table is
        written onto the OUTPUT Voxels' (fresh) spatial cache rather than
        the input's. This keeps the entry scoped to this exact S2C output:
        a paired ``SparseChannel2Spatial`` called directly on ``out`` (as in
        the round-trip unit tests) hits the cache, but the entry cannot leak
        into a sibling/downstream stage that re-uses the INPUT's cache.
        Production paths don't rely on this — the C2S decoder receives its
        input from an intervening sparse conv with its own coords, never
        directly from a paired S2C — but cheap to keep correct for tests
        and any future caller that does want zero-recomputation round trips.
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x: Voxels) -> Voxels:
        f = self.factor
        DIM = x.coords.shape[-1] - 1
        cache = x.spatial_cache
        ck = f"spatial2channel_{f}"
        entry = cache.get(ck)
        if entry is None:
            coord = list(x.coords.unbind(dim=-1))
            for i in range(DIM):
                coord[i + 1] = coord[i + 1] // f
            subidx = x.coords[:, 1:] % f
            subidx = sum(subidx[..., i] * f**i for i in range(DIM))
            MAX = [(s + f - 1) // f for s in _spatial_shape_of(x, cache)]
            OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
            code = sum(c * o for c, o in zip(coord, OFFSET))
            code, idx = code.unique(return_inverse=True)
            new_coords = torch.stack(
                [code // OFFSET[0]] + [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
                dim=-1,
            )
        else:
            new_coords, idx, subidx = entry
        n_per = f**DIM
        new_feats = torch.zeros(
            new_coords.shape[0] * n_per,
            x.feats.shape[1],
            device=x.feats.device,
            dtype=x.feats.dtype,
        )
        new_feats[idx * n_per + subidx] = x.feats
        out = from_feats_coords(new_feats.reshape(new_coords.shape[0], -1), new_coords.int())
        # Forward-direction entry lives on the INPUT cache (shared via
        # replace_features). Inverse-direction entry lives on the OUTPUT
        # cache (fresh dict, scoped to this exact S2C output). Do NOT
        # propagate `cache` onto `out` — that's the bug the prior data_ptr
        # workaround was trying to paper over.
        if entry is None:
            cache[ck] = (new_coords, idx, subidx)
            out.spatial_cache[f"channel2spatial_{f}"] = (x.coords, idx, subidx)
        return out


class SparseChannel2Spatial(nn.Module):
    """Inverse of `SparseSpatial2Channel`. Reads each child slot from the
    channel block and either uses an explicit `subdivision` mask or the cache
    written by a paired `SparseSpatial2Channel`.

    Note: in production, the C2S input never carries an S2C-written cache
    (its coords come from an intervening sparse conv), so the
    ``channel2spatial_{f}`` cache lookup is essentially always a miss and the
    ``subdivision`` branch is taken. The lookup is kept for completeness and
    for callers that explicitly stitch the two ops back-to-back on the same
    Voxels object (e.g. unit tests).
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(
        self,
        x: Voxels,
        subdivision: Voxels | None = None,
    ) -> Voxels:
        f = self.factor
        DIM = x.coords.shape[-1] - 1
        cache = x.spatial_cache
        entry = cache.get(f"channel2spatial_{f}")
        if entry is None:
            if subdivision is None:
                raise ValueError(
                    "SparseChannel2Spatial needs either a cached spatial2channel "
                    "or an explicit subdivision tensor"
                )
            sub = subdivision.feats
            n_leaf = sub.sum(dim=-1)
            subidx = sub.nonzero()[:, -1]
            new_coords = x.coords.clone().detach()
            new_coords[:, 1:] *= f
            new_coords = torch.repeat_interleave(
                new_coords, n_leaf, dim=0, output_size=subidx.shape[0]
            )
            for i in range(DIM):
                new_coords[:, i + 1] += subidx // f**i % f
            idx = torch.repeat_interleave(
                torch.arange(x.coords.shape[0], device=x.device),
                n_leaf,
                dim=0,
                output_size=subidx.shape[0],
            )
        else:
            new_coords, idx, subidx = entry
        n_per = f**DIM
        x_feats = x.feats.reshape(x.feats.shape[0] * n_per, -1)
        new_feats = x_feats[idx * n_per + subidx]
        return from_feats_coords(new_feats, new_coords.int())
