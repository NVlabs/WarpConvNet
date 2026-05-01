# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permutohedral lattice filter (Adams, Baek, Davis 2010).

Reference: "Fast High-Dimensional Filtering Using the Permutohedral Lattice",
https://graphics.stanford.edu/papers/permutohedral/

Algorithm:
    1. Splat: each input point is embedded into a (d+1)-D simplicial lattice;
       its feature is distributed across the (d+1) vertices of the enclosing
       simplex, weighted by barycentric coordinates.
    2. Blur: a separable 3-tap Gaussian blur along each of the (d+1) lattice
       axes. Vertex neighbors along axis ``a`` differ by an integer offset
       with -d at position ``a`` and +1 elsewhere (sums to 0, preserving the
       H_d hyperplane).
    3. Slice: at each query position, gather from the (d+1) enclosing-simplex
       vertices via the same barycentric weights.

Complexity is O(N * d^2) for splat/slice and O(V * d^2) for blur where V is
the number of unique lattice vertices populated by the inputs (V <= N*(d+1)).
This is the standard fast bilateral approach.

Lattice-vertex storage uses ``PackedHashTable128`` (D=7, CoordBits=17) so the
maximum supported feature dim is d=6 (lattice coords are d+1=7).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ._hash_backend import make_hash_table


# --- Lattice math ----------------------------------------------------------


def _embed_lattice(features: Tensor) -> Tensor:
    """Embed (N, d) feature vectors into (N, d+1) lattice coordinates whose
    components sum to zero. Matches Adams' reference C++ ``elevated`` step.

    The (d+1)-D embedding uses an orthogonal basis of H_d (hyperplane sum=0)
    pre-scaled so that one lattice cell ≈ one standard deviation in input
    space — i.e. the caller pre-divides their feature by sigma.
    """
    N, d = features.shape
    device = features.device
    dtype = features.dtype

    inv_std = (d + 1) * (2.0 / 3.0) ** 0.5
    # scale[i] = inv_std / sqrt((i+1)*(i+2)),  i = 0..d-1
    arange_d = torch.arange(1, d + 1, device=device, dtype=dtype)
    scale = inv_std / torch.sqrt(arange_d * (arange_d + 1))  # (d,)

    cf = features * scale  # (N, d)

    # elevated[i] for i=1..d: (sum of cf[j] for j>=i) - i*cf[i-1]
    # elevated[0] = sum of all cf[j]
    elevated = torch.empty((N, d + 1), device=device, dtype=dtype)
    sm = torch.zeros(N, device=device, dtype=dtype)
    for i in range(d, 0, -1):
        cf_i = cf[:, i - 1]
        elevated[:, i] = sm - i * cf_i
        sm = sm + cf_i
    elevated[:, 0] = sm
    return elevated


def _find_enclosing_simplex(elevated: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Given (N, d+1) elevated coords on H_d, find the simplex containing each
    point. Returns:

    - greedy: (N, d+1) int64 — the closest 0-color (sum=0) lattice vertex.
    - rank:   (N, d+1) int64 — rank of (elevated - greedy) along each axis.
              rank[i, j] in [0, d]; ties are broken consistently because
              sum(elevated - greedy) ≈ 0 ensures unique ordering after
              correction.
    - barycentric: (N, d+2) float — barycentric weights; index 0..d are
              vertex weights, index d+1 is a wraparound slot per Adams.

    Mirrors the reference C++ ``compute_simplex`` step.
    """
    N, dp1 = elevated.shape
    d = dp1 - 1
    device = elevated.device
    dtype = elevated.dtype

    inv_dp1 = 1.0 / (d + 1)
    # 1) Round each component to nearest multiple of (d+1).
    v = elevated * inv_dp1
    up = torch.ceil(v) * (d + 1)
    down = torch.floor(v) * (d + 1)
    # Pick whichever is closer.
    pick_up = (up - elevated) < (elevated - down)
    greedy = torch.where(pick_up, up, down).to(torch.int64)  # (N, d+1)

    # Sum of greedy / (d+1) — should be small; corrects toward sum=0.
    sum_g = greedy.sum(dim=-1) // (d + 1)  # (N,)

    # 2) Compute rank: for each axis j, count how many other axes have
    # larger (elevated - greedy) value. Implemented via argsort-of-argsort.
    diff = elevated - greedy.to(dtype)  # (N, d+1)
    # rank[i, j] = number of axes k where diff[i, k] > diff[i, j], breaking
    # ties by axis index to keep determinism.
    # argsort descending on diff gives the order of axes from largest to
    # smallest; rank is the inverse permutation.
    order = torch.argsort(diff, dim=-1, descending=True)  # (N, d+1)
    rank = torch.empty_like(order)
    rank.scatter_(
        dim=-1,
        index=order,
        src=torch.arange(d + 1, device=device).unsqueeze(0).expand(N, d + 1),
    )  # (N, d+1) int64

    # 3) Correct so sum(greedy) == 0 (i.e., sum_g shifts by zero).
    # Adams: when sum_g > 0 we subtract (d+1) from the (sum_g) axes with
    # largest rank; when sum_g < 0 we add (d+1) to the (-sum_g) axes with
    # smallest rank.
    # Vectorize per-batch:
    sum_g_pos = sum_g.clamp(min=0)
    sum_g_neg = (-sum_g).clamp(min=0)

    # Mask of axes whose rank is in the "high" region (rank >= (d+1) - sum_g)
    high_mask = rank >= ((d + 1) - sum_g_pos.unsqueeze(-1))
    low_mask = rank < sum_g_neg.unsqueeze(-1)

    greedy = greedy - (high_mask.to(torch.int64) * (d + 1))
    greedy = greedy + (low_mask.to(torch.int64) * (d + 1))

    # rank update: +sum_g everywhere, then -(d+1) on high_mask, +(d+1) on low_mask
    rank = rank + sum_g.unsqueeze(-1)
    rank = rank - (high_mask.to(torch.int64) * (d + 1))
    rank = rank + (low_mask.to(torch.int64) * (d + 1))

    # 4) Compute barycentric weights.
    # Adams: barycentric[d - rank[j]] += delta[j]; barycentric[d+1 - rank[j]] -= delta[j]
    # then barycentric[0] += 1 + barycentric[d+1].
    delta = (elevated - greedy.to(dtype)) * inv_dp1  # (N, d+1)
    bary = torch.zeros((N, d + 2), device=device, dtype=dtype)
    # idx_pos = d - rank  in [0, d]
    idx_pos = (d - rank).to(torch.int64)
    idx_neg = (d + 1 - rank).to(torch.int64)
    bary.scatter_add_(dim=-1, index=idx_pos, src=delta)
    bary.scatter_add_(dim=-1, index=idx_neg, src=-delta)
    bary[:, 0] = bary[:, 0] + 1.0 + bary[:, d + 1]

    return greedy, rank, bary[:, : d + 1]  # we never read bary[d+1] downstream


def _canonical_simplex_offsets(d: int, device: torch.device) -> Tensor:
    """Adams' canonical simplex table.

    Returns (d+1, d+1) int64. canonical[k, j] is the offset added at
    rank-j to produce vertex k of the simplex:
        canonical[k, j] = k          if j <= d - k
        canonical[k, j] = k - (d+1)  if j >  d - k

    Each row sums to 0 (k*(d+1-k) + (k-(d+1))*k = 0), preserving H_d.
    """
    k = torch.arange(d + 1, device=device).unsqueeze(1)  # (d+1, 1)
    j = torch.arange(d + 1, device=device).unsqueeze(0)  # (1, d+1)
    return torch.where(j <= (d - k), k, k - (d + 1)).to(torch.int64)


# --- Lattice class ---------------------------------------------------------


class PermutohedralLattice:
    """Build-once / filter-many permutohedral lattice.

    Usage:
        lat = PermutohedralLattice.build(positions / sigma)   # positions: (N, d)
        out = lat.filter(features)                            # features: (N, F)
        # or: out = lat.filter(features, query_positions=qp/sigma)

    For the typical "scale spatial by sigma_xyz, color by sigma_rgb" bilateral
    case, just stack and pre-scale features in caller code:
        feat = torch.cat([xyz / sigma_xyz, rgb / sigma_rgb], dim=-1)
        lat = PermutohedralLattice.build(feat)
        out_rgb = lat.filter(rgb)            # bilateral-blurred RGB
    """

    def __init__(
        self,
        unique_keys: Tensor,  # (V, d+1) int32
        inverse: Tensor,  # (N*(d+1),) int64
        bary: Tensor,  # (N, d+1) float
        d: int,
        n_input: int,
        hash_table,
    ):
        self.unique_keys = unique_keys
        self.inverse = inverse
        self.bary = bary
        self.d = d
        self.n_input = n_input
        self.hash_table = hash_table

    @classmethod
    def build(
        cls,
        positions: Tensor,
    ) -> PermutohedralLattice:
        """Construct the lattice from input positions.

        Args:
            positions: (N, d) input positions, **already divided by your
                desired bandwidth(s)** (sigma). One sigma == one lattice cell.
        """
        assert positions.dim() == 2, "positions must be (N, d)"
        assert positions.is_cuda, "permutohedral lattice currently requires CUDA"

        N, d = positions.shape
        device = positions.device

        elevated = _embed_lattice(positions)  # (N, d+1)
        greedy, rank, bary = _find_enclosing_simplex(elevated)  # int64, int64, float
        # bary: (N, d+1)

        canonical = _canonical_simplex_offsets(d, device)  # (d+1, d+1) int64

        # Build (N, d+1, d+1) keys: keys[i, k, j] = greedy[i, j] + canonical[k, rank[i, j]]
        # canonical_at_rank[i, k, j] = canonical[k, rank[i, j]]
        # rank: (N, d+1); we want canonical[k, rank[i, j]] for all k.
        # canonical: (d+1, d+1) — index along axis 1 by rank.
        # Take: canonical_per_axis = canonical[:, rank]  → shape (d+1, N, d+1)
        canonical_per_axis = canonical[:, rank]  # (d+1, N, d+1) int64
        # transpose to (N, d+1, d+1)
        canonical_per_axis = canonical_per_axis.permute(1, 0, 2).contiguous()
        keys = greedy.unsqueeze(1) + canonical_per_axis  # (N, d+1, d+1) int64

        # Flatten for hashing.
        all_keys = keys.reshape(-1, d + 1).to(torch.int32)  # (N*(d+1), d+1)
        # torch.unique on rows.
        unique_keys, inverse = torch.unique(all_keys, dim=0, return_inverse=True)
        # unique_keys: (V, d+1) int32; inverse: (N*(d+1),) int64

        capacity = max(16, int(unique_keys.shape[0] * 2))
        hash_table = make_hash_table(
            unique_keys,
            device=device,
            capacity=capacity,
        )

        return cls(
            unique_keys=unique_keys,
            inverse=inverse,
            bary=bary,
            d=d,
            n_input=N,
            hash_table=hash_table,
        )

    # -- Filter pipeline ----------------------------------------------------

    def _splat(self, features: Tensor) -> Tensor:
        """Distribute (N, F) input features across (V, F) lattice vertices."""
        N, F = features.shape
        assert N == self.n_input, f"feature N={N} != lattice N={self.n_input}"
        d = self.d
        V = self.unique_keys.shape[0]

        # weight[i, k] = bary[i, k]; expand to (N*(d+1), F)
        w = self.bary.reshape(-1).unsqueeze(-1)  # (N*(d+1), 1)
        f_per_vertex = features.unsqueeze(1).expand(N, d + 1, F).reshape(-1, F)  # (N*(d+1), F)
        contrib = w * f_per_vertex

        lattice = torch.zeros((V, F), dtype=features.dtype, device=features.device)
        lattice.index_add_(0, self.inverse, contrib)
        return lattice

    def _blur(self, lattice: Tensor) -> Tensor:
        """Separable 3-tap blur along each of the (d+1) axes.

        Standard reference uses [0.5, 1.0, 0.5] taps then an explicit final
        normalization in the slice step (Adams §4.4): blurred = 0.5*left +
        1.0*center + 0.5*right per axis.

        All 2*(d+1) neighbor lookups are issued as a single fused
        ``batched_search`` launch. The lattice update remains sequential
        across axes (each axis reads the previous axis's output).

        Sentinel-row trick: append a zero row at index V so that misses
        (originally -1) can be remapped to V and gathered without masks.
        """
        d = self.d
        device = lattice.device
        keys = self.unique_keys  # (V, d+1) int32
        V, F = lattice.shape

        K = 2 * (d + 1)
        offsets = torch.full((K, d + 1), 1, dtype=torch.int32, device=device)
        ar = torch.arange(d + 1, device=device)
        offsets[2 * ar, ar] = -d
        offsets[2 * ar + 1] = -offsets[2 * ar]

        results = self.hash_table.batched_search(keys, offsets)  # (K, V) int32, -1 = miss
        # Misses -> V (sentinel row index). Cast to int64 once for indexing.
        results_long = torch.where(results < 0, torch.full_like(results, V), results).long()

        # Append zero sentinel row so gathers from the "miss" index read zero.
        zeros_row = lattice.new_zeros(1, F)
        lattice_pad = torch.cat([lattice, zeros_row], dim=0)  # (V+1, F)

        for axis in range(d + 1):
            fwd = results_long[2 * axis]
            bwd = results_long[2 * axis + 1]
            # Tighter in-place chain (per cuhash maintainer): 4 launches/axis.
            # In-place RAW is empirically tolerated (perturbation is small,
            # subsequent axes wash it out). bpfilter validated this swap.
            combined = lattice_pad[fwd]
            combined.add_(lattice_pad[bwd])
            lattice_pad[:V].add_(combined, alpha=0.5)

        return lattice_pad[:V]

    def _slice(
        self,
        lattice: Tensor,  # (V, F)
        query_inverse: Tensor,  # (M*(d+1),) int64 — vertex idx per (query, k)
        query_bary: Tensor,  # (M, d+1) float
    ) -> Tensor:
        """Gather from lattice into (M, F) using barycentric weights."""
        d = self.d
        M = query_bary.shape[0]
        F = lattice.shape[1]
        # vert_feat: (M*(d+1), F)
        vert_feat = lattice[query_inverse]
        w = query_bary.reshape(-1).unsqueeze(-1)  # (M*(d+1), 1)
        weighted = (w * vert_feat).reshape(M, d + 1, F)
        out = weighted.sum(dim=1)
        # Adams' normalization factor: 1 / (1 + 2^-(d+1)) — corrects for
        # the 0.5 taps in the blur. Apply once at the end.
        alpha = 1.0 / (1.0 + 2.0 ** (-(d + 1)))
        return out * alpha

    def filter(
        self,
        features: Tensor,  # (N, F)
        query_positions: Tensor | None = None,  # (M, d), pre-scaled by sigma
        *,
        normalize: bool = True,
    ) -> Tensor:
        """Run splat → blur → slice.

        If ``query_positions`` is ``None``, slice at the same positions used
        to build the lattice (self-filter, the typical case).

        With ``normalize=True`` (default), an extra "ones" channel is splatted
        and blurred alongside the features; the per-input weight is sliced
        and used to divide. This is the standard homogeneous-coordinate
        normalization (Adams §4.4) that turns the unweighted accumulation
        into a true Gaussian-weighted average. Without it, a constant input
        would not return that constant — the filter weights vary by local
        lattice density.
        """
        # ==================================================================
        # pad-channel workaround for torch gather pessimum at C % 4 == 0
        # ==================================================================
        # observed: torch 2.10.0+cu128 / cuda 12.8, RTX 6000 Ada.
        #
        # `lattice[fwd_idx]` (advanced-indexing gather along dim 0) hits a
        # pessimal kernel when the lattice channel width C is divisible by 4.
        # measured on V=1.3M random int64 indices, fp32 gather:
        #
        #   C=1   0.029 ms        C=11  0.323 ms
        #   C=2   0.031 ms        C=12  0.797 ms <- divisible by 4
        #   C=3   0.037 ms        C=13  0.404 ms
        #   C=4   0.694 ms <-     C=14  0.436 ms
        #   C=5   0.048 ms        C=15  0.476 ms
        #   C=6   0.064 ms        C=16  0.850 ms <- divisible by 4
        #   C=7   0.130 ms        C=17  0.533 ms
        #   C=8   0.711 ms <-     C=20  0.894 ms <- divisible by 4
        #
        # cliff is ~10-15x at C=4, ~3-5x at C>=8. all alternative APIs
        # (`torch.index_select`, `torch.gather`, `lat[idx, :]`) hit the
        # same path. likely a vectorized float4 codegen branch in
        # ATen's IndexKernel.cu that runs slower than scalar for random
        # (uncoalesced) gather; fast path for sequential gather only.
        #
        # workaround: pad C by +1 zero column when C % 4 == 0, run the
        # full splat-blur-slice on the padded lattice, strip at the end.
        # the zero column stays zero through linear ops so output is
        # algebraically identical.
        #
        # measured impact on hero (N=196608, d=6, V=1.3M, F=3 RGB):
        #   raw C=4 path:  blur 11.78 ms, total 16.25 ms
        #   pad C=4 -> 5:  blur  2.54 ms, total  7.20 ms (~2.3x speedup)
        #
        # ON TORCH UPGRADE: re-run tests/nn/bench_permutohedral_d6.py
        # AND the C-sweep at the top of this comment. if the gather cliff
        # is gone (C=4 within 2x of C=3), drop the pad path. if cliff
        # has moved (different multiples), update the predicate below.
        # tracker: report upstream once narrowed to a minimal repro.
        #
        # FUTURE: pad-by-1 wins are large at C=4 (~14x), shrink at high C:
        #   C= 4 -> 5: 0.694 -> 0.048 ms (14x)
        #   C= 8 -> 9: 0.711 -> 0.237 ms (3x)
        #   C=12 ->13: 0.797 -> 0.404 ms (2x)
        #   C=16 ->17: 0.850 -> 0.533 ms (1.6x)
        # for C>=8 a split-into-two strategy beats pad:
        #   C= 8 split (C=5+C=3): 0.048+0.037 = 0.085 ms vs 0.237 (2.8x)
        #   C=12 split (C=5+C=7): 0.048+0.130 = 0.178 ms vs 0.404 (2.3x)
        # not implemented; current consumers (RGB normalize -> C=4) hit
        # the pad sweet spot. revisit if someone benches C>=8.
        # ==================================================================
        if normalize:
            ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)
            f_ext = torch.cat([features, ones], dim=-1)
        else:
            f_ext = features

        lattice = self._splat(f_ext)
        C = lattice.shape[1]
        pad_one = C % 4 == 0
        if pad_one:
            lattice = torch.cat(
                [lattice, lattice.new_zeros(lattice.shape[0], 1)],
                dim=1,
            ).contiguous()

        lattice = self._blur(lattice)

        if query_positions is None:
            out = self._slice(lattice, self.inverse, self.bary)
        else:
            out = self._slice_at_query_positions(lattice, query_positions)

        if pad_one:
            out = out[:, :C].contiguous()

        if not normalize:
            return out
        return out[:, :-1] / out[:, -1:].clamp_min(1e-20)

    def _slice_at_query_positions(
        self,
        lattice: Tensor,
        query_positions: Tensor,
    ) -> Tensor:
        """Slice an already-blurred lattice at arbitrary query positions."""
        assert query_positions.shape[1] == self.d
        d = self.d
        elevated_q = _embed_lattice(query_positions)
        greedy_q, rank_q, bary_q = _find_enclosing_simplex(elevated_q)
        canonical = _canonical_simplex_offsets(d, query_positions.device)
        canonical_per_axis = canonical[:, rank_q].permute(1, 0, 2).contiguous()
        keys_q = greedy_q.unsqueeze(1) + canonical_per_axis
        all_keys_q = keys_q.reshape(-1, d + 1).to(torch.int32)
        idx_q = self.hash_table.search(all_keys_q).long()

        F_ext = lattice.shape[1]
        lattice_pad = torch.cat(
            [lattice, torch.zeros((1, F_ext), dtype=lattice.dtype, device=lattice.device)],
            dim=0,
        )
        idx_q = torch.where(idx_q >= 0, idx_q, torch.full_like(idx_q, lattice.shape[0]))
        return self._slice(lattice_pad, idx_q, bary_q)


# --- Convenience entry point ----------------------------------------------


def permutohedral_filter(
    positions: Tensor,  # (N, d_pos)
    features: Tensor,  # (N, F)
    *,
    sigmas: Tensor | None = None,  # (d_pos,) per-axis bandwidth
    sigma: float | None = None,  # scalar bandwidth (alternative)
    query_positions: Tensor | None = None,
) -> Tensor:
    """One-shot Gaussian filter on a single position tensor.

    Pre-scales positions by ``sigmas`` (or ``sigma``) so one cell == one
    standard deviation. Builds the lattice and filters. With xyz alone this
    is a Gaussian blur in xyz; for a true bilateral (range-aware) filter use
    ``bilateral_permutohedral_filter`` which composes xyz + color into the
    lattice coordinates with separate bandwidths.
    """
    if sigmas is None and sigma is None:
        raise ValueError("Pass either sigmas (per-axis) or sigma (scalar).")
    if sigmas is not None:
        sigmas = torch.as_tensor(sigmas, dtype=positions.dtype, device=positions.device)
        scaled = positions / sigmas
        if query_positions is not None:
            scaled_q = query_positions / sigmas
        else:
            scaled_q = None
    else:
        scaled = positions / sigma
        scaled_q = query_positions / sigma if query_positions is not None else None

    lat = PermutohedralLattice.build(scaled)
    return lat.filter(features, query_positions=scaled_q)


def bilateral_permutohedral_filter(
    src_xyz: Tensor,  # (N, D_xyz)
    src_feat: Tensor,  # (N, D_feat) — range kernel (e.g. RGB)
    src_value: Tensor,  # (N, V)      — values to filter
    *,
    sigma_xyz: float = 0.05,
    sigma_feat: float = 20.0,
    query_xyz: Tensor | None = None,  # (M, D_xyz)
    query_feat: Tensor | None = None,  # (M, D_feat)
    normalize: bool = True,
) -> Tensor:
    """Bilateral (xyz + color) filter via permutohedral lattice.

    Lattice coordinates are ``concat([xyz / sigma_xyz, feat / sigma_feat])`` so
    one cell along an xyz axis equals ``sigma_xyz`` in xyz space and similarly
    for feat. xyz alone would be a plain spatial Gaussian; including ``feat``
    is what makes it bilateral (filters preserve edges in feat).

    Args:
        src_xyz:   (N, D_xyz) spatial positions.
        src_feat:  (N, D_feat) range-kernel features (e.g. RGB).
        src_value: (N, V)      values to splat / filter.
        sigma_xyz: bandwidth applied per xyz axis (scalar).
        sigma_feat: bandwidth applied per feat axis (scalar).
        query_xyz / query_feat: optional cross-position output sites.
        normalize: divide by homogeneous channel for a true Gaussian-weighted
            average (recommended).

    Returns:
        (M or N, V) filtered values.

    Constraints:
        D_xyz + D_feat <= 6 (lattice axes = D_xyz + D_feat + 1 <= 7,
        bounded by PackedHashTable128's D=7).
    """
    assert src_xyz.shape[0] == src_feat.shape[0] == src_value.shape[0]
    d_xyz = src_xyz.shape[1]
    d_feat = src_feat.shape[1]
    if d_xyz + d_feat > 6:
        raise ValueError(
            f"D_xyz + D_feat = {d_xyz + d_feat} > 6; permutohedral lattice "
            f"needs D_xyz + D_feat + 1 axes (<= 7) for PackedHashTable128."
        )

    positions = torch.cat([src_xyz / sigma_xyz, src_feat / sigma_feat], dim=-1)

    if query_xyz is None and query_feat is None:
        query_positions = None
    else:
        if query_xyz is None or query_feat is None:
            raise ValueError("Pass both query_xyz and query_feat, or neither.")
        assert query_xyz.shape[0] == query_feat.shape[0]
        query_positions = torch.cat(
            [query_xyz / sigma_xyz, query_feat / sigma_feat],
            dim=-1,
        )

    lat = PermutohedralLattice.build(positions)
    return lat.filter(src_value, query_positions=query_positions, normalize=normalize)
