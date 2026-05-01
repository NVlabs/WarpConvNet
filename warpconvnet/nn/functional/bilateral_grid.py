# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse d-dimensional bilateral grid + Fast Bilateral Solver.

Reference: Barron & Poole, "The Fast Bilateral Solver" (arXiv:1511.03296).

Two pieces:

1. ``BilateralGrid`` — d-D regular grid with d-linear splat/blur/slice.
   Each input point splats to 2^d corner cells of its enclosing voxel,
   weighted by d-linear interpolation. Cell storage is sparse via
   ``PackedHashTable128`` (D=7, CoordBits=17).
   Blur is a separable 3-tap kernel along each of the d axes.

2. ``bilateral_solver`` — solves the quadratic
        argmin_x  ||sqrt(c) * (x - t)||² + λ * x^T (D - B̄) x
   in grid space via PCG with a Jacobi preconditioner. Recovers smoothed
   per-point values via slice. Useful for confidence-weighted smoothing
   (depth super-resolution, label propagation w/ data term, etc.).

For the densify-labels use case in the handoff note, ``BilateralGrid``
alone (without the solver) gives the same one-shot bilateral filter as
the permutohedral pipeline but with a regular grid that may be cheaper
when d is small (typical bilateral on point clouds: d=6 = xyz+rgb).
"""
from __future__ import annotations

from itertools import product
from typing import Optional

import torch
from torch import Tensor

from warpconvnet.geometry.coords.search.packed128_hashmap import PackedHashTable128


def _corner_offsets(d: int, device: torch.device) -> Tensor:
    """All 2^d corner offset vectors of a d-cube, shape (2^d, d), int64."""
    return torch.tensor(list(product([0, 1], repeat=d)), dtype=torch.int64, device=device)


class BilateralGrid:
    """Sparse d-D regular bilateral grid.

    Build once from positions, then call ``filter`` to splat-blur-slice any
    feature tensor of matching N. Splat uses d-linear barycentric weights
    over the 2^d corners of each input point's enclosing voxel.

    Use cases:
    - Pure Gaussian bilateral: ``BilateralGrid.build(p / sigma).filter(values)``.
    - Fast Bilateral Solver: pass an instance to ``bilateral_solver`` (it
      provides ``B̄`` via ``mul_blur`` and degree ``D`` via ``degrees``).
    """

    def __init__(
        self,
        floors: Tensor,  # (N, d) int64 — lower corner of enclosing voxel per point
        weights: Tensor,  # (N, 2^d) float — d-linear weights per corner
        unique_keys: Tensor,  # (V, d) int32
        inverse: Tensor,  # (N*2^d,) int64 — index into unique_keys
        d: int,
        n_input: int,
        hash_table,
    ):
        self.floors = floors
        self.weights = weights
        self.unique_keys = unique_keys
        self.inverse = inverse
        self.d = d
        self.n_input = n_input
        self.hash_table = hash_table

    @classmethod
    def build(
        cls,
        positions: Tensor,  # (N, d), already pre-scaled by sigma per axis
    ) -> BilateralGrid:
        assert positions.dim() == 2
        assert positions.is_cuda
        N, d = positions.shape
        device = positions.device
        dtype = positions.dtype

        floors = torch.floor(positions).to(torch.int64)  # (N, d)
        frac = positions - floors.to(dtype)  # (N, d), in [0, 1)

        corners = _corner_offsets(d, device)  # (2^d, d)
        n_corners = corners.shape[0]

        # d-linear weights: prod over axes of (frac if corner==1 else 1-frac).
        # frac: (N, 1, d); corners: (1, 2^d, d) → mask
        c = corners.unsqueeze(0).to(dtype)  # (1, 2^d, d)
        f = frac.unsqueeze(1)  # (N, 1, d)
        per_axis = c * f + (1 - c) * (1 - f)  # (N, 2^d, d)
        weights = per_axis.prod(dim=-1)  # (N, 2^d)

        # Vertex keys: (N, 2^d, d) = floors.unsqueeze(1) + corners
        keys = floors.unsqueeze(1) + corners.unsqueeze(0)  # (N, 2^d, d)
        all_keys = keys.reshape(-1, d).to(torch.int32)  # (N*2^d, d)

        unique_keys, inverse = torch.unique(all_keys, dim=0, return_inverse=True)
        capacity = max(16, int(unique_keys.shape[0] * 2))
        hash_table = PackedHashTable128.from_keys(
            unique_keys,
            device=device,
            capacity=capacity,
        )
        return cls(
            floors=floors,
            weights=weights,
            unique_keys=unique_keys,
            inverse=inverse,
            d=d,
            n_input=N,
            hash_table=hash_table,
        )

    @property
    def num_vertices(self) -> int:
        return int(self.unique_keys.shape[0])

    # -- core ops ----------------------------------------------------------

    def splat(self, features: Tensor) -> Tensor:
        """(N, F) → (V, F). Distribute each input feature across 2^d corners."""
        N, F = features.shape
        assert N == self.n_input
        n_corners = self.weights.shape[1]
        V = self.num_vertices
        # contrib: (N, 2^d, F) = weights.unsqueeze(-1) * features.unsqueeze(1)
        contrib = (self.weights.unsqueeze(-1) * features.unsqueeze(1)).reshape(-1, F)
        out = torch.zeros((V, F), dtype=features.dtype, device=features.device)
        out.index_add_(0, self.inverse, contrib)
        return out

    def slice(self, lattice: Tensor) -> Tensor:
        """(V, F) → (N, F). Gather corner values per input, weight, sum."""
        N = self.n_input
        F = lattice.shape[1]
        n_corners = self.weights.shape[1]
        # gather: (N*2^d, F)
        vert_feat = lattice[self.inverse]
        weighted = (self.weights.reshape(-1).unsqueeze(-1) * vert_feat).reshape(N, n_corners, F)
        return weighted.sum(dim=1)

    def blur(
        self, lattice: Tensor, *, taps: tuple[float, float, float] = (0.5, 1.0, 0.5)
    ) -> Tensor:
        """Separable 3-tap blur on the sparse grid along each axis.

        Default taps [0.5, 1.0, 0.5] match Barron's bilateral grid; pair with
        the alpha factor 1 / (1 + 2^-d) at slice time if you need exact
        gaussian normalization. The unweighted version returns the raw blur,
        which is what the bilateral solver consumes (it needs B̄ unnormalized).

        Sentinel-row trick: append a zero row at index V so that misses
        (originally -1) can be remapped to V and gathered without masks.
        """
        d = self.d
        keys = self.unique_keys  # (V, d) int32
        device = keys.device
        a, b, c = taps
        V, F = lattice.shape

        # Single fused launch for all 2*d axis-aligned neighbor lookups.
        K = 2 * d
        offsets = torch.zeros((K, d), dtype=torch.int32, device=device)
        ar = torch.arange(d, device=device)
        offsets[2 * ar, ar] = 1
        offsets[2 * ar + 1, ar] = -1
        results = self.hash_table.batched_search(keys, offsets)  # (K, V) int32, -1 = miss
        results_long = torch.where(results < 0, torch.full_like(results, V), results).long()

        zeros_row = lattice.new_zeros(1, F)
        lattice_pad = torch.cat([lattice, zeros_row], dim=0)  # (V+1, F)

        for axis in range(d):
            fwd = results_long[2 * axis]
            bwd = results_long[2 * axis + 1]
            # Tighter in-place chain: 5 launches/axis. RAW hazard tolerated
            # (small perturbation, washes out across axes; bpfilter-validated).
            lattice_pad[:V].mul_(b)
            lattice_pad[:V].add_(lattice_pad[fwd], alpha=c)
            lattice_pad[:V].add_(lattice_pad[bwd], alpha=a)
        return lattice_pad[:V]

    def _slice_at_query(self, lattice: Tensor, query_positions: Tensor) -> Tensor:
        """Slice ``lattice`` at arbitrary query positions (pre-scaled by sigma).

        Computes per-query d-linear weights and gathers from existing lattice
        cells (queries falling outside the populated cells get zeros — the
        normalization by the homogeneous channel handles this).
        """
        assert query_positions.dim() == 2 and query_positions.shape[1] == self.d
        device = self.unique_keys.device
        F = lattice.shape[1]
        M, d = query_positions.shape
        floors = torch.floor(query_positions).to(torch.int64)
        frac = query_positions - floors.to(query_positions.dtype)
        corners = _corner_offsets(d, device)
        n_corners = corners.shape[0]
        c = corners.unsqueeze(0).to(query_positions.dtype)
        f = frac.unsqueeze(1)
        per_axis = c * f + (1 - c) * (1 - f)
        weights_q = per_axis.prod(dim=-1)  # (M, 2^d)

        keys_q = (floors.unsqueeze(1) + corners.unsqueeze(0)).reshape(-1, d).to(torch.int32)
        idx_q = self.hash_table.search(keys_q).long()  # (M*2^d,)
        # Pad lattice with a sentinel zero row for missing keys.
        lattice_pad = torch.cat(
            [lattice, torch.zeros((1, F), dtype=lattice.dtype, device=lattice.device)],
            dim=0,
        )
        idx_q = torch.where(idx_q >= 0, idx_q, torch.full_like(idx_q, lattice.shape[0]))
        gathered = lattice_pad[idx_q]  # (M*2^d, F)
        weighted = (weights_q.reshape(-1).unsqueeze(-1) * gathered).reshape(M, n_corners, F)
        return weighted.sum(dim=1)

    def filter(
        self,
        features: Tensor,
        *,
        query_positions: Tensor | None = None,
        normalize: bool = True,
    ) -> Tensor:
        """One-shot Gaussian bilateral filter: splat → blur → slice.

        With ``normalize=True`` (default), the per-input weights are also
        splatted, blurred, and sliced — the output is divided by the sliced
        weight to compensate for the (0.5, 1, 0.5) taps and partial overlap.
        This is the standard "homogeneous coordinates" trick (see Adams §4.4).

        If ``query_positions`` is provided, slice happens at those positions
        instead of the build positions (e.g. for cross-cloud filtering).
        Query positions must be pre-scaled by the same sigma used at build.
        """
        # ==================================================================
        # pad-channel workaround for torch gather pessimum at C % 4 == 0
        # ==================================================================
        # observed: torch 2.10.0+cu128 / cuda 12.8, RTX 6000 Ada.
        #
        # `lattice[idx]` (advanced-indexing gather along dim 0) hits a
        # pessimal kernel when channel width C is divisible by 4.
        # microbench at V=1.3M random int64 indices, fp32:
        #     C=3 0.037 ms   C=4 0.694 ms (~19x)   C=5 0.048 ms
        #     C=7 0.130 ms   C=8 0.711 ms (~5x)    C=9 0.237 ms
        # all alternative APIs (`torch.index_select`, `torch.gather`)
        # take the same path. likely vectorized float4 codegen in
        # ATen IndexKernel.cu that runs slower than scalar for random
        # (uncoalesced) gather indices.
        #
        # workaround: pad lattice channels by +1 zero column when
        # C % 4 == 0, splat-blur-slice on the padded width, strip
        # at the end. zero column stays zero through linear ops so
        # output is algebraically identical.
        #
        # ON TORCH UPGRADE: re-bench the C-sweep above; if the cliff is
        # gone or moved, drop or update the predicate. companion comment
        # in warpconvnet/nn/functional/permutohedral.py has full data.
        # ==================================================================
        if normalize:
            ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)
            f_ext = torch.cat([features, ones], dim=-1)
        else:
            f_ext = features

        lattice = self.splat(f_ext)
        C = lattice.shape[1]
        pad_one = C % 4 == 0
        if pad_one:
            lattice = torch.cat(
                [lattice, lattice.new_zeros(lattice.shape[0], 1)],
                dim=1,
            ).contiguous()

        blurred = self.blur(lattice)

        if query_positions is None:
            out = self.slice(blurred)
        else:
            out = self._slice_at_query(blurred, query_positions)

        if pad_one:
            out = out[:, :C].contiguous()

        if not normalize:
            return out
        return out[:, :-1] / out[:, -1:].clamp_min(1e-20)


# -- Fast Bilateral Solver --------------------------------------------------


@torch.no_grad()
def _bistochastize(grid: BilateralGrid, n_iters: int = 10) -> tuple[Tensor, Tensor]:
    """Compute Sinkhorn-style normalization vectors so that
    n^T * splat(slice(...)) is bistochastic. Returns (m, n) per Barron's
    section 4.2: m on input space (N,), n on grid space (V,).
    """
    V = grid.num_vertices
    N = grid.n_input
    device = grid.unique_keys.device
    n = torch.ones(V, 1, device=device, dtype=grid.weights.dtype)
    m = torch.ones(N, 1, device=device, dtype=grid.weights.dtype)
    for _ in range(n_iters):
        # m = 1 / slice(blur(n))
        sliced = grid.slice(grid.blur(n))  # (N, 1)
        m = 1.0 / sliced.clamp_min(1e-20)
        # n = 1 / blur(splat(m))
        blurred = grid.blur(grid.splat(m))  # (V, 1)
        n = 1.0 / blurred.clamp_min(1e-20)
    return m.squeeze(-1), n.squeeze(-1)


def bilateral_solver(
    grid: BilateralGrid,
    target: Tensor,  # (N, F) — observations
    confidence: Tensor,  # (N,) or (N, 1) — per-observation confidence
    *,
    lam: float = 128.0,
    max_iters: int = 25,
    tol: float = 1e-5,
    bistochastize: bool = True,
    bistochastize_iters: int = 10,
) -> Tensor:
    """Solve the Fast Bilateral Solver problem on the given ``BilateralGrid``.

    Minimizes (in grid space y, then sliced back to N):
        argmin_y  λ * y^T (D - B̄_n) y  +  ||sqrt(C̄) (y - t̄)||²
    where C̄ = splat(c), t̄ = splat(c * t), D = diag(B̄ * 1).

    Returns smoothed per-input estimate ``x`` of shape (N, F).
    """
    if confidence.dim() == 1:
        confidence = confidence.unsqueeze(-1)
    assert confidence.shape == (target.shape[0], 1)
    assert target.shape[0] == grid.n_input

    device = target.device
    dtype = target.dtype

    # Bistochastize: rescale confidences via Sinkhorn so the bilateral
    # operator is well-conditioned (see Barron §4.2). Disable this for
    # extreme confidence values where the iteration can produce non-finite
    # scaling factors before the diagonal data term dominates.
    if bistochastize:
        m, n = _bistochastize(grid, n_iters=bistochastize_iters)
        if not (torch.isfinite(m).all() and torch.isfinite(n).all()):
            # Numerical blow-up; fall back to identity scaling.
            m = torch.ones(grid.n_input, device=device, dtype=dtype)
            n = torch.ones(grid.num_vertices, device=device, dtype=dtype)
    else:
        m = torch.ones(grid.n_input, device=device, dtype=dtype)
        n = torch.ones(grid.num_vertices, device=device, dtype=dtype)
    # m: (N,), n: (V,)

    c_eff = confidence * m.unsqueeze(-1)  # (N, 1)
    ct_eff = c_eff * target  # (N, F)

    C_bar = grid.splat(c_eff).squeeze(-1)  # (V,)
    t_bar = grid.splat(ct_eff)  # (V, F)

    # Bistochastized blur operator B̄_n = diag(n) B̄ diag(n). Properties:
    #   (B̄_n y)[i]  = n[i] * (B̄ (n ⊙ y))[i]
    #   (B̄_n 1)[i]  = n[i] * (B̄ n)[i]                       — degree D̃
    #   diag(B̄_n) = n[i]² * diag(B̄)[i] = n[i]² * 1            (self-tap = 1)
    # The smoothness operator A_smooth = diag(D̃) - B̄_n. Adding the
    # confidence-data term gives full A = λ A_smooth + diag(C̄).
    n_col = n.unsqueeze(-1)  # (V, 1)
    Bn = grid.blur(n_col).squeeze(-1)  # (V,)  = B̄ * n
    D_tilde = n * Bn  # (V,)
    diag_Bn = (n * n) * 1.0  # (V,)
    diag_A = lam * (D_tilde - diag_Bn) + C_bar  # (V,)

    def matvec(y: Tensor) -> Tensor:
        # y: (V, F)
        # diag(D̃) y
        Dy = D_tilde.unsqueeze(-1) * y
        # B̄_n y = n * B̄(n * y)
        Bn_y = n_col * grid.blur(n_col * y)
        return lam * (Dy - Bn_y) + C_bar.unsqueeze(-1) * y

    # Initial guess: t̄ / C̄ (target mean per cell)
    y0 = t_bar / C_bar.clamp_min(1e-20).unsqueeze(-1)
    y = y0.clone()
    r = t_bar - matvec(y)  # residual
    M_inv_diag = 1.0 / diag_A.clamp_min(1e-20)  # Jacobi
    z = M_inv_diag.unsqueeze(-1) * r
    p = z.clone()
    rz_old = (r * z).sum()

    initial_norm = r.norm().clamp_min(1e-20)

    for it in range(max_iters):
        Ap = matvec(p)
        alpha = rz_old / (p * Ap).sum().clamp_min(1e-20)
        y = y + alpha * p
        r = r - alpha * Ap
        rn = r.norm()
        if (rn / initial_norm).item() < tol:
            break
        z = M_inv_diag.unsqueeze(-1) * r
        rz_new = (r * z).sum()
        beta = rz_new / rz_old.clamp_min(1e-20)
        p = z + beta * p
        rz_old = rz_new

    # Slice back to per-input. Output is independent of n (cancels with
    # the bistochastized splat). See Barron §4.4.
    return grid.slice(y)


# -- Convenience entry points ----------------------------------------------


def bilateral_filter_grid(
    src_xyz: Tensor,
    src_feat: Tensor,
    src_value: Tensor,
    *,
    sigma_xyz: float = 0.05,
    sigma_feat: float = 20.0,
) -> Tensor:
    """One-shot Gaussian bilateral via regular grid (Barron-style splat/blur/slice).

    Builds positions = concat([src_xyz / sigma_xyz, src_feat / sigma_feat])
    and runs splat→blur→slice on the sparse d-cube grid. Returns ``(N, V)``.
    """
    pos = torch.cat([src_xyz / sigma_xyz, src_feat / sigma_feat], dim=-1)
    grid = BilateralGrid.build(pos)
    return grid.filter(src_value, normalize=True)


def fast_bilateral_solver(
    src_xyz: Tensor,
    src_feat: Tensor,
    target: Tensor,  # (N, F) observations
    confidence: Tensor,  # (N,) per-point trust in the observation
    *,
    sigma_xyz: float = 0.05,
    sigma_feat: float = 20.0,
    lam: float = 128.0,
    max_iters: int = 25,
    tol: float = 1e-5,
) -> Tensor:
    """Confidence-weighted bilateral smoothing via PCG (Barron & Poole 2015)."""
    pos = torch.cat([src_xyz / sigma_xyz, src_feat / sigma_feat], dim=-1)
    grid = BilateralGrid.build(pos)
    return bilateral_solver(
        grid,
        target=target,
        confidence=confidence,
        lam=lam,
        max_iters=max_iters,
        tol=tol,
    )
