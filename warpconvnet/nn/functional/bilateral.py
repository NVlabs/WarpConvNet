# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bilateral filter on point clouds.

For each query point we gather K spatial neighbors and weight them by both
spatial distance (sigma_xyz) and feature distance (sigma_feat), then
aggregate. Two backends:

* ``mode="knn"`` (default): uses warpconvnet's chunked KNN to get a fixed-K
  neighborhood. Predictable memory; recommended.
* ``mode="radius"``: uses warpconvnet's wp.HashGrid radius search to get a
  variable-K neighborhood inside a 3-sigma ball. Lower work per query for
  highly non-uniform densities.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _make_offsets(n: int, device: torch.device) -> Tensor:
    return torch.tensor([0, n], dtype=torch.int32, device=device)


def _gaussian_weights(
    diff_sq: Tensor,  # (M, K) squared distance per neighbor
    inv_two_sigma_sq: float,
) -> Tensor:
    return torch.exp(-diff_sq * inv_two_sigma_sq)


def bilateral_filter(
    src_xyz: Tensor,  # (N, D_xyz)
    src_feat: Tensor,  # (N, D_feat)  — used for the range kernel
    src_value: Tensor,  # (N, V)       — values to filter
    query_xyz: Tensor | None = None,  # (M, D_xyz); default = src_xyz (self-filter)
    query_feat: Tensor | None = None,  # (M, D_feat); default = src_feat
    *,
    sigma_xyz: float = 0.05,
    sigma_feat: float = 20.0,
    k: int = 16,
    mode: str = "knn",
    radius_mult: float = 3.0,
    chunk_size: int = 32768,
) -> Tensor:
    """Bilateral-weighted aggregation of ``src_value`` onto ``query_xyz``.

    Returns ``(M, V)`` of the same dtype as ``src_value``.

    Notes:
        ``sigma_feat`` units match ``src_feat`` (e.g. 0-255 for raw RGB,
        or 0-1 for normalized RGB — be consistent).
        For self-filtering pass nothing for ``query_xyz`` / ``query_feat``.
    """
    assert src_xyz.dim() == 2 and src_feat.dim() == 2 and src_value.dim() == 2
    assert src_xyz.shape[0] == src_feat.shape[0] == src_value.shape[0]
    if query_xyz is None:
        query_xyz = src_xyz
    if query_feat is None:
        query_feat = src_feat
    assert query_xyz.shape[0] == query_feat.shape[0]

    device = src_xyz.device
    M = query_xyz.shape[0]
    inv_2sx2 = 1.0 / (2.0 * sigma_xyz * sigma_xyz)
    inv_2sf2 = 1.0 / (2.0 * sigma_feat * sigma_feat)

    if mode == "knn":
        # Use warpconvnet's chunked KNN. Returns (M, K) int64 neighbor indices.
        from warpconvnet.geometry.coords.search.knn import knn_search

        nbr = knn_search(
            ref_positions=src_xyz,
            query_positions=query_xyz,
            k=k,
            search_method="chunk",
            chunk_size=chunk_size,
        ).long()  # (M, K)

        # Gather neighbor xyz/feat/value
        nbr_xyz = src_xyz[nbr]  # (M, K, D_xyz)
        nbr_feat = src_feat[nbr]  # (M, K, D_feat)
        nbr_value = src_value[nbr]  # (M, K, V)

        d_xyz_sq = ((nbr_xyz - query_xyz.unsqueeze(1)) ** 2).sum(dim=-1)  # (M, K)
        d_feat_sq = ((nbr_feat - query_feat.unsqueeze(1)) ** 2).sum(dim=-1)  # (M, K)
        w = torch.exp(-d_xyz_sq * inv_2sx2 - d_feat_sq * inv_2sf2)  # (M, K)
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-20)
        out = (w.unsqueeze(-1) * nbr_value).sum(dim=1) / w_sum
        return out.to(src_value.dtype)

    elif mode == "radius":
        # Warp's torch interop uses warp.context.runtime which is None until
        # wp.init() runs. Trigger it explicitly to avoid an unhelpful
        # AttributeError deep in wp.from_torch.
        import warp as wp

        if wp.context.runtime is None:
            wp.init()
        from warpconvnet.geometry.coords.search.radius import batched_radius_search

        radius = float(radius_mult * sigma_xyz)
        ref_off = _make_offsets(src_xyz.shape[0], device)
        q_off = _make_offsets(M, device)
        # batched_radius_search returns (neighbor_idx [Q], distances [Q], row_splits [M+1])
        idx, dist, splits = batched_radius_search(
            ref_positions=src_xyz,
            ref_offsets=ref_off,
            query_positions=query_xyz,
            query_offsets=q_off,
            radius=radius,
        )
        idx = idx.long()
        # Per-neighbor squared distance — radius backend may return raw distance, not squared.
        d_xyz_sq = (dist * dist) if dist is not None else None
        if d_xyz_sq is None:
            # Recompute defensively — query index per neighbor:
            row_lens = splits[1:] - splits[:-1]
            qpos_per_nbr = torch.repeat_interleave(query_xyz, row_lens, dim=0)
            d_xyz_sq = ((src_xyz[idx] - qpos_per_nbr) ** 2).sum(dim=-1)

        # Feature distance per neighbor
        row_lens = splits[1:] - splits[:-1]
        qfeat_per_nbr = torch.repeat_interleave(query_feat, row_lens, dim=0)
        d_feat_sq = ((src_feat[idx] - qfeat_per_nbr) ** 2).sum(dim=-1)
        w = torch.exp(-d_xyz_sq * inv_2sx2 - d_feat_sq * inv_2sf2)  # (Q,)

        # Aggregate per query via segment_csr (torch_scatter is in the venv)
        from torch_scatter import segment_csr

        weighted_v = w.unsqueeze(-1) * src_value[idx]  # (Q, V)
        num = segment_csr(weighted_v, splits.long(), reduce="sum")  # (M, V)
        den = segment_csr(w, splits.long(), reduce="sum").clamp_min(1e-20)  # (M,)
        out = num / den.unsqueeze(-1)

        # Queries with zero neighbors get 0 — leave as-is so caller can detect.
        return out.to(src_value.dtype)

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'knn' or 'radius'.")


def bilateral_label_propagate(
    src_xyz: Tensor,  # (N, 3)
    src_feat: Tensor,  # (N, F) — color in the typical case
    src_labels: Tensor,  # (N,)   — int labels (-1 = background)
    dst_xyz: Tensor,  # (M, 3)
    dst_feat: Tensor,  # (M, F)
    *,
    num_classes: int | None = None,
    sigma_xyz: float = 0.03,
    sigma_feat: float = 20.0,
    k: int = 16,
    mode: str = "knn",
    background_label: int = -1,
) -> Tensor:
    """Bilateral label propagation.

    For each dst point, compute the bilateral-weighted vote across its K
    spatial neighbors in ``src``, restricted to neighbors whose label is not
    ``background_label``. Returns ``(M,)`` int64 of propagated labels with
    ``background_label`` for points that had no valid neighbors in range.

    This is the densification primitive needed by
    ``datagen/scripts/densify_labels.py`` to replace the plain-NN baseline.
    """
    assert src_labels.dim() == 1 and src_labels.shape[0] == src_xyz.shape[0]
    device = src_xyz.device

    # Mask out background source points by setting their weight to zero.
    valid = src_labels != background_label  # (N,)

    if num_classes is None:
        max_lbl = int(src_labels[valid].max().item()) if valid.any() else -1
        num_classes = max_lbl + 1
    if num_classes <= 0:
        return torch.full((dst_xyz.shape[0],), background_label, dtype=torch.long, device=device)

    # One-hot for valid labels; background rows stay all-zero so they
    # contribute zero weight to any class.
    onehot = torch.zeros((src_xyz.shape[0], num_classes), dtype=torch.float32, device=device)
    onehot[valid, src_labels[valid].long()] = 1.0

    soft = bilateral_filter(
        src_xyz=src_xyz,
        src_feat=src_feat,
        src_value=onehot,
        query_xyz=dst_xyz,
        query_feat=dst_feat,
        sigma_xyz=sigma_xyz,
        sigma_feat=sigma_feat,
        k=k,
        mode=mode,
    )  # (M, C)

    # Argmax among classes; queries with all-zero rows (no valid neighbors) → background.
    max_v, max_c = soft.max(dim=-1)
    out = torch.where(
        max_v > 0, max_c.long(), torch.full_like(max_c, background_label, dtype=torch.long)
    )
    return out
