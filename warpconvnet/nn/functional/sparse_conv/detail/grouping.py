# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive grouping of kernel map offsets for batched GEMM execution.

Sorts non-identity offsets by pair count, then greedily merges similarly-sized
offsets into buckets with bounded padding waste. Each bucket can be executed as
a single batched GEMM call instead of multiple individual launches.

Reference: Minuet (EuroSys 2024) padding-efficient GEMM grouping.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from warpconvnet.geometry.coords.search.search_results import IntSearchResult


@dataclass
class GroupedKernelMap:
    """Pre-computed grouping of kernel map offsets for batched execution."""

    # Large offsets: process individually (GPU-saturating)
    large_offset_indices: List[int]

    # Small offset buckets: process as batched GEMMs
    buckets: List[List[int]]  # bucket[i] = list of original offset indices

    # Pre-computed concatenated maps per bucket (for implicit GEMM grouped kernel)
    bucket_cat_in_maps: List[Tensor]
    bucket_cat_out_maps: List[Tensor]
    bucket_weight_indices: List[Tensor]  # which weight per pair (for implicit GEMM)

    # Pre-computed flat indices for vectorized gather/scatter into padded (B, max_m) buffers
    # gather_flat_idx[bucket_i][p] = batch_idx * max_m + position_in_batch for pair p
    bucket_gather_flat_idx: List[Tensor]

    # Per-bucket metadata
    bucket_pair_counts: List[List[int]]  # M_k per offset in each bucket
    bucket_max_m: List[int]  # max M per bucket (for padded batching)

    # Original kernel map reference
    kernel_map: IntSearchResult
    identity_map_index: Optional[int]


def generate_padding_buckets(
    pair_counts: List[int],
    offset_indices: List[int],
    threshold: float = 0.1,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Group offsets into buckets with bounded padding waste.

    Sorts offsets by pair count, then greedily merges consecutive offsets
    while the padding redundancy stays below threshold.

    Args:
        pair_counts: M_k for each offset to be grouped.
        offset_indices: Original offset indices corresponding to pair_counts.
        threshold: Maximum fraction of wasted computation from padding.
            E.g., 0.1 means at most 10% of padded computation is wasted.

    Returns:
        Tuple of (buckets_by_original_idx, buckets_by_count) where each bucket
        is a list of original offset indices / pair counts.
    """
    if not pair_counts:
        return [], []

    # Sort by pair count ascending
    sorted_order = sorted(range(len(pair_counts)), key=lambda i: pair_counts[i])
    sorted_counts = [pair_counts[i] for i in sorted_order]
    sorted_indices = [offset_indices[i] for i in sorted_order]

    buckets_idx: List[List[int]] = []
    buckets_count: List[List[int]] = []

    i = 0
    while i < len(sorted_counts):
        bucket_start = i
        actual_total = sorted_counts[i]
        max_m = sorted_counts[i]

        while i + 1 < len(sorted_counts):
            candidate_actual = actual_total + sorted_counts[i + 1]
            candidate_max = max(max_m, sorted_counts[i + 1])
            bucket_size = i + 1 - bucket_start + 1
            candidate_padded = candidate_max * bucket_size
            redundancy = (candidate_padded - candidate_actual) / candidate_actual
            if redundancy > threshold:
                break
            actual_total = candidate_actual
            max_m = candidate_max
            i += 1

        buckets_idx.append(sorted_indices[bucket_start : i + 1])
        buckets_count.append(sorted_counts[bucket_start : i + 1])
        i += 1

    return buckets_idx, buckets_count


def prepare_grouped_kernel_map(
    kernel_map: IntSearchResult,
    grouping_threshold: float = 0.1,
    saturation_m: int = 5000,
) -> GroupedKernelMap:
    """Analyze kernel map and pre-compute grouping for batched execution.

    Args:
        kernel_map: The IntSearchResult from kernel map computation.
        grouping_threshold: Max padding waste fraction for bucket merging.
        saturation_m: Offsets with M_k >= this are processed individually.

    Returns:
        GroupedKernelMap with pre-computed concatenated maps and bucket metadata.
    """
    device = kernel_map.in_maps.device
    iden_idx = kernel_map.identity_map_index
    num_offsets = len(kernel_map)

    # Compute pair counts for all non-identity offsets
    non_iden_indices = []
    non_iden_counts = []
    for k in range(num_offsets):
        if k == iden_idx:
            continue
        count = (kernel_map.offsets[k + 1] - kernel_map.offsets[k]).item()
        if count > 0:
            non_iden_indices.append(k)
            non_iden_counts.append(count)

    # Separate large and small offsets
    large_indices = []
    small_indices = []
    small_counts = []
    for idx, count in zip(non_iden_indices, non_iden_counts):
        if count >= saturation_m:
            large_indices.append(idx)
        else:
            small_indices.append(idx)
            small_counts.append(count)

    # Group small offsets into buckets
    if small_indices:
        buckets, bucket_counts = generate_padding_buckets(
            small_counts, small_indices, threshold=grouping_threshold
        )
    else:
        buckets = []
        bucket_counts = []

    # Pre-compute concatenated maps, weight indices, and flat buffer indices per bucket
    bucket_cat_in_maps = []
    bucket_cat_out_maps = []
    bucket_weight_indices = []
    bucket_gather_flat_idx = []
    bucket_max_m = []

    for bucket_offsets, counts in zip(buckets, bucket_counts):
        in_parts = []
        out_parts = []
        widx_parts = []
        flat_idx_parts = []
        max_m = max(counts)

        for local_k, offset_idx in enumerate(bucket_offsets):
            in_map, out_map = kernel_map[offset_idx]
            n = in_map.shape[0]
            in_parts.append(in_map)
            out_parts.append(out_map)
            widx_parts.append(torch.full((n,), local_k, dtype=torch.int32, device=device))
            # Flat index: batch_idx * max_m + position_within_batch
            flat_idx_parts.append(
                torch.arange(n, dtype=torch.long, device=device) + local_k * max_m
            )

        bucket_cat_in_maps.append(torch.cat(in_parts))
        bucket_cat_out_maps.append(torch.cat(out_parts))
        bucket_weight_indices.append(torch.cat(widx_parts))
        bucket_gather_flat_idx.append(torch.cat(flat_idx_parts))
        bucket_max_m.append(max_m)

    return GroupedKernelMap(
        large_offset_indices=large_indices,
        buckets=buckets,
        bucket_cat_in_maps=bucket_cat_in_maps,
        bucket_cat_out_maps=bucket_cat_out_maps,
        bucket_weight_indices=bucket_weight_indices,
        bucket_gather_flat_idx=bucket_gather_flat_idx,
        bucket_pair_counts=bucket_counts,
        bucket_max_m=bucket_max_m,
        kernel_map=kernel_map,
        identity_map_index=iden_idx,
    )
