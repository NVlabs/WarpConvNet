# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Mask-based fused implicit GEMM for sparse convolution.

Processes all kernel offsets in a single CUDA launch using bitmask-based
offset skipping and mask_argsort for warp-coherent output ordering.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.type_cast import _min_dtype


def _build_pair_table(
    kernel_map: IntSearchResult,
    N_out: int,
    device: torch.device,
) -> Tensor:
    """Build the forward pair_table [K * N_out] from kernel_map."""
    K = len(kernel_map)
    if hasattr(kernel_map, "_pair_table") and kernel_map._pair_table is not None:
        return kernel_map._pair_table.reshape(-1).contiguous()

    pair_table = torch.empty(K * N_out, dtype=torch.int32, device=device)
    pair_table.fill_(-1)
    L = kernel_map.in_maps.shape[0]
    if L > 0 and hasattr(_C.gemm, "csr_to_pair_table_cuda"):
        offsets_gpu = kernel_map.offsets.to(device=device, dtype=torch.int32)
        _C.gemm.csr_to_pair_table_cuda(
            kernel_map.in_maps.int(),
            kernel_map.out_maps.int(),
            offsets_gpu,
            pair_table,
            N_out,
            K,
        )
    return pair_table


def _dispatched_mask_words(K: int) -> int:
    """Round mask_words up to the next DISPATCH_MW template boundary.

    production_bindings.cu's DISPATCH_MW macro picks templates at MW=1, 2,
    4, 8, 12. Kernel templates use their compile-time MW as the stride
    when indexing ``pair_mask[row * MW + word]``. If the caller allocates
    pair_mask with a smaller stride than the dispatched MW, the kernel
    reads past the allocation (or into the wrong row) → illegal address
    and silent-wrong output.

    Ensure the pair_mask stride matches what the kernel will use:
    pad to the next template MW.
    """
    mw_rt = (K + 31) // 32
    for mw in (1, 2, 4, 8, 12):
        if mw_rt <= mw:
            return mw
    return mw_rt  # K > 384: template doesn't cover this, caller must reject


def _build_mask_and_argsort(
    pair_table: Tensor,
    N: int,
    K: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Build pair_mask and mask_argsort from a pair_table [K * N].

    For K <= 32: pair_mask is [N] int32 (single uint32 bitmask per voxel).
    For K > 32: pair_mask is [N * mask_words_padded] int32, interleaved as
                pair_mask[voxel_i * mask_words_padded + word_w].
                mask_words_padded is the next DISPATCH_MW template boundary
                so the kernel's stride matches what the caller allocates.
    mask_argsort is always [N] int32 (voxel permutation).
    """
    mask_words = _dispatched_mask_words(K)
    pair_mask = torch.zeros(N * mask_words, dtype=torch.int32, device=device)
    _C.gemm.build_pair_mask_cuda(pair_table, pair_mask, K, mask_words)
    if mask_words == 1:
        mask_argsort = torch.argsort(pair_mask, stable=True).int()
    else:
        # Sort by first word for warp-coherent grouping
        sort_key = pair_mask[::mask_words]  # word 0 of each voxel
        mask_argsort = torch.argsort(sort_key, stable=True).int()
    return pair_mask, mask_argsort


def _kernel_map_to_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert IntSearchResult to mask-based pair_table + mask + argsort.

    Returns:
        pair_table: [K * N_out] int32, flattened
        pair_mask: [N_out * mask_words] int32 (uint32 bitmask, interleaved)
        mask_argsort: [N_out] int32 permutation
    """
    K = len(kernel_map)
    N_out = num_out_coords
    pair_table = _build_pair_table(kernel_map, N_out, device)
    pair_mask, mask_argsort = _build_mask_and_argsort(pair_table, N_out, K, device)
    return pair_table, pair_mask, mask_argsort


def _build_reverse_mask_data(
    pair_table: Tensor,
    N_in: int,
    N_out: int,
    K: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build reverse pair_table + mask + argsort for atomicAdd-free dgrad.

    The forward pair_table maps (offset_k, out_row) -> in_row.
    The reverse maps (offset_k, in_row) -> out_row, enabling the dgrad
    kernel to iterate over input rows and gather from grad_output.

    Returns:
        reverse_pair_table: [K * N_in] int32
        reverse_pair_mask: [N_in * mask_words] int32 (uint32 bitmask, interleaved)
        reverse_mask_argsort: [N_in] int32 permutation
    """
    pair_table_2d = pair_table.reshape(K, N_out)
    reverse_pair_table = torch.full((K, N_in), -1, dtype=torch.int32, device=device)

    # Vectorized reverse: scatter all K offsets at once
    valid = pair_table_2d >= 0  # [K, N_out] bool
    k_idx, out_idx = torch.where(valid)  # flat indices of valid entries
    in_idx = pair_table_2d[k_idx, out_idx].long()  # corresponding input rows
    reverse_pair_table[k_idx, in_idx] = out_idx.int()

    reverse_flat = reverse_pair_table.reshape(-1).contiguous()
    reverse_pair_mask, reverse_mask_argsort = _build_mask_and_argsort(
        reverse_flat, N_in, K, device
    )
    return reverse_flat, reverse_pair_mask, reverse_mask_argsort


def _get_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get or compute mask data, cached on the kernel_map object."""
    if kernel_map._mask_data is None:
        kernel_map._mask_data = _kernel_map_to_mask_data(kernel_map, num_out_coords, device)
    return kernel_map._mask_data


def _get_reverse_mask_data(
    kernel_map: IntSearchResult,
    num_in_coords: int,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get or compute reverse mask data, cached on the kernel_map object."""
    if kernel_map._reverse_mask_data is None:
        K = len(kernel_map)
        fwd_pair_table, _, _ = _get_mask_data(kernel_map, num_out_coords, device)
        kernel_map._reverse_mask_data = _build_reverse_mask_data(
            fwd_pair_table, num_in_coords, num_out_coords, K, device
        )
    return kernel_map._reverse_mask_data
