# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Mask-based fused implicit GEMM for sparse convolution.

Processes all kernel offsets in a single CUDA launch using bitmask-based
offset skipping and mask_argsort for warp-coherent output ordering.
"""

import os
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.type_cast import _min_dtype


# Sort strategy for mask_argsort. Controls how voxels are ordered prior to
# kernel dispatch:
#   - "mask_bit": stable argsort on the raw uint32 pair_mask word(s). Groups
#                 voxels with identical bitmasks contiguously (default).
#   - "gray_code": treat pair_mask as a Gray code, decode to binary, and sort
#                  by the decoded key. Induces a Gray-order linearization
#                  so consecutive blocks see Hamming-adjacent active-offset
#                  patterns. Expected to improve cache reuse on output rows
#                  when a block transitions between mask groups.
# Override at process start via WARPCONVNET_MASK_SORT={mask_bit,gray_code}.
_MaskSortStrategy = Literal["mask_bit", "gray_code"]


def _default_mask_sort_strategy() -> _MaskSortStrategy:
    val = os.environ.get("WARPCONVNET_MASK_SORT", "mask_bit").strip().lower()
    if val not in ("mask_bit", "gray_code"):
        # Unknown value: fall back to default rather than crash. Mis-typed
        # env vars must not break correctness.
        return "mask_bit"
    return val  # type: ignore[return-value]


def _gray_to_binary_uint32(x: Tensor) -> Tensor:
    """Decode a Gray-code uint32 tensor to its binary representation.

    Standard inverse-Gray: binary[i] = XOR of bits {i, i+1, ..., 31} of gray.
    Implemented as iterated `x ^= x >> shift` doublings (5 steps for 32 bits).
    Operates element-wise; preserves shape and dtype.
    """
    # Cast to int64 to avoid signed-shift surprises while keeping bit-exact
    # uint32 semantics. Final result re-cast to int32 by caller.
    y = x.to(torch.int64) & 0xFFFFFFFF
    y = y ^ (y >> 1)
    y = y ^ (y >> 2)
    y = y ^ (y >> 4)
    y = y ^ (y >> 8)
    y = y ^ (y >> 16)
    return y


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

    mask_gemm_bindings.cu's DISPATCH_MW macro picks templates at MW=1, 2,
    4, 8, 12. Kernel templates use their compile-time MW as the stride
    when indexing ``pair_mask[row * MW + word]``. If the caller allocates
    pair_mask with a smaller stride than the dispatched MW, the kernel
    reads past the allocation (or into the wrong row) → illegal address
    and silent-wrong output.

    The set is intentionally hardcoded here: it mirrors DISPATCH_MW in the
    binding, which is a warpconvnet-side decision independent of per-tile
    warpgemm metadata. Warpgemm's TileMetadata.mask_words reports a tile's
    canonical compiled MW, not the macro-level dispatch ladder.
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
    sort_strategy: Optional[_MaskSortStrategy] = None,
) -> Tuple[Tensor, Tensor]:
    """Build pair_mask and mask_argsort from a pair_table [K * N].

    For K <= 32: pair_mask is [N] int32 (single uint32 bitmask per voxel).
    For K > 32: pair_mask is [N * mask_words_padded] int32, interleaved as
                pair_mask[voxel_i * mask_words_padded + word_w].
                mask_words_padded is the next DISPATCH_MW template boundary
                so the kernel's stride matches what the caller allocates.
    mask_argsort is always [N] int32 (voxel permutation).

    sort_strategy: see _default_mask_sort_strategy() docstring. None reads
    the WARPCONVNET_MASK_SORT env var (default "mask_bit"). The choice is
    semantic-preserving — both yield valid permutations of [0, N).
    """
    mask_words = _dispatched_mask_words(K)
    pair_mask = torch.zeros(N * mask_words, dtype=torch.int32, device=device)
    _C.gemm.build_pair_mask_cuda(pair_table, pair_mask, K, mask_words)

    strategy: _MaskSortStrategy = (
        sort_strategy if sort_strategy is not None else _default_mask_sort_strategy()
    )

    if mask_words == 1:
        word0 = pair_mask
    else:
        # Stride view of word 0 of each voxel. .contiguous() forces a copy
        # so the sort/decoded key isn't strided in subsequent ops.
        word0 = pair_mask[::mask_words].contiguous()

    if strategy == "gray_code":
        # Decode pair_mask (Gray) -> binary, then stable-sort. This puts
        # voxels with Hamming-adjacent mask bits at adjacent positions,
        # improving output-row cache reuse across consecutive blocks.
        # For mask_words > 1, the decoded word-0 key is the dominant
        # signal; ties on word 0 fall back to the natural (stable) order
        # which still groups identical patterns. We deliberately don't
        # multi-key sort the higher words — empirically the fwd kernel
        # only consults word 0 first (NB: a future opt could chain).
        key = _gray_to_binary_uint32(word0).int()
    else:  # "mask_bit" (default, legacy)
        key = word0

    # Fast path: cub::DeviceRadixSort via direct binding bypasses torch.argsort
    # Python+dispatcher overhead. Saves ~150us per call at N=2928 vs
    # torch.argsort(stable=True). Non-stable: voxels with identical mask
    # may be reordered within their group — usually semantic-preserving for
    # cache coherence, but under investigation as a slow-drift training
    # convergence delta vs previous stable-sort behavior.
    #
    # Set WARPCONVNET_FORCE_STABLE_ARGSORT=1 to force torch.argsort(stable=True)
    # — diagnostic A/B for train-trajectory-divergence checks. Adds
    # ~150us per call; only set when investigating numerics.
    _force_stable = os.environ.get("WARPCONVNET_FORCE_STABLE_ARGSORT", "0").strip() in (
        "1",
        "true",
        "True",
    )
    if _force_stable or not hasattr(_C.gemm, "mask_argsort_cuda"):
        mask_argsort = torch.argsort(key, stable=True).int()
    else:
        mask_argsort = torch.empty(N, dtype=torch.int32, device=device)
        _C.gemm.mask_argsort_cuda(key.contiguous(), mask_argsort)
    return pair_mask, mask_argsort


def _kernel_map_to_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
    sort_strategy: Optional[_MaskSortStrategy] = None,
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
    pair_mask, mask_argsort = _build_mask_and_argsort(
        pair_table, N_out, K, device, sort_strategy=sort_strategy
    )
    return pair_table, pair_mask, mask_argsort


def _build_reverse_mask_data(
    pair_table: Tensor,
    N_in: int,
    N_out: int,
    K: int,
    device: torch.device,
    sort_strategy: Optional[_MaskSortStrategy] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build reverse pair_table + mask + argsort for atomicAdd-free dgrad.

    The forward pair_table maps (offset_k, out_row) -> in_row.
    The reverse maps (offset_k, in_row) -> out_row, enabling the dgrad
    kernel to iterate over input rows and gather from grad_output.

    Fast path uses a fused CUDA kernel that emits reverse_pair_table AND
    reverse_pair_mask in a single launch (atomicOr on the bitmask).
    Eliminates ~0.7-1.0ms of host-driven torch.where + scatter + separate
    pair_mask launch at small-N high-K shapes.

    Returns:
        reverse_pair_table: [K * N_in] int32
        reverse_pair_mask: [N_in * mask_words] int32 (uint32 bitmask, interleaved)
        reverse_mask_argsort: [N_in] int32 permutation
    """
    mask_words = _dispatched_mask_words(K)

    if hasattr(_C.gemm, "build_reverse_mask_data_cuda"):
        reverse_pair_table = torch.full((K * N_in,), -1, dtype=torch.int32, device=device)
        reverse_pair_mask = torch.zeros(N_in * mask_words, dtype=torch.int32, device=device)
        _C.gemm.build_reverse_mask_data_cuda(
            pair_table.contiguous(),
            reverse_pair_table,
            reverse_pair_mask,
            N_in,
            N_out,
            K,
            mask_words,
        )
        reverse_flat = reverse_pair_table

        strategy: _MaskSortStrategy = (
            sort_strategy if sort_strategy is not None else _default_mask_sort_strategy()
        )
        if mask_words == 1:
            word0 = reverse_pair_mask
        else:
            word0 = reverse_pair_mask[::mask_words].contiguous()
        if strategy == "gray_code":
            key = _gray_to_binary_uint32(word0).int()
        else:
            key = word0
        if hasattr(_C.gemm, "mask_argsort_cuda"):
            reverse_mask_argsort = torch.empty(N_in, dtype=torch.int32, device=device)
            _C.gemm.mask_argsort_cuda(key.contiguous(), reverse_mask_argsort)
        else:
            reverse_mask_argsort = torch.argsort(key, stable=True).int()
        return reverse_flat, reverse_pair_mask, reverse_mask_argsort

    # Legacy fallback: torch.where + scatter, then separate pair_mask launch.
    pair_table_2d = pair_table.reshape(K, N_out)
    reverse_pair_table = torch.full((K, N_in), -1, dtype=torch.int32, device=device)

    valid = pair_table_2d >= 0
    k_idx, out_idx = torch.where(valid)
    in_idx = pair_table_2d[k_idx, out_idx].long()
    reverse_pair_table[k_idx, in_idx] = out_idx.int()

    reverse_flat = reverse_pair_table.reshape(-1).contiguous()
    reverse_pair_mask, reverse_mask_argsort = _build_mask_and_argsort(
        reverse_flat, N_in, K, device, sort_strategy=sort_strategy
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
