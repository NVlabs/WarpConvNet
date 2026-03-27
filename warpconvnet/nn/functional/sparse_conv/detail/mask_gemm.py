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


def _build_mask_and_argsort(
    pair_table: Tensor,
    N: int,
    K: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Build pair_mask and mask_argsort from a pair_table [K * N]."""
    pair_mask = torch.zeros(N, dtype=torch.int32, device=device)
    if K <= 32 and hasattr(_C.gemm, "build_pair_mask_cuda"):
        _C.gemm.build_pair_mask_cuda(pair_table, pair_mask, K)
    elif K <= 32:
        pair_table_2d = pair_table.reshape(K, N)
        valid = pair_table_2d >= 0
        bit_positions = (1 << torch.arange(K, device=device, dtype=torch.int32)).unsqueeze(1)
        pair_mask = (valid.int() * bit_positions).sum(dim=0).int()
    mask_argsort = torch.argsort(pair_mask, stable=True).int()
    return pair_mask, mask_argsort


def _kernel_map_to_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert IntSearchResult to mask-based pair_table + mask + argsort.

    Returns:
        pair_table: [K * N_out] int32, flattened
        pair_mask: [N_out] int32 (uint32 bitmask)
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
        reverse_pair_mask: [N_in] int32 (uint32 bitmask)
        reverse_mask_argsort: [N_in] int32 permutation
    """
    pair_table_2d = pair_table.reshape(K, N_out)
    reverse_pair_table = torch.full((K, N_in), -1, dtype=torch.int32, device=device)

    for k in range(K):
        valid = pair_table_2d[k] >= 0
        out_rows = torch.where(valid)[0].int()
        in_rows = pair_table_2d[k, valid].long()
        reverse_pair_table[k].scatter_(0, in_rows, out_rows)

    reverse_flat = reverse_pair_table.reshape(-1).contiguous()
    reverse_pair_mask, reverse_mask_argsort = _build_mask_and_argsort(
        reverse_flat, N_in, K, device
    )
    return reverse_flat, reverse_pair_mask, reverse_mask_argsort


# Cache mask data by content hash of kernel_map offsets.
# This survives across different Python objects that represent the same mapping.
# Keys: (offsets_tuple, num_out_coords) for forward data
#        (offsets_tuple, num_out_coords, "reverse", num_in_coords) for reverse dgrad data
_MASK_DATA_CACHE = {}
_MASK_DATA_CACHE_MAX_SIZE = 64  # Evict oldest entries when cache exceeds this


def _content_key(kernel_map: IntSearchResult, num_out_coords: int):
    """Content-based cache key from kernel_map offsets (28 ints for K=27)."""
    offsets = kernel_map.offsets
    if offsets.is_cuda:
        offsets = offsets.cpu()
    return (tuple(offsets.tolist()), num_out_coords)


def _get_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get or compute cached forward mask data for a kernel_map."""
    cache_key = _content_key(kernel_map, num_out_coords)
    if cache_key not in _MASK_DATA_CACHE:
        if len(_MASK_DATA_CACHE) >= _MASK_DATA_CACHE_MAX_SIZE:
            # Evict oldest entries (FIFO)
            for _ in range(len(_MASK_DATA_CACHE) // 4):
                _MASK_DATA_CACHE.pop(next(iter(_MASK_DATA_CACHE)))
        _MASK_DATA_CACHE[cache_key] = _kernel_map_to_mask_data(kernel_map, num_out_coords, device)
    return _MASK_DATA_CACHE[cache_key]


def _get_reverse_mask_data(
    kernel_map: IntSearchResult,
    num_in_coords: int,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get or compute cached reverse mask data for atomicAdd-free dgrad."""
    K = len(kernel_map)
    base_key = _content_key(kernel_map, num_out_coords)
    cache_key = (*base_key, "reverse", num_in_coords)
    if cache_key not in _MASK_DATA_CACHE:
        fwd_pair_table, _, _ = _get_mask_data(kernel_map, num_out_coords, device)
        _MASK_DATA_CACHE[cache_key] = _build_reverse_mask_data(
            fwd_pair_table, num_in_coords, num_out_coords, K, device
        )
    return _MASK_DATA_CACHE[cache_key]


def _mask_implicit_gemm_forward_logic(
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
    block_size: int = 16,
    mma_tile: int = 3,
) -> Tensor:
    """Forward pass using mask-based fused implicit GEMM."""
    device = in_features.device
    feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
    min_dtype = _min_dtype(feature_dtype, weight.dtype)

    _in_features = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight = weight.contiguous().detach().to(dtype=min_dtype)

    N_in, C_in = _in_features.shape
    K, _, C_out = _weight.shape

    output = torch.zeros((num_out_coords, C_out), dtype=min_dtype, device=device)

    if num_out_coords == 0 or K == 0 or C_in == 0 or C_out == 0 or N_in == 0:
        return output.to(dtype=in_features.dtype)

    pair_table, pair_mask, mask_argsort = _get_mask_data(kernel_map, num_out_coords, device)

    # Auto-pad unaligned channels for CuTe tensor core eligibility
    _has_cute = hasattr(_C.gemm, "cute_gemm_mask_fwd")
    vec_width = 16 // _in_features.element_size()  # 8 for fp16/bf16
    orig_C_in, orig_C_out = C_in, C_out
    needs_padding = (C_in % vec_width != 0) or (C_out % vec_width != 0)
    if needs_padding and _has_cute and min_dtype in (torch.float16, torch.bfloat16):
        target_cin = ((C_in + vec_width - 1) // vec_width) * vec_width
        target_cout = ((C_out + vec_width - 1) // vec_width) * vec_width
        _in_features = torch.nn.functional.pad(_in_features, (0, target_cin - C_in))
        _weight = torch.nn.functional.pad(_weight, (0, target_cout - C_out, 0, target_cin - C_in))
        output = torch.zeros((num_out_coords, target_cout), dtype=min_dtype, device=device)
        C_in, C_out = target_cin, target_cout
    aligned = True  # After padding, always aligned

    if _has_cute and aligned and min_dtype in (torch.float16, torch.bfloat16):
        status = _C.gemm.cute_gemm_mask_fwd(
            _in_features,
            _weight,
            output,
            pair_table,
            pair_mask,
            mask_argsort,
            K,
            mma_tile,
            1.0,
        )
        if status == 0:
            if needs_padding:
                output = output[:, :orig_C_out]
            return output.to(dtype=in_features.dtype)
        # CuTe failed — signal error so auto-tuner skips this algo
        raise RuntimeError(
            f"cute_gemm_mask_fwd failed with status {status} "
            f"(N={num_out_coords}, C_in={C_in}, C_out={C_out}, K={K})"
        )

    # No CuTe available or channels unaligned — signal unsupported
    raise RuntimeError(
        f"mask_implicit_gemm requires aligned channels for CuTe " f"(C_in={C_in}, C_out={C_out})"
    )


def _mask_implicit_gemm_backward_logic(
    grad_output: Tensor,
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
    needs_input_grad: Tuple[bool, ...] = (True, True),
    block_size: int = 16,
    mma_tile: int = 3,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Backward pass using mask-based fused implicit GEMM."""
    device = in_features.device
    feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
    min_dtype = _min_dtype(feature_dtype, weight.dtype)

    _grad_output = grad_output.contiguous().detach().to(dtype=min_dtype)
    _in_features = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight = weight.contiguous().detach().to(dtype=min_dtype)

    N_in, C_in = _in_features.shape
    K, _, C_out = _weight.shape

    pair_table, pair_mask, mask_argsort = _get_mask_data(kernel_map, num_out_coords, device)

    grad_in = None
    grad_weight = None

    if needs_input_grad[0]:
        _has_cute_fwd = hasattr(_C.gemm, "cute_gemm_mask_fwd")
        vec_width_bwd = 16 // _grad_output.element_size()
        orig_C_in_bwd, orig_C_out_bwd = C_in, C_out
        _go_bwd, _w_bwd = _grad_output, _weight
        needs_padding_bwd = (C_in % vec_width_bwd != 0) or (C_out % vec_width_bwd != 0)
        if needs_padding_bwd and _has_cute_fwd and min_dtype in (torch.float16, torch.bfloat16):
            tc = ((C_in + vec_width_bwd - 1) // vec_width_bwd) * vec_width_bwd
            tco = ((C_out + vec_width_bwd - 1) // vec_width_bwd) * vec_width_bwd
            _go_bwd = torch.nn.functional.pad(_grad_output, (0, tco - C_out))
            _w_bwd = torch.nn.functional.pad(_weight, (0, tco - C_out, 0, tc - C_in))
            grad_in = torch.zeros((N_in, tc), dtype=min_dtype, device=device)
        else:
            grad_in = torch.zeros((N_in, C_in), dtype=min_dtype, device=device)
        used_cute = False

        if _has_cute_fwd and min_dtype in (torch.float16, torch.bfloat16):
            # AtomicAdd-free dgrad: reuse the forward kernel with reverse
            # pair data. The reverse pair_table maps (offset_k, in_row) ->
            # out_row, so the kernel iterates over input rows and gathers
            # from grad_output — no scatter, no atomicAdd.
            rev_pair_table, rev_pair_mask, rev_argsort = _get_reverse_mask_data(
                kernel_map, N_in, num_out_coords, device
            )
            # Weight transposed: [K, C_in, C_out] -> [K, C_out, C_in]
            # The forward kernel loads B[k] as [C_in_kernel, C_out_kernel].
            # For dgrad: C_in_kernel=C_out, C_out_kernel=C_in, so B[k]
            # should be [C_out, C_in] = weight.transpose(-1,-2).
            _weight_T = _w_bwd.transpose(1, 2).contiguous()
            status = _C.gemm.cute_gemm_mask_fwd(
                _go_bwd,  # "input": grad_output [N_out, C_out]
                _weight_T,  # "weight": [K, C_out, C_in]
                grad_in,  # "output": grad_input [N_in, C_in]
                rev_pair_table,
                rev_pair_mask,
                rev_argsort,
                K,
                mma_tile,
                1.0,
            )
            if status == 0:
                used_cute = True

        if not used_cute:
            # SIMT fallback (still uses atomicAdd scatter)
            status = _C.gemm.mask_implicit_gemm_bwd_dgrad(
                _grad_output,
                _weight,
                grad_in,
                pair_table,
                pair_mask,
                mask_argsort,
                K,
                block_size,
            )
            if status != 0:
                raise RuntimeError(f"mask_implicit_gemm_bwd_dgrad failed: {status}")

        if needs_padding_bwd:
            grad_in = grad_in[:, :orig_C_in_bwd]
        grad_in = grad_in.to(dtype=in_features.dtype)

    if needs_input_grad[1]:
        _has_cute_wgrad = hasattr(_C.gemm, "cute_gemm_mask_wgrad")
        vec_width_w = 16 // _in_features.element_size()
        orig_C_in_w, orig_C_out_w = C_in, C_out
        _in_w, _go_w = _in_features, _grad_output
        needs_padding_w = (C_in % vec_width_w != 0) or (C_out % vec_width_w != 0)
        if needs_padding_w and _has_cute_wgrad and min_dtype in (torch.float16, torch.bfloat16):
            tc = ((C_in + vec_width_w - 1) // vec_width_w) * vec_width_w
            tco = ((C_out + vec_width_w - 1) // vec_width_w) * vec_width_w
            _in_w = torch.nn.functional.pad(_in_features, (0, tc - C_in))
            _go_w = torch.nn.functional.pad(_grad_output, (0, tco - C_out))
            grad_weight = torch.zeros((K, tc, tco), dtype=min_dtype, device=device)
        else:
            grad_weight = torch.zeros((K, C_in, C_out), dtype=min_dtype, device=device)
        used_cute = False

        if _has_cute_wgrad and min_dtype in (torch.float16, torch.bfloat16):
            # Heuristic split_k: target ~4K pairs per split for good parallelism
            # without excessive atomicAdd contention
            split_k = max(1, num_out_coords // 4096)
            # Cap split_k to avoid too many blocks (diminishing returns)
            split_k = min(split_k, 32)
            status = _C.gemm.cute_gemm_mask_wgrad(
                _in_w,
                _go_w,
                grad_weight,
                pair_table,
                pair_mask,
                mask_argsort,
                K,
                mma_tile,
                1.0,
                split_k,
            )
            if status == 0:
                used_cute = True

        if not used_cute:
            # SIMT fallback
            if needs_padding_w:
                grad_weight = torch.zeros((K, C_in, C_out), dtype=min_dtype, device=device)
            status = _C.gemm.mask_implicit_gemm_bwd_wgrad(
                _in_features,
                _grad_output,
                grad_weight,
                pair_table,
                pair_mask,
                K,
                block_size,
            )
            if status != 0:
                raise RuntimeError(f"mask_implicit_gemm_bwd_wgrad failed: {status}")

        if needs_padding_w and used_cute:
            grad_weight = grad_weight[:, :orig_C_in_w, :orig_C_out_w]
        grad_weight = grad_weight.to(dtype=weight.dtype)

    return grad_in, grad_weight
