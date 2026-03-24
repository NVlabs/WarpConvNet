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


def _kernel_map_to_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert IntSearchResult to mask-based pair_table + mask + argsort.

    Uses the cached _pair_table from kernel map generation if available,
    avoiding the expensive Python reconstruction.

    Returns:
        pair_table: [K * N_out] int32, flattened
        pair_mask: [N_out] int32 (uint32 bitmask)
        mask_argsort: [N_out] int32 permutation
    """
    K = len(kernel_map)
    N_out = num_out_coords

    # Fast path: use cached pair_table from kernel map generation
    if hasattr(kernel_map, '_pair_table') and kernel_map._pair_table is not None:
        pair_table = kernel_map._pair_table.reshape(-1).contiguous()  # [K*N_out]
    else:
        # Build pair_table from CSR using CUDA kernel
        pair_table = torch.empty(K * N_out, dtype=torch.int32, device=device)
        pair_table.fill_(-1)
        L = kernel_map.in_maps.shape[0]
        if L > 0 and hasattr(_C.gemm, 'csr_to_pair_table_cuda'):
            offsets_gpu = kernel_map.offsets.to(device=device, dtype=torch.int32)
            _C.gemm.csr_to_pair_table_cuda(
                kernel_map.in_maps.int(),
                kernel_map.out_maps.int(),
                offsets_gpu,
                pair_table,
                N_out,
                K,
            )

    # Build pair_mask using CUDA kernel
    pair_mask = torch.zeros(N_out, dtype=torch.int32, device=device)
    if K <= 32 and hasattr(_C.gemm, 'build_pair_mask_cuda'):
        _C.gemm.build_pair_mask_cuda(pair_table, pair_mask, K)
    elif K <= 32:
        # Fallback: Python vectorized
        pair_table_2d = pair_table.reshape(K, N_out)
        valid = (pair_table_2d >= 0)
        bit_positions = (1 << torch.arange(K, device=device, dtype=torch.int32)).unsqueeze(1)
        pair_mask = (valid.int() * bit_positions).sum(dim=0).int()

    mask_argsort = torch.argsort(pair_mask, stable=True).int()

    return pair_table, pair_mask, mask_argsort


# Cache mask data per kernel_map to avoid recomputation
_MASK_DATA_CACHE = {}


def _get_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get or compute cached mask data for a kernel_map."""
    cache_key = id(kernel_map)
    if cache_key not in _MASK_DATA_CACHE:
        _MASK_DATA_CACHE[cache_key] = _kernel_map_to_mask_data(
            kernel_map, num_out_coords, device
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

    output = torch.zeros(
        (num_out_coords, C_out), dtype=min_dtype, device=device
    )

    if num_out_coords == 0 or K == 0 or C_in == 0 or C_out == 0 or N_in == 0:
        return output.to(dtype=in_features.dtype)

    pair_table, pair_mask, mask_argsort = _get_mask_data(
        kernel_map, num_out_coords, device
    )

    # Try CuTe tensor-core path if available and channels are aligned to 8
    _has_cute = hasattr(_C.gemm, "cute_gemm_mask_fwd")
    vec_width = 16 // _in_features.element_size()  # 8 for fp16/bf16
    aligned = (C_in % vec_width == 0) and (C_out % vec_width == 0)

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
            return output.to(dtype=in_features.dtype)
        # CuTe failed (likely alignment issue) — fall through to SIMT

    # SIMT fallback
    status = _C.gemm.mask_implicit_gemm_fwd(
        _in_features,
        _weight,
        output,
        pair_table,
        pair_mask,
        mask_argsort,
        K,
        block_size,
    )
    if status != 0:
        raise RuntimeError(f"mask_implicit_gemm_fwd failed with status {status}")

    return output.to(dtype=in_features.dtype)


def _mask_implicit_gemm_backward_logic(
    grad_output: Tensor,
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
    needs_input_grad: Tuple[bool, ...] = (True, True),
    block_size: int = 16,
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

    pair_table, pair_mask, mask_argsort = _get_mask_data(
        kernel_map, num_out_coords, device
    )

    grad_in = None
    grad_weight = None

    if needs_input_grad[0]:
        grad_in = torch.zeros(
            (N_in, C_in), dtype=min_dtype, device=device
        )
        # Try CuTe dgrad (tensor cores) if available and channels aligned
        _has_cute_dgrad = hasattr(_C.gemm, "cute_gemm_mask_dgrad")
        vec_width = 16 // _grad_output.element_size()
        aligned = (C_in % vec_width == 0) and (C_out % vec_width == 0)
        used_cute = False

        if _has_cute_dgrad and aligned and min_dtype in (torch.float16, torch.bfloat16):
            # Pre-transpose weight for dgrad: [K, C_in, C_out] → keep same layout
            # The CuTe kernel handles the transpose internally via strided loads
            status = _C.gemm.cute_gemm_mask_dgrad(
                _grad_output, _weight, grad_in,
                pair_table, pair_mask, mask_argsort,
                K, 3, 1.0,
            )
            if status == 0:
                used_cute = True

        if not used_cute:
            # SIMT fallback
            status = _C.gemm.mask_implicit_gemm_bwd_dgrad(
                _grad_output, _weight, grad_in,
                pair_table, pair_mask, mask_argsort,
                K, block_size,
            )
            if status != 0:
                raise RuntimeError(f"mask_implicit_gemm_bwd_dgrad failed: {status}")

        grad_in = grad_in.to(dtype=in_features.dtype)

    if needs_input_grad[1]:
        grad_weight = torch.zeros(
            (K, C_in, C_out), dtype=min_dtype, device=device
        )
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
        grad_weight = grad_weight.to(dtype=weight.dtype)

    return grad_in, grad_weight
