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

    def _prep(t: Tensor, dt: torch.dtype) -> Tensor:
        if t.dtype == dt and t.is_contiguous() and not t.requires_grad:
            return t
        return t.contiguous().detach().to(dtype=dt)

    _in_features = _prep(in_features, min_dtype)
    _weight = _prep(weight, min_dtype)

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

    if _has_cute:
        # CuTe kernels require fp16/bf16 — downcast fp32 inputs
        if min_dtype == torch.float32:
            _in_features = _in_features.half()
            _weight = _weight.half()
            output = output.half()
            min_dtype = torch.float16
            # Recheck padding after dtype change
            vec_width = 16 // _in_features.element_size()
            needs_padding = (orig_C_in % vec_width != 0) or (orig_C_out % vec_width != 0)
            if needs_padding:
                C_in_pad = ((orig_C_in + vec_width - 1) // vec_width) * vec_width
                C_out_pad = ((orig_C_out + vec_width - 1) // vec_width) * vec_width
                _in_features = torch.nn.functional.pad(_in_features, (0, C_in_pad - orig_C_in))
                _weight = torch.nn.functional.pad(
                    _weight, (0, C_out_pad - orig_C_out, 0, C_in_pad - orig_C_in)
                )
                output = torch.zeros((num_out_coords, C_out_pad), dtype=min_dtype, device=device)
                C_in, C_out = C_in_pad, C_out_pad

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
        raise RuntimeError(
            f"cute_gemm_mask_fwd failed with status {status} "
            f"(N={num_out_coords}, C_in={C_in}, C_out={C_out}, K={K})"
        )

    raise RuntimeError(f"mask_implicit_gemm requires CuTe backend (C_in={C_in}, C_out={C_out})")


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
    weight_T: Optional[Tensor] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Backward pass using mask-based fused implicit GEMM."""
    device = in_features.device
    feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
    min_dtype = _min_dtype(feature_dtype, weight.dtype)

    # Avoid redundant copies when tensors already match the target dtype and
    # layout (the caller in unified.py already pre-casts).
    def _prepare(t: Tensor, dt: torch.dtype) -> Tensor:
        if t.dtype == dt and t.is_contiguous() and not t.requires_grad:
            return t
        return t.contiguous().detach().to(dtype=dt)

    _grad_output = _prepare(grad_output, min_dtype)
    _in_features = _prepare(in_features, min_dtype)
    _weight = _prepare(weight, min_dtype)

    N_in, C_in = _in_features.shape
    K, _, C_out = _weight.shape

    pair_table, pair_mask, mask_argsort = _get_mask_data(kernel_map, num_out_coords, device)

    grad_in = None
    grad_weight = None

    # When inputs are fp32, downcast to fp16 for the CuTe tensor-core mask
    # kernel which is ~15x faster than the SIMT fallback. The fp32
    # accumulator inside the CuTe kernel preserves sufficient precision.

    if needs_input_grad[0]:
        _has_cute_fwd = hasattr(_C.gemm, "cute_gemm_mask_fwd")

        cute_dtype = min_dtype
        if _has_cute_fwd and min_dtype == torch.float32:
            cute_dtype = torch.float16

        _go_bwd = _prepare(_grad_output, cute_dtype)
        _w_bwd = _prepare(_weight, cute_dtype)

        vec_width_bwd = 16 // _go_bwd.element_size()
        orig_C_in_bwd, orig_C_out_bwd = C_in, C_out
        needs_padding_bwd = (C_in % vec_width_bwd != 0) or (C_out % vec_width_bwd != 0)
        if needs_padding_bwd and _has_cute_fwd:
            tc = ((C_in + vec_width_bwd - 1) // vec_width_bwd) * vec_width_bwd
            tco = ((C_out + vec_width_bwd - 1) // vec_width_bwd) * vec_width_bwd
            _go_bwd = torch.nn.functional.pad(_go_bwd, (0, tco - C_out))
            _w_bwd = torch.nn.functional.pad(_w_bwd, (0, tco - C_out, 0, tc - C_in))
            grad_in = torch.zeros((N_in, tc), dtype=cute_dtype, device=device)
        else:
            grad_in = torch.zeros((N_in, C_in), dtype=cute_dtype, device=device)
        used_cute = False

        if _has_cute_fwd and cute_dtype in (torch.float16, torch.bfloat16):
            # AtomicAdd-free dgrad: reuse the forward kernel with reverse
            # pair data. The reverse pair_table maps (offset_k, in_row) ->
            # out_row, so the kernel iterates over input rows and gathers
            # from grad_output — no scatter, no atomicAdd.
            rev_pair_table, rev_pair_mask, rev_argsort = _get_reverse_mask_data(
                kernel_map, N_in, num_out_coords, device
            )
            # Weight transposed: [K, C_in, C_out] -> [K, C_out, C_in]
            if weight_T is not None:
                _weight_T = _prepare(weight_T, cute_dtype)
                # Apply same padding if needed
                if needs_padding_bwd:
                    tc = ((C_in + vec_width_bwd - 1) // vec_width_bwd) * vec_width_bwd
                    tco = ((C_out + vec_width_bwd - 1) // vec_width_bwd) * vec_width_bwd
                    _weight_T = torch.nn.functional.pad(_weight_T, (0, tc - C_in, 0, tco - C_out))
            else:
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
            # SIMT fallback (still uses atomicAdd scatter) — also uses
            # cute_dtype (fp16) to avoid the slow fp32 SIMT kernel.
            _go_simt = _prepare(_grad_output, cute_dtype)
            _w_simt = _prepare(_weight, cute_dtype)
            grad_in = torch.zeros((N_in, C_in), dtype=cute_dtype, device=device)
            status = _C.gemm.mask_implicit_gemm_bwd_dgrad(
                _go_simt,
                _w_simt,
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

        cute_dtype_w = min_dtype
        if _has_cute_wgrad and min_dtype == torch.float32:
            cute_dtype_w = torch.float16

        _in_w = _prepare(_in_features, cute_dtype_w)
        _go_w = _prepare(_grad_output, cute_dtype_w)

        vec_width_w = 16 // _in_w.element_size()
        orig_C_in_w, orig_C_out_w = C_in, C_out
        needs_padding_w = (C_in % vec_width_w != 0) or (C_out % vec_width_w != 0)
        if needs_padding_w and _has_cute_wgrad:
            tc = ((C_in + vec_width_w - 1) // vec_width_w) * vec_width_w
            tco = ((C_out + vec_width_w - 1) // vec_width_w) * vec_width_w
            _in_w = torch.nn.functional.pad(_in_w, (0, tc - C_in))
            _go_w = torch.nn.functional.pad(_go_w, (0, tco - C_out))
            grad_weight = torch.zeros((K, tc, tco), dtype=cute_dtype_w, device=device)
        else:
            grad_weight = torch.zeros((K, C_in, C_out), dtype=cute_dtype_w, device=device)
        used_cute = False

        if _has_cute_wgrad and cute_dtype_w in (torch.float16, torch.bfloat16):
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
            # SIMT fallback — also uses cute_dtype_w (fp16) for speed.
            _in_simt = _prepare(_in_features, cute_dtype_w)
            _go_simt = _prepare(_grad_output, cute_dtype_w)
            grad_weight = torch.zeros((K, C_in, C_out), dtype=cute_dtype_w, device=device)
            status = _C.gemm.mask_implicit_gemm_bwd_wgrad(
                _in_simt,
                _go_simt,
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
