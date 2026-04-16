# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm dispatch for sparse convolution forward and backward.

Extracted from unified.py for cleaner separation of concerns.
The backward dispatch supports split dgrad/wgrad execution via
needs_input_grad=(bool, bool).
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.logger import get_logger

from .explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
    _explicit_gemm_forward_grouped,
    _explicit_gemm_backward_grouped,
)
from .implicit_direct import (
    _implicit_gemm_forward_logic,
    _implicit_gemm_backward_logic,
    _implicit_gemm_backward_grouped,
)
from .cutlass import (
    _cutlass_implicit_gemm_forward_logic,
    _cutlass_implicit_gemm_backward_logic,
    _cutlass_implicit_gemm_forward_grouped,
    _cutlass_implicit_gemm_backward_grouped,
)
from .algo_params import (
    _HAS_CUTE_BACKEND,
    _HAS_CUTE_GROUPED,
    _HAS_CUTE_SM90,
    _HAS_CUTE_GROUPED_SM90,
)

logger = get_logger(__name__)

# Lazy imports for optional backends
if _HAS_CUTE_BACKEND:
    from .cute import (
        _cute_implicit_gemm_forward_logic,
        _cute_implicit_gemm_backward_logic,
    )

if _HAS_CUTE_GROUPED:
    from .cute_grouped import (
        _cute_grouped_forward_logic,
        _cute_grouped_backward_logic,
    )

if _HAS_CUTE_SM90:
    from .cute_sm90 import (
        _cute_implicit_gemm_sm90_forward_logic,
        _cute_implicit_gemm_sm90_backward_logic,
    )

if _HAS_CUTE_GROUPED_SM90:
    from .cute_grouped_sm90 import (
        _cute_grouped_sm90_forward_logic,
        _cute_grouped_sm90_backward_logic,
    )


def _execute_forward(
    algo: str,
    params: Dict[str, Any],
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    fwd_block_size: Optional[int],
    groups: int = 1,
    use_fp16_accum: bool = False,
) -> Tensor:
    """Dispatch forward pass to the selected algorithm."""
    if groups > 1 and algo != "production":
        raise ValueError(
            f"Group convolution (groups={groups}) only supported with algo='production', "
            f"got '{algo}'"
        )
    if groups > 1:
        C_in_g = weight.shape[2]
        C_out_g = weight.shape[3]
        if C_in_g < 8 or C_out_g < 8:
            raise ValueError(
                f"Group convolution requires per-group channels >= 8 "
                f"(got C_in/G={C_in_g}, C_out/G={C_out_g}). "
                f"Reduce groups or increase channels."
            )
    if algo == "explicit_gemm":
        return _explicit_gemm_forward_logic(
            in_features, weight, kernel_map, num_out_coords, compute_dtype
        )
    elif algo == "implicit_gemm":
        bs = params.get("fwd_block_size", fwd_block_size or 16)
        return _implicit_gemm_forward_logic(
            in_features, weight, kernel_map, num_out_coords, compute_dtype, bs
        )
    elif algo == "cutlass_implicit_gemm":
        result = _cutlass_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            accumulator_type=params.get("accumulator_type", torch.float32),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"cutlass fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_implicit_gemm":
        result = _cute_implicit_gemm_forward_logic(in_features, weight, kernel_map, num_out_coords)
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"cute fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_grouped":
        result = _cute_grouped_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            mma_tile=params.get("mma_tile", 3),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"cute_grouped fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "explicit_gemm_grouped":
        return _explicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "implicit_gemm_grouped":
        from .implicit_direct import _implicit_gemm_forward_grouped

        return _implicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            fwd_block_size=params.get("fwd_block_size", 16),
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "cutlass_grouped_hybrid":
        result = _cutlass_implicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            accumulator_type=params.get("accumulator_type", torch.float32),
            saturation_m=params.get("saturation_m", 5000),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"cutlass_grouped fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_implicit_gemm_sm90":
        result = _cute_implicit_gemm_sm90_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            mma_tile=params.get("mma_tile", 100),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"cute_sm90 fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "cute_grouped_sm90":
        result = _cute_grouped_sm90_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            mma_tile=params.get("mma_tile", 100),
            use_cp_async=params.get("use_cp_async", True),
        )
        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"cute_grouped_sm90 fwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result))}"
            )
        return result
    elif algo == "production":
        from .mask_gemm import _get_mask_data

        K = len(kernel_map)
        mask_words = (K + 31) // 32

        # Submanifold identity shortcut: center offset maps each voxel to itself
        # Enable when stride=1 (N_in == N_out) and K is odd
        N_in_fwd = in_features.shape[0]
        is_submanifold = (N_in_fwd == num_out_coords) and (K % 2 == 1)
        identity_offset = K // 2 if is_submanifold else -1

        tile_id = params.get("tile_id", 41)
        pair_table, pair_mask, mask_argsort = _get_mask_data(
            kernel_map, num_out_coords, in_features.device
        )

        # Per-group channel dimensions
        if groups > 1:
            # weight: [K, G, C_in_g, C_out_g]
            C_in_g = weight.shape[2]
            C_out_g = weight.shape[3]
        else:
            # weight: [K, C_in, C_out]
            C_in_g = weight.shape[1]
            C_out_g = weight.shape[2]

        # Cast to fp16 if needed (production kernels require fp16/bf16)
        orig_dtype = in_features.dtype
        use_f32_output = orig_dtype == torch.float32
        if orig_dtype == torch.float32:
            _in = in_features.half()
            _w = weight.half()
        else:
            _in = in_features
            _w = weight

        # Select tile based on per-group channel alignment and output dtype
        vec_width = 16 // _in.element_size()
        cin_aligned = C_in_g % vec_width == 0
        cout_aligned = C_out_g % vec_width == 0
        if use_f32_output:
            if cin_aligned and cout_aligned:
                tile_id = 80
            else:
                tile_id = 82
        elif not cin_aligned and not cout_aligned:
            tile_id = 70
        elif not cin_aligned:
            tile_id = 71
        elif not cout_aligned:
            tile_id = 72

        out_dtype = torch.float32 if use_f32_output else _in.dtype

        # Single launch handles groups=1 and groups>1 via grid.z
        C_out_total = C_out_g * groups
        output = torch.zeros(
            (num_out_coords, C_out_total), dtype=out_dtype, device=in_features.device
        )
        status = _C.production.fwd(
            _in,
            _w,
            output,
            pair_table,
            pair_mask,
            mask_argsort,
            K,
            tile_id,
            mask_words,
            identity_offset,
            1.0,
            groups,
        )
        if status != 0:
            raise RuntimeError(f"production fwd failed: status={status}, tile={tile_id}")

        return output.to(dtype=orig_dtype)
    else:
        raise ValueError(f"Unsupported forward algorithm: {algo}")


def _execute_backward(
    algo: str,
    params: Dict[str, Any],
    grad_output: Tensor,
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    needs_input_grad: Tuple[bool, ...],
    weight_T: Optional[Tensor] = None,
    groups: int = 1,
    use_fp16_accum: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Dispatch backward pass to the selected algorithm.

    Args:
        weight_T: Pre-computed weight.transpose(1,2).contiguous() to avoid
            redundant copies when dgrad and wgrad are dispatched separately.

    Returns (grad_in_features, grad_weight). Either can be None if the
    corresponding needs_input_grad flag is False AND the algorithm supports it.
    """
    if algo == "explicit_gemm":
        return _explicit_gemm_backward_logic(
            grad_output, in_features, weight, kernel_map, compute_dtype, device
        )
    elif algo == "implicit_gemm":
        return _implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            gemm_block_size=params.get("bwd_block_size", 16),
            split_k_threads_per_block=params.get("split_k_threads_per_block", 256),
            split_k_factor=params.get("split_k_factor", 4),
            compute_dtype=compute_dtype,
        )
    elif algo == "cutlass_implicit_gemm":
        result = _cutlass_implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            accumulator_type=params.get("accumulator_type", torch.float32),
            device=device,
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"cutlass bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "cute_implicit_gemm":
        result = _cute_implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(needs_input_grad[0], needs_input_grad[1]),
            device=device,
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"cute bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "explicit_gemm_grouped":
        return _explicit_gemm_backward_grouped(
            grad_output,
            in_features,
            weight,
            kernel_map,
            compute_dtype,
            device,
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "implicit_gemm_grouped":
        return _implicit_gemm_backward_grouped(
            grad_output,
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            gemm_block_size=params.get("gemm_block_size", 16),
            split_k_threads_per_block=params.get("split_k_threads_per_block", 256),
            split_k_factor=params.get("split_k_factor", 4),
            compute_dtype=compute_dtype,
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "cutlass_grouped_hybrid":
        result = _cutlass_implicit_gemm_backward_grouped(
            grad_output,
            in_features,
            weight,
            kernel_map,
            accumulator_type=params.get("accumulator_type", torch.float32),
            device=device,
            saturation_m=params.get("saturation_m", 5000),
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"cutlass_grouped bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "cute_grouped":
        # Use fused C++ wgrad when available (works for any stride)
        _has_fused = hasattr(_C.gemm, "sparse_conv_wgrad")
        if _has_fused and needs_input_grad[1] and not needs_input_grad[0]:
            iden = (
                kernel_map.identity_map_index if kernel_map.identity_map_index is not None else -1
            )
            grad_weight = _C.gemm.sparse_conv_wgrad(
                in_features,
                grad_output,
                kernel_map.in_maps,
                kernel_map.out_maps,
                kernel_map.offsets,
                iden,
                len(kernel_map),
                params.get("mma_tile", 3),
            )
            return None, grad_weight
        result = _cute_grouped_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(needs_input_grad[0], needs_input_grad[1]),
            device=device,
            mma_tile=params.get("mma_tile", 3),
            weight_T=weight_T,
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"cute_grouped bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "cute_implicit_gemm_sm90":
        result = _cute_implicit_gemm_sm90_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(needs_input_grad[0], needs_input_grad[1]),
            device=device,
            mma_tile=params.get("mma_tile", 100),
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"cute_sm90 bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo == "cute_grouped_sm90":
        result = _cute_grouped_sm90_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            requires_grad=(needs_input_grad[0], needs_input_grad[1]),
            device=device,
            mma_tile=params.get("mma_tile", 100),
            use_cp_async=params.get("use_cp_async", True),
        )
        if isinstance(result[0], int) and result[0] != 0:
            raise RuntimeError(
                f"cute_grouped_sm90 bwd error: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(result[0]))}"
            )
        return result
    elif algo in ("production", "production_fwd_as_dgrad"):
        from .mask_gemm import _get_mask_data, _get_reverse_mask_data

        use_fwd_for_dgrad = algo == "production_fwd_as_dgrad"

        K = weight.shape[0]
        mask_words = (K + 31) // 32

        tile_id = params.get("tile_id", 60)
        split_k = params.get("split_k", 64)
        N_in = in_features.shape[0]

        # Submanifold identity shortcut for dgrad
        # For submanifold conv, reverse_pair_table[K//2, j] == j (identity)
        is_submanifold_bwd = (N_in == num_out_coords) and (K % 2 == 1)
        identity_offset_bwd = K // 2 if is_submanifold_bwd else -1

        # Per-group channel dimensions
        if groups > 1:
            C_in_g = weight.shape[2]
            C_out_g = weight.shape[3]
            C_in = groups * C_in_g
            C_out = groups * C_out_g
        else:
            C_in = in_features.shape[1]
            C_out = weight.shape[2]
            C_in_g = C_in
            C_out_g = C_out

        # Cast to fp16 if needed (production kernels require fp16/bf16)
        orig_dtype = grad_output.dtype
        use_f32_output = orig_dtype == torch.float32
        if orig_dtype == torch.float32:
            _go = grad_output.half()
            _in = in_features.half()
            _w = weight.half()
        else:
            _go = grad_output
            _in = in_features
            _w = weight

        compute_dtype = _go.dtype

        grad_in = None
        grad_weight = None

        if needs_input_grad[0]:
            rev_pt, rev_pm, rev_as = _get_reverse_mask_data(
                kernel_map, N_in, num_out_coords, grad_output.device
            )

            # production.dgrad expects weight in its NATIVE [K, G, C_in, C_out] layout
            # (the kernel header docstring states: "weight [K, G, C_in, C_out] — NOT
            # transposed"). The mainloop's _load_B_tile addresses each per-K plane as
            # rows of length C_out indexed by C_in (n_local * K_dim + k_local with
            # K_dim=C_out), which matches the un-transposed memory layout. Earlier code
            # passed _w.transpose(1,2) which double-transposed B and gave near-zero
            # correlation with the reference (rdiff ~1.4 across every shape). The
            # production_fwd_as_dgrad path also expects un-transposed weight here.
            _w_dgrad = _w.contiguous()

            # Select dgrad tile based on per-group channel dims
            vec_width = 16 // _go.element_size()
            dgrad_out_dtype = torch.float32 if use_f32_output else compute_dtype

            if use_fwd_for_dgrad:
                dgrad_tile = params.get("tile_id", 41)
                fwd_cin_aligned = C_out_g % vec_width == 0
                fwd_cout_aligned = C_in_g % vec_width == 0
                if use_f32_output:
                    dgrad_tile = 80 if (fwd_cin_aligned and fwd_cout_aligned) else 82
                elif not fwd_cin_aligned and not fwd_cout_aligned:
                    dgrad_tile = 70
                elif not fwd_cin_aligned:
                    dgrad_tile = 71
                elif not fwd_cout_aligned:
                    dgrad_tile = 72

                dgrad_fn = _C.production.fwd
            else:
                cin_aligned = C_in_g % vec_width == 0
                cout_aligned = C_out_g % vec_width == 0
                if not cin_aligned and not cout_aligned:
                    dgrad_tile = 70
                elif not cout_aligned:
                    dgrad_tile = 71
                elif not cin_aligned:
                    dgrad_tile = 72
                elif use_f32_output:
                    dgrad_tile = 81
                else:
                    C = max(C_in_g, C_out_g)
                    is_fp16 = compute_dtype == torch.float16
                    if C <= 48:
                        dgrad_tile = 50
                    elif C <= 96:
                        dgrad_tile = 53 if is_fp16 else 51
                    else:
                        dgrad_tile = 54 if is_fp16 else 52

                dgrad_fn = _C.production.dgrad

            # Single launch handles groups=1 and groups>1 via grid.z
            C_in_total = C_in_g * groups
            grad_in = torch.zeros(
                (N_in, C_in_total), dtype=dgrad_out_dtype, device=grad_output.device
            )
            status = dgrad_fn(
                _go,
                _w_dgrad,
                grad_in,
                rev_pt,
                rev_pm,
                rev_as,
                K,
                dgrad_tile,
                mask_words,
                identity_offset_bwd,
                1.0,
                groups,
            )
            if status != 0:
                raise RuntimeError(f"production dgrad failed: status={status}")

            grad_in = grad_in.to(dtype=orig_dtype)

        if needs_input_grad[1]:
            # Wgrad via production wgrad kernel with reduced_mask
            pair_table, pair_mask, mask_argsort = _get_mask_data(
                kernel_map, num_out_coords, grad_output.device
            )

            # Build reduced_mask (cached on kernel_map)
            if not hasattr(kernel_map, "_reduced_mask") or kernel_map._reduced_mask is None:
                kernel_map._reduced_mask = _C.production.build_reduced_mask(
                    pair_mask, mask_argsort, 32, mask_words
                )

            # Select wgrad tile based on per-group channel dims
            vec_width = 16 // _in.element_size()
            if C_in_g % vec_width != 0 or C_out_g % vec_width != 0:
                wgrad_tile = 73
            else:
                wgrad_tile = tile_id

            # Single launch: [K, G, C_in_g, C_out_g] for groups>1, [K, C_in_g, C_out_g] for groups=1
            if groups == 1:
                grad_weight = torch.zeros(
                    (K, C_in_g, C_out_g), dtype=torch.float32, device=grad_output.device
                )
            else:
                grad_weight = torch.zeros(
                    (K, groups, C_in_g, C_out_g),
                    dtype=torch.float32,
                    device=grad_output.device,
                )
            status = _C.production.wgrad(
                _in,
                _go,
                grad_weight,
                pair_table,
                pair_mask,
                mask_argsort,
                kernel_map._reduced_mask,
                K,
                wgrad_tile,
                split_k,
                1.0,
                groups,
            )
            if status != 0:
                raise RuntimeError(f"production wgrad failed: status={status}")
            grad_weight = grad_weight.to(dtype=weight.dtype)

        return grad_in, grad_weight
    else:
        raise ValueError(f"Unsupported backward algorithm: {algo}")
