# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import warpconvnet._C as _C

from warpconvnet._offset_gemm_constants import (
    BACKEND_CUTE_GROUPED_SM80,
    BACKEND_CUTE_GROUPED_SM90,
    BACKEND_CUTE_SM80,
    BACKEND_CUTE_SM90,
    OP_AD_GATHER_SCATTER,
    OP_TRAB_GATHER,
    KernelVariant,
)


@lru_cache(maxsize=1)
def _kernel_index() -> dict[tuple[str, str, int], KernelVariant]:
    """Build (op, backend, tile_id) → KernelVariant index from the compiled
    dispatch table snapshot exposed by the C++ extension."""

    raw = _C.gemm.offset_gemm_list_kernels()
    return {
        (row["op"], row["backend"], row["tile_id"]): KernelVariant(
            op=row["op"],
            backend=row["backend"],
            tile_id=row["tile_id"],
            grouped=row["grouped"],
            split_k=row["split_k"],
            launch_symbol=row["launch_symbol"],
            tile_tag=row["tile_tag"],
            input_dtype=row["input_dtype"],
            output_dtype=row["output_dtype"],
            acc_dtype=row["acc_dtype"],
            itype=row["itype"],
            block_size=row["block_size"],
            kernel_variant=row["kernel_variant"],
            mma_atom=row["mma_atom"],
        )
        for row in raw
    }


def get_kernel(op: str, backend: str, tile_id: int) -> KernelVariant:
    try:
        return _kernel_index()[(op, backend, tile_id)]
    except KeyError:
        raise KeyError(
            f"No registered offset GEMM kernel for (op={op!r}, backend={backend!r}, "
            f"tile_id={tile_id}). Rebuild the warpconvnet extension or pick a tile "
            "from candidate_kernels()."
        ) from None


def candidate_kernels(op: str, backend: str) -> list[KernelVariant]:
    return [v for (o, b, _), v in _kernel_index().items() if o == op and b == backend]


def _registry_binding_name(op: str) -> str:
    return f"offset_gemm_{op}"


def backend_available(backend: str) -> bool:
    """Return whether a registry backend is compiled into the loaded extension."""

    availability = getattr(_C.gemm, "offset_gemm_backend_available", None)
    if availability is not None:
        return bool(availability(backend))

    legacy_symbols = {
        BACKEND_CUTE_SM90: "cute_gemm_sm90_AD_gather_scatter",
        BACKEND_CUTE_SM80: "cute_gemm_AD_gather_scatter",
        BACKEND_CUTE_GROUPED_SM90: "cute_gemm_sm90_grouped_AD_gather_scatter",
        BACKEND_CUTE_GROUPED_SM80: "cute_gemm_grouped_AD_gather_scatter",
    }
    symbol = legacy_symbols.get(backend)
    return symbol is not None and hasattr(_C.gemm, symbol)


def resolve_ad_gather_scatter_binding(
    variant: KernelVariant,
) -> tuple[Callable, dict[str, object]]:
    """Resolve a registry record to a callable CUDA binding and keyword args.

    Until the extension is rebuilt with the new registry-aware pybind entrypoint,
    this function falls back to the existing backend-specific bindings.
    """

    if variant.op != OP_AD_GATHER_SCATTER:
        raise ValueError(f"unsupported op for runtime resolver: {variant.op}")

    registry_binding = getattr(_C.gemm, _registry_binding_name(variant.op), None)
    if registry_binding is None:
        registry_binding = getattr(_C.gemm, "offset_gemm_AD_gather_scatter", None)
    if registry_binding is not None:
        return registry_binding, {"backend": variant.backend, "tile_id": variant.tile_id}

    if variant.backend == BACKEND_CUTE_SM90:
        legacy_binding = getattr(_C.gemm, "cute_gemm_sm90_AD_gather_scatter", None)
        if legacy_binding is not None:
            return legacy_binding, {"mma_tile": variant.tile_id}
    if variant.backend == BACKEND_CUTE_SM80:
        legacy_binding = getattr(_C.gemm, "cute_gemm_AD_gather_scatter", None)
        if legacy_binding is not None:
            return legacy_binding, {"mma_tile": variant.tile_id}

    raise RuntimeError(
        "No runtime binding available for "
        f"(op={variant.op}, backend={variant.backend}, tile_id={variant.tile_id}). "
        "The extension may need to be rebuilt with the requested backend enabled."
    )


def resolve_grouped_ad_gather_scatter_binding(
    variant: KernelVariant,
) -> tuple[Callable, dict[str, object]]:
    """Resolve a grouped AD gather-scatter record to a callable binding."""

    if variant.op != OP_AD_GATHER_SCATTER or not variant.grouped:
        raise ValueError(
            "unsupported variant for grouped runtime resolver: "
            f"op={variant.op}, grouped={variant.grouped}"
        )

    registry_binding = getattr(_C.gemm, "offset_gemm_grouped_ad_gather_scatter", None)
    if registry_binding is None:
        registry_binding = getattr(_C.gemm, "offset_gemm_grouped_AD_gather_scatter", None)
    if registry_binding is not None:
        return registry_binding, {"backend": variant.backend, "tile_id": variant.tile_id}

    if variant.backend == BACKEND_CUTE_GROUPED_SM90:
        legacy_binding = getattr(_C.gemm, "cute_gemm_sm90_grouped_AD_gather_scatter", None)
        if legacy_binding is not None:
            return legacy_binding, {"mma_tile": variant.tile_id}
    if variant.backend == BACKEND_CUTE_GROUPED_SM80:
        legacy_binding = getattr(_C.gemm, "cute_gemm_grouped_AD_gather_scatter", None)
        if legacy_binding is not None:
            return legacy_binding, {"mma_tile": variant.tile_id}

    raise RuntimeError(
        "No grouped runtime binding available for "
        f"(op={variant.op}, backend={variant.backend}, tile_id={variant.tile_id}). "
        "The extension may need to be rebuilt with the requested backend enabled."
    )


def resolve_trab_gather_binding(variant: KernelVariant) -> tuple[Callable, dict[str, object]]:
    """Resolve a TrAB gather record to a callable CUDA binding."""

    if variant.op != OP_TRAB_GATHER:
        raise ValueError(f"unsupported op for runtime resolver: {variant.op}")

    registry_binding = getattr(_C.gemm, _registry_binding_name(variant.op), None)
    if registry_binding is None:
        registry_binding = getattr(_C.gemm, "offset_gemm_trAB_gather", None)
    if registry_binding is not None:
        return registry_binding, {"backend": variant.backend, "tile_id": variant.tile_id}

    if variant.backend == BACKEND_CUTE_SM80:
        legacy_binding = getattr(_C.gemm, "cute_gemm_trAB_gather", None)
        if legacy_binding is not None:
            return legacy_binding, {"mma_tile": variant.tile_id}

    raise RuntimeError(
        "No TrAB runtime binding available for "
        f"(op={variant.op}, backend={variant.backend}, tile_id={variant.tile_id}). "
        "The extension may need to be rebuilt with the requested backend enabled."
    )


def resolve_grouped_trab_gather_binding(
    variant: KernelVariant,
) -> tuple[Callable, dict[str, object]]:
    """Resolve a grouped TrAB gather record to a callable binding."""

    if variant.op != OP_TRAB_GATHER or not variant.grouped:
        raise ValueError(
            "unsupported variant for grouped TrAB runtime resolver: "
            f"op={variant.op}, grouped={variant.grouped}"
        )

    registry_binding = getattr(_C.gemm, "offset_gemm_grouped_trab_gather", None)
    if registry_binding is None:
        registry_binding = getattr(_C.gemm, "offset_gemm_grouped_trAB_gather", None)
    if registry_binding is not None:
        return registry_binding, {"backend": variant.backend, "tile_id": variant.tile_id}

    if variant.backend == BACKEND_CUTE_GROUPED_SM80:
        legacy_binding = getattr(_C.gemm, "cute_gemm_grouped_trAB_gather", None)
        if legacy_binding is not None:
            return legacy_binding, {"mma_tile": variant.tile_id}

    raise RuntimeError(
        "No grouped TrAB runtime binding available for "
        f"(op={variant.op}, backend={variant.backend}, tile_id={variant.tile_id}). "
        "The extension may need to be rebuilt with the requested backend enabled."
    )


def dispatch_ad_gather_scatter(
    tensor_a,
    tensor_b,
    tensor_c,
    tensor_d,
    indices_a,
    indices_d,
    *,
    backend: str,
    tile_id: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> int:
    variant = get_kernel(OP_AD_GATHER_SCATTER, backend, tile_id)
    binding, kwargs = resolve_ad_gather_scatter_binding(variant)
    return binding(
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_d,
        indices_a,
        indices_d,
        alpha=alpha,
        beta=beta,
        **kwargs,
    )


def dispatch_grouped_ad_gather_scatter(
    tensor_a,
    tensor_d,
    in_map,
    out_map,
    weight_ptrs,
    tile_offsets,
    group_sizes,
    map_offsets,
    total_m_tiles: int,
    *,
    backend: str,
    tile_id: int,
    alpha: float = 1.0,
    use_atomic: bool = True,
    use_cp_async: bool = True,
) -> int:
    variant = get_kernel(OP_AD_GATHER_SCATTER, backend, tile_id)
    binding, kwargs = resolve_grouped_ad_gather_scatter_binding(variant)
    return binding(
        tensor_a,
        tensor_d,
        in_map,
        out_map,
        weight_ptrs,
        tile_offsets,
        group_sizes,
        map_offsets,
        total_m_tiles,
        alpha=alpha,
        use_atomic=use_atomic,
        use_cp_async=use_cp_async,
        **kwargs,
    )


def dispatch_trab_gather(
    tensor_a,
    tensor_b,
    tensor_c,
    tensor_d,
    indices_a,
    indices_b,
    *,
    backend: str,
    tile_id: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> int:
    variant = get_kernel(OP_TRAB_GATHER, backend, tile_id)
    binding, kwargs = resolve_trab_gather_binding(variant)
    return binding(
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_d,
        indices_a,
        indices_b,
        alpha=alpha,
        beta=beta,
        **kwargs,
    )


def dispatch_grouped_trab_gather(
    tensor_a,
    tensor_b,
    in_map,
    out_map,
    output_ptrs,
    gather_sizes,
    map_offsets,
    C_in: int,
    C_out: int,
    output_scalar_type: int,
    *,
    backend: str,
    tile_id: int,
    alpha: float = 1.0,
) -> int:
    variant = get_kernel(OP_TRAB_GATHER, backend, tile_id)
    binding, kwargs = resolve_grouped_trab_gather_binding(variant)
    return binding(
        tensor_a,
        tensor_b,
        in_map,
        out_map,
        output_ptrs,
        gather_sizes,
        map_offsets,
        C_in,
        C_out,
        alpha=alpha,
        output_scalar_type=output_scalar_type,
        **kwargs,
    )
