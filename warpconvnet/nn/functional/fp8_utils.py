# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FP8 quantization utilities for WarpConvNet sparse convolution.

Provides helpers to quantize tensors to FP8 (E4M3 / E5M2) formats and
dequantize them back.  Requires PyTorch >= 2.1 for ``torch.float8_e4m3fn``
and ``torch.float8_e5m2`` dtype support.

Typical usage::

    from warpconvnet.nn.functional.fp8_utils import quantize_to_fp8, dequantize_from_fp8

    q_tensor, scale = quantize_to_fp8(features, fp8_format="e4m3")
    # ... run FP8 GEMM ...
    out = dequantize_from_fp8(q_output, scale)
"""

from __future__ import annotations

from typing import Tuple

import torch

# ---------------------------------------------------------------------------
# FP8 dtype availability check
# ---------------------------------------------------------------------------

_FP8_DTYPES_AVAILABLE = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")

# Maximum representable values for each FP8 format.
# E4M3FN: sign(1) exp(4) mantissa(3), no inf/nan encoding => max = 448.0
# E5M2:   sign(1) exp(5) mantissa(2), IEEE-like             => max = 57344.0
_FP8_FORMAT_INFO = {
    "e4m3": {
        "dtype": getattr(torch, "float8_e4m3fn", None),
        "max_val": 448.0,
    },
    "e5m2": {
        "dtype": getattr(torch, "float8_e5m2", None),
        "max_val": 57344.0,
    },
}


def _check_fp8_available() -> None:
    """Raise a clear error if the current PyTorch build lacks FP8 dtypes."""
    if not _FP8_DTYPES_AVAILABLE:
        raise RuntimeError(
            "FP8 dtypes (torch.float8_e4m3fn / torch.float8_e5m2) are not "
            "available in this PyTorch build.  PyTorch >= 2.1 is required."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantize_to_fp8(
    tensor: torch.Tensor,
    fp8_format: str = "e4m3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating-point tensor to FP8 with per-tensor scaling.

    The scale factor is computed as::

        scale = max(|tensor|) / fp8_max_representable

    so that the dynamic range of *tensor* is mapped onto the full FP8 range.
    A scale of 0 (all-zero input) is replaced with 1.0 to avoid division by
    zero during dequantization.

    Args:
        tensor: Input tensor in FP16, BF16, or FP32.
        fp8_format: ``"e4m3"`` (default, best for inference) or ``"e5m2"``
            (wider dynamic range, useful for training gradients).

    Returns:
        A tuple ``(quantized_tensor, scale_factor)`` where *quantized_tensor*
        has the corresponding ``torch.float8_*`` dtype and *scale_factor* is
        a scalar ``torch.float32`` tensor on the same device.

    Raises:
        RuntimeError: If PyTorch FP8 dtypes are unavailable.
        ValueError: If *fp8_format* is not ``"e4m3"`` or ``"e5m2"``.
    """
    _check_fp8_available()

    if fp8_format not in _FP8_FORMAT_INFO:
        raise ValueError(f"Unknown fp8_format '{fp8_format}'. Must be 'e4m3' or 'e5m2'.")

    info = _FP8_FORMAT_INFO[fp8_format]
    fp8_dtype = info["dtype"]
    fp8_max = info["max_val"]

    # Compute per-tensor absmax in FP32 to avoid overflow with FP16 inputs.
    amax = tensor.detach().float().abs().max()

    # Compute scale: maps tensor range to FP8 range.
    # If the tensor is all zeros, use scale=1.0 to avoid 0/0.
    scale = amax / fp8_max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    # Scale and cast.  We go through FP32 to ensure correct rounding.
    scaled = tensor.float() / scale
    quantized = scaled.to(fp8_dtype)

    return quantized, scale.to(torch.float32)


def dequantize_from_fp8(
    tensor: torch.Tensor,
    scale_factor: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to a higher-precision floating-point type.

    The original values are approximately recovered as::

        output = tensor.float() * scale_factor

    Args:
        tensor: Quantized tensor with ``torch.float8_e4m3fn`` or
            ``torch.float8_e5m2`` dtype.
        scale_factor: Per-tensor scale produced by `quantize_to_fp8`.
        output_dtype: Desired output dtype (default ``torch.float16``).

    Returns:
        Dequantized tensor in *output_dtype*.
    """
    _check_fp8_available()
    return (tensor.float() * scale_factor).to(output_dtype)


# ---------------------------------------------------------------------------
# Dtype detection utility for sparse conv dispatch
# ---------------------------------------------------------------------------


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    """Return True if *dtype* is one of the FP8 floating-point types.

    Safe to call even on older PyTorch versions that lack FP8 dtypes — will
    simply return False.
    """
    if not _FP8_DTYPES_AVAILABLE:
        return False
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def get_fp8_format(dtype: torch.dtype) -> str:
    """Map a PyTorch FP8 dtype to the format string used by this module.

    Returns ``"e4m3"`` for ``torch.float8_e4m3fn`` and ``"e5m2"`` for
    ``torch.float8_e5m2``.

    Raises:
        ValueError: If *dtype* is not an FP8 type.
    """
    _check_fp8_available()
    if dtype == torch.float8_e4m3fn:
        return "e4m3"
    elif dtype == torch.float8_e5m2:
        return "e5m2"
    else:
        raise ValueError(f"{dtype} is not an FP8 dtype.")


def select_fp8_tile_config(dtype: torch.dtype, m: int, n: int, k: int) -> int:
    """Suggest an FP8 MMATile enum value for the given problem dimensions.

    This is a heuristic tile selector for FP8 GEMM on SM90.  It picks a
    tile configuration based on the M, N, K dimensions of the problem,
    preferring larger K tiles when the channel dimension is large enough.

    The returned integer corresponds to the ``MMATile`` enum values defined
    in ``gemm_mma_tiles.h`` (SM90 FP8 tiles start at 200).

    Args:
        dtype: Must be an FP8 dtype.
        m: Number of rows (gathered output points).
        n: Number of output channels.
        k: Number of input channels.

    Returns:
        Integer MMATile enum value.

    Raises:
        ValueError: If *dtype* is not FP8.
    """
    if not is_fp8_dtype(dtype):
        raise ValueError(f"Expected FP8 dtype, got {dtype}")

    # Enum values matching gemm_mma_tiles.h
    SM90_FP8_Tile64x64x64 = 200
    SM90_FP8_Tile64x128x64 = 201
    SM90_FP8_Tile128x64x64 = 202
    SM90_FP8_Tile128x128x64 = 203
    SM90_FP8_Tile64x128x128 = 204
    SM90_FP8_Tile128x128x128 = 205
    SM90_FP8_Tile64x256x128 = 206

    # Prefer tK=128 tiles when K is large enough
    if k >= 128:
        if n >= 256 and m <= 64:
            return SM90_FP8_Tile64x256x128
        if m >= 128 and n >= 128:
            return SM90_FP8_Tile128x128x128
        if n >= 128:
            return SM90_FP8_Tile64x128x128
        # Fall through to tK=64 tiles
    # tK=64 tiles for smaller K or remaining cases
    if m >= 128 and n >= 128:
        return SM90_FP8_Tile128x128x64
    if m >= 128:
        return SM90_FP8_Tile128x64x64
    if n >= 128:
        return SM90_FP8_Tile64x128x64
    return SM90_FP8_Tile64x64x64
