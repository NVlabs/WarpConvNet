# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from typing import Literal, Optional, Tuple, Union

import torch
import warpconvnet._C as _C
from jaxtyping import Int
from torch import Tensor

from warpconvnet.geometry.coords.ops.serialization import SerializationResult
from warpconvnet.utils.ravel import ravel_multi_index_auto_shape
from warpconvnet.utils.unique import unique_segmented

STR2COORD_OFFSET = {
    "random": (None, None, None),
    "zero": (0, 0, 0),
    "x": (0.5, 0, 0),
    "y": (0, 0.5, 0),
    "z": (0, 0, 0.5),
    "xy": (0.5, 0.5, 0),
    "xz": (0.5, 0, 0.5),
    "yz": (0, 0.5, 0.5),
    "xyz": (0.5, 0.5, 0.5),
}

WINDOW_OFFSET_TYPE = Literal["random", "zero", "x", "y", "z", "xy", "xz", "yz", "xyz"]

# Per-forward cache so attention blocks at the same level reuse permutations
# computed by sibling blocks. Call clear_encode_cache() at the start of each
# forward pass to avoid stale entries across iterations.
_ENCODE_CACHE: dict = {}


def clear_encode_cache():
    """Clear the voxel encode cache. Call at the start of each forward pass."""
    _ENCODE_CACHE.clear()


def voxel_encode_cached(
    grid_coord: Int[Tensor, "N 3"],
    batch_offsets: Optional[Int[Tensor, "B+1"]] = None,
    window_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    coord_offset: Union[str, Tuple[float, float, float]] = (0.0, 0.0, 0.0),
    encoding_method: str = "counting_sort",
) -> SerializationResult:
    """Cached `voxel_encode` returning ``perm``, ``inverse_perm`` and ``counts``.

    Reuses results for identical (coords, window_size, offset, method). Skips
    caching when ``coord_offset == "random"``. Call `clear_encode_cache`
    before each forward pass.
    """
    if coord_offset == "random":
        return voxel_encode(
            grid_coord,
            batch_offsets,
            window_size=window_size,
            coord_offset=coord_offset,
            return_perm=True,
            return_inverse=True,
            return_counts=True,
            encoding_method=encoding_method,
        )

    if isinstance(coord_offset, str):
        offset_key = coord_offset
    else:
        offset_key = tuple(float(x) for x in coord_offset)

    if isinstance(window_size, int):
        ws_key = (window_size, window_size, window_size)
    else:
        ws_key = tuple(window_size)

    key = (grid_coord.data_ptr(), grid_coord.shape[0], ws_key, offset_key, encoding_method)

    if key not in _ENCODE_CACHE:
        _ENCODE_CACHE[key] = voxel_encode(
            grid_coord,
            batch_offsets,
            window_size=window_size,
            coord_offset=coord_offset,
            return_perm=True,
            return_inverse=True,
            return_counts=True,
            encoding_method=encoding_method,
        )
    return _ENCODE_CACHE[key]


@torch.no_grad()
def voxel_encode(
    grid_coord: Int[Tensor, "N 3"],
    batch_offsets: Optional[Int[Tensor, "B+1"]] = None,
    window_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    coord_offset: Union[str, Tuple[float, float, float]] = (0.0, 0.0, 0.0),
    return_perm: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    encoding_method: str = "ravel",
) -> Union[Int[Tensor, "N"], SerializationResult]:  # noqa: F821
    """Encode voxel coordinates so that voxels falling within the same window
    share an integer code.

    Args:
        grid_coord: ``(N, 3)`` integer grid coordinates.
        batch_offsets: ``(B+1,)`` cumulative batch offsets.
        window_size: window length per axis (scalar or 3-tuple).
        coord_offset: shift applied before window quantization. Either a 3-tuple
            of float fractions of ``window_size`` or one of the string keys in
            `STR2COORD_OFFSET`.
        return_perm / return_inverse / return_counts: optional outputs.
        encoding_method: ``"morton"``, ``"ravel"``, ``"ravel_fast"``, or
            ``"counting_sort"`` (default for cached path).
    """
    if grid_coord.shape[0] == 0:
        codes = torch.empty(0, dtype=torch.int64, device=grid_coord.device)
        if not return_perm and not return_inverse:
            return codes
        return SerializationResult(
            codes=codes,
            perm=(
                torch.empty(0, dtype=torch.int64, device=grid_coord.device)
                if return_perm
                else None
            ),
            inverse_perm=(
                torch.empty(0, dtype=torch.int64, device=grid_coord.device)
                if return_inverse
                else None
            ),
            counts=(
                torch.empty(0, dtype=torch.int64, device=grid_coord.device)
                if return_counts
                else None
            ),
        )

    assert grid_coord.shape[1] == 3, "grid_coord must be a 3D tensor"
    assert encoding_method in [
        "morton",
        "ravel",
        "ravel_fast",
        "counting_sort",
    ], f"encoding_method must be 'morton', 'ravel', 'ravel_fast', or 'counting_sort', got {encoding_method}"

    if isinstance(coord_offset, str):
        if coord_offset == "random":
            coord_offset = (random.random(), random.random(), random.random())
        else:
            coord_offset = STR2COORD_OFFSET[coord_offset]

    assert (
        isinstance(coord_offset, tuple) and len(coord_offset) == 3
    ), "coord_offset must be a tuple of 3 floats"

    assert window_size is not None, "window_size must be provided"
    if isinstance(window_size, int):
        window_size = (window_size, window_size, window_size)

    assert (
        isinstance(window_size, tuple) and len(window_size) == 3
    ), "window_size must be an integer or a tuple of 3 integers"

    window_size_tensor = torch.tensor(window_size, dtype=torch.int32)

    coord_offset_tensor = (
        torch.round(torch.tensor(coord_offset, dtype=torch.float32) * window_size_tensor.float())
        .int()
        .to(grid_coord.device)
    )

    min_coord = grid_coord.min(dim=0).values.int()

    if encoding_method == "morton":
        N = grid_coord.shape[0]
        grid_coord_int = grid_coord.int().contiguous()
        codes = torch.empty(N, dtype=torch.int64, device=grid_coord.device)

        _C.coords.coord_to_code(
            grid_coord_int,
            coord_offset_tensor.contiguous(),
            min_coord.contiguous(),
            window_size_tensor.to(grid_coord.device).contiguous(),
            N,
            codes,
        )

    elif encoding_method == "ravel_fast":
        device = grid_coord.device
        N = grid_coord.shape[0]
        ws = window_size_tensor.to(device)

        voxel_coord = (grid_coord + coord_offset_tensor - min_coord) // ws

        shape = voxel_coord.max(dim=0).values + 1
        s2 = shape[2].long()
        s1 = shape[1].long() * s2
        codes = (
            voxel_coord[:, 0].long() * s1
            + voxel_coord[:, 1].long() * s2
            + voxel_coord[:, 2].long()
        )

        if not return_perm and not return_inverse and not return_counts:
            return codes

        if batch_offsets is not None:
            batch_idx = torch.searchsorted(
                batch_offsets[1:].to(device=device).long(),
                torch.arange(N, device=device, dtype=torch.int64),
                side="right",
            )
            max_code = codes.max() + 1
            combined = batch_idx * max_code + codes
        else:
            combined = codes

        sorted_codes, perm = torch.sort(combined)
        counts = None
        if return_counts:
            _, counts = torch.unique_consecutive(sorted_codes, return_counts=True)

        inverse_perm = None
        if return_inverse:
            inverse_perm = torch.empty(N, dtype=torch.int64, device=device)
            inverse_perm[perm] = torch.arange(N, device=device)

        return SerializationResult(
            codes=codes,
            perm=perm if return_perm else None,
            inverse_perm=inverse_perm,
            counts=counts,
        )

    elif encoding_method == "counting_sort":
        device = grid_coord.device
        N = grid_coord.shape[0]

        grid_coord_int = grid_coord.int().contiguous()
        ws_dev = window_size_tensor.to(device).contiguous()

        if batch_offsets is None:
            batch_offsets = torch.tensor([0, N], dtype=torch.int32)
        B = batch_offsets.shape[0] - 1
        batch_offsets_dev = batch_offsets.to(device).int().contiguous()

        max_coord = grid_coord.max(dim=0).values.int()
        extent = max_coord + coord_offset_tensor - min_coord + 1
        grid_shape = (
            ((extent + window_size_tensor.to(device) - 1) // window_size_tensor.to(device))
            .int()
            .contiguous()
        )
        W = int(grid_shape[0].item() * grid_shape[1].item() * grid_shape[2].item())

        total_bins = B * W
        MAX_BINS = 4 * 1024 * 1024  # 16 MB of int32
        if total_bins > MAX_BINS:
            return voxel_encode(
                grid_coord,
                batch_offsets,
                window_size,
                coord_offset,
                return_perm,
                return_inverse,
                return_counts,
                encoding_method="ravel_fast",
            )

        codes = torch.empty(N, dtype=torch.int64, device=device)
        histogram = torch.zeros(total_bins, dtype=torch.int32, device=device)
        _C.coords.window_group_histogram(
            grid_coord_int,
            batch_offsets_dev,
            coord_offset_tensor.contiguous(),
            min_coord.contiguous(),
            ws_dev,
            grid_shape,
            codes,
            histogram,
            N,
            B,
            W,
        )

        if not return_perm and not return_inverse and not return_counts:
            return codes

        cumsum = histogram.cumsum(dim=0)
        window_offsets_dense = (cumsum - histogram).int().contiguous()

        counts = None
        if return_counts:
            nonzero_mask = histogram > 0
            counts = histogram[nonzero_mask].long()

        perm = torch.empty(N, dtype=torch.int64, device=device)
        inverse_perm = torch.empty(N, dtype=torch.int64, device=device)
        scatter_counters = torch.zeros(total_bins, dtype=torch.int32, device=device)
        _C.coords.window_group_scatter(
            codes,
            window_offsets_dense,
            scatter_counters,
            perm,
            inverse_perm,
            N,
        )

        return SerializationResult(
            codes=codes,
            perm=perm if return_perm else None,
            inverse_perm=inverse_perm if return_inverse else None,
            counts=counts,
        )

    else:  # encoding_method == "ravel"
        voxel_coord = (grid_coord + coord_offset_tensor - min_coord) // window_size_tensor.to(
            grid_coord.device
        )
        codes = ravel_multi_index_auto_shape(voxel_coord, dim=0)

    if not return_perm and not return_inverse and not return_counts:
        return codes

    counts = None
    if batch_offsets is not None:
        perm, sorted_codes = _C.utils.segmented_sort(
            codes,
            batch_offsets.to(codes.device),
            descending=False,
            return_indices=True,
        )
        unique_codes, counts = unique_segmented(
            sorted_codes, batch_offsets.cpu(), return_counts=return_counts
        )
    else:
        perm = torch.argsort(codes)
        if return_counts:
            counts = torch.unique_consecutive(codes, return_counts=True)[1]

    inverse_perm = None
    if return_inverse:
        inverse_perm = torch.zeros_like(perm).scatter_(
            0, perm, torch.arange(len(perm), device=perm.device)
        )

    return SerializationResult(
        codes=codes,
        perm=perm if return_perm else None,
        inverse_perm=inverse_perm,
        counts=counts,
    )
