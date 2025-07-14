# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import NamedTuple, Optional, Tuple, Union

import torch

import cupy as cp
import math
from jaxtyping import Int
from torch import Tensor

import warpconvnet._C as _C

from warpconvnet.utils.logger import get_logger
from warpconvnet.utils.argsort import argsort
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.utils.cuda_utils import load_kernel

logger = get_logger(__name__)

# cuda_utils.py automatically handles the csrc path for just filename
_assign_order_16bit_kernel = load_kernel(
    kernel_file="morton_code.cu", kernel_name="assign_order_discrete_16bit_kernel"
)
_assign_order_20bit_kernel_4points = load_kernel(
    kernel_file="morton_code.cu", kernel_name="assign_order_discrete_20bit_kernel_4points"
)

# # Hash serialization kernels
# _xorsum_div_kernel = load_kernel(
#     kernel_file="hash_serialization.cu", kernel_name="xorsum_div_kernel"
# )


class POINT_ORDERING(Enum):
    RANDOM = 0
    MORTON_XYZ = 1
    MORTON_XZY = 2
    MORTON_YXZ = 3
    MORTON_YZX = 4
    MORTON_ZXY = 5
    MORTON_ZYX = 6
    # XORSUM_DIV = 7


STR2POINT_ORDERING = {
    "random": POINT_ORDERING.RANDOM,
    "morton_xyz": POINT_ORDERING.MORTON_XYZ,
    "morton_xzy": POINT_ORDERING.MORTON_XZY,
    "morton_yxz": POINT_ORDERING.MORTON_YXZ,
    "morton_yzx": POINT_ORDERING.MORTON_YZX,
    "morton_zxy": POINT_ORDERING.MORTON_ZXY,
    "morton_zyx": POINT_ORDERING.MORTON_ZYX,
}

POINT_ORDERING_TO_MORTON_PERMUTATIONS = {
    POINT_ORDERING.MORTON_XYZ: [0, 1, 2],
    POINT_ORDERING.MORTON_XZY: [0, 2, 1],
    POINT_ORDERING.MORTON_YXZ: [1, 0, 2],
    POINT_ORDERING.MORTON_YZX: [1, 2, 0],
    POINT_ORDERING.MORTON_ZXY: [2, 0, 1],
    POINT_ORDERING.MORTON_ZYX: [2, 1, 0],
}


class SerializationResult(NamedTuple):
    """
    Named tuple containing serialization results.

    Attributes:
        codes: Serialization codes of the grid coordinates
        perm: Permutation that sorts coordinates by their codes (sorted_data = original_data[perm])
        inverse_perm: Inverse permutation to restore original order (original_data = sorted_data[inverse_perm])
        counts: Number of codes with the same value
    """

    codes: Tensor
    perm: Optional[Tensor] = None
    inverse_perm: Optional[Tensor] = None
    counts: Optional[Tensor] = None


@torch.no_grad()
def encode(
    grid_coord: Int[Tensor, "N 3"] | Int[Tensor, "N 4"],
    batch_offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
    order: POINT_ORDERING | str = POINT_ORDERING.MORTON_XYZ,
    return_perm: bool = False,
    return_inverse: bool = False,
    size: Optional[int] = None,
    num_buckets: Optional[int] = None,
) -> Union[Int[Tensor, "N"], SerializationResult]:  # noqa: F821
    """
    Generate ordering of the grid coordinates with optional permutation and inverse permutation.

    Args:
        grid_coord: Grid coordinates (N, 3) or (N, 4)
        batch_offsets: Batch offsets for multi-batch processing.
        order: Coordinate ordering scheme. Options include:
            - Morton codes: MORTON_XYZ, MORTON_XZY, MORTON_YXZ, MORTON_YZX, MORTON_ZXY, MORTON_ZYX
            - Hash functions: XORSUM_DIV, XORSUM_MOD, ZORDER_DIV, ZORDER_MOD, SUM_DIV
            - Other: RANDOM
        return_perm: Whether to return the permutation that sorts the coordinates by their codes.
        return_inverse: Whether to return the inverse permutation to restore original order.
        size: Size parameter for division-based hash functions (XORSUM_DIV, ZORDER_DIV, SUM_DIV).
            If None, uses number of points.
        num_buckets: Number of buckets for modulo-based hash functions (XORSUM_MOD, ZORDER_MOD).
            If None, uses sqrt of number of points.

    Returns:
        If return_perm=False and return_inverse=False:
            codes: serialization codes only (backward compatibility)
        Otherwise:
            SerializationResult with codes and requested permutations

    Examples:
        ```python
        # Just get codes (backward compatibility)
        codes = encode(coords, order=POINT_ORDERING.MORTON_XYZ)

        # Get structured result with permutation
        result = encode(coords, order=POINT_ORDERING.MORTON_XYZ, return_perm=True)
        sorted_coords = coords[result.perm]

        # Get structured result with permutation and inverse (Point Transformer style)
        result = encode(coords, order=POINT_ORDERING.MORTON_XYZ,
                       return_perm=True, return_inverse=True)
        sorted_coords = coords[result.perm]
        restored_coords = sorted_coords[result.inverse_perm]  # Should equal original coords

        # Use hash functions with custom parameters
        result = encode(coords, order=POINT_ORDERING.XORSUM_DIV, size=1000, return_perm=True)
        result = encode(coords, order=POINT_ORDERING.XORSUM_MOD, num_buckets=128, return_perm=True)

        # Access fields
        codes = result.codes
        perm = result.perm
        inverse_perm = result.inverse_perm
        ```
    """
    if isinstance(order, str):
        order = STR2POINT_ORDERING[order.lower()]

    # Early return for backward compatibility when no permutations requested
    if grid_coord.shape[0] == 0:
        codes = torch.empty(0, dtype=torch.int64)
    elif order in POINT_ORDERING_TO_MORTON_PERMUTATIONS.keys():
        codes = morton_code(grid_coord, order=order)
    elif order == POINT_ORDERING.RANDOM:
        codes = torch.randperm(grid_coord.shape[0], device=grid_coord.device)
    else:
        raise NotImplementedError(f"Order '{order}' not supported at the moment")

    # Early return
    if not return_perm and not return_inverse:
        return codes

    # Handle empty grid for structured result
    if (return_perm or return_inverse) and codes.shape[0] == 0:
        empty_tensor = torch.empty(0, dtype=torch.int64)
        return SerializationResult(
            codes=codes,
            perm=empty_tensor if return_perm else None,
            inverse_perm=empty_tensor if return_inverse else None,
        )

    # Generate permutation (when either return_perm or return_inverse is True)
    if batch_offsets is not None:
        perm, _ = _C.utils.segmented_sort(
            codes, batch_offsets.to(codes.device), descending=False, return_indices=True
        )
    else:
        perm = torch.argsort(codes)

    # Generate inverse permutation if requested
    inverse_perm = None
    if return_inverse:
        inverse_perm = torch.zeros_like(perm).scatter_(
            0, perm, torch.arange(len(perm), device=perm.device)
        )

    return SerializationResult(
        codes=codes,
        perm=perm,
        inverse_perm=inverse_perm,
    )


@torch.no_grad()
def morton_code(
    coords: Int[Tensor, "N 3"] | Int[Tensor, "N 4"],  # noqa: F821
    threads_per_block: int = 256,
    order: POINT_ORDERING | str = POINT_ORDERING.MORTON_XYZ,
) -> Int[Tensor, "N"]:  # noqa: F821
    """
    Generate Morton codes for the input coordinates.

    Args:
        coords: Input coordinates (N, 3) or (N, 4)
        threads_per_block: CUDA threads per block
        order: Coordinate ordering scheme (e.g., POINT_ORDERING.MORTON_XYZ, POINT_ORDERING.MORTON_XZY, etc.)

    Returns:
        Morton codes

    The coords must be in the range [0, 2^16 - 1] for 16-bit path (batched)
    or effectively [0, 2^20 - 1] for 20-bit path (single batch, after normalization)
    and the result will be the z-order number of the point.
    """
    if isinstance(order, str):
        order = POINT_ORDERING(order)

    # Assert that the order is morton
    assert (
        order in POINT_ORDERING_TO_MORTON_PERMUTATIONS.keys()
    ), f"Order '{order}' not supported for morton code"

    # Empty grid handling
    if coords.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64)

    min_coord = coords.min(0).values
    coords_normalized = (coords - min_coord).to(dtype=torch.int32).cuda()

    # Apply coordinate transformation based on ordering
    perm = POINT_ORDERING_TO_MORTON_PERMUTATIONS[order]
    if perm != [0, 1, 2]:  # Only apply permutation if it's not standard xyz
        if coords_normalized.shape[1] == 3:
            coords_normalized = coords_normalized[:, perm]
        elif coords_normalized.shape[1] == 4:  # batched coordinates [b, x, y, z]
            # Create permutation for batched coordinates: [b, x, y, z] -> [b, perm[0], perm[1], perm[2]]
            batch_perm = [0] + [p + 1 for p in perm]  # [0, perm[0]+1, perm[1]+1, perm[2]+1]
            coords_normalized = coords_normalized[:, batch_perm]

    device = coords_normalized.device
    num_points = len(coords_normalized)
    result_code_cp = cp.empty(num_points, dtype=cp.int64)

    assert coords_normalized.shape[1] == 3, "coords must be [N, 3]"
    # Single batch path (20-bit)
    coords_cp = cp.from_dlpack(coords_normalized.contiguous())
    # The kernel loads 4 points per thread, so we need to adjust the number of blocks
    blocks_per_grid = math.ceil(num_points / (threads_per_block * 4))
    _assign_order_20bit_kernel_4points(
        (blocks_per_grid,),
        (threads_per_block,),
        (coords_cp, num_points, result_code_cp),
    )

    # Convert result from CuPy array back to PyTorch tensor on the original device
    result_code = torch.as_tensor(result_code_cp, device=device)
    return result_code
