# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union, Generator, Dict, Any, Sequence

import numpy as np
import torch
from jaxtyping import Float, Int
import logging
import cupy as cp
import math
import os
import itertools

from torch import Tensor
from torch.autograd import Function
from torch.utils.dlpack import to_dlpack as torch_to_dlpack, from_dlpack as torch_from_dlpack

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.geometry.coords.search.cache import IntSearchCache, IntSearchCacheKey
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.coords.ops.expand import (
    expand_coords,
)
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING, encode
from warpconvnet.nn.functional.sparse_pool import sparse_reduce
from warpconvnet.utils.ntuple import ntuple
from warpconvnet.utils.cuda_utils import load_kernel
from warpconvnet.utils.logger import get_logger
from warpconvnet.utils.benchmark_cache import (
    load_benchmark_cache,
    save_benchmark_cache,
    mark_benchmark_cache_dirty,
)
from warpconvnet.nn.functional.sparse_conv import (
    SpatiallySparseConvConfig,
    _maybe_cast,
    _get_cuda_kernel,
)
from warpconvnet.utils.type_cast import _min_dtype, _max_dtype

logger = get_logger(__name__)
try:
    import warpconvnet._C as _C
except ImportError as e:
    logger.warning(f"Error importing warpconvnet._C: {e}. Using fallback implementation.")
    _C = None


def _explicit_depthwise_forward_logic(
    in_features: Float[Tensor, "N C"],  # noqa: F821
    weight: Float[Tensor, "K C"],  # noqa: F821
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
) -> Float[Tensor, "M C"]:  # noqa: F821
    device = in_features.device
    comp_in_feats = _maybe_cast(in_features, compute_dtype)
    comp_weight = _maybe_cast(weight, compute_dtype)
    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        output_feature_tensor = comp_in_feats * comp_weight[iden_idx].unsqueeze(0)
    else:
        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=comp_in_feats.dtype
        )

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device)
        out_map = out_map.to(device)
        curr_out_features = comp_in_feats[in_map] * comp_weight[i].unsqueeze(0)
        output_feature_tensor[out_map] += curr_out_features.to(device=device)
    return output_feature_tensor.to(dtype=in_features.dtype)


def _explicit_depthwise_backward_logic(
    grad_output: Float[Tensor, "M C"],  # noqa: F821
    in_features: Float[Tensor, "N C"],  # noqa: F821
    weight: Float[Tensor, "K C"],  # noqa: F821
    kernel_map: IntSearchResult,
    compute_dtype: Optional[torch.dtype] = None,
    device: torch.device = None,
) -> Tuple[Float[Tensor, "N C"], Float[Tensor, "K C"]]:  # noqa: F821
    if device is None:
        device = grad_output.device

    dtype_to_use = compute_dtype if compute_dtype is not None else in_features.dtype
    comp_in_feats = in_features.to(device=device, dtype=dtype_to_use)
    comp_weight = weight.to(device=device, dtype=dtype_to_use)
    comp_grad_output = grad_output.to(device=device, dtype=dtype_to_use)
    grad_weight = torch.zeros_like(comp_weight, device=device)

    # y = x * w
    # L = f(y)
    # dL/dx = dL/dy * dy/dx = dL/dy * w
    # dL/dw = dL/dy * dy/dw = dL/dy * x
    # dL/dy is grad_output
    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = comp_grad_output * comp_weight[iden_idx].unsqueeze(0)
        grad_weight[iden_idx] = torch.sum(comp_in_feats * comp_grad_output, dim=0)
    else:
        grad_in_features = torch.zeros_like(comp_in_feats, device=device)

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue

        curr_grad_output = comp_grad_output[out_map]
        curr_in_feats = comp_in_feats[in_map]
        curr_weight = comp_weight[i]
        grad_in_features[in_map] += curr_grad_output * curr_weight.unsqueeze(0)
        grad_weight[i] += torch.sum(curr_in_feats * curr_grad_output, dim=0)
    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )
