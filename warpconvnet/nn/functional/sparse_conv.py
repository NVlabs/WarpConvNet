# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
from jaxtyping import Float, Int
import math
import itertools

from torch import Tensor
from torch.autograd import Function

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
from warpconvnet.utils.logger import get_logger
from warpconvnet.utils.benchmark_cache import (
    SpatiallySparseConvConfig,
    load_sparse_conv_benchmark_cache,
    mark_benchmark_cache_dirty,
)
from warpconvnet.utils.type_cast import _min_dtype, _maybe_cast
from warpconvnet.utils.timer import CUDATimer
from warpconvnet.utils.ntuple import _pad_tuple
from warpconvnet.constants import (
    WARPCONVNET_FWD_ALGO_MODE,
    WARPCONVNET_BWD_ALGO_MODE,
)

logger = get_logger(__name__)
try:
    import warpconvnet._C as _C
except ImportError as e:
    logger.warning(f"Error importing warpconvnet._C: {e}. Using fallback implementation.")
    _C = None


class STRIDED_CONV_MODE(Enum):
    REDUCE_AND_STRIDE = "reduce_and_stride"  # Apply convolution on the pooled input. This increases the density of the input
    STRIDE_ONLY = "stride_only"


# New Enums for granular algorithm control
class SPARSE_CONV_FWD_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    CUTLASS_IMPLICIT_GEMM = "cutlass_implicit_gemm"
    # EXPLICIT_GEMM_BATCHED = "explicit_gemm_batched" # TODO: Add if supporting
    AUTO = "auto"  # Benchmark and select the best algorithm


class SPARSE_CONV_BWD_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    CUTLASS_IMPLICIT_GEMM = "cutlass_implicit_gemm"
    # EXPLICIT_GEMM_BATCHED = "explicit_gemm_batched" # TODO: Add if supporting
    AUTO = "auto"  # Benchmark and select the best algorithm


# Separate benchmark parameters for independent operations
_BENCHMARK_NUM_RUNS = 2

# AD_gather_scatter operation (used in forward pass and backward pass for input gradients)
_BENCHMARK_AD_GATHER_SCATTER_PARAMS = [
    # Base configurations without split-K
    *[{"mma_tile": tile, "split_k_slices": 1} for tile in range(4)],
    # Successful combinations from current benchmarks
    *[
        {"mma_tile": tile, "split_k_slices": k}
        for tile, k in [
            # Proven successful combinations from forward/backward results
            (0, 2),
            (0, 4),  # Intermediate between successful 2 and 8
            (0, 8),  # MMA=0 patterns
            (1, 2),
            (1, 4),
            (1, 16),
            (1, 32),  # MMA=1 patterns
            (3, 2),
            (3, 4),
            (3, 8),
            (3, 16),
            (3, 32),  # MMA=3 patterns (most successful)
        ]
    ],
]

# trAB_gather operation (used only in backward pass for weight gradients)
_BENCHMARK_TRAB_GATHER_PARAMS = [
    # Base configurations without split-K
    *[{"mma_tile": tile, "split_k_slices": 1} for tile in range(4)],
    # Successful combinations from backward results
    *[
        {"mma_tile": tile, "split_k_slices": k}
        for tile, k in [
            (0, 8),  # MMA=0 for large problems
            (1, 4),  # Intermediate between successful 1 and 8
            (1, 8),
            (1, 16),
            (1, 32),  # MMA=1 TrAB patterns
            (3, 4),
            (3, 8),
            (3, 16),
            (3, 32),  # MMA=3 TrAB patterns (most successful)
        ]
    ],
]

# Forward pass uses only AD_gather_scatter operation
_BENCHMARK_FORWARD_PARAMS = [
    *[
        (SPARSE_CONV_FWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM, params)
        for params in _BENCHMARK_AD_GATHER_SCATTER_PARAMS
    ],
    *[
        (SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM, {"fwd_block_size": block_size})
        for block_size in [4, 16, 32]
    ],
]

# Backward pass combines best AD params with best TrAB params
_BENCHMARK_BACKWARD_PARAMS = [
    # Base combinations without split-K (for compatibility)
    *[
        (
            SPARSE_CONV_BWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM,
            {"mma_tile_ad": ad, "mma_tile_trab": trab},
        )
        for ad, trab in itertools.product(range(4), repeat=2)
    ],
    # Cartesian product of best AD params × best TrAB params
    *[
        (
            SPARSE_CONV_BWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM,
            {
                "mma_tile_ad": ad_params.get("mma_tile", 0),
                "split_k_slices_ad": ad_params.get("split_k_slices", 1),
                "mma_tile_trab": trab_params.get("mma_tile", 0),
                "split_k_slices_trab": trab_params.get("split_k_slices", 1),
            },
        )
        for ad_params in _BENCHMARK_AD_GATHER_SCATTER_PARAMS
        for trab_params in _BENCHMARK_TRAB_GATHER_PARAMS
        # Only include combinations with split-K to avoid duplicating base cases
        if ad_params.get("split_k_slices") is not None
        or trab_params.get("split_k_slices") is not None
    ],
    *[
        (
            SPARSE_CONV_BWD_ALGO_MODE.IMPLICIT_GEMM,
            {
                "gemm_block_size": gemm_block_size,
                "split_k_threads_per_block": split_k_threads_per_block,
                "split_k_factor": split_k_factor,
            },
        )
        for gemm_block_size in [4, 16, 32]
        for split_k_threads_per_block in [256]
        for split_k_factor in [2, 4, 8, 16]
    ],
]

_BENCHMARK_FORWARD_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[SPARSE_CONV_FWD_ALGO_MODE, Dict[str, Any], float]],
] = {}
_BENCHMARK_BACKWARD_RESULTS: Dict[
    SpatiallySparseConvConfig,
    List[Tuple[SPARSE_CONV_BWD_ALGO_MODE, Dict[str, Any], float]],
] = {}


# Load cached benchmark results at module initialization
def _initialize_benchmark_cache():
    """Load cached benchmark results and populate global dictionaries."""
    try:
        cached_forward, cached_backward = load_sparse_conv_benchmark_cache()
        _BENCHMARK_FORWARD_RESULTS.update(cached_forward)
        _BENCHMARK_BACKWARD_RESULTS.update(cached_backward)
        if cached_forward or cached_backward:
            logger.info(
                f"Loaded {len(cached_forward)} forward and {len(cached_backward)} "
                f"backward benchmark configurations from cache"
            )
    except Exception as e:
        logger.warning(f"Failed to initialize benchmark cache: {e}")


# Initialize cache on module load
_initialize_benchmark_cache()


def _filter_benchmark_params_by_env_config(
    all_params: List[
        Tuple[Union[SPARSE_CONV_FWD_ALGO_MODE, SPARSE_CONV_BWD_ALGO_MODE], Dict[str, Any]]
    ],
    env_config: Union[str, List[Union[str, SPARSE_CONV_FWD_ALGO_MODE, SPARSE_CONV_BWD_ALGO_MODE]]],
    is_forward: bool = True,
) -> List[Tuple[Union[SPARSE_CONV_FWD_ALGO_MODE, SPARSE_CONV_BWD_ALGO_MODE], Dict[str, Any]]]:
    """Filter benchmark parameters based on environment variable configuration.

    Args:
        all_params: All available benchmark parameters
        env_config: Environment variable value (string or list of algorithm names)
        is_forward: Whether this is for forward pass (affects enum type)

    Returns:
        Filtered list of benchmark parameters
    """
    if env_config == "auto":
        # When "auto", use all available algorithms
        return all_params

    # Convert environment config to list of algorithm names
    if isinstance(env_config, str):
        algo_names = [env_config]
    else:
        algo_names = env_config

    # Map string algorithm names to enum values
    EnumClass = SPARSE_CONV_FWD_ALGO_MODE if is_forward else SPARSE_CONV_BWD_ALGO_MODE

    # Get target enum values
    target_algos = []
    for algo_name in algo_names:
        if isinstance(algo_name, EnumClass):
            target_algos.append(algo_name)
        else:
            try:
                target_algos.append(EnumClass(algo_name))
            except ValueError:
                logger.warning(f"Unknown algorithm name: {algo_name}")

    if not target_algos:
        logger.warning(f"No valid algorithms found in {algo_names}, using all algorithms")
        return all_params

    # Filter parameters to only include target algorithms
    filtered_params = []
    for algo_enum, params in all_params:
        if algo_enum in target_algos:
            filtered_params.append((algo_enum, params))

    if not filtered_params:
        logger.warning(
            f"No benchmark parameters found for algorithms {algo_names}, using all algorithms"
        )
        return all_params

    return filtered_params


def _get_filtered_forward_params() -> List[Tuple[SPARSE_CONV_FWD_ALGO_MODE, Dict[str, Any]]]:
    """Get forward benchmark parameters filtered by environment variable."""
    return _filter_benchmark_params_by_env_config(
        _BENCHMARK_FORWARD_PARAMS, WARPCONVNET_FWD_ALGO_MODE, is_forward=True
    )


def _get_filtered_backward_params() -> List[Tuple[SPARSE_CONV_BWD_ALGO_MODE, Dict[str, Any]]]:
    """Get backward benchmark parameters filtered by environment variable."""
    return _filter_benchmark_params_by_env_config(
        _BENCHMARK_BACKWARD_PARAMS, WARPCONVNET_BWD_ALGO_MODE, is_forward=False
    )


def _explicit_gemm_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
) -> Float[Tensor, "M C_out"]:
    """Reference implementation of forward pass using explicit GEMM."""
    device = in_features.device
    comp_in_feats = _maybe_cast(in_features, compute_dtype)
    comp_weight = _maybe_cast(weight, compute_dtype)
    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        output_feature_tensor = torch.matmul(comp_in_feats, comp_weight[iden_idx])
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
        curr_out_features = torch.matmul(comp_in_feats[in_map], comp_weight[i])
        output_feature_tensor[out_map] += curr_out_features.to(device=device)
    return output_feature_tensor.to(dtype=in_features.dtype)


def _explicit_gemm_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    compute_dtype: Optional[torch.dtype] = None,
    device: torch.device = None,
) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
    """Backward pass for explicit GEMM convolution."""
    if device is None:
        device = grad_output.device

    dtype_to_use = compute_dtype if compute_dtype is not None else in_features.dtype
    comp_in_feats = in_features.to(device=device, dtype=dtype_to_use)
    comp_weight = weight.to(device=device, dtype=dtype_to_use)
    comp_grad_output = grad_output.to(device=device, dtype=dtype_to_use)
    grad_weight = torch.zeros_like(comp_weight, device=device)

    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = torch.matmul(comp_grad_output, comp_weight[iden_idx].T)
        grad_weight[iden_idx] = torch.matmul(comp_in_feats.T, comp_grad_output)
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
        grad_in_features[in_map] += torch.matmul(curr_grad_output, curr_weight.T)
        grad_weight[i] += torch.matmul(curr_in_feats.T, curr_grad_output)
    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )


def _implicit_gemm_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    fwd_block_size: int,
) -> Float[Tensor, "M C_out"]:
    """Forward pass using implicit GEMM with a custom CUDA kernel."""
    device = in_features.device
    feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
    min_dtype = _min_dtype(feature_dtype, weight.dtype)

    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)

    N_in, C_in = _in_features_detached.shape
    K, _, C_out = _weight_detached.shape
    iden_idx = kernel_map.identity_map_index

    if iden_idx is not None:
        output_feature_tensor = torch.matmul(_in_features_detached, _weight_detached[iden_idx]).to(
            dtype=min_dtype
        )
    else:
        output_feature_tensor = torch.zeros(
            (num_out_coords, C_out), dtype=min_dtype, device=device
        )

    if (
        num_out_coords == 0
        or K == 0
        or C_in == 0
        or C_out == 0
        or _in_features_detached.shape[0] == 0
    ):
        return output_feature_tensor.to(dtype=in_features.dtype)

    for k_idx in range(len(kernel_map)):
        if k_idx == iden_idx:
            continue

        in_map_k, out_map_k = kernel_map[k_idx]
        num_active_pairs = in_map_k.shape[0]
        if num_active_pairs == 0:
            continue

        _C.gemm.implicit_gemm(
            _in_features_detached,
            _weight_detached[k_idx],
            output_feature_tensor,
            in_map_k,
            out_map_k,
            "basic",
            fwd_block_size,
        )

    return output_feature_tensor.to(dtype=in_features.dtype)


def _implicit_gemm_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    gemm_block_size: int,
    split_k_threads_per_block: int,
    split_k_factor: int,
    compute_dtype: Optional[torch.dtype],
) -> Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]]:
    """Backward pass using implicit GEMM kernels."""
    feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
    min_dtype = _min_dtype(feature_dtype, weight.dtype)
    device = in_features.device

    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)
    _grad_output_detached = grad_output.contiguous().detach().to(dtype=min_dtype)

    N_in, C_in = _in_features_detached.shape
    K, _, C_out = _weight_detached.shape
    iden_idx = kernel_map.identity_map_index

    grad_weight_tensor = torch.zeros((K, C_in, C_out), dtype=min_dtype, device=device)
    if iden_idx is not None:
        grad_in_features = torch.matmul(_grad_output_detached, _weight_detached[iden_idx].T)
        grad_weight_tensor[iden_idx] = torch.matmul(
            _in_features_detached.T, _grad_output_detached
        ).to(dtype=min_dtype)
    else:
        grad_in_features = torch.zeros((N_in, C_in), dtype=min_dtype, device=device)

    if (
        num_out_coords == 0
        or K == 0
        or C_in == 0
        or C_out == 0
        or N_in == 0
        or _grad_output_detached.shape[0] == 0
    ):
        return grad_in_features.to(dtype=in_features.dtype), grad_weight_tensor.to(
            dtype=weight.dtype
        )

    for k_idx in range(len(kernel_map)):
        if k_idx == iden_idx:
            continue

        in_map_k, out_map_k = kernel_map[k_idx]
        num_active_pairs = in_map_k.shape[0]
        if num_active_pairs == 0:
            continue

        _C.gemm.implicit_gemm(
            _grad_output_detached,
            _weight_detached[k_idx].T,
            grad_in_features,
            out_map_k,
            in_map_k,
            "basic",
            gemm_block_size,
        )

        _C.gemm.split_k_implicit_gemm(
            _in_features_detached,
            _grad_output_detached,
            grad_weight_tensor[k_idx],
            in_map_k,
            out_map_k,
            split_k_factor=split_k_factor,
            block_size=split_k_threads_per_block,
        )

    return grad_in_features.to(dtype=in_features.dtype), grad_weight_tensor.to(dtype=weight.dtype)


def _cutlass_implicit_gemm_forward_logic(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    accumulator_type: torch.dtype = torch.float32,
    split_k_slices: int = 1,
    mma_tile: int = 0,
) -> Union[Float[Tensor, "M C_out"], int]:
    """Forward pass leveraging CUTLASS implicit GEMM kernels."""
    if _C is None:
        raise ImportError(
            "warpconvnet._C is not available. Please install warpconvnet with cutlass support."
        )

    device = in_features.device
    iden_idx = kernel_map.identity_map_index
    min_dtype = _min_dtype(in_features.dtype, weight.dtype)
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)
    if iden_idx is not None:
        output_feature_tensor = torch.matmul(_in_features_detached, _weight_detached[iden_idx])
    else:
        output_feature_tensor = torch.zeros(
            num_out_coords, weight.shape[-1], device=device, dtype=min_dtype
        )

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device)
        out_map = out_map.to(device)
        status = _C.gemm.cutlass_gemm_AD_gather_scatter(
            _in_features_detached,
            _weight_detached[i],
            output_feature_tensor,
            output_feature_tensor,
            in_map,
            out_map,
            accumulator_type=accumulator_type,
            split_k_slices=split_k_slices,
            mma_tile=mma_tile,
        )
        if status != 0:
            return status
    return output_feature_tensor.to(dtype=in_features.dtype)


def _cutlass_implicit_gemm_backward_logic(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    accumulator_type: torch.dtype = torch.float32,
    split_k_slices_ad: int = 1,
    split_k_slices_trab: int = 1,
    mma_tile_ad: int = 0,
    mma_tile_trab: int = 0,
    requires_grad: Tuple[bool, bool] = (True, True),
    device: torch.device = None,
) -> Union[Tuple[Float[Tensor, "N C_in"], Float[Tensor, "K C_in C_out"]], Tuple[int, int]]:
    """Backward pass leveraging CUTLASS implicit GEMM kernels."""
    if _C is None:
        raise ImportError(
            "warpconvnet._C is not available. Please install warpconvnet with cutlass support."
        )

    if device is None:
        device = in_features.device

    min_dtype = _min_dtype(in_features.dtype, weight.dtype, grad_output.dtype)
    _grad_output_detached = grad_output.contiguous().detach().to(dtype=min_dtype)
    _in_features_detached = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight_detached = weight.contiguous().detach().to(dtype=min_dtype)
    grad_weight = torch.zeros_like(weight, device=device)

    iden_idx = kernel_map.identity_map_index
    if iden_idx is not None:
        grad_in_features = torch.matmul(_grad_output_detached, _weight_detached[iden_idx].T)
        grad_weight[iden_idx] = torch.matmul(_in_features_detached.T, _grad_output_detached)
    else:
        grad_in_features = torch.zeros_like(_in_features_detached, device=device)

    for i in range(len(kernel_map)):
        if i == iden_idx:
            continue

        in_map, out_map = kernel_map[i]
        if in_map.shape[0] == 0:
            continue

        # grad_in_features[in_map] += torch.matmul(grad_output[out_map], curr_weight.T)
        if requires_grad[0]:
            status = _C.gemm.cutlass_gemm_AD_gather_scatter(
                _grad_output_detached,
                _weight_detached[i].T.contiguous(),
                grad_in_features,
                grad_in_features,
                out_map,
                in_map,
                accumulator_type=accumulator_type,
                split_k_slices=split_k_slices_ad,
                mma_tile=mma_tile_ad,
            )
            if status != 0:
                return status, i

        # grad_weight[i] += torch.matmul(curr_in_feats.T, curr_grad_output)
        if requires_grad[1]:
            status = _C.gemm.cutlass_gemm_trAB_gather(
                _in_features_detached,
                _grad_output_detached,
                grad_weight[i],
                grad_weight[i],
                in_map,
                out_map,
                alpha=1.0,
                beta=0,
                accumulator_type=accumulator_type,
                split_k_slices=split_k_slices_trab,
                mma_tile=mma_tile_trab,
            )
            if status != 0:
                return status, i

    return (
        grad_in_features.to(dtype=in_features.dtype),
        grad_weight.to(dtype=weight.dtype),
    )


class SpatiallySparseConvExplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> Float[Tensor, "M C_out"]:
        output_feature_tensor = _explicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
        )
        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.compute_dtype = compute_dtype
        ctx.device = in_features.device
        return output_feature_tensor

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Optional[Float[Tensor, "N C_in"]],
        Optional[Float[Tensor, "K C_in C_out"]],
        None,
        None,
        None,
    ]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        compute_dtype = ctx.compute_dtype
        device = ctx.device

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 5)

        # Basic check for empty inputs, similar to how it was in Unified Function
        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        # Assuming num_out_coords was implicitly handled by grad_output.shape[0] in original explicit backward
        if K == 0 or C_in == 0 or C_out == 0 or N_in == 0 or grad_output.shape[0] == 0:
            grad_in_final = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_weight_final = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in_final, grad_weight_final, 5)

        grad_in_features, grad_weight = _explicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            compute_dtype,
            device,
        )

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 5)


class SpatiallySparseConvImplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        gemm_block_size: int = 16,
        split_k_threads_per_block: int = 128,
        split_k_factor: int = 4,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> Float[Tensor, "M C_out"]:
        output_feature_tensor = _implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            gemm_block_size,
        )
        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.gemm_params = {
            "compute_dtype": compute_dtype,
            "gemm_block_size": gemm_block_size,
            "split_k_threads_per_block": split_k_threads_per_block,
            "split_k_factor": split_k_factor,
        }
        return output_feature_tensor

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Optional[Float[Tensor, "N C_in"]],
        Optional[Float[Tensor, "K C_in C_out"]],
        None,
        None,
        None,
        None,
        None,
    ]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        num_out_coords = grad_output.shape[0]
        gemm_params = ctx.gemm_params

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 7)

        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        if (
            num_out_coords == 0
            or K == 0
            or C_in == 0
            or C_out == 0
            or N_in == 0
            or grad_output.shape[0] == 0
        ):
            grad_in_final = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_weight_final = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in_final, grad_weight_final, 7)

        grad_in_features, grad_weight = _implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            gemm_params["gemm_block_size"],
            gemm_params["split_k_threads_per_block"],
            gemm_params["split_k_factor"],
            gemm_params["compute_dtype"],
        )

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 7)


class SpatiallySparseConvCutlassImplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        accumulator_type: torch.dtype = torch.float32,
        split_k_slices_ad: int = 1,
        split_k_slices_trab: int = 1,
        mma_tile_ad: int = 0,
        mma_tile_trab: int = 0,
    ) -> Union[Float[Tensor, "M C_out"], int]:
        output_feature_tensor = _cutlass_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            accumulator_type,
            split_k_slices=split_k_slices_ad,
            mma_tile=mma_tile_ad,
        )
        if isinstance(output_feature_tensor, int):
            raise RuntimeError(
                f"Error in _cutlass_implicit_gemm_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(output_feature_tensor))}"
            )

        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map
        ctx.cutlass_params = {
            "accumulator_type": accumulator_type,
            "split_k_slices_ad": split_k_slices_ad,
            "split_k_slices_trab": split_k_slices_trab,
            "mma_tile_ad": mma_tile_ad,
            "mma_tile_trab": mma_tile_trab,
        }
        ctx.device = in_features.device
        return output_feature_tensor

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Optional[Float[Tensor, "N C_in"]],
        Optional[Float[Tensor, "K C_in C_out"]],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        cutlass_params = ctx.cutlass_params
        device = ctx.device

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 9)

        # Basic check for empty inputs, similar to how it was in Unified Function
        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        # Assuming num_out_coords was implicitly handled by grad_output.shape[0] in original explicit backward
        if K == 0 or C_in == 0 or C_out == 0 or N_in == 0 or grad_output.shape[0] == 0:
            grad_in_final = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_weight_final = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in_final, grad_weight_final, 9)

        grad_in_features, grad_weight = _cutlass_implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            accumulator_type=cutlass_params["accumulator_type"],
            split_k_slices_ad=cutlass_params["split_k_slices_ad"],
            split_k_slices_trab=cutlass_params["split_k_slices_trab"],
            mma_tile_ad=cutlass_params["mma_tile_ad"],
            mma_tile_trab=cutlass_params["mma_tile_trab"],
            requires_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
            device=device,
        )
        if isinstance(grad_in_features, int):
            raise RuntimeError(
                f"Error in _cutlass_implicit_gemm_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(grad_in_features))}"
            )
        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 9)


class SpatiallySparseConvBatchedExplicitGEMMFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],  # K is batch of kernels here
        kernel_map: IntSearchResult,
        num_out_coords: int,
        matmul_batch_size: int,  # This is kernel_map batching, not to be confused with feature batch
        compute_dtype: torch.dtype = torch.float32,
    ) -> Float[Tensor, "M C_out"]:
        device = in_features.device
        # Output feature tensor has a dummy row for negative indices from out_maps
        output_feature_tensor = torch.zeros(
            num_out_coords + 1,
            weight.shape[-1],
            device=device,
            dtype=in_features.dtype,  # use in_features.dtype for accumulation
        )

        # Ensure weight is on the correct device and dtype for bmm
        comp_weight = weight.to(device=device, dtype=compute_dtype)

        for i_start in range(0, len(kernel_map), matmul_batch_size):
            i_end = min(i_start + matmul_batch_size, len(kernel_map))
            # Get the input and output maps of shape B x N_active (B here is matmul_batch_size)
            in_maps, out_maps = kernel_map.get_batch(
                i_start, i_end, out_format="tensor"
            )  # in_maps can have -1

            # curr_in_features: B x N_active x C_in
            # We need to handle -1 in in_maps which means no input for that position
            # We can use a dummy feature for -1 index, or mask later.
            # Using in_maps.clip(min=0) and then masking based on original in_maps < 0.
            # Create a temporary feature tensor that includes a zero vector for index -1
            # Padded_in_features: (N+1) x C_in
            _padded_in_features = torch.cat(
                (
                    in_features,
                    torch.zeros(1, in_features.shape[1], device=device, dtype=in_features.dtype),
                ),
                dim=0,
            )
            # Map in_maps to use the last row for -1 indices
            _mapped_in_maps = torch.where(in_maps == -1, in_features.shape[0], in_maps)
            curr_in_features = _padded_in_features[_mapped_in_maps].to(
                dtype=compute_dtype
            )  # B x N_active x C_in

            # bmm: B x N_active x C_in @ B x C_in x C_out -> B x N_active x C_out
            curr_out_features_batch = torch.bmm(curr_in_features, comp_weight[i_start:i_end])

            temp_output_batch = torch.zeros_like(
                output_feature_tensor, dtype=curr_out_features_batch.dtype
            )

            # Iterate over the batch dimension of kernel_map processing
            for b_idx in range(in_maps.shape[0]):  # iterate through matmul_batch_size
                # Get specific in_map, out_map for this kernel
                # in_map_b = in_maps[b_idx] # N_active_b
                out_map_b = out_maps[b_idx]  # N_active_b
                # features_b = curr_in_features[b_idx] # N_active_b x C_in
                out_contrib_b = curr_out_features_batch[b_idx]  # N_active_b x C_out

                valid_indices_mask_b = out_map_b != -1
                valid_out_map_b = out_map_b[valid_indices_mask_b]
                valid_out_contrib_b = out_contrib_b[valid_indices_mask_b]

                if valid_out_map_b.numel() > 0:
                    # Use index_add_ for sparse accumulation
                    temp_output_batch.index_add_(0, valid_out_map_b + 1, valid_out_contrib_b)

            output_feature_tensor += temp_output_batch.to(output_feature_tensor.dtype)

        ctx.compute_dtype = compute_dtype
        ctx.kernel_map = kernel_map
        ctx.matmul_batch_size = matmul_batch_size
        ctx.save_for_backward(in_features, weight)
        ctx.device = in_features.device
        ctx.num_in_features_N = in_features.shape[0]

        return output_feature_tensor[1:].clone().to(dtype=in_features.dtype)  # Remove dummy row

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Optional[Float[Tensor, "N C_in"]],
        Optional[Float[Tensor, "K C_in C_out"]],
        None,
        None,
        None,
        None,
    ]:
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        matmul_batch_size = ctx.matmul_batch_size
        compute_dtype = ctx.compute_dtype
        device = ctx.device
        num_in_features_N = ctx.num_in_features_N

        grad_in_features = None
        grad_weight = None

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 6)

        # Add a dummy row to grad_output for consistent indexing with out_maps
        _padded_grad_output = torch.cat(
            (
                torch.zeros(1, grad_output.shape[1], device=device, dtype=grad_output.dtype),
                grad_output,
            ),
            dim=0,
        ).to(dtype=compute_dtype)

        if ctx.needs_input_grad[0]:
            # Add a dummy row for accumulation, similar to original SpatiallySparseConvBatchedExplicitGEMMFunction
            grad_in_features_acc = torch.zeros(
                num_in_features_N + 1,
                in_features.shape[1],
                device=device,
                dtype=compute_dtype,
            )
        if ctx.needs_input_grad[1]:
            grad_weight_acc = torch.zeros_like(weight, dtype=compute_dtype)

        comp_weight = weight.to(device=device, dtype=compute_dtype)
        # Padded_in_features: (N+1) x C_in, for consistent indexing with in_maps
        _padded_in_features = torch.cat(
            (
                in_features,
                torch.zeros(1, in_features.shape[1], device=device, dtype=in_features.dtype),
            ),
            dim=0,
        ).to(dtype=compute_dtype)

        for i_start in range(0, len(kernel_map), matmul_batch_size):
            i_end = min(i_start + matmul_batch_size, len(kernel_map))
            in_maps, out_maps = kernel_map.get_batch(
                i_start, i_end, out_format="tensor"
            )  # B x N_active

            # curr_grad_out_feat: B x N_active x C_out. Indices from out_maps can be -1.
            # Use out_maps + 1 to index _padded_grad_output.
            curr_grad_out_feat = _padded_grad_output[
                out_maps + 1
            ]  # Will pick dummy grad for out_map == -1

            if ctx.needs_input_grad[0]:
                # Calculate grad_in_features contribution
                # dL/dX = dL/dY @ W.T
                # B x N_active x C_out @ B x C_out x C_in -> B x N_active x C_in
                grad_in_contrib = torch.bmm(
                    curr_grad_out_feat, comp_weight[i_start:i_end].transpose(1, 2)
                )

                # Scatter-add to grad_in_features_acc
                # Map in_maps to use the last row for -1 indices
                _mapped_in_maps = torch.where(
                    in_maps == -1, num_in_features_N, in_maps
                )  # B x N_active

                for b_idx in range(in_maps.shape[0]):  # matmul_batch_size
                    in_map_b = _mapped_in_maps[b_idx]  # N_active_b
                    grad_in_contrib_b = grad_in_contrib[b_idx]  # N_active_b x C_in

                    valid_scatter_mask_b = in_maps[b_idx] != -1  # Original in_maps

                    if valid_scatter_mask_b.any():
                        grad_in_features_acc.index_add_(
                            0,
                            in_map_b[valid_scatter_mask_b],
                            grad_in_contrib_b[valid_scatter_mask_b],
                        )

            if ctx.needs_input_grad[1]:
                # Calculate grad_weight contribution
                # dL/dW = X.T @ dL/dY
                # (B x C_in x N_active) @ (B x N_active x C_out) -> B x C_in x C_out
                # Map in_maps to use the last row for -1 indices (for _padded_in_features)
                _mapped_in_maps_for_X = torch.where(in_maps == -1, num_in_features_N, in_maps)
                curr_in_feat_for_grad_w = _padded_in_features[
                    _mapped_in_maps_for_X
                ]  # B x N_active x C_in

                # Zero out contributions where in_map or out_map was -1 originally
                # Mask for grad_output part (dL/dY)
                mask_grad_out = (out_maps != -1).unsqueeze(-1).expand_as(curr_grad_out_feat)
                masked_curr_grad_out_feat = curr_grad_out_feat * mask_grad_out.to(
                    curr_grad_out_feat.dtype
                )

                # Mask for input_feature part (X)
                mask_in_feat = (in_maps != -1).unsqueeze(-1).expand_as(curr_in_feat_for_grad_w)
                masked_curr_in_feat_for_grad_w = curr_in_feat_for_grad_w * mask_in_feat.to(
                    curr_in_feat_for_grad_w.dtype
                )

                grad_weight_acc[i_start:i_end] += torch.bmm(
                    masked_curr_in_feat_for_grad_w.transpose(1, 2),
                    masked_curr_grad_out_feat,
                )

        if ctx.needs_input_grad[0]:
            grad_in_features = (
                grad_in_features_acc[:-1].clone().to(dtype=in_features.dtype)
            )  # Remove dummy row
        if ctx.needs_input_grad[1]:
            grad_weight = grad_weight_acc.to(dtype=weight.dtype)

        return _pad_tuple(grad_in_features, grad_weight, 6)


def _run_forward_benchmarks(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    warmup_iters: int = _BENCHMARK_NUM_RUNS // 2,
    benchmark_iters: int = _BENCHMARK_NUM_RUNS,
    custom_params: Optional[List[Tuple[SPARSE_CONV_FWD_ALGO_MODE, Dict[str, Any]]]] = None,
) -> Tuple[SPARSE_CONV_FWD_ALGO_MODE, Dict[str, Any], float]:
    """
    Benchmark different forward algorithms and return the best one with its parameters and runtime.
    The best is determined by the minimum execution time over benchmark_iters.
    """
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    logger.warning(
        "Using benchmarked forward algo. Until the algorithm finds the best parameters, forward performance will be slow."
    )
    all_benchmark_results: List[Tuple[SPARSE_CONV_FWD_ALGO_MODE, Dict[str, Any], float]] = []
    timer = CUDATimer()

    def _execute_single_fwd_pass(
        algo_mode: SPARSE_CONV_FWD_ALGO_MODE, params_config: Dict[str, Any]
    ) -> Optional[int]:
        if algo_mode == SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM:
            _ = _explicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
            )
        elif algo_mode == SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM:
            _ = _implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                **params_config,
            )
        elif algo_mode == SPARSE_CONV_FWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM:
            status = _cutlass_implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                accumulator_type=params_config.get("accumulator_type", torch.float32),
                split_k_slices=params_config.get("split_k_slices", 1),
                mma_tile=params_config.get("mma_tile", 0),
            )
            if isinstance(status, int) and status != 0:
                return status
        else:
            # Should not happen with current _BENCHMARK_FORWARD_PARAMS
            raise ValueError(f"Unsupported algo_mode in _execute_single_fwd_pass: {algo_mode}")

    params_to_use = custom_params if custom_params is not None else _get_filtered_forward_params()
    for algo_mode, params_config in params_to_use:
        # Warmup runs
        status = None
        for _ in range(warmup_iters):
            status = _execute_single_fwd_pass(algo_mode, params_config)
            if isinstance(status, int) and status != 0:
                # logger.warning(
                #     f"Error in _run_forward_benchmarks: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(status))}"
                # )
                continue

        if status is not None and status != 0:
            continue

        # Benchmark runs
        current_algo_min_time_ms = float("inf")  # Min time for this specific algorithm config

        if benchmark_iters == 0:
            # No benchmark runs, current_algo_min_time_ms remains inf
            # If only warmup_iters > 0, we don't use warmup for min time, benchmark_iters must be > 0
            if warmup_iters == 0:
                continue  # No runs at all for this config, skip
        else:
            for _ in range(benchmark_iters):
                with timer:
                    _execute_single_fwd_pass(algo_mode, params_config)
                current_algo_min_time_ms = min(current_algo_min_time_ms, timer.elapsed_time)

            # If no timed runs were successful (e.g. CUDA not available and no CPU fallback for timing),
            # current_algo_min_time_ms will remain float('inf').

        logger.debug(
            f"Forward benchmark result: {algo_mode.value} {params_config} {current_algo_min_time_ms:.2f}ms"
        )
        if current_algo_min_time_ms != float("inf"):
            all_benchmark_results.append((algo_mode, params_config, current_algo_min_time_ms))

    if not all_benchmark_results:
        logger.warning(
            "Warning: No forward benchmark was successful or no algorithms to test. Defaulting to EXPLICIT_GEMM."
        )
        # Return a default entry indicating failure or no successful benchmarks
        with timer:
            _execute_single_fwd_pass(SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM, {})
        all_benchmark_results.append(
            (SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM, {}, timer.elapsed_time)
        )

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]

    logger.debug(
        f"Best forward algo: {best_algo.value} for log N_in={math.ceil(math.log2(in_features.shape[0])) if in_features.shape[0] > 0 else 0}, log N_out={math.ceil(math.log2(num_out_coords)) if num_out_coords > 0 else 0}, C_in={in_features.shape[1]}, C_out={weight.shape[2]}, K_vol={weight.shape[0]} {best_params} {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results  # Return the sorted list of all results


def _run_backward_benchmarks(
    grad_output: Float[Tensor, "M C_out"],
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    warmup_iters: int = _BENCHMARK_NUM_RUNS // 2,
    benchmark_iters: int = _BENCHMARK_NUM_RUNS,
    custom_params: Optional[List[Tuple[SPARSE_CONV_BWD_ALGO_MODE, Dict[str, Any]]]] = None,
) -> Tuple[SPARSE_CONV_BWD_ALGO_MODE, Dict[str, Any], float]:
    """
    Benchmark different backward algorithms and return the best one with its parameters and runtime.
    The best is determined by the minimum execution time over benchmark_iters.
    """
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[Tuple[SPARSE_CONV_BWD_ALGO_MODE, Dict[str, Any], float]] = []
    timer = CUDATimer()

    def _execute_single_bwd_pass(
        algo_mode: SPARSE_CONV_BWD_ALGO_MODE, params_config: Dict[str, Any]
    ) -> Optional[int]:
        status = None

        if algo_mode == SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM:
            _, _ = _explicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                compute_dtype,
                device,
            )
        elif algo_mode == SPARSE_CONV_BWD_ALGO_MODE.IMPLICIT_GEMM:
            _, _ = _implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                params_config.get("gemm_block_size", 16),
                params_config.get("split_k_threads_per_block", 128),
                params_config.get("split_k_factor", 4),
                compute_dtype,
            )
        elif algo_mode == SPARSE_CONV_BWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM:
            status, _ = _cutlass_implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                accumulator_type=params_config.get("accumulator_type", torch.float32),
                split_k_slices_ad=params_config.get("split_k_slices_ad", 1),
                split_k_slices_trab=params_config.get("split_k_slices_trab", 1),
                mma_tile_ad=params_config.get("mma_tile_ad", 0),
                mma_tile_trab=params_config.get("mma_tile_trab", 0),
                device=device,
            )
            if isinstance(status, int) and status != 0:
                return status
        else:
            raise ValueError(f"Unsupported algo_mode in _execute_single_bwd_pass: {algo_mode}")

    params_to_use = custom_params if custom_params is not None else _get_filtered_backward_params()
    for algo_mode, params_config in params_to_use:
        status = None
        for _ in range(warmup_iters):
            status = _execute_single_bwd_pass(algo_mode, params_config)
            if isinstance(status, int) and status != 0:
                # logger.warning(
                #     f"Error in _run_backward_benchmarks: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(status))}"
                # )
                continue

        if status is not None and status != 0:
            continue

        # Benchmark runs
        current_algo_min_time_ms = float("inf")  # Min time for this specific algorithm config

        if benchmark_iters == 0:
            if warmup_iters == 0:
                continue
        else:
            for _ in range(benchmark_iters):
                with timer:
                    _execute_single_bwd_pass(algo_mode, params_config)
                current_algo_min_time_ms = min(current_algo_min_time_ms, timer.elapsed_time)

        logger.debug(
            f"Backward benchmark result: {algo_mode.value} {params_config} {current_algo_min_time_ms:.2f}ms"
        )
        if current_algo_min_time_ms != float("inf"):
            all_benchmark_results.append((algo_mode, params_config, current_algo_min_time_ms))

    if not all_benchmark_results:
        logger.warning(
            "Warning: No backward benchmark was successful or no algorithms to test. Defaulting to EXPLICIT_GEMM."
        )
        with timer:
            _execute_single_bwd_pass(SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM, {})
        all_benchmark_results.append(
            (SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM, {}, timer.elapsed_time)
        )

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    logger.debug(
        f"Best backward algo: {best_algo.value} for log N_in={math.ceil(math.log2(in_features.shape[0])) if in_features.shape[0] > 0 else 0}, log N_out={math.ceil(math.log2(num_out_coords)) if num_out_coords > 0 else 0}, C_in={in_features.shape[1]}, C_out={weight.shape[2]}, K_vol={weight.shape[0]} {best_params} {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results  # Return the sorted list of all results


class UnifiedSpatiallySparseConvFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_features: Float[Tensor, "N C_in"],
        weight: Float[Tensor, "K C_in C_out"],
        kernel_map: IntSearchResult,
        num_out_coords: int,
        fwd_algo: Union[SPARSE_CONV_FWD_ALGO_MODE, List[SPARSE_CONV_FWD_ALGO_MODE]],
        bwd_algo: Union[SPARSE_CONV_BWD_ALGO_MODE, List[SPARSE_CONV_BWD_ALGO_MODE]],
        compute_dtype: Optional[torch.dtype],
        fwd_block_size: Optional[int],  # For implicit GEMM if not AUTO
        bwd_block_size: Optional[int],  # For implicit GEMM if not AUTO
    ) -> Float[Tensor, "M C_out"]:
        global _BENCHMARK_FORWARD_RESULTS  # noqa: F824
        output_feature_tensor = None

        # Convert string inputs to enum if needed
        if isinstance(fwd_algo, str):
            fwd_algo = SPARSE_CONV_FWD_ALGO_MODE(fwd_algo)
        if isinstance(bwd_algo, str):
            bwd_algo = SPARSE_CONV_BWD_ALGO_MODE(bwd_algo)

        if isinstance(fwd_algo, list):
            fwd_algo = [
                SPARSE_CONV_FWD_ALGO_MODE(algo) if isinstance(algo, str) else algo
                for algo in fwd_algo
            ]
        if isinstance(bwd_algo, list):
            bwd_algo = [
                SPARSE_CONV_BWD_ALGO_MODE(algo) if isinstance(algo, str) else algo
                for algo in bwd_algo
            ]

        # UNIFIED APPROACH: Always benchmark within filtered algorithm space
        # Step 1: Determine algorithm filter set
        if isinstance(fwd_algo, list):
            algorithm_filter = fwd_algo
        elif fwd_algo == SPARSE_CONV_FWD_ALGO_MODE.AUTO:
            algorithm_filter = "auto"  # All algorithms
        else:
            # Single algorithm - create list for consistent processing
            algorithm_filter = [fwd_algo]

        # Step 2: Generate configuration for caching
        config = SpatiallySparseConvConfig(
            num_in_coords=in_features.shape[0],
            num_out_coords=num_out_coords,
            in_channels=in_features.shape[1],
            out_channels=weight.shape[2],
            kernel_volume=weight.shape[0],
            in_dtype=in_features.dtype,
        )

        # Step 3: Check cache first
        cached_result = _BENCHMARK_FORWARD_RESULTS.get(config)
        if cached_result is not None:
            if algorithm_filter == "auto":
                # Use best overall result
                chosen_fwd_algo, chosen_fwd_params, _ = cached_result[0]
            else:
                # Filter cached results to only include algorithms in filter set
                filtered_cached_results = []
                for algo, params, time in cached_result:
                    if algo in algorithm_filter:
                        filtered_cached_results.append((algo, params, time))

                if filtered_cached_results:
                    # Use the best performing algorithm from the filtered results
                    chosen_fwd_algo, chosen_fwd_params, _ = filtered_cached_results[0]
                else:
                    raise ValueError(
                        f"No cached algorithms from {algorithm_filter} found. Please check your algorithm list and try again. If you need to reset the cache, please run `rm -rf ~/.cache/warpconvnet/`."
                    )
        else:
            # Step 4: No cache - always benchmark within filtered space
            if algorithm_filter == "auto":
                # Benchmark all algorithms
                filtered_params = _BENCHMARK_FORWARD_PARAMS
            else:
                # Filter benchmark parameters to only include algorithms in filter set
                filtered_params = _filter_benchmark_params_by_env_config(
                    _BENCHMARK_FORWARD_PARAMS, algorithm_filter, is_forward=True
                )

            # Always run benchmarks to find optimal parameters
            all_fwd_benchmark_results = _run_forward_benchmarks(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                custom_params=filtered_params,
            )
            _BENCHMARK_FORWARD_RESULTS[config] = all_fwd_benchmark_results
            chosen_fwd_algo, chosen_fwd_params, _ = all_fwd_benchmark_results[0]  # Best is first

            mark_benchmark_cache_dirty()

        # Step 5: Execute with optimal algorithm and parameters
        if chosen_fwd_algo == SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM:
            output_feature_tensor = _explicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
            )
        elif chosen_fwd_algo == SPARSE_CONV_FWD_ALGO_MODE.IMPLICIT_GEMM:
            current_fwd_block_size = chosen_fwd_params.get("fwd_block_size")
            if current_fwd_block_size is None:  # Fallback if somehow not set
                current_fwd_block_size = fwd_block_size if fwd_block_size is not None else 16
                logger.warning(
                    f"fwd_block_size not found in chosen_fwd_params for IMPLICIT_GEMM, using fallback {current_fwd_block_size}"
                )
            output_feature_tensor = _implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                current_fwd_block_size,
            )
        elif chosen_fwd_algo == SPARSE_CONV_FWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM:
            output_feature_tensor = _cutlass_implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                split_k_slices=chosen_fwd_params.get("split_k_slices", 1),
                accumulator_type=chosen_fwd_params.get("accumulator_type", torch.float32),
            )
            if isinstance(output_feature_tensor, int) and output_feature_tensor != 0:
                raise RuntimeError(
                    f"Error in _cutlass_implicit_gemm_forward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(output_feature_tensor))}"
                )
        else:
            raise ValueError(f"Unsupported forward algorithm: {chosen_fwd_algo}")

        ctx.save_for_backward(in_features, weight)
        ctx.kernel_map = kernel_map

        # For SpatiallySparseConvConfig in backward if bwd_algo is AUTO
        ctx.config_params_for_bwd = {
            "num_in_coords": in_features.shape[0],
            "num_out_coords": num_out_coords,
            "in_channels": in_features.shape[1],
            "out_channels": weight.shape[2],
            "kernel_volume": weight.shape[0],
            "implicit_matmul_fwd_block_size": chosen_fwd_params.get(
                "fwd_block_size", fwd_block_size
            ),  # from fwd decision
            "implicit_matmul_bwd_block_size": bwd_block_size,  # from user input for bwd
            "compute_dtype": compute_dtype,
            "device": in_features.device,
            "initial_bwd_algo": bwd_algo,
            "initial_bwd_block_size": bwd_block_size,
        }

        # Return structure for backward:
        # grads for: in_features, weight, kernel_map, num_out_coords, fwd_algo, bwd_algo, compute_dtype, fwd_block_size, bwd_block_size
        return output_feature_tensor

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Optional[Float[Tensor, "N C_in"]],
        Optional[Float[Tensor, "K C_in C_out"]],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        global _BENCHMARK_BACKWARD_RESULTS  # noqa: F824
        in_features, weight = ctx.saved_tensors
        kernel_map = ctx.kernel_map
        config_params = ctx.config_params_for_bwd
        num_out_coords = config_params["num_out_coords"]
        compute_dtype = config_params["compute_dtype"]
        device = config_params["device"]
        initial_bwd_algo = config_params["initial_bwd_algo"]
        initial_bwd_block_size = config_params["initial_bwd_block_size"]
        # explicit_matmul_batch_size = ctx.explicit_matmul_batch_size # TODO

        # Convert string to enum if needed
        if isinstance(initial_bwd_algo, str):
            initial_bwd_algo = SPARSE_CONV_BWD_ALGO_MODE(initial_bwd_algo)

        if isinstance(initial_bwd_algo, list):
            initial_bwd_algo = [
                SPARSE_CONV_BWD_ALGO_MODE(algo) if isinstance(algo, str) else algo
                for algo in initial_bwd_algo
            ]

        grad_in_features, grad_weight = None, None

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 9)

        N_in, C_in = in_features.shape
        K, _, C_out = weight.shape
        if (
            num_out_coords == 0
            or K == 0
            or C_in == 0
            or C_out == 0
            or N_in == 0
            or grad_output.shape[0] == 0
        ):
            grad_in_final = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_weight_final = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in_final, grad_weight_final, 9)

        # UNIFIED APPROACH: Always benchmark within filtered algorithm space
        # Step 1: Determine algorithm filter set
        if isinstance(initial_bwd_algo, list):
            algorithm_filter = initial_bwd_algo
        elif initial_bwd_algo == SPARSE_CONV_BWD_ALGO_MODE.AUTO:
            algorithm_filter = "auto"  # All algorithms
        else:
            # Single algorithm - create list for consistent processing
            algorithm_filter = [initial_bwd_algo]

        # Step 2: Generate configuration for caching
        config_params = ctx.config_params_for_bwd
        config = SpatiallySparseConvConfig(
            num_in_coords=config_params["num_in_coords"],
            num_out_coords=config_params["num_out_coords"],
            in_channels=config_params["in_channels"],
            out_channels=config_params["out_channels"],
            kernel_volume=config_params["kernel_volume"],
            in_dtype=grad_output.dtype,
        )

        # Step 3: Check cache first
        cached_result = _BENCHMARK_BACKWARD_RESULTS.get(config)
        if cached_result is not None:
            if algorithm_filter == "auto":
                # Use best overall result
                chosen_bwd_algo, chosen_bwd_params, _ = cached_result[0]
            else:
                # Filter cached results to only include algorithms in filter set
                filtered_cached_results = []
                for algo, params, time in cached_result:
                    if algo in algorithm_filter:
                        filtered_cached_results.append((algo, params, time))

                if filtered_cached_results:
                    # Use the best performing algorithm from the filtered results
                    chosen_bwd_algo, chosen_bwd_params, _ = filtered_cached_results[
                        0
                    ]  # Best is first
                else:
                    # Fallback to best cached result if none from filter set
                    logger.warning(
                        f"No cached algorithms from {algorithm_filter} found, using best cached"
                    )
                    chosen_bwd_algo, chosen_bwd_params, _ = cached_result[0]
        else:
            # Step 4: No cache - always benchmark within filtered space
            if algorithm_filter == "auto":
                # Benchmark all algorithms
                filtered_params = _BENCHMARK_BACKWARD_PARAMS
            else:
                # Filter benchmark parameters to only include algorithms in filter set
                filtered_params = _filter_benchmark_params_by_env_config(
                    _BENCHMARK_BACKWARD_PARAMS, algorithm_filter, is_forward=False
                )

            # Always run benchmarks to find optimal parameters
            all_bwd_benchmark_results = _run_backward_benchmarks(
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                device,
                custom_params=filtered_params,
            )
            _BENCHMARK_BACKWARD_RESULTS[config] = all_bwd_benchmark_results
            chosen_bwd_algo, chosen_bwd_params, _ = all_bwd_benchmark_results[0]  # Best is first

            mark_benchmark_cache_dirty()

        if chosen_bwd_algo == SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM:
            grad_in_features, grad_weight = _explicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                compute_dtype,
                device,
            )
        elif chosen_bwd_algo == SPARSE_CONV_BWD_ALGO_MODE.IMPLICIT_GEMM:
            grad_in_features, grad_weight = _implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                gemm_block_size=chosen_bwd_params.get("bwd_block_size", 16),
                split_k_threads_per_block=chosen_bwd_params.get("split_k_threads_per_block", 256),
                split_k_factor=chosen_bwd_params.get("split_k_factor", 4),
                compute_dtype=compute_dtype,
            )
        elif chosen_bwd_algo == SPARSE_CONV_BWD_ALGO_MODE.CUTLASS_IMPLICIT_GEMM:
            grad_in_features, grad_weight = _cutlass_implicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
                split_k_slices_ad=chosen_bwd_params.get("split_k_slices_ad", 1),
                split_k_slices_trab=chosen_bwd_params.get("split_k_slices_trab", 1),
                mma_tile_ad=chosen_bwd_params.get("mma_tile_ad", 0),
                mma_tile_trab=chosen_bwd_params.get("mma_tile_trab", 0),
                accumulator_type=chosen_bwd_params.get("accumulator_type", torch.float32),
                device=device,
            )
            if isinstance(grad_in_features, int) and grad_in_features != 0:
                raise RuntimeError(
                    f"Error in _cutlass_implicit_gemm_backward_logic: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(grad_in_features))}"
                )

        # elif chosen_bwd_algo == SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM_BATCHED: # TODO
        # if explicit_matmul_batch_size is None:
        #     raise ValueError("explicit_matmul_batch_size is required for batched explicit GEMM backward.")
        # grad_in_features, grad_weight = _batched_explicit_gemm_backward_logic(...)
        else:
            raise ValueError(f"Unsupported backward algorithm: {chosen_bwd_algo}")

        if not ctx.needs_input_grad[0]:
            grad_in_features = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None

        return _pad_tuple(grad_in_features, grad_weight, 9)


def spatially_sparse_conv(
    input_sparse_tensor: Geometry,
    weight: Float[Tensor, "K C_in C_out"],
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    stride: Union[int, List[int], Tuple[int, ...]] = 1,
    kernel_dilation: Union[int, List[int], Tuple[int, ...]] = 1,
    bias: Optional[Float[Tensor, "C_out"]] = None,  # noqa: F821
    kernel_matmul_batch_size: int = 2,
    generative: bool = False,
    output_spatially_sparse_tensor: Optional[Geometry] = None,
    transposed: bool = False,
    fwd_algo: Union[
        SPARSE_CONV_FWD_ALGO_MODE, List[SPARSE_CONV_FWD_ALGO_MODE]
    ] = SPARSE_CONV_FWD_ALGO_MODE.EXPLICIT_GEMM,
    bwd_algo: Union[
        SPARSE_CONV_BWD_ALGO_MODE, List[SPARSE_CONV_BWD_ALGO_MODE]
    ] = SPARSE_CONV_BWD_ALGO_MODE.EXPLICIT_GEMM,
    stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
    stride_reduce: str = "max",
    order: POINT_ORDERING = POINT_ORDERING.RANDOM,
    compute_dtype: Optional[torch.dtype] = None,  # Use None to default to in_features.dtype
    implicit_matmul_fwd_block_size: Optional[int] = 16,  # Default, can be None if not implicit
    implicit_matmul_bwd_block_size: Optional[int] = 16,  # Default, can be None if not implicit
) -> Geometry:  # Should return Voxels or a base Geometry type compatible with Voxels
    """
    Perform spatially sparse convolution on the input tensor
    Spatially sparse and feature sparse is not supported yet.

    If stride is not 1, the kernel map will be generated by stride_mode.

    If generative, the output coordinates will be expanded by (kernel size // 2) all directions.

    For transposed convolution, the output coordinates should be provided along with the
    output coordinate stride.
    """
    if not isinstance(input_sparse_tensor, Voxels):
        raise TypeError(
            f"Native spatially_sparse_conv expects input_sparse_tensor of type Voxels, got {type(input_sparse_tensor)}"
        )

    if output_spatially_sparse_tensor is not None and not isinstance(
        output_spatially_sparse_tensor, Voxels
    ):
        raise TypeError(
            f"Native spatially_sparse_conv expects output_spatially_sparse_tensor of type Voxels or None, got {type(output_spatially_sparse_tensor)}"
        )

    num_spatial_dims = input_sparse_tensor.num_spatial_dims
    _kernel_size = ntuple(kernel_size, ndim=num_spatial_dims)
    _kernel_dilation = ntuple(kernel_dilation, ndim=num_spatial_dims)
    _stride = ntuple(stride, ndim=num_spatial_dims)

    num_total_kernels = np.prod(_kernel_size)
    if np.prod(_kernel_size) == 1 and np.prod(_stride) == 1:
        out_feature_tensor = input_sparse_tensor.feature_tensor @ weight[0]
        if bias is not None:
            out_feature_tensor += bias
        return input_sparse_tensor.replace(
            batched_features=out_feature_tensor,
        )

    in_tensor_stride = input_sparse_tensor.tensor_stride
    if in_tensor_stride is None:
        in_tensor_stride = ntuple(1, ndim=num_spatial_dims)

    if transposed and not generative:
        assert (
            output_spatially_sparse_tensor is not None
        ), "Output spatially sparse tensor is required for transposed convolution"

    out_tensor_stride: Tuple[int, ...]
    if not transposed:
        out_tensor_stride = tuple(o * s for o, s in zip(_stride, in_tensor_stride))
    else:  # transposed
        if (
            output_spatially_sparse_tensor is not None
            and output_spatially_sparse_tensor.tensor_stride is not None
        ):
            out_tensor_stride = output_spatially_sparse_tensor.tensor_stride
        else:
            out_tensor_stride = ntuple(1, ndim=num_spatial_dims)
        # At least one of the output stride dimensions should be smaller than the input stride dimensions
        assert any(
            o < i for o, i in zip(out_tensor_stride, in_tensor_stride)
        ), "Output stride is larger than input stride"

    # Determine effective compute_dtype
    effective_compute_dtype = (
        compute_dtype if compute_dtype is not None else input_sparse_tensor.feature_tensor.dtype
    )

    if stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and any(s != 1 for s in _stride):
        reduced_input_voxels = sparse_reduce(
            input_sparse_tensor,
            kernel_size=_stride,
            stride=_stride,
            reduction=stride_reduce,
        )
        current_input_features_for_gemm = reduced_input_voxels.feature_tensor
        # The `kernel_map` indices (in_map) should refer to indices within `reduced_input_voxels`.
        # `generate_kernel_map` (called by `generate_output_coords_and_kernel_map`) must ensure this mapping is correct
        # when `stride_mode` is `REDUCE_AND_STRIDE`.
        input_sparse_tensor = reduced_input_voxels
        _stride = ntuple(1, ndim=num_spatial_dims)
    else:
        current_input_features_for_gemm = input_sparse_tensor.feature_tensor

    batch_indexed_out_coords, out_offsets, kernel_map = generate_output_coords_and_kernel_map(
        input_sparse_tensor=input_sparse_tensor,
        kernel_size=_kernel_size,
        kernel_dilation=_kernel_dilation,
        stride=_stride,
        generative=generative,
        transposed=transposed,
        output_spatially_sparse_tensor=output_spatially_sparse_tensor,
        stride_mode=stride_mode,
        order=order,
    )
    num_out_coords = batch_indexed_out_coords.shape[0]

    out_feature_tensor = UnifiedSpatiallySparseConvFunction.apply(
        current_input_features_for_gemm,
        weight,
        kernel_map,
        num_out_coords,
        fwd_algo,
        bwd_algo,
        effective_compute_dtype,
        implicit_matmul_fwd_block_size,
        implicit_matmul_bwd_block_size,
    )

    if bias is not None:
        out_feature_tensor += bias

    out_offsets_cpu = out_offsets.cpu().int()
    return input_sparse_tensor.replace(
        batched_coordinates=IntCoords(
            batch_indexed_out_coords[:, 1:],
            offsets=out_offsets_cpu,
        ),
        batched_features=out_feature_tensor,
        tensor_stride=out_tensor_stride,
    )


def generate_output_coords_and_kernel_map(
    input_sparse_tensor: Voxels,  # Ensure this is Voxels
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    stride: Tuple[int, ...],
    generative: bool,
    transposed: bool,
    output_spatially_sparse_tensor: Optional[Voxels],
    stride_mode: STRIDED_CONV_MODE = STRIDED_CONV_MODE.STRIDE_ONLY,
    order: POINT_ORDERING = POINT_ORDERING.RANDOM,
) -> Tuple[Tensor, Int[Tensor, "B+1"], IntSearchResult]:
    """
    Perform spatially sparse convolution on the input tensor
    Spatially sparse and feature sparse is not supported yet.

    If stride is not 1, the kernel map will be generated by stride_mode.

    If generative, the output coordinates will be expanded by (kernel size // 2) all directions.

    For transposed convolution, the output coordinates should be provided along with the
    output coordinate stride.

    Args:
        skip_symmetric_kernel_map: If True, skip the symmetric parts of the kernel map
            for odd-sized kernels (e.g., for 3x3x3 kernels, only use half of the kernel positions).
    """
    batch_indexed_in_coords = input_sparse_tensor.batch_indexed_coordinates
    in_to_out_stride_ratio = stride

    if input_sparse_tensor.coordinates.dtype not in (torch.int32, torch.int64):
        assert (
            input_sparse_tensor.voxel_size is not None
        ), "Voxel size is required for non-integer coordinates"
        # TODO(cchoy): Implement a voxel size aware coordinate mapping

    # Out coords and offsets generation
    if output_spatially_sparse_tensor is not None:
        assert (
            not generative
        ), "Output spatially sparse tensor is not supported with generative convolution"
        batch_indexed_out_coords = output_spatially_sparse_tensor.batch_indexed_coordinates
        out_offsets = output_spatially_sparse_tensor.offsets
    elif generative and all(s == 1 for s in stride):
        assert not transposed, "Transposed and generative convolution is not supported yet"
        batch_indexed_out_coords, out_offsets = expand_coords(
            batch_indexed_in_coords,
            kernel_size=kernel_size,
            kernel_dilation=kernel_dilation,
        )
    elif any(s != 1 for s in stride):
        batch_indexed_out_coords, out_offsets = stride_coords(
            batch_indexed_in_coords,
            stride,
        )
        # if generative, we need to expand the coordinates in addition
        if generative and stride_mode == STRIDED_CONV_MODE.STRIDE_ONLY:
            batch_indexed_out_coords, out_offsets = expand_coords(
                batch_indexed_out_coords,
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
            )
        elif generative and stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE:
            batch_indexed_expanded_coords, expanded_offsets = expand_coords(
                batch_indexed_out_coords,
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
            )
            # rename
            batch_indexed_in_coords = batch_indexed_out_coords
            batch_indexed_out_coords = batch_indexed_expanded_coords
            out_offsets = expanded_offsets
    elif all(s == 1 for s in stride):
        batch_indexed_out_coords, out_offsets = (
            batch_indexed_in_coords,
            input_sparse_tensor.offsets,
        )
    else:
        raise ValueError(
            f"Unsupported case. stride_mode: {stride_mode}, generative: {generative}, transposed: {transposed}"
        )

    # Order the output coordinates
    if order != POINT_ORDERING.RANDOM:
        code_result = encode(
            grid_coord=batch_indexed_out_coords[:, 1:],
            batch_offsets=out_offsets,
            order=order,
            return_perm=True,
        )
        batch_indexed_out_coords = batch_indexed_out_coords[code_result.perm]

    # if input_sparse_tensor.cache is not None, check the cache first
    kernel_map_cache_key = IntSearchCacheKey(
        kernel_size=kernel_size,
        kernel_dilation=kernel_dilation,
        transposed=transposed,
        generative=generative,
        stride_mode=str(stride_mode),
        skip_symmetric_kernel_map=False,
        in_offsets=input_sparse_tensor.offsets,
        out_offsets=out_offsets,
    )
    if input_sparse_tensor.cache is not None:
        kernel_map = input_sparse_tensor.cache.get(kernel_map_cache_key)
        if kernel_map is not None:
            return batch_indexed_out_coords, out_offsets, kernel_map

    # Kernel map generation
    if transposed and not generative:
        if input_sparse_tensor.cache is not None:
            # Check if the kernel map for non transposed case exists
            kernel_map_cache_key_non_transposed = IntSearchCacheKey(
                kernel_size=kernel_size,
                kernel_dilation=kernel_dilation,
                transposed=False,
                generative=generative,
                stride_mode=str(stride_mode),
                skip_symmetric_kernel_map=False,
                in_offsets=out_offsets,
                out_offsets=input_sparse_tensor.offsets,
            )
            kernel_map_non_transposed = input_sparse_tensor.cache.get(
                kernel_map_cache_key_non_transposed
            )
            if kernel_map_non_transposed is not None:
                # Swap in and out maps for transposed kernel map generation and swap it back
                kernel_map = IntSearchResult(
                    in_maps=kernel_map_non_transposed.out_maps,
                    out_maps=kernel_map_non_transposed.in_maps,
                    offsets=kernel_map_non_transposed.offsets,
                )
                return batch_indexed_out_coords, out_offsets, kernel_map
            else:
                logger.warning(
                    "No kernel map found for non-transposed case. Generating new kernel map."
                )

        # Swap in and out maps for transposed kernel map generation and swap it back
        kernel_map = generate_kernel_map(
            batch_indexed_out_coords,
            batch_indexed_in_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            skip_symmetric_kernel_map=False,
        )
        kernel_map = IntSearchResult(
            in_maps=kernel_map.out_maps,
            out_maps=kernel_map.in_maps,
            offsets=kernel_map.offsets,
        )
    elif stride_mode == STRIDED_CONV_MODE.STRIDE_ONLY:
        kernel_map = generate_kernel_map(
            batch_indexed_in_coords,
            batch_indexed_out_coords,
            in_to_out_stride_ratio,
            kernel_size,
            kernel_dilation,
            skip_symmetric_kernel_map=False,
        )
    elif stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and not generative:
        # Compute mapping from output to output since it will be reduced
        kernel_map = generate_kernel_map(
            batch_indexed_out_coords,
            batch_indexed_out_coords,
            ntuple(1, ndim=input_sparse_tensor.num_spatial_dims),
            kernel_size,
            kernel_dilation,
            skip_symmetric_kernel_map=False,
        )
    elif stride_mode == STRIDED_CONV_MODE.REDUCE_AND_STRIDE and generative:
        kernel_map = generate_kernel_map(
            batch_indexed_in_coords,
            batch_indexed_out_coords,
            ntuple(1, ndim=input_sparse_tensor.num_spatial_dims),
            kernel_size,
            kernel_dilation,
            skip_symmetric_kernel_map=False,
        )
    else:
        raise ValueError(
            f"Unsupported case. stride_mode: {stride_mode}, generative: {generative}, transposed: {transposed}"
        )

    if input_sparse_tensor.cache is None:
        input_sparse_tensor._extra_attributes["_cache"] = IntSearchCache()

    input_sparse_tensor.cache.put(kernel_map_cache_key, kernel_map)
    return batch_indexed_out_coords, out_offsets, kernel_map


def _skip_symmetric_kernel_parts(
    kernel_map: IntSearchResult, kernel_size: Tuple[int, ...]
) -> IntSearchResult:
    """
    Skip symmetric parts of the kernel map for odd-sized kernels.
    For example, for a 3x3x3 kernel, only use the first half of the kernel positions.
    """
    # Check if kernel is odd and potentially symmetric
    is_odd = all(k % 2 == 1 for k in kernel_size)

    if not is_odd:
        # If not odd, return the original kernel map
        return kernel_map

    kv = int(np.prod(kernel_size))
    if kv <= 1:
        return kernel_map

    # For symmetric kernels, keep only the first half (excluding center if odd volume)
    center_idx = kv // 2

    # Update offsets to match the reduced number of kernels
    new_offsets = kernel_map.offsets[: center_idx + 1].clone()

    # Create a new kernel map with only the first half of kernels
    tot_num_kernels = new_offsets[-1].item()
    in_maps_filtered = kernel_map.in_maps[:tot_num_kernels]
    out_maps_filtered = kernel_map.out_maps[:tot_num_kernels]

    return IntSearchResult(
        in_maps=in_maps_filtered,
        out_maps=out_maps_filtered,
        offsets=new_offsets,
        identity_map_index=center_idx,
    )
