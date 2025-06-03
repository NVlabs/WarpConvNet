# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Optional, Tuple, Union, Literal

import torch

from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.coords.search.cache import IntSearchCache, IntSearchCacheKey
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.geometry.coords.search.torch_discrete import (
    generate_kernel_map,
)
from warpconvnet.geometry.types.voxels import (
    Voxels,
)
from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.ops.reductions import REDUCTIONS, row_reduction
from warpconvnet.utils.ntuple import ntuple
from warpconvnet.utils.unique import unique_inverse


def sparse_reduce(
    spatially_sparse_tensor: Voxels,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Optional[Union[int, Tuple[int, ...]]] = None,
    reduction: Union[REDUCTIONS, str] = REDUCTIONS.MAX,
    out_code_backend: Literal["hashmap", "ravel", "unique", "morton"] = "hashmap",
) -> Voxels:
    """
    Max pooling for spatially sparse tensors.
    """
    if isinstance(reduction, str):
        reduction = REDUCTIONS(reduction)

    if stride is None:
        stride = kernel_size
    ndim = spatially_sparse_tensor.num_spatial_dims
    stride = ntuple(stride, ndim=ndim)
    kernel_size = ntuple(kernel_size, ndim=ndim)

    in_tensor_stride = spatially_sparse_tensor.stride
    if in_tensor_stride is None:
        in_tensor_stride = ntuple(1, ndim=ndim)
    out_tensor_stride = tuple(o * s for o, s in zip(stride, in_tensor_stride))

    batch_indexed_in_coords = spatially_sparse_tensor.batch_indexed_coordinates
    batch_indexed_out_coords, output_offsets = stride_coords(
        batch_indexed_in_coords, stride, backend=out_code_backend
    )
    from warpconvnet.nn.functional.sparse_conv import STRIDED_CONV_MODE

    kernel_map_cache_key = IntSearchCacheKey(
        kernel_size=kernel_size,
        kernel_dilation=ntuple(1, ndim=ndim),
        transposed=False,
        generative=False,
        stride_mode=str(STRIDED_CONV_MODE.STRIDE_ONLY),
        skip_symmetric_kernel_map=False,
        in_offsets=spatially_sparse_tensor.offsets,
        out_offsets=output_offsets,
    )
    kernel_map = None
    if spatially_sparse_tensor.cache is not None:
        kernel_map = spatially_sparse_tensor.cache.get(kernel_map_cache_key)

    if kernel_map is None:
        # Find mapping from in to out
        kernel_map: IntSearchResult = generate_kernel_map(
            batch_indexed_in_coords,
            batch_indexed_out_coords,
            in_to_out_stride_ratio=stride,
            kernel_size=kernel_size,
            kernel_dilation=ntuple(1, ndim=ndim),
            skip_symmetric_kernel_map=False,
        )

    if spatially_sparse_tensor.cache is None:
        spatially_sparse_tensor._extra_attributes["_cache"] = IntSearchCache()
    spatially_sparse_tensor.cache.put(kernel_map_cache_key, kernel_map)

    in_maps, unique_out_maps, map_offsets = kernel_map.to_csr()
    in_features = spatially_sparse_tensor.feature_tensor
    device = in_features.device

    out_features = row_reduction(in_features[in_maps], map_offsets.to(device), reduction)

    if len(unique_out_maps) != batch_indexed_out_coords.shape[0]:
        warnings.warn(
            f"Some output coordinates don't have any input maps. {batch_indexed_out_coords.shape[0] - len(unique_out_maps)} output coordinates are missing.",
            stacklevel=2,
        )

        # cchoy: This is a rare case where some output coordinates don't have any input maps.
        # We need to zero out the features for those coordinates.
        new_out_features = torch.zeros(
            batch_indexed_out_coords.shape[0],
            in_features.shape[1],
            device=spatially_sparse_tensor.device,
        )
        new_out_features[unique_out_maps] = out_features
        out_features = new_out_features

    output_offsets = output_offsets.cpu()
    return spatially_sparse_tensor.replace(
        batched_coordinates=IntCoords(
            batch_indexed_out_coords[:, 1:],
            output_offsets,
        ),
        batched_features=out_features,
        stride=out_tensor_stride,
    )


def sparse_max_pool(
    spatially_sparse_tensor: Voxels,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Voxels:
    """
    Max pooling for spatially sparse tensors.
    """
    return sparse_reduce(spatially_sparse_tensor, kernel_size, stride, reduction=REDUCTIONS.MAX)


def sparse_avg_pool(
    spatially_sparse_tensor: Voxels,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Voxels:
    """
    Average pooling for spatially sparse tensors.
    """
    return sparse_reduce(spatially_sparse_tensor, kernel_size, stride, reduction=REDUCTIONS.MEAN)


def sparse_unpool(
    pooled_st: Voxels,
    unpooled_st: Voxels,
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]],
    concat_unpooled_st: bool = False,
) -> Voxels:
    """
    Unpooling for spatially sparse tensors.
    """
    ndim = pooled_st.num_spatial_dims
    stride = ntuple(stride, ndim=ndim)
    kernel_size = ntuple(kernel_size, ndim=ndim)

    # use the cache for the transposed case to get the kernel map
    from warpconvnet.nn.functional.sparse_conv import STRIDED_CONV_MODE

    kernel_map_cache_key = IntSearchCacheKey(
        kernel_size=kernel_size,
        kernel_dilation=ntuple(1, ndim=ndim),
        transposed=False,
        generative=False,
        stride_mode=str(STRIDED_CONV_MODE.STRIDE_ONLY),
        skip_symmetric_kernel_map=False,
        in_offsets=unpooled_st.offsets,
        out_offsets=pooled_st.offsets,
    )
    assert pooled_st.cache is not None
    kernel_map = pooled_st.cache.get(kernel_map_cache_key)
    assert kernel_map is not None

    # Switch
    unpooled_maps = kernel_map.in_maps
    pooled_maps = kernel_map.out_maps

    perm = torch.argsort(unpooled_maps)
    rep_feats = pooled_st.feature_tensor[pooled_maps[perm]]
    if concat_unpooled_st:
        rep_feats = torch.cat([unpooled_st.feature_tensor, rep_feats], dim=-1)

    return unpooled_st.replace(batched_features=rep_feats)
