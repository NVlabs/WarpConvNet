# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Union
from jaxtyping import Float

import torch
from torch import Tensor
from torch_scatter import segment_csr

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels


def _global_pool(
    x: Geometry,
    reduce: Literal["max", "mean", "sum"],
) -> Tensor:
    """
    Global pooling that generates a single feature per batch.
    The coordinates of the output are the simply the 0 vector.
    """
    features = x.feature_tensor
    input_offsets = x.offsets.long().to(features.device)
    output_features = segment_csr(src=features, indptr=input_offsets, reduce=reduce)
    return output_features


def global_pool(x: Geometry, reduce: Literal["max", "mean", "sum"]) -> Geometry:
    """Pool over all coordinates and return a single feature per batch.

    Args:
        x: Input geometry instance to pool.
        reduce: Reduction type used to combine features.

    Returns:
        Geometry object with a single coordinate and feature per batch.
    """
    B = x.batch_size
    num_spatial_dims = x.num_spatial_dims
    # Generate output coordinates
    output_coords = torch.zeros(B, num_spatial_dims, dtype=torch.int32, device=x.device)
    output_offsets = torch.arange(B + 1, dtype=torch.int32)

    # Generate output features
    output_features = _global_pool(x, reduce)

    return x.replace(
        batched_coordinates=x.batched_coordinates.__class__(
            output_coords, output_offsets
        ),
        batched_features=x.batched_features.__class__(output_features, output_offsets),
        offsets=output_offsets,
    )


def global_scale(x: Geometry, scale: Float[Tensor, "B C"]) -> Geometry:
    """
    Global scaling that generates a single feature per batch.
    The coordinates of the output are the simply the 0 vector.
    """
    offsets = x.offsets
    diff = offsets.diff()
    B = diff.shape[0]
    # assert that the scale has the same batch size
    assert scale.shape[0] == B, "Scale must have the same batch size as the input"
    # repeat scale for each batch
    scaled_features = x.feature_tensor * torch.repeat_interleave(
        scale, diff.to(scale.device), dim=0
    )
    return x.replace(
        batched_features=scaled_features,
    )
