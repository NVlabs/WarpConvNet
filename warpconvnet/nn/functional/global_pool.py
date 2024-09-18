from typing import Literal, Optional, Union

import torch
import torch.nn as nn
from torch_scatter import segment_csr

from warpconvnet.geometry.point_collection import PointCollection
from warpconvnet.geometry.spatially_sparse_tensor import SpatiallySparseTensor


def global_pool(
    x: Union[PointCollection, SpatiallySparseTensor],
    pool_type: Literal["max", "mean", "sum"],
) -> Union[PointCollection, SpatiallySparseTensor]:
    """
    Global pooling that generates a single feature per batch.
    The coordinates of the output are the simply the 0 vector.
    """
    B = x.batch_size
    num_spatial_dims = x.num_spatial_dims
    # Generate output coordinates
    output_coords = torch.zeros(B, num_spatial_dims, dtype=torch.int32, device=x.device)
    output_offsets = torch.arange(B + 1, dtype=torch.int32)  # [0, 1, 2, ..., B]
    features = x.features
    input_offsets = x.offsets.long().to(features.device)
    output_features = segment_csr(src=features, indptr=input_offsets, reduce=pool_type)
    return x.replace(
        batched_coordinates=output_coords,
        batched_features=output_features,
        offsets=output_offsets,
    )
