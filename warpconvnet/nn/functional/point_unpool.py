# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.coords.search.knn import batched_knn_search
from warpconvnet.utils.unique import ToUnique


class FEATURE_UNPOOLING_MODE(Enum):
    REPEAT = "repeat"


def _unpool_features(
    pooled_pc: "Points",  # noqa: F821
    unpooled_pc: "Points",  # noqa: F821
    to_unique: Optional[ToUnique] = None,
    unpooling_mode: Optional[
        Union[str, FEATURE_UNPOOLING_MODE]
    ] = FEATURE_UNPOOLING_MODE.REPEAT,
) -> Float[Tensor, "M C"]:
    """Compute unpooled features for ``unpooled_pc`` from ``pooled_pc``."""
    if isinstance(unpooling_mode, str):
        unpooling_mode = FEATURE_UNPOOLING_MODE(unpooling_mode)

    if unpooling_mode == FEATURE_UNPOOLING_MODE.REPEAT and to_unique is not None:
        return to_unique.to_original(pooled_pc.features)
    elif unpooling_mode == FEATURE_UNPOOLING_MODE.REPEAT and to_unique is None:
        unpooled2pooled_idx = batched_knn_search(
            ref_positions=pooled_pc.coordinate_tensor,
            ref_offsets=pooled_pc.offsets,
            query_positions=unpooled_pc.coordinate_tensor,
            query_offsets=unpooled_pc.offsets,
            k=1,
        ).squeeze(-1)
        return pooled_pc.features[unpooled2pooled_idx]

    raise NotImplementedError(f"Unpooling mode {unpooling_mode} not implemented")


def point_unpool(
    pooled_pc: "BatchedSpatialFeatures",  # noqa: F821
    unpooled_pc: "Points",  # noqa: F821
    concat_unpooled_pc: bool,
    unpooling_mode: Optional[
        Union[str, FEATURE_UNPOOLING_MODE]
    ] = FEATURE_UNPOOLING_MODE.REPEAT,
    to_unique: Optional[ToUnique] = None,
) -> "Points":  # noqa: F821
    """Unpool features to a denser set of points."""
    unpooled_features = _unpool_features(
        pooled_pc=pooled_pc,
        unpooled_pc=unpooled_pc,
        to_unique=to_unique,
        unpooling_mode=unpooling_mode,
    )
    if concat_unpooled_pc:
        unpooled_features = torch.cat(
            (unpooled_features, unpooled_pc.feature_tensor), dim=-1
        )
    return unpooled_pc.replace(batched_features=unpooled_features)
