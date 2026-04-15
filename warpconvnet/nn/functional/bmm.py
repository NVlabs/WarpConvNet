# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from jaxtyping import Float
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.ops.convert import (
    cat_to_pad_tensor,
    pad_to_cat_tensor,
)
from warpconvnet.geometry.features.pad import PadFeatures


def bmm(
    sf: Geometry,
    weights: Float[Tensor, "B C_in C_out"],
) -> Geometry:
    """
    Batch matrix multiplication.
    """
    assert sf.batch_size == weights.shape[0]
    if isinstance(sf.batched_features, CatFeatures):
        bat_features = cat_to_pad_tensor(sf.feature_tensor, sf.offsets)  # BxNxC_in
        out_bat_features = torch.bmm(bat_features, weights)
        out_features = pad_to_cat_tensor(out_bat_features, sf.offsets)
        out_features = CatFeatures(out_features, sf.offsets)
    elif isinstance(sf.batched_features, PadFeatures):
        bat_features = sf.feature_tensor  # BxMxC_in
        out_bat_features = torch.bmm(bat_features, weights)  # BxMxC_out
        out_features = PadFeatures(out_bat_features, sf.offsets)
    else:
        raise ValueError(f"Unsupported batched features type: {type(sf.batched_features)}")
    return sf.replace(
        batched_features=out_features,
    )
