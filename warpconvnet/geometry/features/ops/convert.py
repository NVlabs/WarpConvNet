# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, TYPE_CHECKING
from jaxtyping import Float, Int

import torch
from torch import Tensor

from ..batch_copy import copy_batch_torch

if TYPE_CHECKING:
    from ..cat import CatFeatures
    from ..pad import PadFeatures
    from ..grid import GridFeatures


def cat_to_pad_tensor(
    in_features: Float[Tensor, "N F"],  # noqa: F821
    row_splits: Int[Tensor, "B+1"],  # noqa: F821
    pad_multiple: Optional[int] = None,
) -> Float[Tensor, "B M F"]:
    """
    Convert a concatenated 2D tensor to a batched 3D tensor.
    """
    assert in_features.ndim == 2
    return copy_batch_torch(in_features, row_splits, pad_multiple)


def cat_to_pad(
    cat_features: "CatFeatures", pad_multiple: Optional[int] = None
) -> "PadFeatures":
    """Convert concatenated features to padded format."""
    from ..pad import PadFeatures
    from ..cat import CatFeatures

    assert isinstance(
        cat_features, CatFeatures
    ), f"cat_features must be a CatFeatures, got {type(cat_features)}"

    batched_tensor = cat_to_pad_tensor(
        cat_features.batched_tensor,
        cat_features.offsets,
        pad_multiple=pad_multiple,
    )
    return PadFeatures(batched_tensor, cat_features.offsets, pad_multiple)


def pad_to_cat_tensor(
    in_features: Float[Tensor, "B M F"],
    row_splits: Int[Tensor, "B+1"],  # noqa: F821
) -> Float[Tensor, "N F"]:
    """
    Convert a batched 3D tensor to a concatenated 2D tensor.
    """
    assert in_features.ndim == 3
    tot_num = row_splits[-1]
    num_points = row_splits.diff()
    out_features = torch.zeros(
        (tot_num, in_features.shape[2]),
        dtype=in_features.dtype,
        device=in_features.device,
    )
    for batch_idx, num_points_in_batch in enumerate(num_points):
        out_features[row_splits[batch_idx] : row_splits[batch_idx + 1]] = in_features[
            batch_idx
        ][:num_points_in_batch]
    return out_features


def pad_to_cat(pad_features: "PadFeatures") -> "CatFeatures":
    """Convert padded features to concatenated format."""
    from ..cat import CatFeatures

    batched_tensor = pad_to_cat_tensor(
        pad_features.batched_tensor, pad_features.offsets
    )
    return CatFeatures(batched_tensor, pad_features.offsets)


def to_batched_features(
    features: Union["CatFeatures", "PadFeatures", "GridFeatures", Tensor],
    offsets: Int[Tensor, "B+1"],  # noqa: F821
    device: Optional[str] = None,
) -> Union["CatFeatures", "PadFeatures", "GridFeatures"]:
    from ..cat import CatFeatures
    from ..pad import PadFeatures
    from ..grid import GridFeatures

    if isinstance(features, Tensor):
        if features.ndim == 2:
            return CatFeatures(features, offsets, device=device)
        elif features.ndim == 3:
            return PadFeatures(features, offsets, device=device)
        else:
            raise ValueError(f"Invalid features tensor shape {features.shape}")
    else:
        valid_types = [CatFeatures, PadFeatures, GridFeatures]

        if any(isinstance(features, t) for t in valid_types):
            if device is not None:
                features = features.to(device)
            return features
        else:
            valid_types_str = ", ".join([t.__name__ for t in valid_types])
            raise TypeError(
                f"Features must be a tensor or one of {valid_types_str}, "
                f"got {type(features)}"
            )
