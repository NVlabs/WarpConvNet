# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule

__all__ = [
    "Transform",
    "Cat",
    "Sum",
]


class Transform(BaseSpatialModule):
    """
    Point transform module that applies a feature transform to the input point collection.
    No spatial operations are performed.

    Hydra config example usage:

    .. code-block:: yaml

            model:
            feature_transform:
                _target_: warpconvnet.nn.point_transform.Transform
                feature_transform_fn: _target_: torch.nn.ReLU
    """

    def __init__(self, feature_transform_fn: nn.Module):
        super().__init__()
        self.feature_transform_fn = feature_transform_fn

    def forward(self, *sfs: Tuple[Geometry, ...]) -> Geometry:
        """
        Apply the feature transform to the input point collection

        Args:
            pc: Input point collection

        Returns:
            Transformed point collection
        """
        if isinstance(sfs, Geometry):
            return sfs.replace(
                batched_features=self.feature_transform_fn(sfs.feature_tensor)
            )

        # When input is not a single BatchedSpatialFeatures, we assume the inputs are features
        assert [isinstance(sf, Geometry) for sf in sfs] == [True] * len(sfs)
        # Assert that all spatial features have the same offsets
        assert all(torch.allclose(sf.offsets, sfs[0].offsets) for sf in sfs)
        sf = sfs[0]
        features = [sf.feature_tensor for sf in sfs]

        out_features = self.feature_transform_fn(*features)
        return sf.replace(
            batched_features=out_features,
        )


class Cat(Transform):
    def __init__(self):
        # concatenation
        super().__init__(lambda *x: torch.concatenate(x, dim=-1))


class Sum(Transform):
    @staticmethod
    def _sum_fn(cls, xs: Tuple[Tensor, ...]) -> Tensor:
        feat = xs[0].clone()
        for x in xs[1:]:
            feat += x
        return feat

    def __init__(self):
        super().__init__(self._sum_fn)
