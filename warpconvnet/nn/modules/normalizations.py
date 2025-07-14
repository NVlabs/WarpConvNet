# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.transforms import apply_feature_transform
from warpconvnet.nn.functional.normalizations import segmented_layer_norm

__all__ = [
    "NormalizationBase",
    "BatchNorm",
    "LayerNorm",
    "SegmentedLayerNorm",
    "InstanceNorm",
    "GroupNorm",
    "RMSNorm",
]


class NormalizationBase(BaseSpatialModule):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def __repr__(self):
        return f"{self.__class__.__name__}({self.norm})"

    def forward(
        self,
        input: Union[Geometry, Tensor],
    ):
        return apply_feature_transform(input, self.norm)


class BatchNorm(NormalizationBase):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(nn.BatchNorm1d(num_features, eps=eps, momentum=momentum))


class LayerNorm(NormalizationBase):
    def __init__(
        self,
        normalized_shape: List[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__(
            nn.LayerNorm(
                normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias
            )
        )


class SegmentedLayerNorm(nn.LayerNorm):

    def __init__(
        self, channels: int, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True
    ):
        super().__init__([channels], eps=eps, elementwise_affine=elementwise_affine, bias=bias)

    def forward(self, x: Geometry):
        # Only works for geometry with batched features
        assert isinstance(
            x, Geometry
        ), f"SegmentedLayerNorm only works for Geometry, got {type(x)}"
        out_feature = segmented_layer_norm(
            x.feature_tensor,
            x.offsets,
            gamma=self.weight if self.elementwise_affine else None,
            beta=self.bias if self.elementwise_affine else None,
            eps=self.eps,
        )
        return x.replace(batched_feature=out_feature)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine})"


class InstanceNorm(NormalizationBase):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__(nn.InstanceNorm1d(num_features, eps=eps))


class GroupNorm(NormalizationBase):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__(nn.GroupNorm(num_groups, num_channels, eps=eps))


class RMSNorm(NormalizationBase):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(nn.RMSNorm(dim, eps=eps))
