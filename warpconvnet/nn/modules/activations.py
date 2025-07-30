# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from torch import Tensor, nn

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.transforms import (
    apply_feature_transform,
    elu,
    gelu,
    leaky_relu,
    log_softmax,
    sigmoid,
    silu,
    softmax,
    tanh,
)

__all__ = [
    "ReLU",
    "GELU",
    "SiLU",
    "Tanh",
    "Sigmoid",
    "LeakyReLU",
    "ELU",
    "Softmax",
    "LogSoftmax",
    "DropPath",
]


class ReLU(BaseSpatialModule):
    """Apply the ReLU activation to ``Geometry`` features.

    Parameters
    ----------
    inplace : bool, optional
        Whether to perform the operation in-place. Defaults to ``False``.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def __repr__(self):
        return f"{self.__class__.__name__}(inplace={self.relu.inplace})"

    def forward(self, input: Geometry):  # noqa: F821
        return apply_feature_transform(input, self.relu)


class GELU(BaseSpatialModule):
    """Applies the GELU activation to ``Geometry`` features."""

    def forward(self, input: Geometry):  # noqa: F821
        return gelu(input)


class SiLU(BaseSpatialModule):
    """Applies the SiLU activation to ``Geometry`` features."""

    def forward(self, input: Geometry):  # noqa: F821
        return silu(input)


class Tanh(BaseSpatialModule):
    """Applies the ``tanh`` activation to ``Geometry`` features."""

    def forward(self, input: Geometry):  # noqa: F821
        return tanh(input)


class Sigmoid(BaseSpatialModule):
    """Applies the sigmoid activation to ``Geometry`` features."""

    def forward(self, input: Geometry):  # noqa: F821
        return sigmoid(input)


class LeakyReLU(nn.Module):
    """Applies the LeakyReLU activation to ``Geometry`` features."""

    def forward(self, input: Geometry):  # noqa: F821
        return leaky_relu(input)


class ELU(BaseSpatialModule):
    """Applies the ELU activation to ``Geometry`` features."""

    def forward(self, input: Geometry):  # noqa: F821
        return elu(input)


class Softmax(BaseSpatialModule):
    """Applies the ``softmax`` activation to ``Geometry`` features."""

    def forward(self, input: Geometry):  # noqa: F821
        return softmax(input)


class LogSoftmax(BaseSpatialModule):
    """Applies the ``log_softmax`` activation to ``Geometry`` features."""

    def forward(self, input: Geometry):  # noqa: F821
        return log_softmax(input)


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Apply stochastic depth to the input tensor.

    Parameters
    ----------
    x : ``torch.Tensor``
        Input tensor to apply stochastic depth to.
    drop_prob : float, optional
        Probability of dropping a sample. Defaults to ``0.0``.
    training : bool, optional
        Whether the module is in training mode. Defaults to ``False``.
    scale_by_keep : bool, optional
        If ``True`` the output is scaled by ``1 - drop_prob``. Defaults to ``True``.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(BaseSpatialModule):
    """Stochastic depth regularization.

    Parameters
    ----------
    drop_prob : float, optional
        Probability of dropping a sample. Defaults to ``0.0``.
    scale_by_keep : bool, optional
        If ``True`` the output is scaled by ``1 - drop_prob``. Defaults to ``True``.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Union[Geometry, Tensor]):  # noqa: F821
        if isinstance(x, Geometry):
            return x.replace(
                batched_features=drop_path(
                    x.feature_tensor, self.drop_prob, self.training, self.scale_by_keep
                )
            )
        else:
            return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3): 0.3f}"
