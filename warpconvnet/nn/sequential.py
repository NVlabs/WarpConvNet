from typing import Tuple

import torch
import torch.nn as nn

from warpconvnet.geometry.base_geometry import SpatialFeatures
from warpconvnet.nn.base_module import BaseSpatialModule


def run_forward(module: nn.Module, x: SpatialFeatures, in_sf: SpatialFeatures):
    if isinstance(module, BaseSpatialModule) and isinstance(x, SpatialFeatures):
        return module(x), in_sf
    elif not isinstance(module, BaseSpatialModule) and isinstance(x, SpatialFeatures):
        in_sf = x
        x = module(x.feature_tensor)
    elif isinstance(x, torch.Tensor) and isinstance(module, BaseSpatialModule):
        x = in_sf.replace(batched_features=x)
        x = module(x)
    else:
        x = module(x)

    return x, in_sf


def tuple_run_forward(module: nn.Module, xs: Tuple[SpatialFeatures], in_sf: SpatialFeatures):
    if isinstance(module, BaseSpatialModule) and isinstance(xs[0], SpatialFeatures):
        return module(xs[0], *xs[1:]), in_sf
    elif not isinstance(module, BaseSpatialModule) and isinstance(xs[0], SpatialFeatures):
        in_sf = xs[0]
        xs = (module(xs[0].feature_tensor), *xs[1:])
    elif isinstance(xs[0], torch.Tensor) and isinstance(module, BaseSpatialModule):
        xs = (in_sf.replace(batched_features=xs[0]), *xs[1:])
        xs = module(*xs)
    else:
        xs = module(*xs)

    return xs, in_sf


class Sequential(nn.Sequential, BaseSpatialModule):
    """
    Sequential module that allows for spatial and non-spatial layers to be chained together.

    If the module has multiple consecutive non-spatial layers, then it will not create an intermediate
    spatial features object and will become more efficient.
    """

    def forward(self, x: SpatialFeatures):
        assert isinstance(
            x, SpatialFeatures
        ), f"Expected BatchedSpatialFeatures, got {type(x)}"

        in_sf = x
        for module in self:
            x, in_sf = run_forward(module, x, in_sf)

        if isinstance(x, torch.Tensor):
            x = in_sf.replace(batched_features=x)

        return x


class TupleSequential(Sequential, BaseSpatialModule):
    """
    Sequential module that allows multiple inputs for a specified layer.
    """

    def __init__(self, *args, tuple_layer: int):
        super().__init__(*args)
        self.tuple_layer = tuple_layer

    def forward(self, *xs: Tuple[SpatialFeatures]):
        x = xs[0]
        in_sf = x
        for i, module in enumerate(self):
            if i == self.tuple_layer:
                x, in_sf = tuple_run_forward(module, (x, *xs[1:]), in_sf)
            else:
                x, in_sf = run_forward(module, x, in_sf)

        if isinstance(x, torch.Tensor):
            x = in_sf.replace(batched_features=x)

        return x
