# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test for all class imports. The file structure could cause circular import errors.
"""

import pytest
from dataclasses import is_dataclass

from warpconvnet.geometry.base.batched import BatchedTensor
from warpconvnet.geometry.base import Coords, Features, Geometry

from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.pad import PadFeatures
from warpconvnet.geometry.features.patch import CatPatchFeatures, PadPatchFeatures

from warpconvnet.geometry.coords.real import RealCoords
from warpconvnet.geometry.coords.integer import IntCoords

from warpconvnet.geometry.coords.search.search_results import IntSearchResult, RealSearchResult
from warpconvnet.geometry.coords.search.search_configs import IntSearchConfig, RealSearchConfig

from warpconvnet.geometry.coords.search.cache import (
    IntSearchCache,
    IntSearchCacheKey,
    RealSearchCache,
    RealSearchCacheKey,
)

from warpconvnet.geometry.coords.search.continuous import neighbor_search
from warpconvnet.geometry.coords.search.torch_discrete import (
    kernel_offsets_from_size,
    generate_kernel_map,
)

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels

# Import nn modules
from warpconvnet.nn.modules.activations import ReLU, GELU, SiLU, Tanh, Sigmoid, LeakyReLU
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.modules.point_conv import PointConv
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv, SparseConv2d, SparseConv3d

# TODO: Import nn functionals


def test_features_import():
    assert issubclass(Features, BatchedTensor)
    assert issubclass(CatFeatures, Features)
    assert issubclass(PadFeatures, Features)
    assert issubclass(CatPatchFeatures, Features)
    assert issubclass(PadPatchFeatures, Features)


def test_coords_import():
    assert issubclass(Coords, BatchedTensor)
    assert issubclass(RealCoords, Coords)
    assert issubclass(IntCoords, Coords)


def test_geometry_import():
    assert issubclass(Points, Geometry)
    assert issubclass(Voxels, Geometry)


def test_search_results_import():
    assert is_dataclass(RealSearchResult)
    assert is_dataclass(IntSearchResult)


def test_search_configs_import():
    assert is_dataclass(RealSearchConfig)
    assert is_dataclass(IntSearchConfig)


def test_search_cache_import():
    assert is_dataclass(IntSearchCacheKey)
    # IntSearchCache is a dict subclass keyed by IntSearchCacheKey, not a
    # dataclass (it stopped being one when it became a container).
    assert issubclass(IntSearchCache, dict)

    assert is_dataclass(RealSearchCache)
    assert is_dataclass(RealSearchCacheKey)
