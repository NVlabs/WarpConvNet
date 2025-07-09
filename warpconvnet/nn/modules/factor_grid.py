# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Neural network modules for FactorGrid operations.

This module provides neural network layers and operations specifically designed
for working with FactorGrid geometries in the FIGConvNet architecture.
"""

from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.features.grid import GridMemoryFormat
from warpconvnet.geometry.types.factor_grid import FactorGrid
from warpconvnet.geometry.types.grid import Grid
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.functional.factor_grid import (
    factor_grid_transform,
    factor_grid_cat,
    factor_grid_pool,
    factor_grid_intra_communication,
    factor_grid_intra_communications,
)

__all__ = [
    "FactorGridTransform",
    "FactorGridCat",
    "FactorGridPool",
    "FactorGridIntraCommunication",
]


class FactorGridTransform(BaseSpatialModule):
    """Apply a transform operation to all grids in a FactorGrid.

    This is equivalent to GridFeatureGroupTransform but works with FactorGrid objects.
    """

    def __init__(self, transform: nn.Module, in_place: bool = True) -> None:
        super().__init__()
        self.transform = transform
        self.in_place = in_place

    def forward(self, factor_grid: FactorGrid) -> FactorGrid:
        """Apply transform to all grids in the FactorGrid."""
        return factor_grid_transform(factor_grid, self.transform, self.in_place)


class FactorGridCat(BaseSpatialModule):
    """Concatenate features of two FactorGrid objects.

    This is equivalent to GridFeatureGroupCat but works with FactorGrid objects.
    """

    def __init__(self):
        super().__init__()

    def forward(self, factor_grid1: FactorGrid, factor_grid2: FactorGrid) -> FactorGrid:
        """Concatenate features from two FactorGrid objects."""
        return factor_grid_cat(factor_grid1, factor_grid2)


class FactorGridPool(BaseSpatialModule):
    """Pooling operation for FactorGrid.

    This is equivalent to GridFeatureGroupPool but works with FactorGrid objects.
    """

    def __init__(
        self,
        pooling_type: Literal["max", "mean", "attention"] = "max",
    ):
        super().__init__()
        self.pooling_type = pooling_type

        # Pooling operation
        if pooling_type == "max":
            self.pool_op = nn.AdaptiveMaxPool1d(1)
        elif pooling_type == "mean":
            self.pool_op = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == "attention":
            # For now, use simple attention mechanism
            # Note: attention layer dimensions will depend on actual feature dimensions
            self.attention = None  # Will be set based on input if needed
            self.pool_op = None
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

    def forward(self, factor_grid: FactorGrid) -> Tensor:
        """Pool features from FactorGrid to a single tensor."""
        return factor_grid_pool(
            factor_grid,
            self.pooling_type,
            pool_op=self.pool_op,
            attention_layer=getattr(self, "attention", None),
        )


class FactorGridIntraCommunication(BaseSpatialModule):
    """Intra-communication between grids in a FactorGrid.

    This is equivalent to GridFeaturesGroupIntraCommunication but works with FactorGrid objects.
    """

    def __init__(self, communication_types: List[Literal["sum", "mul"]] = ["sum"]) -> None:
        super().__init__()
        assert len(communication_types) > 0, "At least one communication type must be provided"
        assert len(communication_types) <= 2, "At most two communication types can be provided"
        self.communication_types = communication_types

    def forward(self, factor_grid: FactorGrid) -> FactorGrid:
        """Perform intra-communication between grids in the FactorGrid."""
        return factor_grid_intra_communications(factor_grid, self.communication_types)
