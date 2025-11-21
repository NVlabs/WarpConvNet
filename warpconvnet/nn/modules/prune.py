# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from jaxtyping import Bool
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.functional.sparse_ops import prune_spatially_sparse_tensor
from warpconvnet.nn.modules.base_module import BaseSpatialModule


class SparsePrune(BaseSpatialModule):
    """
    Module wrapper around ``prune_spatially_sparse_tensor`` so pruning can be composed in nn.Sequential.

    Forward Args
    -----------
    spatial_tensor : Geometry
        Sparse geometry (e.g., Voxels) whose coordinates/features will be filtered.
    mask : Bool[Tensor, "N"]
        Boolean mask aligned with ``spatial_tensor.coordinate_tensor``.
    """

    def forward(
        self,
        spatial_tensor: Geometry,
        mask: Bool[Tensor, "N"],  # noqa: F821
    ) -> Geometry:
        return prune_spatially_sparse_tensor(spatial_tensor, mask)
