# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch
from jaxtyping import Bool
from torch import Tensor

from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.types.voxels import Voxels


def cat_spatially_sparse_tensors(
    *sparse_tensors: Sequence[Voxels],
) -> Voxels:
    """
    Concatenate a list of spatially sparse tensors.
    """
    # Check that all sparse tensors have the same offsets
    offsets = sparse_tensors[0].offsets
    for sparse_tensor in sparse_tensors:
        if not torch.allclose(sparse_tensor.offsets.to(offsets), offsets):
            raise ValueError("All sparse tensors must have the same offsets")

    # Concatenate the features tensors
    features_tensor = torch.cat(
        [sparse_tensor.feature_tensor for sparse_tensor in sparse_tensors], dim=-1
    )
    return sparse_tensors[0].replace(batched_features=features_tensor)


def prune_spatially_sparse_tensor(
    spatial_tensor: Geometry,
    mask: Bool[Tensor, "N"],  # noqa: F821
) -> Geometry:
    """
    Prune a spatially sparse tensor using a boolean mask.

    Args:
        spatial_tensor: Geometry instance whose coordinates/features will be filtered.
        mask: Boolean mask of shape (N,) aligned with the flattened coordinates/features.

    Returns:
        New Geometry instance containing only the entries where mask == True.
    """
    if mask.shape[0] != spatial_tensor.coordinate_tensor.shape[0]:
        raise ValueError(
            f"Mask length {mask.shape[0]} must match number of coordinates {spatial_tensor.coordinate_tensor.shape[0]}"
        )

    mask = mask.to(spatial_tensor.device)
    if mask.dtype != torch.bool:
        mask = mask.bool()

    coords = spatial_tensor.batched_coordinates
    if not hasattr(coords, "prune"):
        raise TypeError(f"{coords.__class__.__name__} does not implement prune()")

    pruned_coords = coords.prune(mask)
    pruned_features = spatial_tensor.feature_tensor[mask]
    return spatial_tensor.replace(
        batched_coordinates=pruned_coords,
        batched_features=pruned_features,
    )
