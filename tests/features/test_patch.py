# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.patch import CatPatchFeatures


@pytest.fixture
def setup_features():
    B, min_N, max_N, C = 3, 1000, 10000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    features = [torch.rand((N, C)) for N in Ns]
    cat_features = CatFeatures(
        torch.cat(features, dim=0),
        torch.tensor([0] + list(torch.cumsum(Ns, dim=0).numpy())),
    )
    patch_size = 16
    return cat_features, patch_size


def test_cat_to_patch(setup_features):
    cat_features, patch_size = setup_features

    # Test conversion to patch features
    patch_features = CatPatchFeatures.from_cat(cat_features, patch_size)
    patch_features.clear_padding(-torch.inf)
    assert patch_features.num_channels == cat_features.num_channels
    assert (patch_features.offsets == cat_features.offsets).all()

    # Test conversion back to cat features
    cat_features_converted = patch_features.to_cat()
    assert (cat_features_converted.offsets == cat_features.offsets).all()
    assert (cat_features_converted.batched_tensor == cat_features.batched_tensor).all()
