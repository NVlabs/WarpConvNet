# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #9: PointMinkUNet passed unsupported ``return_type="sparse"``.

Before fix: ``PointMinkUNetBase.forward`` crashed with an assertion from
``warpconvnet.nn.functional.point_pool.point_pool`` because the functional
signature is ``Literal["point", "voxel"]`` but the model passed ``"sparse"``.

These tests would have failed before the fix and pass after.
"""

import pytest
import torch

from warpconvnet.geometry.types.points import Points
from warpconvnet.models.mink_unet import PointMinkUNet18
from warpconvnet.nn.functional.point_pool import point_pool
from warpconvnet.nn.modules.point_pool import (
    PointAvgPool,
    PointMaxPool,
    PointSumPool,
)


@pytest.fixture
def setup_points():
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B, min_N, max_N, C = 2, 500, 2000, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features).to(device)


class TestIssue9ModelCaller:
    """The regression that motivated issue #9: PointMinkUNet18 forward pass works."""

    def test_point_mink_unet_18_forward_backward_smoke(self, setup_points):
        """PointMinkUNet18 must accept a Points input and run forward+backward.

        Fails before the fix with:
            AssertionError: return_type must be 'point' or 'voxel'
        """
        model = PointMinkUNet18(in_channels=7, out_channels=5).to(setup_points.device)
        out = model(setup_points)
        assert isinstance(out, Points)
        assert out.batch_size == setup_points.batch_size
        # Backward smoke test — drives grads through the pool/unpool path.
        out.features.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


class TestIssue9PrecedenceDocumentation:
    """Functional docstring says voxel-size wins when both are provided;
    the code actually honors max_num_points first. Fix pins the *code* behavior
    as the contract and updates the doc; these tests lock that in."""

    def test_both_arguments_prefers_max_num_points(self, setup_points):
        """When both ``downsample_max_num_points`` and ``downsample_voxel_size``
        are supplied, max-num-points determines the output size.

        Locking in the existing code path: max-num-points branch executes and
        yields the same result regardless of whether ``downsample_voxel_size``
        is passed alongside it."""
        max_pts = 64
        voxel_size = 0.1

        # Seed both identically to make random sampling deterministic
        torch.manual_seed(42)
        out_max_only = point_pool(
            setup_points,
            reduction="mean",
            downsample_max_num_points=max_pts,
            return_type="point",
            return_to_unique=False,
        )

        torch.manual_seed(42)
        out_both = point_pool(
            setup_points,
            reduction="mean",
            downsample_max_num_points=max_pts,
            downsample_voxel_size=voxel_size,  # would voxelize much coarser if it won
            return_type="point",
            return_to_unique=False,
        )

        # If voxel-size had won, out_both would differ from out_max_only.
        assert out_max_only.batch_size == setup_points.batch_size
        assert out_both.batch_size == setup_points.batch_size
        assert torch.equal(out_both.offsets, out_max_only.offsets)
        assert len(out_both.features) == len(out_max_only.features), (
            f"max-num-points branch should produce identical output with/without "
            f"downsample_voxel_size; got {len(out_both.features)} vs "
            f"{len(out_max_only.features)}"
        )
        assert torch.allclose(
            out_both.coordinate_tensor, out_max_only.coordinate_tensor
        ), "coordinates differ when downsample_voxel_size added on top"
        assert torch.allclose(
            out_both.features, out_max_only.features
        ), "features differ when downsample_voxel_size added on top"

        # Sanity: voxel-only produces a *different* output size, proving
        # the two branches actually differ.
        out_voxel_only = point_pool(
            setup_points,
            reduction="mean",
            downsample_voxel_size=voxel_size,
            return_type="point",
            return_to_unique=False,
        )
        assert out_voxel_only.batch_size == setup_points.batch_size
        assert len(out_voxel_only.features) != len(out_max_only.features), (
            f"voxel-only path produced same output as max-points path "
            f"({len(out_voxel_only.features)}); test no longer discriminates"
        )

    def test_wrapper_modules_reject_sparse_return_type(self, setup_points):
        """Wrapper annotations used to advertise ``'sparse'`` as a valid
        ``return_type``. It isn't — ``point_pool`` accepts only
        ``'point'`` / ``'voxel'``. Wrappers now carry the corrected
        ``Literal['point', 'voxel']``."""
        for cls in (PointMaxPool, PointAvgPool, PointSumPool):
            with pytest.raises(AssertionError, match="return_type"):
                cls(downsample_voxel_size=0.1, return_type="sparse")(setup_points)
