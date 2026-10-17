# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #11: dtype mismatch in point_pool max-points branch.

Reporter's scenario: ``point_pool(..., downsample_max_num_points=N,
return_type="voxel")`` on default float-coordinate ``Points`` first crashed
inside ``IntCoords`` with "Discrete coordinates must be integers", and when
the caller cast coordinates to int as a workaround, the max-points kNN
reduction then crashed with ``RuntimeError: cdist only supports
floating-point dtypes``.

Resolution: reject the unsupported combination
``downsample_max_num_points=...`` + ``return_type='voxel'`` clearly in
``point_pool``, with a message pointing to the supported alternative
(``downsample_voxel_size``). Max-point sampling operates on continuous
coordinates and has no defined voxel grid to discretise onto — silently
casting coordinates to int would change spatial semantics and break
nearest-neighbour reduction (``torch.cdist`` requires floats, as the
reporter's second traceback demonstrated).
"""

import pytest
import torch

from warpconvnet.geometry.types.points import Points
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.point_pool import point_pool


@pytest.fixture
def setup_points():
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, min_N, max_N, C = 2, 500, 1500, 7
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [torch.rand((N, 3)) for N in Ns]
    features = [torch.rand((N, C)) for N in Ns]
    return Points(coords, features).to(device)


class TestIssue11Rejection:
    """The unsupported combination must be rejected with a clear message."""

    def test_voxel_return_with_max_num_points_rejected(self, setup_points):
        """The reporter's exact failure path now errors with a helpful
        message instead of an internal Voxels constructor assertion."""
        with pytest.raises(ValueError, match="return_type='voxel'"):
            point_pool(
                setup_points,
                reduction="mean",
                downsample_max_num_points=64,
                return_type="voxel",
            )

    def test_voxel_return_with_both_arguments_rejected(self, setup_points):
        """Voxel size does not override the max-point-count branch's rejection."""
        with pytest.raises(ValueError, match="voxel"):
            point_pool(
                setup_points,
                reduction="max",
                downsample_max_num_points=32,
                downsample_voxel_size=0.1,
                return_type="voxel",
            )


class TestIssue11SupportedCombinations:
    """Supported combinations still work after the rejection was added."""

    def test_max_num_points_with_point_return_type_works(self, setup_points):
        out = point_pool(
            setup_points,
            reduction="mean",
            downsample_max_num_points=64,
            return_type="point",
        )
        # Should return a Points instance, not Voxels, and not error
        assert isinstance(out, Points)
        assert out.batch_size == setup_points.batch_size
        # Coordinates stay continuous (float) — no silent voxelisation
        assert out.coordinate_tensor.dtype in (torch.float32, torch.float64)
        # Output is smaller than input per-batch (pooling actually happened)
        per_batch_in = (setup_points.offsets.diff()).cpu()
        per_batch_out = (out.offsets.diff()).cpu()
        assert (
            per_batch_out <= per_batch_in
        ).all(), f"max-num-points branch should not grow; in={per_batch_in} out={per_batch_out}"
        assert (
            per_batch_out <= 64
        ).all(), f"each batch should sample at most 64 points; got {per_batch_out}"

    def test_max_num_points_with_both_args_still_point(self, setup_points):
        """When both args are given + return_type='point', max-points path
        (with continuous output) is taken — preserves issue-#9 precedence."""
        out = point_pool(
            setup_points,
            reduction="mean",
            downsample_max_num_points=32,
            downsample_voxel_size=0.05,
            return_type="point",
        )
        assert isinstance(out, Points)
        assert out.batch_size == setup_points.batch_size
        assert out.coordinate_tensor.dtype in (torch.float32, torch.float64)

    def test_voxel_size_path_returns_voxels(self, setup_points):
        """Sanity: the supported way to get a Voxels output still works."""
        out = point_pool(
            setup_points,
            reduction="mean",
            downsample_voxel_size=0.1,
            return_type="voxel",
        )
        assert isinstance(out, Voxels)
        assert out.batch_size == setup_points.batch_size
        # Voxels coordinates must be integer
        assert out.coordinate_tensor.dtype in (
            torch.int32,
            torch.int64,
        ), "Voxels coordinates must be integer dtype"
        # Batch offsets must be internally consistent with the coordinates
        assert out.offsets[-1].item() == out.coordinate_tensor.shape[0], (
            f"offsets[-1]={out.offsets[-1].item()} != coord rows "
            f"{out.coordinate_tensor.shape[0]}"
        )
        # Feature/coordinate row counts must match
        assert out.coordinate_tensor.shape[0] == out.features.shape[0], (
            f"coords rows {out.coordinate_tensor.shape[0]} != features rows "
            f"{out.features.shape[0]}"
        )


class TestIssue11DtypeContract:
    """Coordinate/feature/batch-offset consistency around the dtype fix."""

    def test_no_silent_int_cast_in_point_branch(self, setup_points):
        """Regression guard: the fix must not silently cast continuous
        coordinates to integers on the 'point' return_type path. The
        spatial semantics (and downstream kNN / cdist) rely on float."""
        out = point_pool(
            setup_points,
            reduction="mean",
            downsample_max_num_points=48,
            return_type="point",
        )
        # Same dtype as input coordinates
        assert out.batch_size == setup_points.batch_size
        assert out.coordinate_tensor.dtype == setup_points.coordinate_tensor.dtype
        # Coordinates must be a subset of the input's coordinates
        # (the branch samples; it does not voxelise/average).
        in_coords = setup_points.coordinate_tensor
        out_coords = out.coordinate_tensor
        # Every output row must exist in the input (sampled rows).
        # Cheapest check: match via nearest float equality per row.
        for row in out_coords[:10]:  # spot-check first 10
            matches = (in_coords == row).all(dim=1).any()
            assert matches, f"sampled coord {row} not in input — was it modified?"
