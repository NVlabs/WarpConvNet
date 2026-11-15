# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from warpconvnet.nn.functional.voxel_encode import voxel_encode


@pytest.mark.parametrize("window_size", [2, 4, (3, 5, 2)])
def test_voxel_encode_ravel_counts_unbatched(window_size):
    """Without batch_offsets, counts must describe the windows in perm order."""
    torch.manual_seed(0)
    coords = torch.randint(0, 20, (300, 3), dtype=torch.int32)

    result = voxel_encode(
        coords,
        batch_offsets=None,
        window_size=window_size,
        return_perm=True,
        return_counts=True,
        encoding_method="ravel",
    )

    ws = np.broadcast_to(np.asarray(window_size), (3,))
    windows = (coords.numpy() - coords.numpy().min(axis=0)) // ws
    _, ref_counts = np.unique(windows, axis=0, return_counts=True)

    counts = result.counts.cpu().numpy()
    assert counts.sum() == coords.shape[0]
    np.testing.assert_array_equal(counts, ref_counts)

    # Each counts-sized segment of the permuted codes must be a single window.
    sorted_codes = result.codes[result.perm].cpu().numpy()
    for segment in np.split(sorted_codes, np.cumsum(counts)[:-1]):
        assert (segment == segment[0]).all()
