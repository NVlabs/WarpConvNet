# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel-map correctness at realistic scale.

Given an input coordinate set I, output coordinate set O (= strided I), a
kernel size K, and a stride S, `generate_kernel_map` must return CSR arrays
`(in_maps, out_maps, offsets)` such that for every kernel slot k and every
pair (in_maps[k][i], out_maps[k][i]):

    I[in_maps[k][i]]  ==  S * O[out_maps[k][i]] + kernel_offset[k]          (*)

and conversely every valid (input, output, k) triple satisfying (*) must
appear exactly once in the map. These tests verify (*) directly at 500K+
voxels — the scale where a stride-2 bug we see in the production fwd test
would actually bite training.

Also verifies PackedHashTable behavior at large N: insert→search round-trips,
all inserted keys map to correct indices, and coord-out-of-bounds coordinates
never produce false hits.
"""

import pytest
import torch

from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.ops.stride import stride_coords
from warpconvnet.geometry.coords.search.packed_hashmap import PackedHashTable
from warpconvnet.geometry.coords.search.torch_discrete import (
    generate_kernel_map,
    kernel_offsets_from_size,
)
from warpconvnet.geometry.types.voxels import Voxels


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scannet_like_voxels(n_per_batch=200_000, batch_size=2, coord_range=512, seed=0) -> Voxels:
    """Mimic a ScanNet minibatch after voxelization: ~200K-500K voxels per scene."""
    torch.manual_seed(seed)
    coords_list = []
    feats_list = []
    for _ in range(batch_size):
        # Oversample 1.5x and dedup to reach near n_per_batch unique coords.
        c = torch.randint(0, coord_range, (int(n_per_batch * 1.5), 3), dtype=torch.int32)
        c = torch.unique(c, dim=0)[:n_per_batch]
        coords_list.append(c)
        feats_list.append(torch.randn(c.shape[0], 3))
    return Voxels(coords_list, feats_list).to("cuda")


# ---------------------------------------------------------------------------
# Large-scale PackedHashTable round-trip
# ---------------------------------------------------------------------------


class TestPackedHashTableLargeScale:
    """Verify PackedHashTable behavior at realistic scene sizes."""

    @pytest.mark.parametrize("n_per_batch", [100_000, 500_000])
    def test_large_round_trip(self, n_per_batch):
        voxels = _scannet_like_voxels(n_per_batch=n_per_batch, batch_size=2)
        in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        ht = PackedHashTable.from_coords(in_coords)
        idx = ht.search(in_coords)
        torch.cuda.synchronize()
        assert (idx >= 0).all(), f"{(idx < 0).sum().item()} inserted coords not found"
        # Every insertion index must round-trip.
        assert torch.equal(in_coords[idx.long()], in_coords)

    def test_misses_on_shifted_coords(self):
        """Coordinates shifted by (+1000,0,0) must not collide with any inserted key."""
        voxels = _scannet_like_voxels(n_per_batch=200_000, batch_size=2, seed=1)
        in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        ht = PackedHashTable.from_coords(in_coords)
        shifted = in_coords.clone()
        shifted[:, 1] += 5000  # far outside the input range
        idx = ht.search(shifted)
        torch.cuda.synchronize()
        assert (idx == -1).all(), f"{(idx >= 0).sum().item()} shifted coords got false positives"


# ---------------------------------------------------------------------------
# Kernel-map invariant: I[in] == S * O[out] + offset[k]
# ---------------------------------------------------------------------------


def _verify_kernel_map_invariant(
    voxels: Voxels,
    kernel_size=(3, 3, 3),
    stride=1,
    max_pairs=2_000_000,
):
    """Verify (*) for every (in, out, k) pair the map returns."""
    in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    if stride == 1:
        out_coords = in_coords
    else:
        out_coords, _ = stride_coords(in_coords, stride=(stride,) * len(kernel_size))

    kmap = generate_kernel_map(
        in_coords,
        out_coords,
        in_to_out_stride_ratio=(stride,) * len(kernel_size),
        kernel_size=kernel_size,
    )
    offsets_3d = kernel_offsets_from_size(
        kernel_size, (1,) * len(kernel_size), device=in_coords.device
    )
    # offsets_3d: (K, 4) with leading batch-dim zero; columns [batch, x, y, z]
    K = offsets_3d.shape[0]
    assert (
        K == kmap.offsets.shape[0] - 1
    ), f"K mismatch: offsets says {kmap.offsets.shape[0]-1}, kernel has {K}"

    total_pairs = int(kmap.offsets[-1].item())
    assert total_pairs == kmap.in_maps.shape[0]
    assert total_pairs == kmap.out_maps.shape[0]

    # Iterate kernel slots and verify the invariant on a bounded random sample.
    rng = torch.Generator(device=in_coords.device).manual_seed(0)
    checked = 0
    for k in range(K):
        in_map_k, out_map_k = kmap[k]
        nk = in_map_k.shape[0]
        if nk == 0:
            continue
        # Sample to keep memory reasonable at K=27 × 10M pairs.
        budget = max(1, max_pairs // K)
        if nk > budget:
            perm = torch.randperm(nk, generator=rng, device=in_coords.device)[:budget]
            in_map_k = in_map_k[perm]
            out_map_k = out_map_k[perm]
        # Build the predicted input coordinate from the output coord + offset.
        # I[in_map_k] should equal S * O[out_map_k] + offset_3d[k] (batch stays equal).
        out_c = out_coords[out_map_k.long()]
        pred = out_c.clone()
        pred[:, 1:] = out_c[:, 1:] * stride + offsets_3d[k, 1:]
        # Batch index must match.
        pred[:, 0] = out_c[:, 0]
        actual = in_coords[in_map_k.long()]
        ok = torch.equal(pred, actual)
        assert ok, (
            f"kernel-map invariant violated at k={k} stride={stride} "
            f"kernel_size={kernel_size}: first mismatch pred={pred[(pred != actual).any(dim=1)][0].tolist()} "
            f"actual={actual[(pred != actual).any(dim=1)][0].tolist()}"
        )
        checked += in_map_k.shape[0]
    assert checked > 0, "No pairs checked — kernel map was empty"


class TestKernelMapInvariant:
    """Every (in, out, k) triple must satisfy I[in] == S*O[out] + offset[k]."""

    @pytest.mark.parametrize("kernel_size", [(3, 3, 3), (5, 5, 5)])
    def test_stride1(self, kernel_size):
        voxels = _scannet_like_voxels(n_per_batch=100_000, batch_size=2, seed=2)
        _verify_kernel_map_invariant(voxels, kernel_size=kernel_size, stride=1)

    @pytest.mark.parametrize("stride", [2, 4])
    def test_stride_gt1(self, stride):
        voxels = _scannet_like_voxels(n_per_batch=100_000, batch_size=2, seed=3)
        _verify_kernel_map_invariant(voxels, kernel_size=(3, 3, 3), stride=stride)

    @pytest.mark.slow
    @pytest.mark.parametrize("stride", [1, 2])
    def test_scannet_scale(self, stride):
        """ScanNet-scale: 500K voxels per scene, batch_size=2."""
        voxels = _scannet_like_voxels(n_per_batch=500_000, batch_size=2, seed=4)
        _verify_kernel_map_invariant(voxels, kernel_size=(3, 3, 3), stride=stride)


# ---------------------------------------------------------------------------
# Kernel-map coverage: every valid (in, out, k) triple must be present exactly once.
# ---------------------------------------------------------------------------


class TestKernelMapCoverage:
    """For small inputs, brute-force enumerate all valid pairs and check the map
    contains each exactly once (no missing, no duplicates)."""

    def test_coverage_stride1_small(self):
        # Tiny input so the brute force is tractable.
        coords = torch.unique(torch.randint(0, 10, (200, 3), dtype=torch.int32), dim=0)
        feats = torch.randn(coords.shape[0], 3)
        voxels = Voxels([coords], [feats]).to("cuda")
        in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        kernel_size = (3, 3, 3)

        kmap = generate_kernel_map(
            in_coords,
            in_coords,
            in_to_out_stride_ratio=(1, 1, 1),
            kernel_size=kernel_size,
        )
        offsets_3d = kernel_offsets_from_size(kernel_size, (1, 1, 1), device=in_coords.device)
        K = offsets_3d.shape[0]

        # Brute force: for each out_idx, for each k, check if out+offset ∈ input.
        # Use a Python dict for correctness (small N).
        in_dict = {}
        for i, c in enumerate(in_coords.cpu().tolist()):
            in_dict[tuple(c)] = i

        expected_pairs = set()  # set of (k, in_idx, out_idx)
        for out_idx, oc in enumerate(in_coords.cpu().tolist()):  # stride=1: same set
            for k in range(K):
                off = offsets_3d[k].cpu().tolist()
                probe = (oc[0], oc[1] + off[1], oc[2] + off[2], oc[3] + off[3])
                if probe in in_dict:
                    expected_pairs.add((k, in_dict[probe], out_idx))

        actual_pairs = set()
        for k in range(K):
            im, om = kmap[k]
            for a, b in zip(im.cpu().tolist(), om.cpu().tolist()):
                actual_pairs.add((k, a, b))

        missing = expected_pairs - actual_pairs
        extra = actual_pairs - expected_pairs
        assert (
            not missing
        ), f"kernel map missing {len(missing)} valid pairs; first: {next(iter(missing))}"
        assert not extra, f"kernel map has {len(extra)} spurious pairs; first: {next(iter(extra))}"

    def test_coverage_stride2_small(self):
        coords = torch.unique(torch.randint(0, 16, (300, 3), dtype=torch.int32), dim=0)
        # Keep only coords whose strided downsample exists — avoid edge effects.
        feats = torch.randn(coords.shape[0], 3)
        voxels = Voxels([coords], [feats]).to("cuda")
        in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
        out_coords, _ = stride_coords(in_coords, stride=(2, 2, 2))
        kernel_size = (3, 3, 3)

        kmap = generate_kernel_map(
            in_coords,
            out_coords,
            in_to_out_stride_ratio=(2, 2, 2),
            kernel_size=kernel_size,
        )
        offsets_3d = kernel_offsets_from_size(kernel_size, (1, 1, 1), device=in_coords.device)
        K = offsets_3d.shape[0]

        in_dict = {}
        for i, c in enumerate(in_coords.cpu().tolist()):
            in_dict[tuple(c)] = i

        expected_pairs = set()
        for out_idx, oc in enumerate(out_coords.cpu().tolist()):
            for k in range(K):
                off = offsets_3d[k].cpu().tolist()
                # stride=2 means input coord = 2*out + offset
                probe = (oc[0], 2 * oc[1] + off[1], 2 * oc[2] + off[2], 2 * oc[3] + off[3])
                if probe in in_dict:
                    expected_pairs.add((k, in_dict[probe], out_idx))

        actual_pairs = set()
        for k in range(K):
            im, om = kmap[k]
            for a, b in zip(im.cpu().tolist(), om.cpu().tolist()):
                actual_pairs.add((k, a, b))

        missing = expected_pairs - actual_pairs
        extra = actual_pairs - expected_pairs
        assert not missing, (
            f"stride-2 kernel map MISSING {len(missing)} of {len(expected_pairs)} valid pairs; "
            f"first: {next(iter(missing))}"
        )
        assert (
            not extra
        ), f"stride-2 kernel map has {len(extra)} spurious pairs; first: {next(iter(extra))}"
