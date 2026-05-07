# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 5 tests: TRELLIS.2 sparse spatial ops on warpconvnet ``Voxels``.

Validates that the sparse-tensor surface used by the TRELLIS.2 port is
byte-exact against the upstream ``trellis2.modules.sparse`` reference.
"""
import os
import sys

import pytest
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.trellis2.sparse_ops import (
    from_feats_coords,
    sparse_cat,
    sparse_unbind,
)
from warpconvnet.models.trellis2.sparse_spatial import (
    SparseChannel2Spatial,
    SparseDownsample,
    SparseSpatial2Channel,
    SparseSubdivide,
    SparseUpsample,
)

_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
_HAS_REF = False
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH):
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
    if _TRELLIS2_PATH not in sys.path:
        sys.path.insert(0, _TRELLIS2_PATH)
    try:
        from trellis2.modules.sparse.basic import SparseTensor as RefSparseTensor
        from trellis2.modules.sparse.spatial.basic import (
            SparseDownsample as RefSparseDownsample,
        )
        from trellis2.modules.sparse.spatial.spatial2channel import (
            SparseChannel2Spatial as RefSparseChannel2Spatial,
            SparseSpatial2Channel as RefSparseSpatial2Channel,
        )

        _HAS_REF = True
    except Exception:  # noqa: BLE001
        _HAS_REF = False


# -----------------------------------------------------------------------------
# Random sparse fixture (CPU). Each batch gets distinct random voxel coords
# inside an R^3 grid (no overlaps within a batch).
# -----------------------------------------------------------------------------
def _make_sparse(B: int = 2, N_per: int = 32, C: int = 8, R: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    coords_list, feats_list = [], []
    for b in range(B):
        flat = torch.randperm(R**3, generator=g)[:N_per]
        x = flat // (R * R)
        y = (flat // R) % R
        z = flat % R
        coords_list.append(torch.stack([torch.full_like(x, b), x, y, z], dim=-1).int())
        feats_list.append(torch.randn(N_per, C, generator=g))
    return torch.cat(feats_list, dim=0), torch.cat(coords_list, dim=0)


@pytest.fixture
def st():
    feats, coords = _make_sparse()
    return from_feats_coords(feats, coords)


# -----------------------------------------------------------------------------
# Voxels-on-trellis surface
# -----------------------------------------------------------------------------
def test_voxels_basic_attrs(st):
    assert isinstance(st, Voxels)
    assert st.batch_size == 2
    assert st.feats.shape[0] == st.coords.shape[0]
    assert st.coords.shape[1] == 4  # [batch, x, y, z]


def test_voxels_replace_features_keeps_coords(st):
    new = st.replace_features(st.feats * 2)
    torch.testing.assert_close(new.feats, st.feats * 2)
    torch.testing.assert_close(new.coords, st.coords)


def test_voxels_to_dense_roundtrip(st):
    dense = st.to_dense(channel_dim=1)
    bxyz = st.coords.long()
    sampled = dense[bxyz[:, 0], :, bxyz[:, 1], bxyz[:, 2], bxyz[:, 3]]
    torch.testing.assert_close(sampled, st.feats)


def test_voxels_elemwise_add(st):
    other = st.replace_features(torch.ones_like(st.feats))
    out = st + other
    torch.testing.assert_close(out.feats, st.feats + 1)


def test_sparse_cat_unbind_roundtrip(st):
    parts = sparse_unbind(st, dim=0)
    assert len(parts) == 2
    re = sparse_cat(parts, dim=0)
    torch.testing.assert_close(re.feats, st.feats)


# -----------------------------------------------------------------------------
# Spatial ops
# -----------------------------------------------------------------------------
def _coord_code(c):
    return c[:, 0] * 1_000_000 + c[:, 1] * 10_000 + c[:, 2] * 100 + c[:, 3]


def test_downsample_shrinks_unique_coords(st):
    out = SparseDownsample(2)(st)
    code = _coord_code(out.coords)
    assert code.unique().numel() == code.numel()


def test_downsample_upsample_recovers_coords(st):
    down = SparseDownsample(2)(st)
    up = SparseUpsample(2)(down)
    torch.testing.assert_close(
        _coord_code(up.coords).sort().values,
        _coord_code(st.coords).sort().values,
    )


def test_spatial2channel_channel_count(st):
    out = SparseSpatial2Channel(2)(st)
    assert out.feats.shape[1] == st.feats.shape[1] * 8


def test_spatial2channel_then_channel2spatial_recovers_feats(st):
    packed = SparseSpatial2Channel(2)(st)
    unpacked = SparseChannel2Spatial(2)(packed)
    src_order = torch.argsort(_coord_code(st.coords))
    out_order = torch.argsort(_coord_code(unpacked.coords))
    torch.testing.assert_close(st.feats[src_order], unpacked.feats[out_order])


def test_subdivide_multiplies_voxel_count(st):
    sub = SparseSubdivide(2)(st)
    assert sub.coords.shape[0] == st.coords.shape[0] * 8
    assert sub.feats.shape[0] == st.feats.shape[0] * 8


# -----------------------------------------------------------------------------
# Reference parity (upstream trellis2)
# -----------------------------------------------------------------------------
def _to_ref(st: Voxels):
    return RefSparseTensor(st.feats.clone(), st.coords.clone())


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_downsample_matches_reference(st):
    ours = SparseDownsample(2)(st)
    ref = RefSparseDownsample(2)(_to_ref(st))
    o_idx = torch.argsort(_coord_code(ours.coords))
    r_idx = torch.argsort(_coord_code(ref.coords))
    torch.testing.assert_close(ours.coords[o_idx], ref.coords[r_idx])
    torch.testing.assert_close(ours.feats[o_idx], ref.feats[r_idx], rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_spatial2channel_matches_reference(st):
    ours = SparseSpatial2Channel(2)(st)
    ref = RefSparseSpatial2Channel(2)(_to_ref(st))
    o_idx = torch.argsort(_coord_code(ours.coords))
    r_idx = torch.argsort(_coord_code(ref.coords))
    torch.testing.assert_close(ours.coords[o_idx], ref.coords[r_idx])
    torch.testing.assert_close(ours.feats[o_idx], ref.feats[r_idx], rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 ref not on PYTHONPATH")
def test_spatial2channel_channel2spatial_paired_match(st):
    ours = SparseSpatial2Channel(2)(st)
    ours_back = SparseChannel2Spatial(2)(ours)
    ref_in = _to_ref(st)
    ref = RefSparseSpatial2Channel(2)(ref_in)
    ref_back = RefSparseChannel2Spatial(2)(ref)
    o_idx = torch.argsort(_coord_code(ours_back.coords))
    r_idx = torch.argsort(_coord_code(ref_back.coords))
    torch.testing.assert_close(ours_back.coords[o_idx], ref_back.coords[r_idx])
    torch.testing.assert_close(ours_back.feats[o_idx], ref_back.feats[r_idx], rtol=1e-6, atol=1e-6)
