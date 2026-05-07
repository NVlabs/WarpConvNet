# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS-port-specific sparse helpers.

The generic constructor / cat / unbind helpers were promoted to
`warpconvnet.geometry.utils.voxel_ops`; this module re-exports them and adds
the fractional ``_trellis_scale`` tracking used by paired Down/Up sparse
blocks (specific to TRELLIS-style inference).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Tuple

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.utils.voxel_ops import (
    from_feats_coords as _generic_from_feats_coords,
    sparse_cat,
    sparse_unbind,
)


__all__ = [
    "from_feats_coords",
    "get_scale",
    "get_scale_keyed_cache",
    "set_scale",
    "sparse_cat",
    "sparse_unbind",
]


_SCALE_KEY = "_trellis_scale"


def from_feats_coords(feats, coords, *, scale=None, spatial_cache=None, **kwargs):
    """Wraps the generic builder with optional `_trellis_scale` injection."""
    extras = dict(kwargs)
    if scale is not None:
        extras[_SCALE_KEY] = scale
    if spatial_cache is not None:
        extras["_spatial_cache"] = spatial_cache
    return _generic_from_feats_coords(feats, coords, **extras)


def get_scale(v: Voxels) -> tuple[Fraction, Fraction, Fraction]:
    s = v._extra_attributes.get(_SCALE_KEY)
    if s is None:
        s = (Fraction(1, 1),) * 3
        v._extra_attributes[_SCALE_KEY] = s
    return s


def set_scale(v: Voxels, new_scale: tuple[Fraction, Fraction, Fraction]) -> None:
    v._extra_attributes[_SCALE_KEY] = new_scale


def get_scale_keyed_cache(v: Voxels) -> dict:
    """Return the per-scale sub-dict of the voxels' spatial cache."""
    return v.spatial_cache.setdefault(str(get_scale(v)), {})
