# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim — sparse resampling promoted to ``warpconvnet.nn.modules``."""

from warpconvnet.nn.modules.sparse_resample import (
    SparseChannel2Spatial,
    SparseDownsample,
    SparseSpatial2Channel,
    SparseSubdivide,
    SparseUpsample,
)


__all__ = [
    "SparseChannel2Spatial",
    "SparseDownsample",
    "SparseSpatial2Channel",
    "SparseSubdivide",
    "SparseUpsample",
]
