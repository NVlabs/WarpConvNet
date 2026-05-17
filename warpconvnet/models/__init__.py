# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model re-exports.

Importing ``warpconvnet.models`` previously eagerly loaded every model in the
package, which dragged in heavy optional deps even when the caller only wanted, say,
``warpconvnet.models.trellis2``. We use PEP-562 ``__getattr__`` so each public
name imports the underlying module on first access — preserving the public
API while letting callers avoid pulling in modules they don't use.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DGCNN": ("warpconvnet.models.dgcnn", "DGCNN"),
    "DGCNNEncoder": ("warpconvnet.models.dgcnn", "DGCNNEncoder"),
    "FIGConvNet": ("warpconvnet.models.figconv", "FIGConvNet"),
    "FIGConvNetDrivAer": ("warpconvnet.models.figconv", "FIGConvNetDrivAer"),
    "MaskFormer": ("warpconvnet.models.maskformer", "MaskFormer"),
    "MaskTransformer": ("warpconvnet.models.maskformer", "MaskTransformer"),
    "MinkUNet18": ("warpconvnet.models.mink_unet", "MinkUNet18"),
    "MinkUNet34": ("warpconvnet.models.mink_unet", "MinkUNet34"),
    "MinkUNet50": ("warpconvnet.models.mink_unet", "MinkUNet50"),
    "MinkUNet101": ("warpconvnet.models.mink_unet", "MinkUNet101"),
    "MinkUNetBase": ("warpconvnet.models.mink_unet", "MinkUNetBase"),
    "PointMinkUNet18": ("warpconvnet.models.mink_unet", "PointMinkUNet18"),
    "PointMinkUNet34": ("warpconvnet.models.mink_unet", "PointMinkUNet34"),
    "PointMinkUNetBase": ("warpconvnet.models.mink_unet", "PointMinkUNetBase"),
    "PointNet": ("warpconvnet.models.pointnet", "PointNet"),
    "PointTransformerV3": ("warpconvnet.models.point_transformer_v3", "PointTransformerV3"),
    "SpaCeFormer": ("warpconvnet.models.space_former", "SpaCeFormer"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


__all__ = sorted(_LAZY_IMPORTS.keys())
