# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpaceFormer: mixed space/curve sparse attention for 3D point clouds.

See ``README.md`` for usage and the project page. Two models live here:

- ``SpaCeFormer`` (``space_former.py``): the sparse attention U-Net backbone; with
  ``out_channels=num_classes`` it is a per-point **semantic segmentation** model.
- ``SpaCeFormerInstSeg`` (``space_former_seg.py``): a proposal-free mask-query
  decoder on top of that backbone for open-vocabulary **instance segmentation**.
"""
from .space_former import SpaCeFormer
from .space_former_seg import (
    SpaCeFormerInstSeg,
    build_backbone,
    build_spaceformer,
    load_spaceformer_checkpoint,
)

__all__ = [
    "SpaCeFormer",
    "SpaCeFormerInstSeg",
    "build_backbone",
    "build_spaceformer",
    "load_spaceformer_checkpoint",
]
