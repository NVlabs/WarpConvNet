# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry utility helpers.

Keep this initializer import-light: low-level geometry modules import
``warpconvnet.geometry.utils.list_to_batch`` during package construction, and
eagerly importing voxel helpers here creates a circular import through
``Voxels``.
"""

__all__: list[str] = []
