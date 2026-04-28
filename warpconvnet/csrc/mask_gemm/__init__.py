# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask GEMM artifacts emitted by warpgemm.

Holds the tile_metadata.py snapshot that drives mask GEMM tile selection
in nn/.../sparse_conv/detail/dispatch.py. The mask GEMM kernel headers
(MaskGemm_*.h) live in csrc/include/ for historical reasons; future
warpgemm emits may move them under this package.
"""
