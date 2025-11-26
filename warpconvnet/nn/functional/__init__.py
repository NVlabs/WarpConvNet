# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Functional operators used by WarpConvNet modules.

The package intentionally keeps implementations in separate submodules
(`sparse_conv`, `point_pool`, etc.) to avoid large monolithic files.
Defining this initializer makes the namespace explicit so documentation
generators like mkdocstrings can traverse it.
"""

__all__: list[str] = []
