# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from warpconvnet.utils.cupy_alloc import set_cupy_allocator

set_cupy_allocator()

# Import constants to set the default values
from warpconvnet.constants import WARPCONVNET_SKIP_SYMMETRIC_KERNEL_MAP
