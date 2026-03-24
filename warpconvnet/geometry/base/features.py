# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
from jaxtyping import Float

from .batched import BatchedTensor
from torch import Tensor


class Features(BatchedTensor):
    """Base class for features."""

    @property
    def num_channels(self):
        return self.batched_tensor.shape[-1]

    @property
    def is_cat(self):
        """By default, the features class uses concatenated features"""
        return True

    @property
    def is_pad(self):
        """By default, the features class uses concatenated features"""
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(offsets={self.offsets}, shape={self.batched_tensor.shape}, device={self.device}, dtype={self.dtype})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(offsets={self.offsets}, shape={self.batched_tensor.shape})"
