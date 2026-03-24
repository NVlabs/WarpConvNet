# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
import torch.nn.functional as F

from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.features.pad import PadFeatures


@dataclass
class CatPatchFeatures(CatFeatures):
    """
    A data class for batched features that are divided into patches.

    From a [N1, N2, ..., N_B] batch of points, we pad each batch to be of size K * ceil(Ni / K), where K is the patch size.
    The padding is added at the end of each batch.

    The new padded batch size is [K * ceil(Ni / K) for Ni in [N1, N2, ..., N_B]].

    The patch_offsets tensor keeps track of the starting index of each patch in the padded batch.

    References: OctFormer https://arxiv.org/abs/2305.03045
    """

    patch_size: int
    patch_offsets: Int[Tensor, "B+1"]  # noqa: F821

    def __init__(
        self,
        batched_tensor: Float[Tensor, "M C"],
        offsets: Int[Tensor, "B+1"],  # noqa: F821
        patch_size: int,
        patch_offsets: Int[Tensor, "B+1"],  # noqa: F821
    ):
        self.patch_size = patch_size
        self.patch_offsets = patch_offsets
        super().__init__(batched_tensor, offsets)

    def check(self):
        assert (
            self.batched_tensor.shape[0] % self.patch_size == 0
        ), "Number of (padded) points must be divisible by patch size"
        assert (
            self.patch_offsets.shape[0] == self.offsets.shape[0]
        ), "Patch offsets must have the same batch size as offsets"
        assert (
            self.batched_tensor.shape[0] == self.patch_offsets[-1]
        ), "Number of (padded) points must be equal to the last patch offset"
        assert self.patch_size > 0, "Patch size must be positive"

    @classmethod
    def from_cat(cls, features: CatFeatures, patch_size: int):
        # Convert the [N1, N2, ..., N_B] batch of points into [(N1 + K - 1) // K * K, (N2 + K - 1) // K * K, ..., (N_B + K - 1) // K * K]
        # by padding each batch with Ni % K points.
        num_points = features.offsets.diff()
        num_points_padded = (num_points + patch_size - 1) // patch_size * patch_size
        patch_offsets = torch.cumsum(F.pad(num_points_padded, (1, 0)), dim=0)
        batch_tensor = torch.zeros(
            patch_offsets[-1],
            features.shape[1],
            dtype=features.dtype,
            device=features.device,
        )
        for i in range(features.batch_size):
            batch_tensor[patch_offsets[i] : patch_offsets[i] + num_points[i], :] = (
                features.batched_tensor[
                    features.offsets[i] : features.offsets[i] + num_points[i], :
                ]
            )
        return cls(batch_tensor, features.offsets, patch_size, patch_offsets)

    def clear_padding(self, clear_value: float = 0.0):
        num_points = self.offsets.diff()
        for i in range(self.batch_size):
            self.batched_tensor[
                self.patch_offsets[i] + num_points[i] : self.patch_offsets[i + 1], :
            ] = clear_value
        return self

    def __getitem__(self, idx: int) -> Float[Tensor, "N C"]:  # noqa: F722,F821
        N = self.offsets[idx + 1] - self.offsets[idx]
        return self.batched_tensor[
            self.patch_offsets[idx] : self.patch_offsets[idx] + N, :
        ]

    def to_cat(self) -> CatFeatures:
        batch_tensor = torch.empty(
            self.offsets[-1],
            self.num_channels,
            dtype=self.dtype,
            device=self.device,
        )
        for i in range(self.batch_size):
            batch_tensor[self.offsets[i] : self.offsets[i + 1]] = self[i]
        return CatFeatures(batch_tensor, self.offsets)

    def replace(
        self,
        batched_tensor: Optional[Float[Tensor, "B M C"]] = None,  # noqa: F821
        offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
        patch_size: Optional[int] = None,
        patch_offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
        **kwargs,
    ):
        batched_tensor = (
            batched_tensor if batched_tensor is not None else self.batched_tensor
        )
        return self.__class__(
            batched_tensor=batched_tensor,
            offsets=(offsets if offsets is not None else self.offsets),
            patch_size=patch_size if patch_size is not None else self.patch_size,
            patch_offsets=(
                patch_offsets if patch_offsets is not None else self.patch_offsets
            ),
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(offsets={self.offsets}, patch_size={self.patch_size}, patch_offsets={self.patch_offsets}, shape={self.batched_tensor.shape})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(offsets={self.offsets}, patch_size={self.patch_size}, patch_offsets={self.patch_offsets}, shape={self.batched_tensor.shape})"


@dataclass
class PadPatchFeatures(PadFeatures):
    pass
