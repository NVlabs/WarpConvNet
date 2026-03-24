# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List
from jaxtyping import Float, Int
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from warpconvnet.geometry.base.features import Features
from warpconvnet.geometry.utils.list_to_batch import list_to_pad_tensor
from warpconvnet.geometry.features.ops.convert import pad_to_cat, cat_to_pad


@dataclass
class PadFeatures(Features):
    batched_tensor: Float[Tensor, "B M C"]  # noqa: F722,F821
    offsets: Int[Tensor, "B+1"]  # noqa: F722,F821
    pad_multiple: Optional[int] = None

    def __init__(
        self,
        batched_tensor: (
            List[Float[Tensor, "N C"]] | Float[Tensor, "B M C"]
        ),  # noqa: F722,F821
        offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F722,F821
        pad_multiple: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if isinstance(batched_tensor, list):
            assert offsets is None, "If batched_tensors is a list, offsets must be None"
            batched_tensor, offsets, _ = list_to_pad_tensor(
                batched_tensor, pad_multiple
            )

        if isinstance(batched_tensor, torch.Tensor) and offsets is None:
            assert (
                pad_multiple is not None
            ), "pad_multiple must be provided if batched_tensor is a tensor"
            if batched_tensor.ndim == 2:
                batched_tensor = batched_tensor.unsqueeze(0)
            offsets = [0, batched_tensor.shape[1]]

        assert batched_tensor.ndim == 3, "Batched tensor must be 3D"

        self.batched_tensor = batched_tensor
        self.offsets = offsets
        self.pad_multiple = pad_multiple
        if device is not None:
            self.batched_tensor = self.batched_tensor.to(device)

    def check(self):
        super().check()
        assert self.batched_tensor.ndim == 2, "Batched tensor must be 2D"
        assert (
            self.batched_tensor.shape[0] == self.offsets[-1]
        ), f"Offsets {self.offsets} does not match tensors {self.batched_tensor.shape}"

    @property
    def batch_size(self):
        return self.batched_tensor.shape[0]

    @property
    def max_num_points(self):
        return self.batched_tensor.shape[1]

    @property
    def is_cat(self):
        return False

    @property
    def is_pad(self):
        return True

    def __getitem__(self, idx: int) -> Float[Tensor, "N C"]:  # noqa: F722,F821
        return self.batched_tensor[idx]

    def to(self, device: str) -> "PadFeatures":
        return PadFeatures(
            batched_tensor=self.batched_tensor.to(device),
            offsets=self.offsets.to(device),
            pad_multiple=self.pad_multiple,
        )

    def equal_shape(self, value: object) -> bool:
        if not isinstance(value, PadFeatures):
            return False
        return (
            self.offsets == value.offsets
        ).all() and self.batched_tensor.shape == value.batched_tensor.shape

    def equal_rigorous(self, value: "PadFeatures") -> bool:
        if not isinstance(value, PadFeatures):
            return False
        return (
            self.equal_shape(value)
            and (self.batched_tensor == value.batched_tensor).all()
        )

    def to_cat(self) -> "CatFeatures":  # noqa: F821
        return pad_to_cat(self)

    @classmethod
    def from_cat(
        cls,
        batched_object: "CatFeatures",
        pad_multiplier: Optional[int] = None,  # noqa: F821
    ) -> "PadFeatures":  # noqa: F821
        return cat_to_pad(batched_object, pad_multiplier)

    def clear_padding(self, clear_value: float = 0.0) -> "PadFeatures":  # noqa: F821
        """
        Clear the padded part of the tensor
        """
        num_points = self.offsets.diff()
        for i in range(self.batch_size):
            self.batched_tensor[i, num_points[i] :, :] = clear_value
        return self

    def replace(
        self,
        batched_tensor: Optional[Float[Tensor, "B M C"]] = None,
        offsets: Optional[Int[Tensor, "B+1"]] = None,  # noqa: F821
        pad_multiple: Optional[int] = None,
        **kwargs,
    ):
        batched_tensor = (
            batched_tensor if batched_tensor is not None else self.batched_tensor
        )
        if pad_multiple is not None:
            # pad the tensor to the same multiple as the original tensor
            new_num_points = (
                (batched_tensor.shape[1] + pad_multiple - 1)
                // pad_multiple
                * pad_multiple
            )
            if new_num_points > batched_tensor.shape[1]:
                batched_tensor = F.pad(
                    batched_tensor, (0, 0, 0, new_num_points - batched_tensor.shape[1])
                )
        return self.__class__(
            batched_tensor=batched_tensor,
            offsets=(offsets if offsets is not None else self.offsets),
            pad_multiple=pad_multiple,
            **kwargs,
        )
