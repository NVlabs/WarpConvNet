# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Sequence
from jaxtyping import Float, Int
from dataclasses import dataclass

import torch
from torch import Tensor

from warpconvnet.geometry.utils.list_to_batch import list_to_cat_tensor


@dataclass
class BatchedTensor:
    """A class for handling batched tensors with variable sizes.

    This class provides a way to store and manipulate tensors of varying sizes in a batch,
    using a concatenated tensor and offset indices to track individual tensor boundaries similar to torch.nested.nested_tensor. However, this class provides a more flexible interface for operations and is dataclass-based to support fully sharded data parallel training.

    Attributes:
        batched_tensor (Float[Tensor, "N C"]): The concatenated tensor containing all batched data
        offsets (Int[Tensor, "B + 1"]): Indices marking the start/end of each tensor in the batch
    """

    batched_tensor: Float[Tensor, "N C"]  # noqa: F722,F821
    offsets: Int[Tensor, "B + 1"]  # noqa: F722,F821

    def __init__(
        self,
        batched_tensor: List[Float[Tensor, "N C"]] | Float[Tensor, "N C"],  # noqa: F722,F821
        offsets: Optional[List[int] | Tensor] = None,
        device: Optional[str] = None,
    ):
        """Initialize a batched object with a list of tensors.

        Args:
            batched_tensor: List of tensors to batch or a single pre-concatenated tensor.
                If providing a concatenated tensor, offsets must also be provided.
            offsets: List of indices marking the boundaries between tensors in the batch.
                Required if batched_tensor is a pre-concatenated tensor.
            device: Target device for the batched tensor. If None, uses the input tensor's device.

        Raises:
            AssertionError: If invalid combination of inputs is provided or if tensor format is incorrect.
        """
        if isinstance(batched_tensor, Sequence):
            assert offsets is None, "If batched_tensors is a list, offsets must be None."
            batched_tensor, offsets, _ = list_to_cat_tensor(batched_tensor)
        else:
            assert isinstance(
                batched_tensor, torch.Tensor
            ), "Batched tensor must be a tensor or a list"
            if offsets is None:
                offsets = [0, batched_tensor.shape[0]]

        if isinstance(offsets, Sequence):
            offsets = torch.LongTensor(offsets)
        elif isinstance(offsets, Tensor):
            offsets = offsets.cpu()
        else:
            raise ValueError(f"Invalid offsets type {type(offsets)}")

        # Prevent from triggering __getattribute__ used in GridCoords
        object.__setattr__(self, "offsets", offsets)
        if device is not None:
            batched_tensor = batched_tensor.to(device)
        object.__setattr__(self, "batched_tensor", batched_tensor)

        self.check()

    @property
    def batch_size(self) -> int:
        """Get the number of tensors in the batch.

        Returns:
            int: The batch size
        """
        return len(self.offsets) - 1

    def check(self):
        # offset check
        assert isinstance(
            self.offsets, (torch.IntTensor, torch.LongTensor)
        ), f"Offsets must be a cpu IntTensor or cpu LongTensor, got {self.offsets}"
        assert self.offsets.requires_grad is False, "Offsets must not require grad"
        assert (
            len(self.offsets) == self.batch_size + 1
        ), f"Offsets {self.offsets} does not match batch size {self.batch_size}"
        # batched_tensor check
        assert isinstance(self.batched_tensor, torch.Tensor)

    def _to(self, device: str) -> "BatchedTensor":
        """Internal method to move the batched tensor to the device. This method modifies the current object in place."""
        self.batched_tensor = self.batched_tensor.to(device)
        return self

    def to(
        self,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "BatchedTensor":
        """Move the batched tensor to the specified device. This creates a new BatchedTensor object on the specified device.

        Args:
            device: Target device to move the tensor to

        Returns:
            BatchedTensor: A new BatchedTensor on the specified device
        """
        if device is None:
            device = self.device
        return self.__class__(
            batched_tensor=self.batched_tensor.to(device=device, dtype=dtype),
            offsets=self.offsets,
        )

    @property
    def device(self):
        return self.batched_tensor.device

    @property
    def shape(self):
        return self.batched_tensor.shape

    @property
    def dtype(self):
        return self.batched_tensor.dtype

    def half(self):
        return self.__class__(
            batched_tensor=self.batched_tensor.half(),
            offsets=self.offsets,
        )

    def float(self):
        return self.__class__(
            batched_tensor=self.batched_tensor.float(),
            offsets=self.offsets,
        )

    def double(self):
        return self.__class__(
            batched_tensor=self.batched_tensor.double(),
            offsets=self.offsets,
        )

    def numel(self):
        return self.batched_tensor.numel()

    def __len__(self) -> int:
        return len(self.batched_tensor)

    def __getitem__(self, idx: int) -> Float[Tensor, "N C"]:  # noqa: F722,F821
        """Get a single tensor from the batch.

        Args:
            idx: Index of the tensor to retrieve

        Returns:
            Tensor: The tensor at the specified index

        Raises:
            AssertionError: If idx is not an integer
        """
        assert isinstance(idx, int), "Index must be an integer"
        return self.batched_tensor[self.offsets[idx] : self.offsets[idx + 1]]

    def equal_shape(self, value: "BatchedTensor") -> bool:
        return (self.offsets == value.offsets).all() and self.numel() == value.numel()

    def equal_rigorous(self, value: "BatchedTensor") -> bool:
        if not isinstance(value, BatchedTensor):
            return False
        return self.equal_shape(value) and (self.batched_tensor == value.batched_tensor).all()

    def __eq__(self, value: "BatchedTensor") -> bool:
        """
        Accelerated version that only checks length and offsets
        """
        return self.equal_shape(value)

    def binary_op(self, value: object, op: str) -> "BatchedTensor":
        """Apply a binary operation to the batched tensor.

        Supports operations with scalars and other BatchedTensors of the same shape.

        Args:
            value: The value to operate with (scalar, single-element tensor, or BatchedTensor)
            op: The operation to perform (e.g., "__add__", "__mul__")

        Returns:
            BatchedTensor: Result of the binary operation

        Raises:
            AssertionError: If operating with a BatchedTensor of different shape
        """
        if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
            return self.__class__(
                batched_tensor=getattr(self.batched_tensor, op)(value),
                offsets=self.offsets,
            )

        assert self.equal_shape(value)
        return self.__class__(
            batched_tensor=getattr(self.batched_tensor, op)(value.batched_tensor),
            offsets=self.offsets,
        )

    def __add__(self, value: object) -> "BatchedTensor":
        return self.binary_op(value, "__add__")

    def __sub__(self, value: object) -> "BatchedTensor":
        return self.binary_op(value, "__sub__")

    def __mul__(self, value: object) -> "BatchedTensor":
        return self.binary_op(value, "__mul__")

    def __truediv__(self, value: object) -> "BatchedTensor":
        return self.binary_op(value, "__truediv__")

    def __floordiv__(self, value: object) -> "BatchedTensor":
        return self.binary_op(value, "__floordiv__")

    def __mod__(self, value: object) -> "BatchedTensor":
        return self.binary_op(value, "__mod__")

    def __pow__(self, value: object) -> "BatchedTensor":
        return self.binary_op(value, "__pow__")

    def __str__(self) -> str:
        """Short representation of the object."""
        return (
            f"{self.__class__.__name__}(offsets={self.offsets}, shape={self.batched_tensor.shape})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the object."""
        return f"{self.__class__.__name__}(offsets={self.offsets}, shape={self.batched_tensor.shape}, device={self.device}, dtype={self.dtype})"

    def to_nested(self) -> torch.Tensor:
        """Convert to a PyTorch nested tensor.

        Returns:
            torch.Tensor: A nested tensor containing the same data
        """
        return torch.nested.nested_tensor(
            [self[i] for i in range(self.batch_size)],
            requires_grad=self.batched_tensor.requires_grad,
        )

    @classmethod
    def from_nested(cls, nested: torch.Tensor) -> "BatchedTensor":
        """Create a BatchedTensor from a PyTorch nested tensor.

        Args:
            nested: The nested tensor to convert

        Returns:
            BatchedTensor: A new BatchedTensor containing the same data
        """
        rg = nested.requires_grad
        return cls([t.requires_grad_(rg) for t in nested.unbind()])
