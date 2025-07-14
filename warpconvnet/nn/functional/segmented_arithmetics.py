# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function

from torch_scatter import segment_csr

import warpconvnet._C as _C


class SegmentedArithmeticFunction(Function):
    """
    Custom autograd function for segmented arithmetic operations.

    This ensures proper gradient flow through our segmented arithmetic operations.
    """

    @staticmethod
    def forward(
        ctx: Any, x: Tensor, y: Tensor, offsets: Tensor, operation: str, eps: float = 1e-5
    ) -> Tensor:
        """
        Forward pass for segmented arithmetic.

        Args:
            ctx: Context for backward pass
            x: Input tensor of shape (N, D)
            y: Segment-wise tensor of shape (K, D)
            offsets: Segment boundaries of shape (K+1,)
            operation: Operation type ("add", "subtract", "multiply", "divide")
            eps: Epsilon value for numerical stability
        Returns:
            Result tensor of shape (N, D)
        """
        # Perform the operation
        output = torch.zeros_like(x)
        _C.utils.segmented_arithmetic(x, y, output, offsets, operation)

        # Save for backward pass
        ctx.save_for_backward(x, y, offsets, eps)
        ctx.operation = operation

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass for segmented arithmetic.

        Returns:
            Gradients w.r.t. (x, y, offsets, operation)
        """
        x, y, offsets, eps = ctx.saved_tensors
        operation = ctx.operation

        grad_x = None
        grad_y = None

        # Gradient w.r.t. x
        if ctx.needs_input_grad[0]:
            if operation == "add":
                grad_x = grad_output
            elif operation == "subtract":
                grad_x = grad_output
            elif operation == "multiply":
                # grad_x = grad_output * y (broadcast y to segments)
                grad_x = torch.zeros_like(x)
                _C.utils.segmented_arithmetic(grad_output, y, grad_x, offsets, "multiply")
            elif operation == "divide":
                # grad_x = grad_output / y (broadcast y to segments)
                grad_x = torch.zeros_like(x)
                _C.utils.segmented_arithmetic(grad_output, y, grad_x, offsets, "divide")

        # Gradient w.r.t. y
        if ctx.needs_input_grad[1]:
            if operation == "add":
                # grad_y = sum(grad_output) per segment
                grad_y = segment_csr(grad_output, offsets.to(torch.int64), reduce="sum")
            elif operation == "subtract":
                # grad_y = -sum(grad_output) per segment
                grad_y_sum = segment_csr(grad_output, offsets.to(torch.int64), reduce="sum")
                grad_y = -grad_y_sum
            elif operation == "multiply":
                # grad_y = sum(grad_output * x) per segment
                grad_y_input = grad_output * x
                grad_y = segment_csr(grad_y_input, offsets.to(torch.int64), reduce="sum")
            elif operation == "divide":
                # grad_y = -sum(grad_output * x / y^2) per segment
                # First compute x / y per segment, then multiply by grad_output
                x_div_y_sq = torch.zeros_like(x)
                y_squared = y * y + eps
                _C.utils.segmented_arithmetic(x, y_squared, x_div_y_sq, offsets, "divide")
                grad_y_input = -grad_output * x_div_y_sq
                grad_y = segment_csr(grad_y_input, offsets.to(torch.int64), reduce="sum")

        return grad_x, grad_y, None, None


def segmented_add(x: Tensor, y: Tensor, offsets: Tensor, eps: float = 1e-5) -> Tensor:
    return SegmentedArithmeticFunction.apply(x, y, offsets, "add", eps)  # type: ignore[return-value]


def segmented_subtract(x: Tensor, y: Tensor, offsets: Tensor, eps: float = 1e-5) -> Tensor:
    return SegmentedArithmeticFunction.apply(x, y, offsets, "subtract", eps)  # type: ignore[return-value]


def segmented_multiply(x: Tensor, y: Tensor, offsets: Tensor, eps: float = 1e-5) -> Tensor:
    return SegmentedArithmeticFunction.apply(x, y, offsets, "multiply", eps)  # type: ignore[return-value]


def segmented_divide(x: Tensor, y: Tensor, offsets: Tensor, eps: float = 1e-5) -> Tensor:
    return SegmentedArithmeticFunction.apply(x, y, offsets, "divide", eps)  # type: ignore[return-value]
