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
from typing import Optional, Tuple, Any
from jaxtyping import Float, Int

import torch
from torch import Tensor
from torch.autograd import Function

from torch_scatter import segment_csr
import warpconvnet._C as _C


class SegmentedLayerNormFunction(Function):
    """
    Custom autograd function for segmented layer normalization (core normalization only).

    This function implements both forward and backward passes for the core normalization:
    (x - mean) / std, without gamma and beta scaling/bias parameters.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, offsets: Tensor, eps: float = 1e-5) -> Tensor:
        """
        Forward pass for segmented layer normalization (core normalization only).

        Args:
            ctx: Context for saving tensors needed in backward pass
            x: Input tensor of shape (N, D)
            offsets: Segment boundaries of shape (K+1,)
            eps: Epsilon for numerical stability

        Returns:
            Normalized tensor of shape (N, D) with mean=0, std=1 per segment
        """
        N, D = x.shape
        K = offsets.shape[0] - 1

        # Convert offsets to appropriate types
        offsets = offsets.to(dtype=torch.int64)  # For segment_csr
        d_offsets = offsets.to(x.device)

        # Compute mean and variance using segmented reduction
        mean = segment_csr(x, d_offsets, reduce="mean")  # Shape: (K, D)
        x_squared = x * x
        mean_squared = segment_csr(x_squared, d_offsets, reduce="mean")  # Shape: (K, D)
        variance = mean_squared - mean * mean  # Shape: (K, D)

        # Compute standard deviation
        std = torch.sqrt(variance + eps)  # Shape: (K, D)

        # Normalize: (x - mean) / std
        output = torch.zeros_like(x)

        # Subtract mean from each element
        _C.utils.segmented_arithmetic(x, mean, output, d_offsets, "subtract")

        # Divide by standard deviation
        _C.utils.segmented_arithmetic(output, std, output, d_offsets, "divide")

        # Save tensors for backward pass
        ctx.save_for_backward(x, mean, std, d_offsets)
        ctx.eps = eps
        ctx.N = N
        ctx.D = D
        ctx.K = K

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass for segmented layer normalization (core normalization only).

        Treats mean and std as constants (detached from gradient computation).
        This simplifies the backward pass significantly.

        Args:
            ctx: Context containing saved tensors
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradients w.r.t. (x, offsets, eps)
        """
        x, mean, std, d_offsets = ctx.saved_tensors

        # Detach mean and std to treat them as constants
        mean = mean.detach()
        std = std.detach()

        grad_x = None

        # Gradient w.r.t. x (simplified with mean and std as constants)
        if ctx.needs_input_grad[0]:
            # With mean and std as constants, the gradient simplifies to:
            # grad_x = grad_output / std (broadcast std to segments)
            grad_x = torch.zeros_like(x)
            _C.utils.segmented_arithmetic(grad_output, std, grad_x, d_offsets, "divide")

        # Return gradients in the same order as forward inputs
        # (x, offsets, eps)
        return grad_x, None, None


def segmented_norm(x: Tensor, offsets: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Segmented normalization.
    """
    return SegmentedLayerNormFunction.apply(x, offsets, eps)  # type: ignore[assignment]


# LayerNorm
def segmented_layer_norm(
    x: Float[Tensor, "N D"],
    offsets: Int[Tensor, "K+1"],
    gamma: Optional[Float[Tensor, "K D"]] = None,
    beta: Optional[Float[Tensor, "K D"]] = None,
    eps: float = 1e-5,
) -> Float[Tensor, "N D"]:
    r"""
    Layer normalization on segmented data.

    This is a segmented reduction of the form:

    .. math::
        \gamma_k \frac{x_i - \mu_k}{\sigma_k + \epsilon} + \beta_k

    where :math:`\mu_k` and :math:`\sigma_k` are the mean and standard deviation of the :math:`k`-th segment,
    and :math:`\gamma_k` and :math:`\beta_k` are optional learnable parameters for the :math:`k`-th segment.

    Args:
        x: Input tensor of shape (N, D)
        offsets: Segment boundaries of shape (K+1,) where K is the number of segments
        gamma: Optional learnable scale parameters of shape (D,)
        beta: Optional learnable bias parameters of shape (D,)
        eps: Epsilon value for numerical stability

    Returns:
        Normalized tensor of shape (N, D)
    """
    # Apply core normalization using the autograd function
    normalized: Tensor = segmented_norm(x, offsets, eps)  # type: ignore[assignment]

    if gamma is not None and beta is not None:
        normalized = torch.addcmul(beta, gamma, normalized)
    elif gamma is not None:
        normalized = torch.mul(gamma.unsqueeze(0), normalized)
    elif beta is not None:
        normalized = torch.add(beta.unsqueeze(0), normalized)

    return normalized
