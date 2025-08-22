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
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple

import torch


def rand_clamped(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    scale: float = 0.1,
    distribution: str = "uniform",
) -> torch.Tensor:
    """Uniform [0, scale) or Normal(0, 1) then clamped to [-scale, scale]."""
    if distribution == "uniform":
        return torch.rand(shape, dtype=dtype, device=device) * scale
    elif distribution == "normal":
        return torch.clamp(torch.randn(shape, dtype=dtype, device=device) * scale, -scale, scale)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


def rand_indices(size: int, indices_size: int, device: str) -> torch.Tensor:
    """Sorted unique indices of dtype int (int32)."""
    assert indices_size <= size, "indices_size must be less than or equal to size"
    return torch.sort(torch.randperm(size, device=device)[:indices_size], dim=0)[0].int()


def compare_results(result_auto, d_ref, verbose=True) -> Tuple[float, float]:
    """Verbose compare with finite check, absolute and relative diffs.

    Args:
        result_auto: torch.Tensor
        d_ref: torch.Tensor
        verbose: bool

    Returns:
        max_abs_diff: float
        max_rel_diff: float
    """
    if not torch.all(torch.isfinite(result_auto)) or not torch.all(torch.isfinite(d_ref)):
        print("❌ Results contain NaNs or Infs!")

    abs_diff = torch.abs(result_auto.to(torch.float32) - d_ref.to(torch.float32))
    max_diff_idx = torch.argmax(abs_diff)
    max_diff = abs_diff.view(-1)[max_diff_idx]

    rel_diff = torch.abs(abs_diff / (d_ref.to(torch.float32) + 1e-6))
    max_rel_diff_idx = torch.argmax(rel_diff)
    max_rel_diff = rel_diff.view(-1)[max_rel_diff_idx]

    if verbose:
        print(
            f"Max abs diff (all): {max_diff.item()} and value at max abs diff: {result_auto.view(-1)[max_diff_idx].item()}, {d_ref.view(-1)[max_diff_idx].item()}"
        )
        print(
            f"Max rel diff (all): {max_rel_diff.item()} and value at max rel diff: {result_auto.view(-1)[max_rel_diff_idx].item()}, {d_ref.view(-1)[max_rel_diff_idx].item()}"
        )

    return max_diff, max_rel_diff
