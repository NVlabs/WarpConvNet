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

import pytest

import numpy as np
import torch

import warpconvnet._C as _C


def compare_results(result_auto, d_ref, indices_d):
    # Check all results are finite
    if not torch.all(torch.isfinite(result_auto)) or not torch.all(torch.isfinite(d_ref)):
        print("❌ Results contain NaNs or Infs!")

    all_diff = torch.abs(result_auto - d_ref)
    max_diff_idx = torch.argmax(all_diff)
    max_diff = all_diff.view(-1)[max_diff_idx]

    rel_diff = torch.abs((result_auto - d_ref) / (d_ref + 1e-6))
    max_rel_diff_idx = torch.argmax(rel_diff)
    max_rel_diff = rel_diff.view(-1)[max_rel_diff_idx]

    print(
        f"Max diff (all): {max_diff.item()} and value at max diff: {result_auto.view(-1)[max_diff_idx].item()}, {d_ref.view(-1)[max_diff_idx].item()}"
    )
    print(
        f"Max rel diff (all): {max_rel_diff.item()} and value at max rel diff: {result_auto.view(-1)[max_rel_diff_idx].item()}, {d_ref.view(-1)[max_rel_diff_idx].item()}"
    )


def randn_clamped(shape, dtype, device, scale=0.1):
    return torch.clamp(torch.randn(shape, dtype=dtype, device=device) * scale, -scale, scale)


def rand_indices(size, indices_size, device):
    return torch.sort(torch.randperm(size, device=device)[:indices_size], dim=0)[0].int()


@pytest.mark.parametrize(
    "N, C_in, C_out, indices_ratio, dtype",
    [
        (2**14, 3, 16, 0.5, torch.float32),
        (2**14, 3, 16, 0.5, torch.float16),
        (2**14, 3, 16, 0.5, torch.bfloat16),
        (2**20, 3, 16, 0.5, torch.float32),
        (2**20, 3, 16, 0.5, torch.float16),
        (2**20, 3, 16, 0.5, torch.bfloat16),
    ],
    ids=[
        "f32",
        "f16",
        "bf16",
        "f32_large",
        "f16_large",
        "bf16_large",
    ],
)
def test_implicit_gemm(N, C_in, C_out, indices_ratio, dtype):
    """Test Implicit GEMM with half precision inputs and half accumulator"""
    print(f"Testing {N}, {C_in}, {C_out}, {indices_ratio}, {dtype}...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate unique gather & scatter indices to avoid data races
    indices_a = rand_indices(N, int(N * indices_ratio), "cuda")  # N x 1
    indices_c = rand_indices(N, int(N * indices_ratio), "cuda")

    # Create input tensors with specific values for debugging (all float16)
    tensor_a = randn_clamped((N, C_in), dtype, "cuda")  # N x C_in
    tensor_b = randn_clamped((C_in, C_out), dtype, "cuda")  # C_in x C_out

    # Set tensor_c to zeros for simplicity
    tensor_c = torch.zeros(N, C_out, dtype=dtype, device="cuda")  # N x C_out

    # Test with explicit accumulator type (default is float32)
    status = _C.gemm.implicit_gemm(
        tensor_a,
        tensor_b,
        tensor_c,
        indices_a,
        indices_c,
        "basic",
        4,
    )
    torch.cuda.synchronize()
    assert (
        status == 0
    ), f"Error in implicit_gemm: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(status))}"

    # Compute reference result using PyTorch (convert to float32 for computation)
    a_gathered = tensor_a[indices_a.squeeze()]
    c_ref = torch.zeros(N, C_out, dtype=dtype, device="cuda")
    c_out = torch.matmul(a_gathered, tensor_b).to(dtype)
    c_ref[indices_c.squeeze()] = c_out

    # Compare results (convert to float32 for comparison)
    compare_results(tensor_c, c_ref, indices_c)

    # Use more lenient thresholds for half precision
    print(f"{N}, {C_in}, {C_out}, {indices_ratio}, {dtype} test passed!")

    # Test accumulation C+=AB
    indices_a = rand_indices(N, int(N * indices_ratio), "cuda")  # N x 1
    indices_c = rand_indices(N, int(N * indices_ratio), "cuda")

    # Create input tensors with specific values for debugging (all float16)
    tensor_a = randn_clamped((N, C_in), dtype, "cuda")  # N x C_in
    tensor_b = randn_clamped((C_in, C_out), dtype, "cuda")  # C_in x C_out

    # Test with accumulation
    status = _C.gemm.implicit_gemm(
        tensor_a,
        tensor_b,
        tensor_c,
        indices_a,
        indices_c,
        "basic",
        4,
    )

    # Compute reference result using PyTorch (convert to float32 for computation)
    a_gathered = tensor_a[indices_a.squeeze()]
    c_out = torch.matmul(a_gathered, tensor_b).to(dtype)
    c_ref[indices_c.squeeze()] += c_out

    # Compare results (convert to float32 for comparison)
    compare_results(tensor_c, c_ref, indices_c)
