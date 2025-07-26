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


def rand_clamped(shape, dtype, device, scale=0.1):
    return torch.rand(shape, dtype=dtype, device=device) * scale


def rand_indices(size, indices_size, device):
    return torch.sort(torch.randperm(size, device=device)[:indices_size], dim=0)[0].int()


@pytest.mark.parametrize(
    "N, C_a, C_b, indices_ratio, dtype",
    [
        (2**14, 3, 16, 0.5, torch.float32),
        (2**14, 3, 16, 0.5, torch.float16),
        (2**14, 3, 16, 0.5, torch.bfloat16),
        (2**20, 3, 16, 0.5, torch.float32),
        (2**20, 3, 16, 0.5, torch.float16),
        (2**20, 3, 16, 0.5, torch.bfloat16),
    ],
    ids=[
        "f32_small",
        "f16_small",
        "bf16_small",
        "f32",
        "f16",
        "bf16",
    ],
)
def test_split_k_implicit_gemm(N, C_a, C_b, indices_ratio, dtype):
    """Test Split-K Implicit GEMM: C += transpose(A[indices_a]) @ B[indices_b]"""
    print(f"Testing {N}, {C_a}, {C_b}, {indices_ratio}, {dtype}...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Number of indices to select
    K = int(N * indices_ratio)

    # Generate unique indices for A and B
    indices_a = rand_indices(N, K, "cuda")  # K indices for A
    indices_b = rand_indices(N, K, "cuda")  # K indices for B

    # Create input tensors with smaller scale for numerical stability
    scale = 0.01 if dtype == torch.float32 else 0.005  # Even smaller for half precision
    tensor_a = rand_clamped((N, C_a), dtype, "cuda", scale=scale)  # N x C_a
    tensor_b = rand_clamped((N, C_b), dtype, "cuda", scale=scale)  # N x C_b

    # Initialize output tensor C to zeros for cleaner testing
    tensor_c = torch.zeros((C_a, C_b), dtype=dtype, device="cuda")
    tensor_c_original = tensor_c.clone()

    # Debug: Check initial values
    print(f"Before kernel: C sum = {tensor_c.sum().item():.6f}")
    print(f"A range: [{tensor_a.min().item():.6f}, {tensor_a.max().item():.6f}]")
    print(f"B range: [{tensor_b.min().item():.6f}, {tensor_b.max().item():.6f}]")
    print(f"Indices A: {indices_a[:5]} ... {indices_a[-5:]}")
    print(f"Indices B: {indices_b[:5]} ... {indices_b[-5:]}")

    # Test split-K implicit GEMM
    status = _C.gemm.split_k_implicit_gemm(
        tensor_a,
        tensor_b,
        tensor_c,
        indices_a,
        indices_b,
        split_k_factor=4,
        block_size=16,
    )
    torch.cuda.synchronize()
    assert status == 0, f"Error in split_k_implicit_gemm: status {status}"

    # Debug: Check after kernel
    print(f"After kernel: C sum = {tensor_c.sum().item():.6f}")
    print(f"C range: [{tensor_c.min().item():.6f}, {tensor_c.max().item():.6f}]")

    # Compute reference result using PyTorch
    # C += transpose(A[indices_a]) @ B[indices_b]
    a_gathered = tensor_a[indices_a.squeeze()]  # K x C_a
    b_gathered = tensor_b[indices_b.squeeze()]  # K x C_b

    # Reference: C_ref = C_original + transpose(A_gathered) @ B_gathered
    c_ref = tensor_c_original + torch.matmul(a_gathered.T, b_gathered)

    # Compare results
    compare_results(tensor_c, c_ref, None)

    # Check numerical accuracy
    max_abs_diff = torch.max(torch.abs(tensor_c - c_ref)).item()
    max_rel_diff = torch.max(torch.abs((tensor_c - c_ref) / (c_ref + 1e-6))).item()

    # Set tolerance based on data type
    if dtype == torch.float32:
        abs_tol, rel_tol = 1e-4, 1e-3
    elif dtype == torch.float16:
        abs_tol, rel_tol = 1e-2, 1e-1
    else:  # bfloat16
        abs_tol, rel_tol = 5e-1, 5e-1  # Very relaxed for bfloat16's 7-bit mantissa + accumulation

    assert (
        max_abs_diff < abs_tol
    ), f"Max absolute difference {max_abs_diff} exceeds tolerance {abs_tol}"
    assert (
        max_rel_diff < rel_tol
    ), f"Max relative difference {max_rel_diff} exceeds tolerance {rel_tol}"

    print(
        f"{N}, {C_a}, {C_b}, {indices_ratio}, {dtype} test passed! Max abs diff: {max_abs_diff:.6f}, Max rel diff: {max_rel_diff:.6f}"
    )
