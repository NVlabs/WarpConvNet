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

from ..common import compare_results, rand_clamped, rand_indices


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
def test_wmma_split_k_implicit_gemm(N, C_a, C_b, indices_ratio, dtype):
    """Test WMMA Split-K Implicit GEMM: C += transpose(A[indices_a]) @ B[indices_b]"""
    print(f"Testing WMMA {N}, {C_a}, {C_b}, {indices_ratio}, {dtype}...")

    torch.manual_seed(42)
    np.random.seed(42)

    K = int(N * indices_ratio)

    indices_a = rand_indices(N, K, "cuda")
    indices_b = rand_indices(N, K, "cuda")

    scale = 0.01 if dtype == torch.float32 else 0.005
    tensor_a = rand_clamped((N, C_a), dtype, "cuda", scale=scale)
    tensor_b = rand_clamped((N, C_b), dtype, "cuda", scale=scale)

    # For float32 inputs, kernel converts A/B to f16 and accumulates into f32 C
    c_dtype = torch.float32 if dtype == torch.float32 else dtype
    tensor_c = torch.zeros((C_a, C_b), dtype=c_dtype, device="cuda")
    tensor_c_original = tensor_c.clone()

    print(f"Before kernel: C sum = {tensor_c.sum().item():.6f}")
    print(f"A range: [{tensor_a.min().item():.6f}, {tensor_a.max().item():.6f}]")
    print(f"B range: [{tensor_b.min().item():.6f}, {tensor_b.max().item():.6f}]")
    print(f"Indices A: {indices_a[:5]} ... {indices_a[-5:]}")
    print(f"Indices B: {indices_b[:5]} ... {indices_b[-5:]}")

    status = _C.gemm.wmma_split_k_implicit_gemm_sm80(
        tensor_a,
        tensor_b,
        tensor_c,
        indices_a,
        indices_b,
        split_k_factor=4,
    )
    torch.cuda.synchronize()
    assert status == 0, f"Error in wmma_split_k_implicit_gemm_sm80: status {status}"

    print(f"After kernel: C sum = {tensor_c.sum().item():.6f}")
    print(f"C range: [{tensor_c.min().item():.6f}, {tensor_c.max().item():.6f}]")

    a_gathered = tensor_a[indices_a.squeeze()]
    b_gathered = tensor_b[indices_b.squeeze()]
    c_ref = tensor_c_original + torch.matmul(a_gathered.T, b_gathered)

    # Cast reference to match output dtype
    c_ref = c_ref.to(tensor_c.dtype)

    max_abs_diff, max_rel_diff = compare_results(tensor_c, c_ref, verbose=False)

    if tensor_c.dtype == torch.float32:
        # WMMA path multiplies with half-precision inputs; allow slightly higher tolerance
        abs_tol, rel_tol = 5e-2, 2e-3
    elif tensor_c.dtype == torch.float16:
        abs_tol, rel_tol = 1e-2, 1e-1
    else:  # bfloat16
        abs_tol, rel_tol = 5e-1, 5e-1

    assert (
        max_abs_diff < abs_tol
    ), f"Max absolute difference {max_abs_diff} exceeds tolerance {abs_tol}"
    assert (
        max_rel_diff < rel_tol
    ), f"Max relative difference {max_rel_diff} exceeds tolerance {rel_tol}"

    print(
        f"WMMA {N}, {C_a}, {C_b}, {indices_ratio}, {dtype} test passed! Max abs diff: {max_abs_diff:.6f}, Max rel diff: {max_rel_diff:.6f}"
    )
