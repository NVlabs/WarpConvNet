# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np
import torch

import warpconvnet._C as _C

from ..common import rand_clamped, rand_indices, compare_results


@pytest.mark.parametrize(
    "N, C_in, C_out, indices_ratio, in_dtype, out_dtype",
    [
        (4096, 32, 64, 0.5, torch.float16, torch.float16),
        (4096, 32, 64, 0.5, torch.float16, torch.float32),
        (4096, 32, 64, 0.5, torch.bfloat16, torch.bfloat16),
        (4096, 32, 64, 0.5, torch.bfloat16, torch.float32),
    ],
    ids=["f16->f16", "f16->f32", "bf16->bf16", "bf16->f32"],
)
def test_wmma_implicit_gemm_sm80(N, C_in, C_out, indices_ratio, in_dtype, out_dtype):
    torch.manual_seed(123)
    np.random.seed(123)

    # Problem sizes
    M = N
    K = C_in
    P = int(N * indices_ratio)
    Q = N

    # Indices
    indices_a = rand_indices(M, P, "cuda")
    indices_d = rand_indices(Q, P, "cuda")

    # Inputs
    A = rand_clamped((M, K), in_dtype, "cuda", distribution="normal")
    B = rand_clamped((K, C_out), in_dtype, "cuda", distribution="normal")
    C = torch.zeros((M, C_out), dtype=in_dtype, device="cuda")

    # Output
    D = torch.zeros((Q, C_out), dtype=out_dtype, device="cuda")

    # Run kernel: D[indices_d] = alpha * (A[indices_a] @ B) + beta * C[...] ; here alpha=1, beta=0
    status = _C.gemm.wmma_implicit_gemm_sm80(A, B, C, D, indices_a, indices_d, 1.0, 0.0)
    torch.cuda.synchronize()
    assert status == 0

    # Reference
    a_gathered = A[indices_a]
    prod = torch.matmul(a_gathered.to(torch.float32), B.to(torch.float32))
    prod = prod.to(out_dtype)
    D_ref = torch.zeros_like(D)
    D_ref[indices_d] = prod

    compare_results(D, D_ref)

    # Accumulation test: call again with beta=0, should add a second copy of prod
    status = _C.gemm.wmma_implicit_gemm_sm80(A, B, C, D, indices_a, indices_d, 1.0, 0.0)
    torch.cuda.synchronize()
    assert status == 0
    D_ref[indices_d] += prod
    compare_results(D, D_ref)
