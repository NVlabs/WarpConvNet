# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for CuTe 3.x GEMM with gather/scatter via layout composition.

import pytest
import torch

try:
    import warpconvnet._C as _C

    HAS_CUTE_GEMM = hasattr(_C.gemm, "cute_gemm_AD_gather_scatter")
except ImportError:
    HAS_CUTE_GEMM = False

pytestmark = pytest.mark.skipif(
    not HAS_CUTE_GEMM, reason="CuTe GEMM bindings not available"
)


# ---------------------------------------------------------------------------
# Test 1: AD Gather-Scatter Correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [0, 1, 2, 3])
def test_cute_gemm_ad_gather_scatter(dtype, tile):
    """D[out_map] = alpha * A[in_map] @ B + beta * C[out_map]"""
    M, K, N, idx_size = 4096, 128, 128, 2048
    A = torch.randn(M, K, dtype=dtype, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=dtype, device="cuda") * 0.1
    idx_a = torch.randperm(M, device="cuda")[:idx_size].int()
    idx_d = torch.randperm(M, device="cuda")[:idx_size].int()
    D = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    status = _C.gemm.cute_gemm_AD_gather_scatter(
        A, B, D, D, idx_a, idx_d, mma_tile=tile, alpha=1.0, beta=0.0
    )
    assert status == 0, f"Kernel returned status {status}"

    # Reference
    D_ref = torch.zeros_like(D)
    D_ref[idx_d] = A[idx_a].float() @ B.float()
    torch.testing.assert_close(D, D_ref, atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 1b: AD Gather-Scatter with fp16/bf16 output (16-true mode)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [0, 3])
def test_cute_gemm_ad_gather_scatter_fp16_output(dtype, tile):
    """D[out_map] = alpha * A[in_map] @ B with C/D in fp16/bf16."""
    M, K, N, idx_size = 4096, 128, 128, 2048
    A = torch.randn(M, K, dtype=dtype, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=dtype, device="cuda") * 0.1
    idx_a = torch.randperm(M, device="cuda")[:idx_size].int()
    idx_d = torch.randperm(M, device="cuda")[:idx_size].int()
    D = torch.zeros(M, N, dtype=dtype, device="cuda")

    status = _C.gemm.cute_gemm_AD_gather_scatter(
        A, B, D, D, idx_a, idx_d, mma_tile=tile, alpha=1.0, beta=0.0
    )
    assert status == 0, f"Kernel returned status {status}"

    # Reference in float32
    D_ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    D_ref[idx_d] = A[idx_a].float() @ B.float()
    torch.testing.assert_close(D.float(), D_ref, atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 2: AD Gather-Scatter with beta accumulation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("tile", [3])  # Tile64x64x32
def test_cute_gemm_ad_gather_scatter_beta(dtype, tile):
    """Test alpha/beta linear combination: D = alpha*A[in_map]@B + beta*C"""
    M, K, N, idx_size = 2048, 64, 64, 1024
    A = torch.randn(M, K, dtype=dtype, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=dtype, device="cuda") * 0.1
    idx_a = torch.randperm(M, device="cuda")[:idx_size].int()
    idx_d = torch.randperm(M, device="cuda")[:idx_size].int()

    C = torch.randn(M, N, dtype=torch.float32, device="cuda") * 0.1
    D = C.clone()

    alpha, beta = 0.5, 0.3
    status = _C.gemm.cute_gemm_AD_gather_scatter(
        A, B, C, D, idx_a, idx_d, mma_tile=tile, alpha=alpha, beta=beta
    )
    assert status == 0

    D_ref = C.clone()
    D_ref[idx_d] = alpha * (A[idx_a].float() @ B.float()) + beta * C[idx_d]
    torch.testing.assert_close(D, D_ref, atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 3: Cross-Validation vs CUTLASS 2.x
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tile", [0, 1, 2, 3])
def test_cute_vs_2x_cross_validate(tile):
    """Both backends must produce similar results on same inputs."""
    M, K, N, idx_size = 2048, 64, 64, 1024

    # Alignment check
    if K % 8 != 0 or N % 8 != 0:
        pytest.skip("K or N not aligned to 8")

    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.1
    idx_a = torch.randperm(M, device="cuda")[:idx_size].int()
    idx_d = torch.randperm(M, device="cuda")[:idx_size].int()

    D_2x = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    D_cute = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    status_2x = _C.gemm.cutlass_gemm_AD_gather_scatter(
        A, B, D_2x, D_2x, idx_a, idx_d, mma_tile=tile
    )
    status_cute = _C.gemm.cute_gemm_AD_gather_scatter(
        A, B, D_cute, D_cute, idx_a, idx_d, mma_tile=tile
    )

    if status_2x != 0:
        pytest.skip(f"CUTLASS 2.x returned status {status_2x}")
    assert status_cute == 0, f"CuTe returned status {status_cute}"

    torch.testing.assert_close(D_2x, D_cute, atol=1e-2, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test 4: TrAB Gather Correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [0, 1, 2, 3])
def test_cute_gemm_trAB_gather(dtype, tile):
    """D[k,n] = alpha * A[idx_a]^T @ B[idx_b] + beta * C[k,n]"""
    M_A, K, M_B, N, idx_size = 4096, 64, 4096, 128, 2048
    A = torch.randn(M_A, K, dtype=dtype, device="cuda") * 0.1
    B = torch.randn(M_B, N, dtype=dtype, device="cuda") * 0.1
    idx_a = torch.randperm(M_A, device="cuda")[:idx_size].int()
    idx_b = torch.randperm(M_B, device="cuda")[:idx_size].int()
    D = torch.zeros(K, N, dtype=torch.float32, device="cuda")

    status = _C.gemm.cute_gemm_trAB_gather(
        A, B, D, D, idx_a, idx_b, mma_tile=tile, alpha=1.0, beta=0.0
    )
    assert status == 0, f"Kernel returned status {status}"

    # Reference
    D_ref = torch.zeros_like(D)
    D_ref = A[idx_a].float().T @ B[idx_b].float()
    torch.testing.assert_close(D, D_ref, atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 4b: TrAB Gather with beta accumulation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("tile", [3])
def test_cute_gemm_trAB_gather_beta(dtype, tile):
    """Test alpha/beta: D = alpha * A[idx_a]^T @ B[idx_b] + beta * C"""
    M_A, K, M_B, N, idx_size = 2048, 64, 2048, 64, 1024
    A = torch.randn(M_A, K, dtype=dtype, device="cuda") * 0.1
    B = torch.randn(M_B, N, dtype=dtype, device="cuda") * 0.1
    idx_a = torch.randperm(M_A, device="cuda")[:idx_size].int()
    idx_b = torch.randperm(M_B, device="cuda")[:idx_size].int()

    C = torch.randn(K, N, dtype=torch.float32, device="cuda") * 0.1
    D = C.clone()

    alpha, beta = 0.5, 0.3
    status = _C.gemm.cute_gemm_trAB_gather(
        A, B, C, D, idx_a, idx_b, mma_tile=tile, alpha=alpha, beta=beta
    )
    assert status == 0

    D_ref = alpha * (A[idx_a].float().T @ B[idx_b].float()) + beta * C
    torch.testing.assert_close(D, D_ref, atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 4c: TrAB Gather with fp16/bf16 output (16-true mode)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [0, 3])
def test_cute_gemm_trAB_gather_fp16_output(dtype, tile):
    """D[k,n] = alpha * A[idx_a]^T @ B[idx_b] with C/D in fp16/bf16."""
    M_A, K, M_B, N, idx_size = 4096, 64, 4096, 128, 2048
    A = torch.randn(M_A, K, dtype=dtype, device="cuda") * 0.1
    B = torch.randn(M_B, N, dtype=dtype, device="cuda") * 0.1
    idx_a = torch.randperm(M_A, device="cuda")[:idx_size].int()
    idx_b = torch.randperm(M_B, device="cuda")[:idx_size].int()
    D = torch.zeros(K, N, dtype=dtype, device="cuda")

    status = _C.gemm.cute_gemm_trAB_gather(
        A, B, D, D, idx_a, idx_b, mma_tile=tile, alpha=1.0, beta=0.0
    )
    assert status == 0, f"Kernel returned status {status}"

    D_ref = A[idx_a].float().T @ B[idx_b].float()
    torch.testing.assert_close(D.float(), D_ref, atol=1e-1, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 5: Edge Cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "M,K,N,idx",
    [
        (256, 64, 64, 1),  # single row
        (256, 64, 64, 7),  # odd number of indices
        (128, 8, 128, 100),  # minimum K for alignment=8
        (4096, 128, 128, 4096),  # full gather (all rows distinct)
    ],
)
def test_cute_gemm_edge_cases(M, K, N, idx):
    """Edge cases for AD gather-scatter."""
    if K < 8:
        pytest.skip("K < 8 alignment requirement")

    A = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
    B = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.1
    idx_a = torch.randperm(M, device="cuda")[:idx].int()
    idx_d = torch.randperm(M, device="cuda")[:idx].int()
    D = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    status = _C.gemm.cute_gemm_AD_gather_scatter(
        A, B, D, D, idx_a, idx_d, mma_tile=3, alpha=1.0, beta=0.0
    )
    assert status == 0

    D_ref = torch.zeros_like(D)
    D_ref[idx_d] = A[idx_a].float() @ B.float()
    torch.testing.assert_close(D, D_ref, atol=1e-1, rtol=1e-2)
