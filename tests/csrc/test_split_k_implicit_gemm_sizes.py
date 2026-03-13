# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stress-test split_k_implicit_gemm across a wide range of (N, C_a, C_b, K)
to catch illegal memory accesses, allocation failures, and correctness bugs.

Run:
    pytest tests/csrc/test_split_k_implicit_gemm_sizes.py -v
    pytest tests/csrc/test_split_k_implicit_gemm_sizes.py -v -k "fp16"
"""

import pytest
import torch

import warpconvnet._C as _C

from ..common import compare_results, rand_clamped, rand_indices


# ---------------------------------------------------------------------------
# Parametrize over a broad range of shapes that exercise edge cases:
#   - K < split_k_factor  (single split, no temp allocation)
#   - K = 1               (degenerate)
#   - C large             (512 — matches hidden_dim in training config)
#   - N small / large
#   - N != K              (partial index selection)
# ---------------------------------------------------------------------------

_SIZES = [
    # (N, C_a, C_b, K) — K is number of active index pairs
    # Edge: very small K (fewer than split_k_factor=4)
    (64, 8, 8, 1),
    (64, 8, 8, 2),
    (64, 8, 8, 3),
    (64, 8, 8, 4),
    # Edge: K == split_k_factor boundary
    (128, 16, 16, 4),
    (128, 16, 16, 5),
    # Small N, varying C
    (32, 4, 4, 16),
    (32, 16, 16, 16),
    (32, 32, 32, 16),
    (32, 64, 64, 32),
    (32, 128, 128, 32),
    (32, 256, 256, 32),
    # Medium N, large C (matches training hidden_dim)
    (256, 256, 256, 128),
    (256, 384, 256, 128),
    (256, 512, 512, 128),
    (256, 512, 384, 256),
    # Large N (typical point cloud sizes)
    (4096, 64, 64, 2048),
    (4096, 128, 128, 2048),
    (4096, 256, 256, 4096),
    (4096, 512, 512, 2048),
    # Very large N
    (16384, 128, 128, 8192),
    (16384, 512, 512, 8192),
    # Asymmetric C_a != C_b
    (256, 64, 128, 128),
    (256, 128, 64, 128),
    (256, 256, 512, 128),
    (256, 512, 256, 128),
    # K == N (all indices active)
    (128, 64, 64, 128),
    (512, 256, 256, 512),
    # K == 1 with large C (degenerate single-pair)
    (1024, 256, 256, 1),
    (1024, 512, 512, 1),
]

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_SPLIT_K_FACTORS = [1, 2, 4, 8]
_BLOCK_SIZES = [128, 256]


def _make_id(N, C_a, C_b, K, dtype, split_k, block):
    dt = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
    return f"N{N}_Ca{C_a}_Cb{C_b}_K{K}_{dt}_sk{split_k}_bs{block}"


@pytest.mark.parametrize(
    "N, C_a, C_b, K, dtype, split_k_factor, block_size",
    [
        (*size, dt, sk, bs)
        for size in _SIZES
        for dt in _DTYPES
        for sk in _SPLIT_K_FACTORS
        for bs in _BLOCK_SIZES
    ],
    ids=[
        _make_id(*size, dt, sk, bs)
        for size in _SIZES
        for dt in _DTYPES
        for sk in _SPLIT_K_FACTORS
        for bs in _BLOCK_SIZES
    ],
)
def test_split_k_sizes(N, C_a, C_b, K, dtype, split_k_factor, block_size):
    """Test split_k_implicit_gemm: C += transpose(A[indices_a]) @ B[indices_b]"""
    torch.manual_seed(42)
    device = "cuda"

    # Generate random index pairs
    indices_a = rand_indices(N, K, device)
    indices_b = rand_indices(N, K, device)

    # Scale down inputs for numerical stability in half precision
    scale = 0.01 if dtype == torch.float32 else 0.005
    tensor_a = rand_clamped((N, C_a), dtype, device, scale=scale)
    tensor_b = rand_clamped((N, C_b), dtype, device, scale=scale)

    # Output: C_a x C_b, initialized to small random values to test accumulation
    tensor_c = rand_clamped((C_a, C_b), dtype, device, scale=scale)
    tensor_c_orig = tensor_c.clone()

    # Run kernel
    _C.gemm.split_k_implicit_gemm(
        tensor_a,
        tensor_b,
        tensor_c,
        indices_a,
        indices_b,
        split_k_factor=split_k_factor,
        block_size=block_size,
    )
    torch.cuda.synchronize()

    # Reference: C += A[indices_a].T @ B[indices_b]
    a_gathered = tensor_a[indices_a.long()]  # K x C_a
    b_gathered = tensor_b[indices_b.long()]  # K x C_b
    c_ref = tensor_c_orig.float() + a_gathered.float().T @ b_gathered.float()
    c_ref = c_ref.to(dtype)

    # Compare
    max_abs, max_rel = compare_results(tensor_c, c_ref, verbose=False)

    if dtype == torch.float32:
        abs_tol, rel_tol = 1e-3, 1e-2
    elif dtype == torch.float16:
        abs_tol, rel_tol = 5e-2, 2e-1
    else:  # bfloat16
        abs_tol, rel_tol = 1.0, 1.0

    # Scale tolerance with K — accumulation error grows with more additions
    abs_tol *= max(1.0, (K / 256) ** 0.5)

    assert max_abs < abs_tol, (
        f"Abs diff {max_abs:.4e} > tol {abs_tol:.4e} "
        f"(N={N}, C_a={C_a}, C_b={C_b}, K={K}, {dtype}, sk={split_k_factor}, bs={block_size})"
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp16", "bf16"])
def test_split_k_no_crash_training_config(dtype):
    """Reproduce the training configuration that triggered the original crash:
    hidden_dim=512, dec_channels=[512,384,256,256], 16-mixed precision.
    """
    torch.manual_seed(777)
    device = "cuda"

    # Simulate multiple backward passes with different layer shapes
    layer_configs = [
        # (N_in, M_out, C_in, C_out, num_offsets)
        (8192, 4096, 512, 512, 27),  # encoder layer
        (4096, 2048, 512, 384, 27),
        (2048, 1024, 384, 256, 27),
        (1024, 512, 256, 256, 27),  # decoder layer
    ]

    for N_in, M_out, C_in, C_out, num_offsets in layer_configs:
        scale = 0.005
        in_features = rand_clamped((N_in, C_in), dtype, device, scale=scale)
        grad_output = rand_clamped((M_out, C_out), dtype, device, scale=scale)
        grad_weight = torch.zeros((C_in, C_out), dtype=dtype, device=device)

        for k in range(num_offsets):
            # Random subset of active pairs per offset
            num_pairs = torch.randint(1, min(N_in, M_out), (1,)).item()
            in_map = rand_indices(N_in, num_pairs, device)
            out_map = rand_indices(M_out, num_pairs, device)

            _C.gemm.split_k_implicit_gemm(
                in_features,
                grad_output,
                grad_weight,
                in_map,
                out_map,
                split_k_factor=4,
                block_size=256,
            )

        torch.cuda.synchronize()

        # Basic sanity: result should be finite
        assert torch.all(torch.isfinite(grad_weight)), (
            f"Non-finite values in grad_weight for layer "
            f"({N_in}, {M_out}, {C_in}, {C_out}), dtype={dtype}"
        )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp16", "bf16"])
def test_split_k_repeated_calls_same_output(dtype):
    """Multiple kernel offsets accumulating into the same output tensor,
    mimicking the backward loop in _implicit_gemm_backward_logic."""
    torch.manual_seed(123)
    device = "cuda"
    N, C_a, C_b = 2048, 256, 256
    num_offsets = 10

    tensor_a = rand_clamped((N, C_a), dtype, device, scale=0.005)
    tensor_b = rand_clamped((N, C_b), dtype, device, scale=0.005)
    tensor_c = torch.zeros((C_a, C_b), dtype=dtype, device=device)
    c_ref = torch.zeros((C_a, C_b), dtype=torch.float32, device=device)

    for _ in range(num_offsets):
        K = torch.randint(1, N, (1,)).item()
        indices_a = rand_indices(N, K, device)
        indices_b = rand_indices(N, K, device)

        _C.gemm.split_k_implicit_gemm(
            tensor_a,
            tensor_b,
            tensor_c,
            indices_a,
            indices_b,
            split_k_factor=4,
            block_size=256,
        )

        # Reference accumulation in float32
        a_g = tensor_a[indices_a.long()].float()
        b_g = tensor_b[indices_b.long()].float()
        c_ref += a_g.T @ b_g

    torch.cuda.synchronize()

    max_abs, _ = compare_results(tensor_c, c_ref.to(dtype), verbose=False)

    if dtype == torch.float32:
        assert max_abs < 1e-1, f"Accumulated abs diff {max_abs:.4e} too large"
    elif dtype == torch.float16:
        assert max_abs < 2.0, f"Accumulated abs diff {max_abs:.4e} too large"
    else:
        assert max_abs < 5.0, f"Accumulated abs diff {max_abs:.4e} too large"
