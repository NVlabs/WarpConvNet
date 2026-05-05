# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Thorough correctness tests for the fused RoPE + QKV reshape CUDA kernel
# (`warpconvnet._C.fused_rope.qkv`) and its Python wrappers
# `warpconvnet.nn.functional.fused_rope.fused_rope_qkv` and
# `warpconvnet.nn.modules.rope.VoxelRotaryPositionalEmbeddings`.
#
# Coverage:
#   * Numerical parity vs a pure-PyTorch reference for fwd and bwd
#   * fp32 / fp16 / bf16 dtypes
#   * AMP autocast contexts (fp16 + bf16)
#   * Coord dtypes (int32, int64, float32, float64) and non-contiguous coords
#   * head_dim = (..., 6, 7, 8, 12, 24, 64, 96) including non-multiple-of-6
#   * rope_dim == 0 path (head_dim < 6)
#   * M = 0 edge case, large M, non-contiguous qkv
#   * Negative / large coordinates
#   * R(theta)^T == R(-theta) property (forward(conjugate=1) inverts forward(conjugate=0))

import math

import pytest
import torch

import warpconvnet._C as _C
from warpconvnet.nn.functional.fused_rope import fused_rope_qkv
from warpconvnet.nn.modules.rope import VoxelRotaryPositionalEmbeddings

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Pure-PyTorch reference. Mirrors warpconvnet/csrc/fused_rope_kernel.cu exactly.
# ---------------------------------------------------------------------------
def _torch_rope_ref(
    qkv: torch.Tensor,
    coords: torch.Tensor,
    theta: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    conjugate: int = 0,
) -> torch.Tensor:
    """Reference impl. Inputs ``qkv`` [M,3,C], ``coords`` [M,3] (any numeric).

    Computes math in fp32 and casts back to ``qkv.dtype`` — matches the
    kernel's ``static_cast<float>`` accumulators.
    """
    M, _, C = qkv.shape
    head_dim = C // num_heads
    out = qkv.reshape(M, 3, num_heads, head_dim).clone()
    if rope_dim == 0 or M == 0:
        return out

    coords_f = coords.float()
    coord_min = coords_f.min(dim=0).values
    cx = coords_f[:, 0] - coord_min[0] + 1.0
    cy = coords_f[:, 1] - coord_min[1] + 1.0
    cz = coords_f[:, 2] - coord_min[2] + 1.0

    half_rope = rope_dim // 2
    theta_len = rope_dim // 6

    hr = torch.arange(half_rope, device=qkv.device)
    axis = hr // theta_len
    t_idx = hr % theta_len

    coord_stack = torch.stack([cx, cy, cz], dim=1)  # [M, 3]
    coord_axis = coord_stack[:, axis]  # [M, half_rope]
    angle = coord_axis * theta[t_idx].float()
    cos_v = torch.cos(angle)
    sin_v = torch.sin(angle)
    if conjugate:
        sin_v = -sin_v

    qk = out[:, :2, :, :].float()
    real = qk[..., 0:rope_dim:2]
    imag = qk[..., 1:rope_dim:2]
    cos_b = cos_v.view(M, 1, 1, half_rope)
    sin_b = sin_v.view(M, 1, 1, half_rope)
    new_real = real * cos_b - imag * sin_b
    new_imag = real * sin_b + imag * cos_b
    out[:, :2, :, 0:rope_dim:2] = new_real.to(qkv.dtype)
    out[:, :2, :, 1:rope_dim:2] = new_imag.to(qkv.dtype)
    return out


def _make_theta(rope_dim: int, base: int = 250) -> torch.Tensor:
    """Match `VoxelRotaryPositionalEmbeddings.rope_init`."""
    if rope_dim == 0:
        return torch.zeros(0, dtype=torch.float32, device="cuda")
    return (
        1.0 / (base ** (torch.arange(0, rope_dim // 3, 2, dtype=torch.float32) / (rope_dim // 3)))
    ).cuda()


def _make_inputs(M, num_heads, head_dim, dtype, coord_max=64, coord_dtype=torch.int32, seed=0):
    torch.manual_seed(seed)
    C = num_heads * head_dim
    qkv = torch.randn(M, 3, C, device="cuda", dtype=dtype)
    coords = torch.randint(0, coord_max, (M, 3), device="cuda", dtype=coord_dtype)
    return qkv, coords


def _tol(dtype):
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    if dtype == torch.float16:
        return dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    raise ValueError(dtype)


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "num_heads,head_dim",
    [
        (1, 6),  # smallest rope_dim multiple of 6
        (2, 12),
        (4, 16),  # head_dim not multiple of 6 (rope_dim = 12)
        (8, 24),
        (8, 32),  # head_dim=32, rope_dim=30, pass_dim=2
        (16, 64),
        (32, 96),
    ],
)
@pytest.mark.parametrize("M", [1, 4, 128, 1024])
def test_forward_parity(dtype, num_heads, head_dim, M):
    qkv, coords = _make_inputs(M, num_heads, head_dim, dtype)
    rope_dim = (head_dim // 6) * 6
    theta = _make_theta(rope_dim)

    out_kernel = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    out_ref = _torch_rope_ref(qkv, coords, theta, num_heads, rope_dim, conjugate=0)

    assert out_kernel.shape == (M, 3, num_heads, head_dim)
    assert out_kernel.dtype == dtype
    assert torch.isfinite(out_kernel).all()
    assert torch.allclose(out_kernel.float(), out_ref.float(), **_tol(dtype))


# ---------------------------------------------------------------------------
# rope_dim == 0 path (head_dim < 6): straight reshape
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [1, 2, 4, 5])
def test_rope_dim_zero_passthrough(dtype, head_dim):
    M, num_heads = 128, 4
    qkv, coords = _make_inputs(M, num_heads, head_dim, dtype)

    # VoxelRotaryPositionalEmbeddings derives rope_dim = (head_dim // 6) * 6 = 0,
    # so its forward should bypass the kernel and just reshape.
    rope = VoxelRotaryPositionalEmbeddings(
        dim=num_heads * head_dim, num_heads=num_heads, base=250
    ).cuda()
    assert rope.rope_dim == 0
    out = rope(qkv, coords)
    expected = qkv.reshape(M, 3, num_heads, head_dim)
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# M = 0 edge case
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_M_zero(dtype):
    num_heads, head_dim = 4, 24
    qkv = torch.empty(0, 3, num_heads * head_dim, device="cuda", dtype=dtype)
    coords = torch.empty(0, 3, device="cuda", dtype=torch.int32)
    rope_dim = 24
    theta = _make_theta(rope_dim)
    out = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    assert out.shape == (0, 3, num_heads, head_dim)
    assert out.dtype == dtype


# ---------------------------------------------------------------------------
# Backward parity vs torch autograd through the reference impl
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "num_heads,head_dim,M",
    [
        (4, 24, 64),
        (8, 32, 256),
        (16, 64, 128),
    ],
)
def test_backward_parity(dtype, num_heads, head_dim, M):
    rope_dim = (head_dim // 6) * 6
    theta = _make_theta(rope_dim)
    qkv, coords = _make_inputs(M, num_heads, head_dim, dtype, seed=1)

    qkv_a = qkv.detach().clone().requires_grad_(True)
    qkv_b = qkv.detach().clone().requires_grad_(True)

    out_a = fused_rope_qkv(qkv_a, coords, theta, num_heads, rope_dim)
    out_b = _torch_rope_ref(qkv_b, coords, theta, num_heads, rope_dim, conjugate=0)

    grad = torch.randn_like(out_a)
    out_a.backward(grad)
    out_b.backward(grad)

    assert qkv_a.grad is not None and qkv_b.grad is not None
    assert torch.isfinite(qkv_a.grad).all()
    # Compare in float — fp16/bf16 backward goes through fp32 accumulators on
    # both paths but cast results back to the input dtype.
    assert torch.allclose(qkv_a.grad.float(), qkv_b.grad.float(), **_tol(dtype))


# ---------------------------------------------------------------------------
# Conjugate property: kernel(conjugate=1)(kernel(conjugate=0)(x)) == x for Q/K
# (V is always pass-through, so trivially holds.)
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_conjugate_inverts_rotation(dtype):
    M, num_heads, head_dim = 256, 8, 24
    rope_dim = 24
    theta = _make_theta(rope_dim)
    qkv, coords = _make_inputs(M, num_heads, head_dim, dtype, seed=2)

    C = num_heads * head_dim
    coords_f = coords.float().contiguous()

    # Apply forward (conjugate=0).
    fwd_out = torch.empty(M, 3, num_heads, head_dim, device="cuda", dtype=dtype)
    _C.fused_rope.qkv(qkv.contiguous(), coords_f, theta, fwd_out, num_heads, rope_dim, 0)

    # Reshape [M,3,H,D] -> [M,3,C] and apply conjugate=1.
    fwd_3d = fwd_out.reshape(M, 3, C).contiguous()
    inv_out = torch.empty(M, 3, num_heads, head_dim, device="cuda", dtype=dtype)
    _C.fused_rope.qkv(fwd_3d, coords_f, theta, inv_out, num_heads, rope_dim, 1)

    expected = qkv.reshape(M, 3, num_heads, head_dim)
    # Q (idx 0) and K (idx 1) should round-trip; V (idx 2) is always copy.
    qk_diff = (inv_out[:, :2].float() - expected[:, :2].float()).abs().max().item()
    v_diff = (inv_out[:, 2].float() - expected[:, 2].float()).abs().max().item()
    tol = _tol(dtype)["atol"] * 2  # two casts incur 2x rounding budget
    assert qk_diff < tol, f"Q/K round-trip diff {qk_diff} > tol {tol}"
    assert v_diff < tol, f"V copy diff {v_diff} > tol {tol}"


# ---------------------------------------------------------------------------
# AMP autocast: forward dtype + finite gradients
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_autocast_forward_backward(amp_dtype):
    """Wrap the call in torch.autocast and verify dtype propagates + grads finite.

    The kernel itself is dtype-agnostic; what we're checking here is that the
    autograd Function's manual dtype handling does not silently up-/down-cast
    inputs in a way that breaks AMP.
    """
    M, num_heads, head_dim = 512, 8, 32
    rope_dim = (head_dim // 6) * 6
    theta = _make_theta(rope_dim)

    # Master copy lives in fp32 (typical AMP setup).
    qkv = torch.randn(
        M, 3, num_heads * head_dim, device="cuda", dtype=torch.float32, requires_grad=True
    )
    coords = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        # Inside autocast, downstream linear ops would produce amp_dtype inputs.
        # Simulate that by casting qkv before the call.
        qkv_amp = qkv.to(amp_dtype)
        out = fused_rope_qkv(qkv_amp, coords, theta, num_heads, rope_dim)

    assert out.dtype == amp_dtype, f"out dtype {out.dtype} != amp dtype {amp_dtype}"
    assert torch.isfinite(out).all()

    loss = out.float().sum()
    loss.backward()
    assert qkv.grad is not None
    assert torch.isfinite(qkv.grad).all()


@cuda_only
@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_autocast_module_e2e(amp_dtype):
    """Full module path under AMP: VoxelRotaryPositionalEmbeddings."""
    M, num_heads, head_dim = 256, 4, 24
    C = num_heads * head_dim
    rope = VoxelRotaryPositionalEmbeddings(dim=C, num_heads=num_heads, base=250).cuda()

    qkv = torch.randn(M, 3, C, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        qkv_amp = qkv.to(amp_dtype)
        out = rope(qkv_amp, coords)

    assert out.dtype == amp_dtype
    assert out.shape == (M, 3, num_heads, head_dim)
    assert torch.isfinite(out).all()

    out.float().sum().backward()
    assert torch.isfinite(qkv.grad).all()


# ---------------------------------------------------------------------------
# Mixed-precision: backward grad dtype matches forward output dtype
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grad_dtype_matches_input(dtype):
    M, num_heads, head_dim = 128, 4, 24
    rope_dim = 24
    theta = _make_theta(rope_dim)
    qkv = torch.randn(M, 3, num_heads * head_dim, device="cuda", dtype=dtype, requires_grad=True)
    coords = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)

    out = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    out.sum().backward()
    assert qkv.grad.dtype == dtype


# ---------------------------------------------------------------------------
# Coordinate dtype handling: int32 / int64 / float32 / float64 must agree
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("coord_dtype", [torch.int32, torch.int64, torch.float32, torch.float64])
def test_coord_dtype_invariance(coord_dtype):
    M, num_heads, head_dim = 256, 4, 24
    rope_dim = 24
    theta = _make_theta(rope_dim)

    torch.manual_seed(0)
    qkv = torch.randn(M, 3, num_heads * head_dim, device="cuda", dtype=torch.float32)
    coords_i32 = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)
    coords_other = coords_i32.to(coord_dtype)

    out_i32 = fused_rope_qkv(qkv, coords_i32, theta, num_heads, rope_dim)
    out_other = fused_rope_qkv(qkv, coords_other, theta, num_heads, rope_dim)
    assert torch.allclose(out_i32, out_other, atol=1e-6)


# ---------------------------------------------------------------------------
# Non-contiguous inputs: kernel binding's `.contiguous()` must handle these
# ---------------------------------------------------------------------------
@cuda_only
def test_non_contiguous_qkv():
    M, num_heads, head_dim = 128, 4, 24
    C = num_heads * head_dim
    rope_dim = 24
    theta = _make_theta(rope_dim)

    # Build qkv as a transposed view -> non-contiguous.
    base = torch.randn(3, M, C, device="cuda", dtype=torch.float32)
    qkv = base.transpose(0, 1)  # [M, 3, C], non-contig
    assert not qkv.is_contiguous()
    coords = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)

    out = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    out_ref = _torch_rope_ref(qkv, coords, theta, num_heads, rope_dim)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)


@cuda_only
def test_non_contiguous_coords():
    M, num_heads, head_dim = 128, 4, 24
    rope_dim = 24
    theta = _make_theta(rope_dim)

    qkv = torch.randn(M, 3, num_heads * head_dim, device="cuda", dtype=torch.float32)
    coords_full = torch.randint(0, 64, (M, 6), device="cuda", dtype=torch.int32)
    coords = coords_full[
        :, :3
    ]  # contiguous view of [:, :3] is still contig in stride-1; force slice that's not
    coords_strided = coords_full[:, ::2]  # [M, 3] non-contiguous
    assert not coords_strided.is_contiguous()

    out_a = fused_rope_qkv(qkv, coords.contiguous(), theta, num_heads, rope_dim)
    out_b = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    assert torch.allclose(out_a, out_b, atol=1e-6)

    # And strided variant should also work without crashing.
    out_c = fused_rope_qkv(qkv, coords_strided, theta, num_heads, rope_dim)
    assert torch.isfinite(out_c).all()


# ---------------------------------------------------------------------------
# Coordinate range: negative + large coords (kernel subtracts coord_min, +1)
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("coord_offset", [-1000, 0, 10_000])
def test_coordinate_offset_invariance(coord_offset):
    """Forward output should depend only on relative voxel position
    (coords - coords.min() + 1), not on a uniform shift."""
    M, num_heads, head_dim = 256, 4, 24
    rope_dim = 24
    theta = _make_theta(rope_dim)

    torch.manual_seed(3)
    qkv = torch.randn(M, 3, num_heads * head_dim, device="cuda", dtype=torch.float32)
    base = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)
    shifted = base + coord_offset

    out_base = fused_rope_qkv(qkv, base, theta, num_heads, rope_dim)
    out_shifted = fused_rope_qkv(qkv, shifted, theta, num_heads, rope_dim)
    # Subtraction of coord_min cancels the uniform shift, so outputs match.
    assert torch.allclose(out_base, out_shifted, atol=1e-5, rtol=1e-5)


@cuda_only
def test_large_coordinates():
    """Sanity check at high coord magnitudes — angles wrap cleanly via cosf/sinf."""
    M, num_heads, head_dim = 128, 4, 24
    rope_dim = 24
    theta = _make_theta(rope_dim, base=250)

    torch.manual_seed(4)
    qkv = torch.randn(M, 3, num_heads * head_dim, device="cuda", dtype=torch.float32)
    coords = torch.randint(0, 1_000_000, (M, 3), device="cuda", dtype=torch.int32)

    out = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    out_ref = _torch_rope_ref(qkv, coords, theta, num_heads, rope_dim)
    # cosf/sinf at very large angles diverge a bit from torch.cos/sin (libm vs
    # device). Allow 1e-4.
    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Larger workload: bigger M to exercise grid scheduling
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("M", [4096, 65_537])  # crosses BLOCK_M=128 boundaries
def test_large_M(M):
    num_heads, head_dim = 8, 32
    rope_dim = (head_dim // 6) * 6
    theta = _make_theta(rope_dim)
    qkv, coords = _make_inputs(M, num_heads, head_dim, torch.float32, coord_max=128)

    out = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    out_ref = _torch_rope_ref(qkv, coords, theta, num_heads, rope_dim)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Module-level: VoxelRotaryPositionalEmbeddings forward path
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_module_forward_matches_functional(dtype):
    M, num_heads, head_dim = 256, 4, 24
    C = num_heads * head_dim
    rope = VoxelRotaryPositionalEmbeddings(dim=C, num_heads=num_heads, base=250).cuda()

    qkv = torch.randn(M, 3, C, device="cuda", dtype=dtype)
    coords = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)

    out_module = rope(qkv, coords)
    out_func = fused_rope_qkv(qkv, coords, rope.theta, num_heads, rope.rope_dim)
    assert torch.equal(out_module, out_func)


@cuda_only
def test_module_accepts_2d_qkv():
    """Module forward should accept [M, 3*C] flat layout and reshape internally."""
    M, num_heads, head_dim = 64, 4, 24
    C = num_heads * head_dim
    rope = VoxelRotaryPositionalEmbeddings(dim=C, num_heads=num_heads, base=250).cuda()

    qkv_3d = torch.randn(M, 3, C, device="cuda", dtype=torch.float32)
    qkv_2d = qkv_3d.reshape(M, 3 * C).contiguous()
    coords = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)

    out_3d = rope(qkv_3d, coords)
    out_2d = rope(qkv_2d, coords)
    assert torch.equal(out_3d, out_2d)


# ---------------------------------------------------------------------------
# V (qkv_idx == 2) is always pass-through (no rotation)
# ---------------------------------------------------------------------------
@cuda_only
def test_v_is_passthrough():
    M, num_heads, head_dim = 128, 4, 24
    rope_dim = 24
    theta = _make_theta(rope_dim)
    qkv, coords = _make_inputs(M, num_heads, head_dim, torch.float32, seed=5)

    out = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    expected_v = qkv[:, 2].reshape(M, num_heads, head_dim)
    assert torch.equal(out[:, 2], expected_v)


# ---------------------------------------------------------------------------
# Determinism: same inputs -> bitwise-identical outputs
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_deterministic(dtype):
    M, num_heads, head_dim = 256, 4, 24
    rope_dim = 24
    theta = _make_theta(rope_dim)
    qkv, coords = _make_inputs(M, num_heads, head_dim, dtype, seed=6)

    out1 = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    out2 = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Numerical bound at rope_dim boundary (head_dim not divisible by 6)
# ---------------------------------------------------------------------------
@cuda_only
def test_pass_through_dims_unchanged():
    """When head_dim > rope_dim, the trailing pass_dim entries copy verbatim."""
    M, num_heads, head_dim = 128, 4, 32  # rope_dim=30, pass_dim=2
    rope_dim = (head_dim // 6) * 6
    assert rope_dim == 30 and head_dim - rope_dim == 2
    theta = _make_theta(rope_dim)
    qkv, coords = _make_inputs(M, num_heads, head_dim, torch.float32, seed=7)

    out = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    qkv_4d = qkv.reshape(M, 3, num_heads, head_dim)
    # Trailing pass_dim columns of Q/K/V copy bit-for-bit (no math).
    assert torch.equal(out[..., rope_dim:], qkv_4d[..., rope_dim:])


# ---------------------------------------------------------------------------
# Gradient finiteness with extreme inputs (catches NaN-producing kernel bugs)
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_extreme_input_gradients_finite(dtype):
    """Large feature magnitudes shouldn't produce NaN gradients in the kernel."""
    M, num_heads, head_dim = 256, 4, 24
    rope_dim = 24
    theta = _make_theta(rope_dim)
    torch.manual_seed(8)
    scale = 10.0 if dtype != torch.float16 else 4.0  # avoid fp16 overflow
    qkv = (
        torch.randn(M, 3, num_heads * head_dim, device="cuda", dtype=dtype) * scale
    ).requires_grad_(True)
    coords = torch.randint(0, 64, (M, 3), device="cuda", dtype=torch.int32)

    out = fused_rope_qkv(qkv, coords, theta, num_heads, rope_dim)
    assert torch.isfinite(out).all()
    out.float().sum().backward()
    assert torch.isfinite(qkv.grad).all()
