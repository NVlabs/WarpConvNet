# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fused RoPE + QKV reshape via custom CUDA kernel.
# Replaces qkv.chunk(3) -> rope(Q) -> rope(K) -> cat([Q,K,V]) -> reshape(M,3,H,D)
# with a single-pass kernel. Forward and backward share one kernel; backward
# negates sin to apply the inverse rotation.

import torch

import warpconvnet._C as _C


class _FusedRopeQKVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, coords, theta, num_heads, rope_dim):
        M, _, C = qkv.shape
        head_dim = C // num_heads
        out = torch.empty(M, 3, num_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
        # Voxel coords arrive as int32 from the upstream pipeline; the kernel
        # requires float32 for the cos/sin phase math. .float() allocates a
        # fresh M*3*4-byte tensor each forward. Future optimization: template
        # the CUDA kernel on coord dtype and skip this conversion.
        coords_f = coords.float().contiguous()
        _C.fused_rope.qkv(
            qkv.contiguous(),
            coords_f,
            theta.contiguous(),
            out,
            num_heads,
            rope_dim,
            0,
        )
        ctx.save_for_backward(coords_f, theta)
        ctx.num_heads = num_heads
        ctx.rope_dim = rope_dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        coords_f, theta = ctx.saved_tensors
        num_heads = ctx.num_heads
        rope_dim = ctx.rope_dim

        M, _, H, D = grad_output.shape
        C = H * D

        grad_out_3d = grad_output.contiguous().reshape(M, 3, C)
        grad_qkv_4d = torch.empty(M, 3, H, D, dtype=grad_output.dtype, device=grad_output.device)
        _C.fused_rope.qkv(
            grad_out_3d,
            coords_f,
            theta.contiguous(),
            grad_qkv_4d,
            num_heads,
            rope_dim,
            1,  # conjugate: invert rotation
        )

        grad_qkv = grad_qkv_4d.reshape(M, 3, C)
        return grad_qkv, None, None, None, None


def fused_rope_qkv(
    qkv: torch.Tensor,
    coords: torch.Tensor,
    theta: torch.Tensor,
    num_heads: int,
    rope_dim: int,
) -> torch.Tensor:
    """Fused RoPE + QKV reshape via CUDA kernel.

    Takes ``[M, 3, C] -> [M, 3, H, D]`` with RoPE applied to Q and K.
    Backward applies the inverse rotation through the same kernel.
    """
    M = qkv.shape[0]
    C = qkv.shape[2]
    head_dim = C // num_heads

    if M == 0:
        return torch.empty(M, 3, num_heads, head_dim, dtype=qkv.dtype, device=qkv.device)

    return _FusedRopeQKVFunction.apply(qkv, coords, theta, num_heads, rope_dim)
