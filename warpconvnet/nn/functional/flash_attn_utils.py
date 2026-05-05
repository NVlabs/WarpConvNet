# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

try:
    import flash_attn
except ImportError:
    flash_attn = None

# CUDA grid.x dimension limit
MAX_FLASH_ATTN_SEQS = 65535


def flash_attn_varlen_qkvpacked(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    window_size=(-1, -1),
) -> torch.Tensor:
    """Wrap ``flash_attn_varlen_qkvpacked_func`` with chunking when the number of
    sequences exceeds the CUDA ``gridDim.x`` limit (65535).

    Args:
        qkv: ``(total_tokens, 3, num_heads, head_dim)``
        cu_seqlens: ``(num_seqs + 1,)`` cumulative sequence lengths.
        max_seqlen: maximum sequence length.
        dropout_p: dropout probability.
        softmax_scale: softmax scale.
        window_size: sliding window attention window size.

    Returns:
        ``(total_tokens, num_heads, head_dim)``
    """
    assert flash_attn is not None, "flash_attn is required"
    num_seqs = cu_seqlens.shape[0] - 1

    if num_seqs <= MAX_FLASH_ATTN_SEQS:
        return flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            window_size=window_size,
        )

    out = torch.empty(
        qkv.shape[0],
        qkv.shape[2],
        qkv.shape[3],
        dtype=qkv.dtype,
        device=qkv.device,
    )

    chunk_start_seq = 0
    while chunk_start_seq < num_seqs:
        chunk_end_seq = min(chunk_start_seq + MAX_FLASH_ATTN_SEQS, num_seqs)
        token_start = cu_seqlens[chunk_start_seq].item()
        token_end = cu_seqlens[chunk_end_seq].item()

        chunk_cu_seqlens = cu_seqlens[chunk_start_seq : chunk_end_seq + 1] - token_start
        chunk_qkv = qkv[token_start:token_end]

        chunk_max_seqlen = (
            chunk_cu_seqlens.diff().max().item() if chunk_end_seq > chunk_start_seq else 0
        )

        chunk_out = flash_attn.flash_attn_varlen_qkvpacked_func(
            chunk_qkv,
            chunk_cu_seqlens.contiguous(),
            max_seqlen=chunk_max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            window_size=window_size,
        )
        out[token_start:token_end] = chunk_out
        chunk_start_seq = chunk_end_seq

    return out
