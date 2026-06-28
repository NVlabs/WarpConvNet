# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Volt: a Volume Transformer for sparse-voxel semantic segmentation (WarpConvNet port).

See ``README.md`` for the architecture description and the ScanNet ablation. Every
variant below is a pure keyword-argument change to `Volt` — no new code paths,
just configuration.
"""
from .volt import (
    Block,
    ConvBlockTokenizer,
    Decoder,
    Detokenizer,
    RoPE,
    RoPEAttention,
    SparseResBlock,
    Tokenizer,
    TokenConv,
    Volt,
)

# Volt-S backbone defaults — the configuration that reproduces the reference Volt-S
# (76.06 TTA on ScanNet val, within noise of the published 77.3 once the improvements
# below are added). All variants are expressed as overrides on top of these.
VOLT_S_DEFAULTS = dict(
    in_channels=6,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    qk_norm=True,
    drop_path=0.3,
    stride=5,
    kernel_size=5,
    up_mlp_dim=128,
    increase_drop_path=True,
)

# name -> kwargs overriding VOLT_S_DEFAULTS. The trailing comment is the verified
# ScanNet val mIoU (test-time augmentation) for that configuration.
VOLT_VARIANTS = {
    "volt-s": dict(),  # 76.06  (faithful port baseline)
    "volt-b": dict(embed_dim=768, num_heads=12),  # 76.53  (4x width)
    "volt-convattn": dict(conv_before_attn=True),  # 76.41  (per-block conv before attn)
    "volt-convblock": dict(tokenizer_type="convblock"),  # 77.01  (ResNet conv-stem tokenizer)
    "volt-blockattn": dict(
        tokenizer_type="convblock", conv_before_attn=True
    ),  # 78.00  (BEST — beats published 77.3)
    "volt-all3": dict(
        embed_dim=768, num_heads=12, tokenizer_type="convblock", conv_before_attn=True
    ),  # large variant (see README)
}


def build_volt(variant: str = "volt-s", **overrides) -> Volt:
    """Construct a Volt variant by name. Extra ``overrides`` win over the preset.

    >>> model = build_volt("volt-blockattn")        # the best ScanNet config
    >>> model = build_volt("volt-b", drop_path=0.4)  # a preset with a tweak
    """
    if variant not in VOLT_VARIANTS:
        raise ValueError(f"unknown Volt variant {variant!r}; choose from {sorted(VOLT_VARIANTS)}")
    cfg = {**VOLT_S_DEFAULTS, **VOLT_VARIANTS[variant], **overrides}
    return Volt(**cfg)


__all__ = [
    "Volt",
    "build_volt",
    "VOLT_VARIANTS",
    "VOLT_S_DEFAULTS",
    "Tokenizer",
    "ConvBlockTokenizer",
    "SparseResBlock",
    "RoPE",
    "RoPEAttention",
    "TokenConv",
    "Block",
    "Decoder",
    "Detokenizer",
]
