# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small neural-network utilities shared by model ports and reusable modules."""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = [
    "DEFAULT_MIXED_PRECISION_MODULES",
    "convert_module_parameters_to",
    "convert_module_to_f16",
    "convert_module_to_f32",
    "manual_cast",
    "str_to_dtype",
    "zero_module",
]


DEFAULT_MIXED_PRECISION_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)


def manual_cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast ``tensor`` to ``dtype`` only when AMP autocast is not active."""
    if not torch.is_autocast_enabled():
        return tensor.type(dtype)
    return tensor


def str_to_dtype(s: str) -> torch.dtype:
    """Parse common dtype spellings used in model config files."""
    return {
        "f16": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "f32": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[s]


def zero_module(module: nn.Module) -> nn.Module:
    """Zero all parameters of ``module`` in-place and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def convert_module_parameters_to(
    module: nn.Module,
    dtype: torch.dtype,
    module_types: tuple[type[nn.Module], ...] = DEFAULT_MIXED_PRECISION_MODULES,
) -> None:
    """Cast parameters for selected leaf module families in-place.

    Intended for diffusion/flow model ports that keep normalization modules in
    fp32 while casting Linear/Conv layers to fp16 or bf16.
    """
    if isinstance(module, module_types):
        for p in module.parameters():
            p.data = p.data.to(dtype)


def convert_module_to_f16(
    module: nn.Module,
    module_types: tuple[type[nn.Module], ...] = DEFAULT_MIXED_PRECISION_MODULES,
) -> None:
    convert_module_parameters_to(module, torch.float16, module_types=module_types)


def convert_module_to_f32(
    module: nn.Module,
    module_types: tuple[type[nn.Module], ...] = DEFAULT_MIXED_PRECISION_MODULES,
) -> None:
    convert_module_parameters_to(module, torch.float32, module_types=module_types)
