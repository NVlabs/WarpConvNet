# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified WarpConvNet gradient-checkpointing contract."""

from __future__ import annotations

import torch
import torch.nn as nn

from warpconvnet.nn.modules.dit import (
    ModulatedTransformerBlock,
    ModulatedTransformerCrossBlock,
)
from warpconvnet.nn.modules.gradient_checkpointing import (
    GradientCheckpointingMixin,
    GradientCheckpointingModelMixin,
    configure_gradient_checkpointing,
)
from warpconvnet.nn.modules.sparse_dit import (
    ModulatedSparseTransformerBlock,
    ModulatedSparseTransformerCrossBlock,
)


class _CheckpointedLinear(GradientCheckpointingMixin, nn.Module):
    def __init__(self, use_checkpoint: bool = False):
        super().__init__()
        self._init_gradient_checkpointing(use_checkpoint)
        self.linear = nn.Linear(8, 8)
        self.calls = 0

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.linear(x).tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._gradient_checkpointed_call(self._forward, x)


class _CheckpointedModel(GradientCheckpointingModelMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_CheckpointedLinear(), _CheckpointedLinear()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _run_linear_backward(module: _CheckpointedLinear, features: torch.Tensor):
    features = features.detach().clone().requires_grad_(True)
    output = module(features)
    output.square().mean().backward()
    parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), features.grad.detach(), parameter_grads


def test_mixin_checkpoint_backward_matches_uncheckpointed():
    torch.manual_seed(11)
    reference = _CheckpointedLinear(use_checkpoint=False)
    checkpointed = _CheckpointedLinear(use_checkpoint=True)
    checkpointed.load_state_dict(reference.state_dict())
    features = torch.randn(4, 8)

    ref_out, ref_input_grad, ref_parameter_grads = _run_linear_backward(reference, features)
    out, input_grad, parameter_grads = _run_linear_backward(checkpointed, features)

    assert reference.calls == 1
    assert checkpointed.calls == 2
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(input_grad, ref_input_grad)
    assert parameter_grads.keys() == ref_parameter_grads.keys()
    for name in parameter_grads:
        torch.testing.assert_close(parameter_grads[name], ref_parameter_grads[name])


def test_model_mixin_enable_disable_and_kwargs():
    model = _CheckpointedModel()
    assert model.supports_gradient_checkpointing
    assert not model.is_gradient_checkpointing

    model.gradient_checkpointing_enable({"preserve_rng_state": False})
    assert model.is_gradient_checkpointing
    assert all(block.gradient_checkpointing for block in model.blocks)
    assert all(
        block._gradient_checkpointing_func.keywords
        == {"preserve_rng_state": False, "use_reentrant": False}
        for block in model.blocks
    )

    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing
    assert not any(block.gradient_checkpointing for block in model.blocks)


def test_configure_gradient_checkpointing_on_plain_container():
    model = nn.Sequential(_CheckpointedLinear(), _CheckpointedLinear())
    assert configure_gradient_checkpointing(model, enable=True) == 2
    assert all(block.gradient_checkpointing for block in model)
    assert configure_gradient_checkpointing(model, enable=False) == 2
    assert not any(block.gradient_checkpointing for block in model)

    empty = nn.Sequential(nn.Linear(2, 2))
    assert configure_gradient_checkpointing(empty, strict=False) == 0


def test_mixin_rejects_reentrant_checkpointing():
    model = _CheckpointedModel()
    try:
        model.gradient_checkpointing_enable({"use_reentrant": True})
    except ValueError as error:
        assert "use_reentrant=False" in str(error)
    else:
        raise AssertionError("reentrant checkpointing should be rejected")


def test_all_dit_block_families_share_checkpointing_contract():
    blocks = nn.ModuleList(
        [
            ModulatedTransformerBlock(channels=16, num_heads=4),
            ModulatedTransformerCrossBlock(channels=16, ctx_channels=8, num_heads=4),
            ModulatedSparseTransformerBlock(channels=16, num_heads=4),
            ModulatedSparseTransformerCrossBlock(channels=16, ctx_channels=8, num_heads=4),
        ]
    )

    assert configure_gradient_checkpointing(blocks, enable=True) == 4
    assert all(block.gradient_checkpointing for block in blocks)
    assert all(block.use_checkpoint for block in blocks)


def test_dense_dit_checkpoint_backward_matches_uncheckpointed():
    torch.manual_seed(23)
    reference = ModulatedTransformerBlock(
        channels=16,
        num_heads=4,
        mlp_ratio=2.0,
        use_checkpoint=False,
    )
    checkpointed = ModulatedTransformerBlock(
        channels=16,
        num_heads=4,
        mlp_ratio=2.0,
        use_checkpoint=True,
    )
    checkpointed.load_state_dict(reference.state_dict())

    features = torch.randn(2, 5, 16)
    modulation = torch.randn(2, 16)

    def run(block):
        x = features.detach().clone().requires_grad_(True)
        mod = modulation.detach().clone().requires_grad_(True)
        calls = 0
        original_forward = block._forward

        def counted_forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_forward(*args, **kwargs)

        block._forward = counted_forward
        output = block(x, mod)
        output.square().mean().backward()
        parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in block.named_parameters()
            if parameter.grad is not None
        }
        return (
            output.detach(),
            x.grad.detach(),
            mod.grad.detach(),
            parameter_grads,
            calls,
        )

    ref_out, ref_x_grad, ref_mod_grad, ref_parameter_grads, ref_calls = run(reference)
    out, x_grad, mod_grad, parameter_grads, calls = run(checkpointed)

    assert ref_calls == 1
    assert calls == 2
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(x_grad, ref_x_grad)
    torch.testing.assert_close(mod_grad, ref_mod_grad)
    assert parameter_grads.keys() == ref_parameter_grads.keys()
    for name in parameter_grads:
        torch.testing.assert_close(parameter_grads[name], ref_parameter_grads[name])
