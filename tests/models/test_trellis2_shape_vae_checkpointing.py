# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient-checkpointing coverage for the TRELLIS.2 shape encoder."""

from __future__ import annotations

import pytest
import torch

from warpconvnet.models.trellis2.shape_vae import (
    FlexiDualGridVaeEncoder,
    SparseResBlockS2C3d,
)
from warpconvnet.models.trellis2.sparse_ops import from_feats_coords


_HAS_CUDA = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")


def _tiny_encoder() -> FlexiDualGridVaeEncoder:
    return FlexiDualGridVaeEncoder(
        model_channels=[32, 64],
        latent_channels=8,
        num_blocks=[1, 1],
        block_type=["SparseConvNeXtBlock3d"] * 2,
        down_block_type=["SparseResBlockS2C3d"],
        block_args=[{}, {}],
        use_fp16=False,
    )


def _full_grid(R: int, channels: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    coords = torch.stack(
        torch.meshgrid(torch.arange(R), torch.arange(R), torch.arange(R), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    coords = torch.cat(
        [torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords.int()],
        dim=1,
    )
    features = torch.randn(coords.shape[0], channels, generator=generator)
    return from_feats_coords(features, coords)


def test_encoder_gradient_checkpointing_toggle():
    encoder = _tiny_encoder()
    checkpointable = [m for m in encoder.modules() if hasattr(m, "gradient_checkpointing")]

    assert encoder.supports_gradient_checkpointing
    assert len(checkpointable) == 3
    assert not encoder.is_gradient_checkpointing

    encoder.gradient_checkpointing_enable({"preserve_rng_state": False})
    assert encoder.is_gradient_checkpointing
    assert all(m.gradient_checkpointing for m in checkpointable)
    assert all(
        m._gradient_checkpointing_func.keywords["use_reentrant"] is False for m in checkpointable
    )
    assert all(
        m._gradient_checkpointing_func.keywords["preserve_rng_state"] is False
        for m in checkpointable
    )

    encoder.gradient_checkpointing_disable()
    assert not encoder.is_gradient_checkpointing
    assert not any(m.gradient_checkpointing for m in checkpointable)


def test_encoder_rejects_reentrant_checkpointing():
    encoder = _tiny_encoder()
    with pytest.raises(ValueError, match="use_reentrant=False"):
        encoder.gradient_checkpointing_enable({"use_reentrant": True})


def test_hf_parent_discovers_encoder_checkpoint_blocks():
    """Transformers' standard module walk must reach the TRELLIS leaves."""
    transformers = pytest.importorskip("transformers")

    class ParentModel(torch.nn.Module):
        supports_gradient_checkpointing = True
        main_input_name = "voxels"
        gradient_checkpointing_enable = transformers.PreTrainedModel.gradient_checkpointing_enable
        gradient_checkpointing_disable = (
            transformers.PreTrainedModel.gradient_checkpointing_disable
        )
        _set_gradient_checkpointing = transformers.PreTrainedModel._set_gradient_checkpointing

        def __init__(self):
            super().__init__()
            self.encoder = _tiny_encoder()

    model = ParentModel()
    checkpointable = [m for m in model.encoder.modules() if hasattr(m, "gradient_checkpointing")]

    # This is the path exercised by TrainingArguments(
    #     gradient_checkpointing=True
    # ) when no checkpointing kwargs are supplied.
    model.gradient_checkpointing_enable()
    assert all(m.gradient_checkpointing for m in checkpointable)
    assert all(
        m._gradient_checkpointing_func.keywords["use_reentrant"] is False for m in checkpointable
    )

    model.gradient_checkpointing_disable()
    assert not any(m.gradient_checkpointing for m in checkpointable)

    model.gradient_checkpointing_enable({"preserve_rng_state": False})
    assert all(
        m._gradient_checkpointing_func.keywords["preserve_rng_state"] is False
        for m in checkpointable
    )

    model.gradient_checkpointing_disable()
    assert not any(m.gradient_checkpointing for m in checkpointable)


@_skip_no_cuda
def test_s2c_checkpoint_backward_matches_uncheckpointed():
    torch.manual_seed(17)
    reference = SparseResBlockS2C3d(32, 64, use_checkpoint=False).cuda()
    checkpointed = SparseResBlockS2C3d(32, 64, use_checkpoint=True).cuda()

    # Avoid the zero-initialized second convolution masking upstream gradients.
    with torch.no_grad():
        reference.conv2.weight.normal_(0, 0.02)
        reference.conv2.bias.normal_(0, 0.02)
    checkpointed.load_state_dict(reference.state_dict())

    template = _full_grid(R=4, channels=32, seed=17).to("cuda")

    def run(block):
        features = template.feats.detach().clone().requires_grad_(True)
        voxels = template.replace_features(features)
        calls = 0
        original_forward = block._forward

        def counted_forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_forward(*args, **kwargs)

        block._forward = counted_forward
        output = block(voxels).feats
        output.float().square().mean().backward()
        parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in block.named_parameters()
            if parameter.grad is not None
        }
        return output.detach(), features.grad.detach(), parameter_grads, calls

    ref_out, ref_input_grad, ref_parameter_grads, ref_calls = run(reference)
    out, input_grad, parameter_grads, calls = run(checkpointed)

    assert ref_calls == 1
    assert calls == 2
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(input_grad, ref_input_grad)
    assert parameter_grads.keys() == ref_parameter_grads.keys()
    for name in parameter_grads:
        torch.testing.assert_close(parameter_grads[name], ref_parameter_grads[name])


@_skip_no_cuda
def test_tiny_encoder_checkpointed_backward():
    torch.manual_seed(5)
    encoder = _tiny_encoder().cuda().train()
    encoder.gradient_checkpointing_enable()

    geometry = _full_grid(R=4, channels=6, seed=5).to("cuda")
    vertices = geometry.replace_features(geometry.feats[:, :3].sigmoid())
    intersected = geometry.replace_features((geometry.feats[:, 3:] > 0).float())

    latent = encoder(vertices, intersected)
    assert latent.feats.shape == (8, 8)
    latent.feats.float().square().mean().backward()

    assert all(parameter.grad is not None for parameter in encoder.parameters())
