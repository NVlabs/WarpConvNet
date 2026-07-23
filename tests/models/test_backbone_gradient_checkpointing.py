# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient-checkpointing tests for SpaCeFormer and PointTransformerV3."""

from __future__ import annotations

import copy
import importlib.util
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING
from warpconvnet.geometry.types.conversion.to_voxels import points_to_voxels
from warpconvnet.geometry.types.points import Points
from warpconvnet.models.point_transformer_v3 import (
    PatchAttentionBlock,
    PointTransformerV3,
)
from warpconvnet.models.spaceformer import SpaCeFormer
from warpconvnet.nn.modules.space_attention import (
    PostNormBlock,
    PreNormBlock,
    StreamNormBlock,
)


_HAS_CUDA = torch.cuda.is_available()
_HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
_skip_without_sparse_attention = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_FLASH_ATTN),
    reason="CUDA and flash-attn are required",
)


class _GeometryLinear(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(channels, channels)

    def forward(self, x):
        return x.replace_features(self.linear(x.feature_tensor))


class _GeometryAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(channels, channels)

    def forward(self, x, order=None):
        del order
        return x.replace_features(self.linear(x.feature_tensor))


def _replace_compute_layers(block, channels: int):
    """Keep the real block forward while using deterministic CPU operations."""
    block.conv = _GeometryLinear(channels)
    block.conv_shortcut = nn.Identity()
    block.norm1 = nn.Identity()
    block.attention = _GeometryAttention(channels)
    block.norm2 = nn.Identity()
    block.mlp = _GeometryLinear(channels)
    block.drop_path = nn.Identity()
    return block


def _make_points(channels: int) -> Points:
    generator = torch.Generator().manual_seed(17)
    return Points(
        [
            torch.randn(7, 3, generator=generator),
            torch.randn(5, 3, generator=generator),
        ],
        [
            torch.randn(7, channels, generator=generator),
            torch.randn(5, channels, generator=generator),
        ],
    )


def _run_block_backward(block, template: Points):
    features = template.feature_tensor.detach().clone().requires_grad_(True)
    points = template.replace_features(features)
    calls = 0
    original_forward = block._forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    block._forward = counted_forward
    output = block(points, POINT_ORDERING.MORTON_XYZ).feature_tensor
    output.square().mean().backward()
    parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in block.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), features.grad.detach(), parameter_grads, calls


@pytest.mark.parametrize(
    "block_cls",
    [PatchAttentionBlock, PreNormBlock, PostNormBlock, StreamNormBlock],
)
def test_attention_block_checkpoint_backward_matches_uncheckpointed(block_cls):
    torch.manual_seed(29)
    channels = 8
    reference = _replace_compute_layers(
        block_cls(
            in_channels=channels,
            attention_channels=channels,
            patch_size=4,
            num_heads=2,
            drop_path=0.0,
            use_checkpoint=False,
        ),
        channels,
    )
    checkpointed = copy.deepcopy(reference)
    checkpointed.use_checkpoint = True
    template = _make_points(channels)

    ref_out, ref_input_grad, ref_parameter_grads, ref_calls = _run_block_backward(
        reference, template
    )
    out, input_grad, parameter_grads, calls = _run_block_backward(checkpointed, template)

    assert ref_calls == 1
    assert calls == 2
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(input_grad, ref_input_grad)
    assert parameter_grads.keys() == ref_parameter_grads.keys()
    for name in parameter_grads:
        torch.testing.assert_close(parameter_grads[name], ref_parameter_grads[name])


@_skip_without_sparse_attention
@pytest.mark.parametrize("block_cls", [PatchAttentionBlock, PreNormBlock])
def test_real_sparse_attention_block_checkpoint_backward(block_cls):
    torch.manual_seed(37)
    channels = 16
    points = Points(
        [torch.rand(64, 3, device="cuda")],
        [torch.randn(64, channels, device="cuda")],
    )
    template = points_to_voxels(points, voxel_size=0.05)
    block = block_cls(
        in_channels=channels,
        attention_channels=channels,
        patch_size=16,
        num_heads=4,
        drop_path=0.0,
        use_checkpoint=True,
    ).cuda()
    features = template.feature_tensor.detach().clone().requires_grad_(True)
    voxels = template.replace_features(features)
    calls = 0
    original_forward = block._forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    block._forward = counted_forward
    with torch.autocast("cuda", dtype=torch.float16):
        output = block(voxels).feature_tensor
        loss = output.float().square().mean()
    loss.backward()

    assert calls == 2
    assert features.grad is not None
    assert any(parameter.grad is not None for parameter in block.parameters())


def _make_ptv3(use_checkpoint: bool = False) -> PointTransformerV3:
    return PointTransformerV3(
        in_channels=4,
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(2, 4),
        enc_patch_size=(8, 8),
        dec_depths=(1,),
        dec_channels=(8,),
        dec_num_head=(2,),
        dec_patch_size=(8,),
        shuffle_orders=False,
        use_checkpoint=use_checkpoint,
    )


def _make_spaceformer(use_checkpoint: bool = False) -> SpaCeFormer:
    return SpaCeFormer(
        in_channels=4,
        enc_depths=(1, 1),
        enc_channels=(8, 16),
        enc_num_head=(2, 4),
        enc_patch_size=(8, 8),
        enc_attn_types="cc",
        dec_depths=(1,),
        dec_channels=(8,),
        dec_num_head=(2,),
        dec_patch_size=(8,),
        dec_attn_types="c",
        shuffle_orders=False,
        conv_norm_layer=None,
        use_checkpoint=use_checkpoint,
    )


@pytest.mark.parametrize("model_factory", [_make_ptv3, _make_spaceformer])
def test_backbone_model_checkpointing_enable_disable(
    model_factory: Callable[[bool], nn.Module],
):
    model = model_factory(False)
    boundaries = [
        module for module in model.modules() if hasattr(module, "gradient_checkpointing")
    ]

    assert model.supports_gradient_checkpointing
    assert len(boundaries) == 3
    assert not model.is_gradient_checkpointing

    model.gradient_checkpointing_enable({"preserve_rng_state": False})
    assert model.is_gradient_checkpointing
    assert all(module.gradient_checkpointing for module in boundaries)
    assert all(
        module._gradient_checkpointing_func.keywords
        == {"preserve_rng_state": False, "use_reentrant": False}
        for module in boundaries
    )

    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing
    assert not any(module.gradient_checkpointing for module in boundaries)


@pytest.mark.parametrize("model_factory", [_make_ptv3, _make_spaceformer])
def test_backbone_constructor_can_enable_checkpointing(
    model_factory: Callable[[bool], nn.Module],
):
    model = model_factory(True)
    boundaries = [
        module for module in model.modules() if hasattr(module, "gradient_checkpointing")
    ]

    assert boundaries
    assert model.is_gradient_checkpointing
    assert all(module.gradient_checkpointing for module in boundaries)
    assert all("checkpoint" not in name for name in model.state_dict())
