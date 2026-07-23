# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient-checkpointing tests for MinkUNet and training-mode BatchNorm."""

from __future__ import annotations

import copy
import importlib.util
from contextlib import contextmanager
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from warpconvnet.geometry.types.conversion.to_voxels import points_to_voxels
from warpconvnet.geometry.types.points import Points
from warpconvnet.models.mink_unet import (
    BasicBlock,
    BottleneckBlock,
    MinkUNet18,
    MinkUNetBase,
)
from warpconvnet.nn.modules.gradient_checkpointing import preserve_module_buffers
from warpconvnet.nn.modules.sequential import Sequential


_HAS_CUDA = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
_HAS_EXTENSION = importlib.util.find_spec("warpconvnet._C") is not None
_skip_no_extension = pytest.mark.skipif(
    not _HAS_EXTENSION,
    reason="WarpConvNet extension required",
)
_BATCH_NORM_BUFFERS = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


class _FailOnSecondCall(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("recompute failed")
        return x


def _linear_conv(
    in_channels: int,
    out_channels: int,
    momentum: float | None,
    *,
    activation: bool = True,
    track_running_stats: bool = True,
) -> Sequential:
    return Sequential(
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(
            out_channels,
            momentum=momentum,
            track_running_stats=track_running_stats,
        ),
        nn.ReLU() if activation else nn.Identity(),
    )


def _replace_sparse_convs(
    block,
    in_channels: int,
    out_channels: int,
    momentum: float | None,
    *,
    track_running_stats: bool = True,
):
    if isinstance(block, BottleneckBlock):
        mid_channels = out_channels // block.expansion
        block.conv1 = _linear_conv(
            in_channels,
            mid_channels,
            momentum,
            track_running_stats=track_running_stats,
        )
        block.conv2 = _linear_conv(
            mid_channels,
            mid_channels,
            momentum,
            track_running_stats=track_running_stats,
        )
        block.conv3 = _linear_conv(
            mid_channels,
            out_channels,
            momentum,
            activation=False,
            track_running_stats=track_running_stats,
        )
    else:
        block.conv1 = _linear_conv(
            in_channels,
            out_channels,
            momentum,
            track_running_stats=track_running_stats,
        )
        block.conv2 = _linear_conv(
            out_channels,
            out_channels,
            momentum,
            activation=False,
            track_running_stats=track_running_stats,
        )

    if block.downsample is not None:
        block.downsample = _linear_conv(
            in_channels,
            out_channels,
            momentum,
            activation=False,
            track_running_stats=track_running_stats,
        )
    return block


def _make_points(channels: int, *, device: str = "cpu") -> Points:
    generator = torch.Generator(device=device).manual_seed(41)
    return Points(
        [torch.randn(12, 3, generator=generator, device=device)],
        [
            torch.randn(
                12,
                channels,
                generator=generator,
                device=device,
            )
        ],
    )


def _batch_norm_state(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for module_name, child in module.named_modules():
        if not isinstance(child, nn.modules.batchnorm._BatchNorm):
            continue
        for buffer_name in _BATCH_NORM_BUFFERS:
            buffer = getattr(child, buffer_name)
            if buffer is not None:
                state[f"{module_name}.{buffer_name}"] = buffer.detach().clone()
    return state


def _assert_state_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name])


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
    output = block(points).feature_tensor
    state_after_forward = _batch_norm_state(block)
    output.square().mean().backward()
    parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in block.named_parameters()
        if parameter.grad is not None
    }
    return (
        output.detach(),
        features.grad.detach(),
        parameter_grads,
        state_after_forward,
        _batch_norm_state(block),
        calls,
    )


@pytest.mark.parametrize("momentum", [0.1, 1.0, None])
@pytest.mark.parametrize(
    "block_cls,in_channels,out_channels",
    [
        (BasicBlock, 8, 8),
        (BottleneckBlock, 8, 16),
    ],
)
def test_mink_block_checkpoint_preserves_gradients_and_batch_norm_state(
    block_cls,
    in_channels,
    out_channels,
    momentum,
):
    torch.manual_seed(43)
    reference = _replace_sparse_convs(
        block_cls(in_channels, out_channels, use_checkpoint=False),
        in_channels,
        out_channels,
        momentum,
    )
    checkpointed = copy.deepcopy(reference)
    checkpointed.use_checkpoint = True
    template = _make_points(in_channels)

    (
        ref_out,
        ref_input_grad,
        ref_parameter_grads,
        ref_forward_state,
        ref_backward_state,
        ref_calls,
    ) = _run_block_backward(reference, template)
    (
        out,
        input_grad,
        parameter_grads,
        forward_state,
        backward_state,
        calls,
    ) = _run_block_backward(checkpointed, template)

    assert ref_calls == 1
    assert calls == 2
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(input_grad, ref_input_grad)
    assert parameter_grads.keys() == ref_parameter_grads.keys()
    for name in parameter_grads:
        torch.testing.assert_close(parameter_grads[name], ref_parameter_grads[name])

    _assert_state_close(ref_forward_state, ref_backward_state)
    _assert_state_close(forward_state, ref_forward_state)
    _assert_state_close(backward_state, ref_forward_state)


def test_mink_block_checkpoint_supports_stateless_batch_norm():
    torch.manual_seed(47)
    reference = _replace_sparse_convs(
        BasicBlock(8, 8),
        8,
        8,
        momentum=0.1,
        track_running_stats=False,
    )
    checkpointed = copy.deepcopy(reference)
    checkpointed.use_checkpoint = True
    template = _make_points(8)

    ref_out, ref_input_grad, ref_parameter_grads, _, _, ref_calls = _run_block_backward(
        reference, template
    )
    out, input_grad, parameter_grads, forward_state, backward_state, calls = _run_block_backward(
        checkpointed, template
    )

    assert ref_calls == 1
    assert calls == 2
    assert not forward_state
    assert not backward_state
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(input_grad, ref_input_grad)
    for name in parameter_grads:
        torch.testing.assert_close(parameter_grads[name], ref_parameter_grads[name])


def test_mink_block_composes_user_and_state_preservation_contexts():
    events = []

    @contextmanager
    def marked_context(name):
        events.append(f"{name}:enter")
        try:
            yield
        finally:
            events.append(f"{name}:exit")

    def user_context_fn():
        return marked_context("forward"), marked_context("recompute")

    block = _replace_sparse_convs(
        BasicBlock(8, 8),
        8,
        8,
        momentum=0.1,
    )
    block.gradient_checkpointing_enable({"context_fn": user_context_fn})
    _, _, _, state_after_forward, state_after_backward, calls = _run_block_backward(
        block, _make_points(8)
    )

    assert calls == 2
    assert events == [
        "forward:enter",
        "forward:exit",
        "recompute:enter",
        "recompute:exit",
    ]
    _assert_state_close(state_after_backward, state_after_forward)


def test_mink_block_preserves_batch_norm_across_retained_graph_backwards():
    block = _replace_sparse_convs(
        BasicBlock(8, 8, use_checkpoint=True),
        8,
        8,
        momentum=0.1,
    )
    points = _make_points(8)
    features = points.feature_tensor.detach().clone().requires_grad_(True)
    output = block(points.replace_features(features)).feature_tensor
    loss = output.square().mean()
    state_after_forward = _batch_norm_state(block)

    loss.backward(retain_graph=True)
    _assert_state_close(_batch_norm_state(block), state_after_forward)

    loss.backward()
    _assert_state_close(_batch_norm_state(block), state_after_forward)


def test_mink_block_preserves_batch_norm_during_gradient_accumulation():
    block = _replace_sparse_convs(
        BasicBlock(8, 8, use_checkpoint=True),
        8,
        8,
        momentum=0.1,
    )
    first = _make_points(8)
    second = _make_points(8)
    first_output = block(first).feature_tensor
    second_output = block(second).feature_tensor
    state_after_forwards = _batch_norm_state(block)

    (first_output.square().mean() + second_output.square().mean()).backward()

    _assert_state_close(_batch_norm_state(block), state_after_forwards)
    assert all(
        child.num_batches_tracked.item() == 2
        for child in block.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    )


def test_mink_block_preserves_batch_norm_with_autograd_grad():
    block = _replace_sparse_convs(
        BasicBlock(8, 8, use_checkpoint=True),
        8,
        8,
        momentum=0.1,
    )
    points = _make_points(8)
    features = points.feature_tensor.detach().clone().requires_grad_(True)
    output = block(points.replace_features(features)).feature_tensor
    state_after_forward = _batch_norm_state(block)

    (input_grad,) = torch.autograd.grad(output.square().mean(), features)

    assert input_grad is not None
    _assert_state_close(_batch_norm_state(block), state_after_forward)


def test_preserve_module_buffers_restores_after_exception():
    batch_norm = nn.BatchNorm1d(4)
    original = _batch_norm_state(batch_norm)

    with pytest.raises(RuntimeError, match="recompute failed"):
        with preserve_module_buffers(
            batch_norm,
            _BATCH_NORM_BUFFERS,
            module_types=nn.modules.batchnorm._BatchNorm,
        ):
            batch_norm.running_mean.add_(3)
            batch_norm.running_var.mul_(5)
            batch_norm.num_batches_tracked.add_(7)
            raise RuntimeError("recompute failed")

    _assert_state_close(_batch_norm_state(batch_norm), original)


def test_mink_block_restores_batch_norm_after_recompute_exception():
    failure = _FailOnSecondCall()
    block = BasicBlock(8, 8, use_checkpoint=True)
    block.conv1 = Sequential(
        nn.Linear(8, 8, bias=False),
        nn.BatchNorm1d(8),
        failure,
    )
    block.conv2 = _linear_conv(
        8,
        8,
        momentum=0.1,
        activation=False,
    )
    output = block(_make_points(8)).feature_tensor
    state_after_forward = _batch_norm_state(block)

    with pytest.raises(RuntimeError, match="recompute failed"):
        output.square().mean().backward()

    assert failure.calls == 2
    _assert_state_close(_batch_norm_state(block), state_after_forward)


def test_mink_checkpoint_rejects_torch_compile_context():
    block = _replace_sparse_convs(
        BasicBlock(8, 8, use_checkpoint=True),
        8,
        8,
        momentum=0.1,
    )

    with mock.patch("torch.compiler.is_compiling", return_value=True):
        with pytest.raises(RuntimeError, match="torch.compile"):
            block(_make_points(8))


def test_mink_unet_model_checkpointing_enable_disable():
    model = MinkUNet18(in_channels=3, out_channels=20)
    boundaries = [
        module for module in model.modules() if hasattr(module, "gradient_checkpointing")
    ]
    state_dict_keys = set(model.state_dict())

    assert model.supports_gradient_checkpointing
    assert len(boundaries) == 16
    assert not model.is_gradient_checkpointing

    model.gradient_checkpointing_enable({"preserve_rng_state": False})
    assert model.is_gradient_checkpointing
    assert all(module.gradient_checkpointing for module in boundaries)
    assert all(
        module._gradient_checkpointing_func.keywords
        == {"preserve_rng_state": False, "use_reentrant": False}
        for module in boundaries
    )
    assert set(model.state_dict()) == state_dict_keys

    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing


def test_mink_unet_constructor_can_enable_checkpointing():
    model = MinkUNet18(in_channels=3, out_channels=20, use_checkpoint=True)
    boundaries = [
        module for module in model.modules() if hasattr(module, "gradient_checkpointing")
    ]

    assert len(boundaries) == 16
    assert model.is_gradient_checkpointing
    assert all(module.gradient_checkpointing for module in boundaries)


def _ddp_checkpoint_worker(
    rank: int,
    world_size: int,
    init_method: str,
    broadcast_buffers: bool,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(59)
        block = _replace_sparse_convs(
            BasicBlock(8, 8, use_checkpoint=True),
            8,
            8,
            momentum=0.1,
        )
        model = DistributedDataParallel(
            block,
            broadcast_buffers=broadcast_buffers,
        )
        points = _make_points(8)
        features = points.feature_tensor.detach().clone() + rank
        output = model(points.replace_features(features)).feature_tensor
        output.square().mean().backward()

        assert all(
            child.num_batches_tracked.item() == 1
            for child in block.modules()
            if isinstance(child, nn.modules.batchnorm._BatchNorm)
        )
        assert all(
            parameter.grad is not None
            for parameter in block.parameters()
            if parameter.requires_grad
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="Gloo distributed backend required",
)
@pytest.mark.parametrize("broadcast_buffers", [False, True])
def test_mink_checkpoint_with_ddp(tmp_path, broadcast_buffers):
    init_path = tmp_path / f"ddp_init_{int(broadcast_buffers)}"
    mp.spawn(
        _ddp_checkpoint_worker,
        args=(
            2,
            f"file://{init_path}",
            broadcast_buffers,
        ),
        nprocs=2,
        join=True,
    )


@_skip_no_cuda
@_skip_no_extension
def test_real_sparse_mink_block_checkpoint_backward():
    torch.manual_seed(53)
    reference = BasicBlock(16, 16).cuda()
    checkpointed = copy.deepcopy(reference)
    checkpointed.use_checkpoint = True
    points = Points(
        [torch.rand(64, 3, device="cuda")],
        [torch.randn(64, 16, device="cuda")],
    )
    template = points_to_voxels(points, voxel_size=0.05)

    def run(block):
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
        state_after_forward = _batch_norm_state(block)
        loss.backward()
        parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in block.named_parameters()
            if parameter.grad is not None
        }
        return (
            output.detach(),
            features.grad.detach(),
            parameter_grads,
            state_after_forward,
            _batch_norm_state(block),
            calls,
        )

    (
        ref_out,
        ref_input_grad,
        ref_parameter_grads,
        ref_forward_state,
        ref_backward_state,
        ref_calls,
    ) = run(reference)
    (
        out,
        input_grad,
        parameter_grads,
        forward_state,
        backward_state,
        calls,
    ) = run(checkpointed)

    assert ref_calls == 1
    assert calls == 2
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(input_grad, ref_input_grad)
    assert parameter_grads.keys() == ref_parameter_grads.keys()
    for name in parameter_grads:
        torch.testing.assert_close(parameter_grads[name], ref_parameter_grads[name])
    _assert_state_close(ref_forward_state, ref_backward_state)
    _assert_state_close(forward_state, ref_forward_state)
    _assert_state_close(backward_state, ref_forward_state)


@_skip_no_cuda
@_skip_no_extension
def test_tiny_mink_unet_checkpointed_backward():
    torch.manual_seed(61)
    points = Points(
        [torch.rand(512, 3, device="cuda")],
        [torch.randn(512, 3, device="cuda")],
    )
    voxels = points_to_voxels(points, voxel_size=0.04)
    model = MinkUNetBase(
        in_channels=3,
        out_channels=5,
        planes=(8, 16, 24, 32, 32, 24, 16, 8),
        layers=(1,) * 8,
        init_dim=8,
        BLOCK=BasicBlock,
        use_checkpoint=True,
    ).cuda()

    with torch.autocast("cuda", dtype=torch.float16):
        output = model(voxels).feature_tensor
        loss = output.float().square().mean()
    loss.backward()

    assert output.shape == (voxels.feature_tensor.shape[0], 5)
    assert all(
        child.num_batches_tracked.item() == 1
        for child in model.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    )
    assert any(parameter.grad is not None for parameter in model.parameters())
