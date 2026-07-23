# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified activation-checkpointing support for WarpConvNet modules.

The mixins in this module follow the attribute and method conventions used by
Hugging Face ``PreTrainedModel`` without depending on Transformers. Models can
therefore be controlled directly or discovered through an HF parent model.

Checkpointable leaf modules use ``GradientCheckpointingMixin`` and route
their forward implementation through ``_gradient_checkpointed_call``. Model
containers use ``GradientCheckpointingModelMixin``, or callers can apply
``configure_gradient_checkpointing`` to any module tree.

Typical leaf integration::

    class Block(GradientCheckpointingMixin, nn.Module):
        def __init__(self, use_checkpoint=False):
            super().__init__()
            self._init_gradient_checkpointing(use_checkpoint)

        def _forward(self, x):
            return self.layers(x)

        def forward(self, x):
            return self._gradient_checkpointed_call(self._forward, x)
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from functools import partial
from typing import Any, Callable, ContextManager

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


__all__ = [
    "GradientCheckpointingMixin",
    "GradientCheckpointingModelMixin",
    "configure_gradient_checkpointing",
    "preserve_module_buffers",
]


def _checkpointing_function(
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
) -> Callable:
    kwargs = dict(gradient_checkpointing_kwargs or {})
    kwargs.setdefault("use_reentrant", False)
    if kwargs["use_reentrant"]:
        raise ValueError(
            "WarpConvNet gradient checkpointing requires use_reentrant=False "
            "because inputs and outputs may be carried inside Geometry objects"
        )
    return partial(checkpoint, **kwargs)


def _normalize_checkpointing_function(function: Callable) -> Callable:
    """Enforce non-reentrant mode on a standard PyTorch checkpoint partial."""
    if function is checkpoint:
        return partial(checkpoint, use_reentrant=False)
    if not isinstance(function, partial) or function.func is not checkpoint:
        return function

    kwargs = dict(function.keywords or {})
    kwargs.setdefault("use_reentrant", False)
    if kwargs["use_reentrant"]:
        raise ValueError(
            "WarpConvNet gradient checkpointing requires use_reentrant=False "
            "because inputs and outputs may be carried inside Geometry objects"
        )
    return partial(checkpoint, *function.args, **kwargs)


class _PreserveModuleBuffers:
    """Reusable context that restores in-place buffer mutations on exit."""

    def __init__(
        self,
        module: nn.Module,
        buffer_names: tuple[str, ...],
        module_types: type[nn.Module] | tuple[type[nn.Module], ...] | None,
    ):
        self.module = module
        self.buffer_names = buffer_names
        self.module_types = module_types
        self._snapshot_stack: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

    def __enter__(self) -> None:
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        for child in self.module.modules():
            if self.module_types is not None and not isinstance(
                child,
                self.module_types,
            ):
                continue
            for name in self.buffer_names:
                buffer = getattr(child, name, None)
                if isinstance(buffer, torch.Tensor):
                    snapshots.append((buffer, buffer.detach().clone()))
        self._snapshot_stack.append(snapshots)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        snapshots = self._snapshot_stack.pop()
        with torch.no_grad():
            for buffer, snapshot in snapshots:
                buffer.copy_(snapshot)
        return False


def preserve_module_buffers(
    module: nn.Module,
    buffer_names: tuple[str, ...],
    module_types: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
) -> ContextManager:
    """Return a reusable context that restores selected in-place buffers.

    This is intended for non-reentrant checkpoint recomputation. Snapshots are
    taken on every context entry, after the logical forward has already updated
    its state, and restored after each replay, including exceptional exits and
    retained-graph backward calls.

    Args:
        module: Root whose descendants should be inspected.
        buffer_names: Attribute names of tensor buffers to preserve.
        module_types: Optional module type filter.
    """
    return _PreserveModuleBuffers(module, buffer_names, module_types)


class _CombinedContexts:
    """Combine context objects without making reusable inputs one-shot."""

    def __init__(self, contexts: tuple[ContextManager, ...]):
        self.contexts = contexts
        self._exit_stack: list[ExitStack] = []

    def __enter__(self) -> None:
        stack = ExitStack()
        try:
            for context in self.contexts:
                stack.enter_context(context)
        except BaseException:
            stack.__exit__(*sys.exc_info())
            raise
        self._exit_stack.append(stack)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        stack = self._exit_stack.pop()
        return bool(stack.__exit__(exc_type, exc_value, traceback))


def _compose_checkpoint_context_fns(
    first: Callable[[], tuple[ContextManager, ContextManager]],
    second: Callable[[], tuple[ContextManager, ContextManager]],
) -> Callable[[], tuple[ContextManager, ContextManager]]:
    def composed() -> tuple[ContextManager, ContextManager]:
        first_forward, first_recompute = first()
        second_forward, second_recompute = second()
        return (
            _CombinedContexts((first_forward, second_forward)),
            _CombinedContexts((first_recompute, second_recompute)),
        )

    return composed


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is not None and is_compiling():
        return True

    dynamo = getattr(torch, "_dynamo", None)
    is_dynamo_compiling = getattr(dynamo, "is_compiling", None)
    return bool(is_dynamo_compiling is not None and is_dynamo_compiling())


def _checkpoint_with_context(
    checkpointing_function: Callable,
    context_fn: Callable[[], tuple[ContextManager, ContextManager]],
    function: Callable,
    *args,
    **kwargs,
):
    if _is_torch_compiling():
        raise RuntimeError(
            "WarpConvNet checkpoint boundaries that preserve recomputation "
            "state are not supported under torch.compile because PyTorch "
            "requires compiled checkpoint context_fn values to be "
            "TorchDispatchMode instances"
        )

    if checkpointing_function is checkpoint:
        existing_context_fn = None
    elif isinstance(checkpointing_function, partial) and checkpointing_function.func is checkpoint:
        existing_context_fn = (checkpointing_function.keywords or {}).get("context_fn")
    else:
        raise TypeError(
            "Recomputation state preservation requires the standard "
            "torch.utils.checkpoint.checkpoint callable"
        )

    if existing_context_fn is not None:
        context_fn = _compose_checkpoint_context_fns(
            existing_context_fn,
            context_fn,
        )

    return checkpointing_function(
        function,
        *args,
        context_fn=context_fn,
        **kwargs,
    )


def configure_gradient_checkpointing(
    module: nn.Module,
    enable: bool = True,
    gradient_checkpointing_func: Callable | None = None,
    *,
    strict: bool = True,
) -> int:
    """Configure all checkpoint-aware descendants of ``module``.

    Args:
        module: Root of the module tree to update.
        enable: Whether checkpointing should be active.
        gradient_checkpointing_func: Optional checkpoint callable injected by
            Hugging Face or built by ``gradient_checkpointing_enable``.
        strict: Raise when the tree has no checkpoint-aware modules.

    Returns:
        Number of modules updated.
    """
    if gradient_checkpointing_func is not None:
        gradient_checkpointing_func = _normalize_checkpointing_function(
            gradient_checkpointing_func
        )

    count = 0
    for child in module.modules():
        if not hasattr(child, "gradient_checkpointing"):
            continue
        child.gradient_checkpointing = bool(enable)
        if gradient_checkpointing_func is not None:
            child._gradient_checkpointing_func = gradient_checkpointing_func
        count += 1

    if strict and count == 0:
        raise ValueError(
            f"{module.__class__.__name__} contains no gradient-checkpointable modules"
        )
    return count


class GradientCheckpointingModelMixin:
    """Model-level enable/disable API compatible with Hugging Face."""

    supports_gradient_checkpointing = True

    @property
    def use_checkpoint(self) -> bool:
        """Aggregate backward-compatible alias for model containers."""
        return self.is_gradient_checkpointing

    @use_checkpoint.setter
    def use_checkpoint(self, value: bool) -> None:
        # Some existing model constructors assign this before creating their
        # block list. ``strict=False`` makes that assignment a harmless no-op;
        # the constructor still forwards the value to each block.
        configure_gradient_checkpointing(self, enable=bool(value), strict=False)

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable | None = None,
    ) -> None:
        configure_gradient_checkpointing(
            self,
            enable=enable,
            gradient_checkpointing_func=gradient_checkpointing_func,
        )

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=_checkpointing_function(gradient_checkpointing_kwargs),
        )

    def gradient_checkpointing_disable(self) -> None:
        self._set_gradient_checkpointing(enable=False)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return any(
            child.gradient_checkpointing
            for child in self.modules()
            if hasattr(child, "gradient_checkpointing")
        )


class GradientCheckpointingMixin(GradientCheckpointingModelMixin):
    """Mixin for a leaf module that owns a checkpoint boundary.

    Subclasses call ``_init_gradient_checkpointing`` from ``__init__`` and
    route their ordinary implementation through
    ``_gradient_checkpointed_call`` from ``forward``.
    """

    def _init_gradient_checkpointing(self, use_checkpoint: bool = False) -> None:
        self.gradient_checkpointing = bool(use_checkpoint)
        self._gradient_checkpointing_func = _checkpointing_function()

    @property
    def _gradient_checkpointing_func(self) -> Callable:
        return self.__gradient_checkpointing_func

    @_gradient_checkpointing_func.setter
    def _gradient_checkpointing_func(self, function: Callable) -> None:
        # Hugging Face assigns this attribute directly while walking a parent
        # model, so normalize at the leaf boundary as well as in our tree helper.
        self.__gradient_checkpointing_func = _normalize_checkpointing_function(function)

    def _get_gradient_checkpointing_context_fn(
        self,
    ) -> Callable[[], tuple[ContextManager, ContextManager]] | None:
        """Return optional forward/recompute contexts for this boundary."""
        return None

    @property
    def use_checkpoint(self) -> bool:
        """Backward-compatible alias for ``gradient_checkpointing``."""
        return self.gradient_checkpointing

    @use_checkpoint.setter
    def use_checkpoint(self, value: bool) -> None:
        self.gradient_checkpointing = bool(value)

    def _gradient_checkpointed_call(self, function: Callable, *args, **kwargs):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            context_fn = self._get_gradient_checkpointing_context_fn()
            if context_fn is not None:
                return _checkpoint_with_context(
                    self._gradient_checkpointing_func,
                    context_fn,
                    function,
                    *args,
                    **kwargs,
                )
            return self._gradient_checkpointing_func(function, *args, **kwargs)
        return function(*args, **kwargs)
