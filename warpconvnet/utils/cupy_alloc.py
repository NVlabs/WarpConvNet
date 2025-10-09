# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

import cupy as cp
import torch


_cupy_allocator = None


def _torch_alloc(size: int, device_id: int) -> Any:
    torch_stream_ptr = torch.cuda.current_stream().cuda_stream
    cupy_stream_ptr = cp.cuda.get_current_stream().ptr
    if torch_stream_ptr != cupy_stream_ptr:
        raise RuntimeError("The current stream set in PyTorch and CuPy must be same.")
    return torch.cuda.caching_allocator_alloc(size, device_id, torch_stream_ptr)


def _torch_free(mem_ptr: int, device_id: int) -> None:
    if torch is None:
        return
    torch.cuda.caching_allocator_delete(mem_ptr)


def set_cupy_allocator(
    alloc_fn: Callable[[int, int], Any] = _torch_alloc,
    free_fn: Callable[[int, int], None] = _torch_free,
) -> None:
    """Set the CuPy allocator to use the PyTorch memory pool."""
    global _cupy_allocator
    if _cupy_allocator is None:
        _cupy_allocator = cp.cuda.memory.PythonFunctionAllocator(alloc_fn, free_fn)
        cp.cuda.set_allocator(_cupy_allocator.malloc)
        print("CuPy allocator set to PyTorch memory pool.")
