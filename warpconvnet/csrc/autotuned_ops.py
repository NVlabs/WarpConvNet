# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Tuple
import math

import torch

import warpconvnet._C as _C
from warpconvnet.utils.benchmark_cache import make_autotuned_op


def _device_sm_dtype_key(*tensors: torch.Tensor) -> Tuple[int, Tuple[int, int], str]:
    dev = tensors[0].device.index if tensors else torch.cuda.current_device()
    sm = torch.cuda.get_device_capability(dev)
    # Use first tensor's dtype as representative; ops below are homogeneous dtypes by design
    dtype = str(tensors[0].dtype) if tensors else str(torch.float16)
    return dev, sm, dtype


def _key_trAB_gather(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    tensor_d: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    *,
    accumulator_type: torch.dtype = torch.float32,
    split_k_slices: int = 1,
    mma_tile: int = 0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[Any, ...]:
    dev, sm, dtype = _device_sm_dtype_key(tensor_a)
    # Shapes: A[M_A, K], B[M_B, N], output K x N
    # Keep channel sizes exact (K, N); bucket indices length in log2 space
    _, K = tensor_a.shape
    _, N = tensor_b.shape
    len_a = int(indices_a.numel())
    len_b = int(indices_b.numel())
    log_len_a = int(math.ceil(math.log2(len_a))) if len_a > 0 else 0
    log_len_b = int(math.ceil(math.log2(len_b))) if len_b > 0 else 0
    return (
        "cutlass_gemm_trAB_gather",
        dev,
        sm,
        dtype,
        str(accumulator_type),
        K,
        N,
        log_len_a,
        log_len_b,
    )


def _key_AD_gather_scatter(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    tensor_d: torch.Tensor,
    indices_a: torch.Tensor,
    indices_d: torch.Tensor,
    *,
    accumulator_type: torch.dtype = torch.float32,
    split_k_slices: int = 1,
    mma_tile: int = 0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[Any, ...]:
    dev, sm, dtype = _device_sm_dtype_key(tensor_a)
    # Shapes: A[M, K], B[K, N], output D[out_size, N]
    _, K = tensor_a.shape
    K2, N = tensor_b.shape
    assert K2 == K
    len_a = int(indices_a.numel())
    len_d = int(indices_d.numel())
    log_len_a = int(math.ceil(math.log2(len_a))) if len_a > 0 else 0
    log_len_d = int(math.ceil(math.log2(len_d))) if len_d > 0 else 0
    # Keep channel sizes exact (K, N); indices lengths in log2 space
    return (
        "cutlass_gemm_AD_gather_scatter",
        dev,
        sm,
        dtype,
        str(accumulator_type),
        K,
        N,
        log_len_a,
        log_len_d,
    )


# AD_gather_scatter operation (used in forward pass and backward pass for input gradients)
_BENCHMARK_AD_GATHER_SCATTER_PARAMS = [
    # Base configurations without split-K
    *[{"mma_tile": tile, "split_k_slices": 1} for tile in range(4)],
    # Successful combinations from current benchmarks
    *[
        {"mma_tile": tile, "split_k_slices": k}
        for tile, k in [
            # Proven successful combinations from forward/backward results
            (0, 2),
            (0, 4),  # Intermediate between successful 2 and 8
            (0, 8),  # MMA=0 patterns
            (1, 2),
            (1, 4),
            (1, 16),
            (1, 32),  # MMA=1 patterns
            (3, 2),
            (3, 4),
            (3, 8),
            (3, 16),
            (3, 32),  # MMA=3 patterns (most successful)
        ]
    ],
]

# trAB_gather operation (used only in backward pass for weight gradients)
_BENCHMARK_TRAB_GATHER_PARAMS = [
    # Base configurations without split-K
    *[{"mma_tile": tile, "split_k_slices": 1} for tile in range(4)],
    # Successful combinations from backward results
    *[
        {"mma_tile": tile, "split_k_slices": k}
        for tile, k in [
            (0, 8),  # MMA=0 for large problems
            (1, 4),  # Intermediate between successful 1 and 8
            (1, 8),
            (1, 16),
            (1, 32),  # MMA=1 TrAB patterns
            (3, 4),
            (3, 8),
            (3, 16),
            (3, 32),  # MMA=3 TrAB patterns (most successful)
        ]
    ],
]


def _run_and_time_status(
    c_fn,
    candidate: Dict[str, Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    output_positions: Tuple[int, int],
) -> float | int:
    """
    Returns:
        float: The minimum latency in milliseconds.
        int: The status code returned by the kernel.
    """
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    def _clone_outputs_for_args(a: Tuple[Any, ...]) -> Tuple[Any, ...]:
        a_list = list(a)
        pos_c, pos_d = output_positions
        if isinstance(a_list[pos_c], torch.Tensor):
            a_list[pos_c] = a_list[pos_c].clone()
        if isinstance(a_list[pos_d], torch.Tensor):
            a_list[pos_d] = a_list[pos_d].clone()
        return tuple(a_list)

    # Warmup on cloned outputs
    warm_args = _clone_outputs_for_args(args)
    status = c_fn(*warm_args, **kwargs, **candidate)
    if status != 0:
        return status
    torch.cuda.current_stream().synchronize()

    # Timed iterations on fresh cloned outputs each time; return min latency
    iters = 3
    best_ms = float("inf")
    for _ in range(iters):
        run_args = _clone_outputs_for_args(args)
        starter.record()
        status = c_fn(*run_args, **kwargs, **candidate)
        ender.record()
        ender.synchronize()
        if status != 0:
            return status
        ms = float(starter.elapsed_time(ender))
        if ms < best_ms:
            best_ms = ms
    return best_ms


# Public autotuned wrappers
cutlass_gemm_trAB_gather_autotuned = make_autotuned_op(
    namespace="implicit_gemm_trAB_gather",
    c_fn=_C.gemm.cutlass_gemm_trAB_gather,
    param_space=_BENCHMARK_TRAB_GATHER_PARAMS,
    key_fn=_key_trAB_gather,
    run_and_time_fn=lambda cand, a_kw, k_kw: _run_and_time_status(
        _C.gemm.cutlass_gemm_trAB_gather, cand, a_kw, k_kw, output_positions=(2, 3)
    ),
    record_failures_as_inf=False,
)

cutlass_gemm_AD_gather_scatter_autotuned = make_autotuned_op(
    namespace="implicit_gemm_AD_gather_scatter",
    c_fn=_C.gemm.cutlass_gemm_AD_gather_scatter,
    param_space=_BENCHMARK_AD_GATHER_SCATTER_PARAMS,
    key_fn=_key_AD_gather_scatter,
    run_and_time_fn=lambda cand, a_kw, k_kw: _run_and_time_status(
        _C.gemm.cutlass_gemm_AD_gather_scatter, cand, a_kw, k_kw, output_positions=(2, 3)
    ),
    record_failures_as_inf=False,
)
