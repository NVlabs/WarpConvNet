# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Tuple
from contextlib import nullcontext

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import (
    SpatiallySparseConv,
    SPATIALLY_SPARSE_CONV_ALGO_MODE,
    STRIDED_CONV_MODE,
)


def pytest_benchmark_generate_json(config, benchmarks, output_json):
    """Customize benchmark JSON output."""
    # Filter out unwanted stats
    for benchmark in benchmarks:
        benchmark.stats = {"min": benchmark.stats["min"]}


def pytest_addoption(parser):
    """Add custom options to pytest."""
    parser.addoption(
        "--benchmark-only-min",
        action="store_true",
        default=True,
        help="Only show minimum time in benchmark results",
    )


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """Update benchmark JSON format."""
    for benchmark in output_json["benchmarks"]:
        benchmark["stats"] = {"min": benchmark["stats"]["min"]}


@pytest.fixture
def setup_voxel_data():
    """Setup fixed coordinate and feature data for benchmarking."""
    torch.manual_seed(0)
    device = "cuda:0"

    # Fixed configuration for data generation
    B, min_N, max_N = 3, 100000, 1000000
    base_channels = 32  # Use maximum channel size for feature generation

    # Generate fixed coordinates and features
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    # default dtype is float32
    coords = [(torch.rand((N, 3), device=device) / voxel_size).int() for N in Ns]
    features = [torch.randn((N, base_channels), device=device) for N in Ns]

    return coords, features


def _gen_dtype_params(C_in=32, C_out=64, algo=SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM):
    """Generate dtype test parameters."""
    base_config = (C_in, C_out)
    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "none": None,
    }

    params = []
    # Format: (in_dtype, compute_dtype) pairs
    configs = [
        ("fp32", "none"),
        ("fp32", "fp32"),
        ("fp16", "fp32"),
        ("fp16", "fp16"),
        ("fp16", "bf16"),
        ("fp16", "none"),
        ("bf16", "fp16"),
        ("bf16", "bf16"),
        ("bf16", "none"),
        ("none", "none"),
    ]

    for dtype_key, compute_key in configs:
        params.append(
            pytest.param(
                (*base_config, dtypes[dtype_key], dtypes[compute_key], algo),
                id=f"{dtype_key}_{compute_key}",
            )
        )

    return params


@pytest.fixture(params=_gen_dtype_params())
def setup_sparse_conv(
    request, setup_voxel_data
) -> Tuple[SpatiallySparseConv, Voxels, torch.dtype]:
    """Setup sparse convolution with different configurations."""
    C_in, C_out, dtype, compute_dtype, algo_mode = request.param
    device = "cuda:0"

    coords, base_features = setup_voxel_data

    # Slice features if needed to match in_channels
    features = [f[:, :C_in] for f in base_features]
    input_voxels = Voxels(coords, features, device=device).unique()

    # Create conv layer with various configurations
    conv = SpatiallySparseConv(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        stride=(2, 2, 2),
        conv_algo=algo_mode,
        stride_mode=STRIDED_CONV_MODE.REDUCE_AND_STRIDE,
        out_code_backend="morton",
        compute_dtype=compute_dtype,
    ).to(device)

    return conv, input_voxels, dtype


@pytest.mark.benchmark(group="sparse_conv_forward")
def test_forward_sparse_conv(setup_sparse_conv, benchmark):
    """Benchmark forward pass of sparse convolution."""
    conv, input_voxels, test_dtype = setup_sparse_conv
    if test_dtype is not None:
        input_voxels = input_voxels.to(dtype=test_dtype)

    use_amp = test_dtype in [torch.float16, torch.bfloat16]

    def run_forward():
        with torch.no_grad():
            # Use autocast for FP16, nullcontext for FP32
            ctx = torch.cuda.amp.autocast(dtype=test_dtype) if use_amp else nullcontext()
            with ctx:
                return conv(input_voxels)

    result = benchmark.pedantic(
        run_forward,
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="sparse_conv_backward")
def test_forward_backward_sparse_conv(setup_sparse_conv, benchmark):
    """Benchmark forward + backward pass of sparse convolution."""
    conv, input_voxels, test_dtype = setup_sparse_conv
    if test_dtype is not None:
        input_voxels = input_voxels.to(dtype=test_dtype)

    # Create GradScaler for FP16
    use_amp = test_dtype in [torch.float16, torch.bfloat16]
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    optimizer = torch.optim.Adam(conv.parameters())

    def run_forward_backward():
        # Use autocast for FP16, nullcontext for FP32
        ctx = torch.cuda.amp.autocast(dtype=test_dtype) if use_amp else nullcontext()
        with ctx:
            output = conv(input_voxels)
            loss = output.features.mean()

        if scaler is not None:
            # FP16 training path
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32 training path
            loss.backward()
            optimizer.step()

        return output

    # Reset gradients before each benchmark iteration
    def run_with_reset():
        optimizer.zero_grad(set_to_none=True)
        return run_forward_backward()

    result = benchmark.pedantic(
        run_with_reset,
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )


# New parameter generation function
def _gen_algo_block_size_dtype_params(C_in=32, C_out=64):
    params = []
    dtypes = {
        "fp32": torch.float32,
        "none": None,  # Represents using the input tensor's dtype from setup_voxel_data (fp32)
    }
    # (input_dtype_key, compute_dtype_key)
    dtype_configs = [
        ("fp32", "fp32"),
    ]

    algos_to_test = [
        SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM,
        SPATIALLY_SPARSE_CONV_ALGO_MODE.IMPLICIT_GEMM,
    ]
    # (fwd_block_size, bwd_block_size) for Implicit GEMM
    implicit_gemm_block_config = [(8, 8), (16, 16), (32, 32)]

    for algo_mode in algos_to_test:
        for in_dtype_key, compute_dtype_key in dtype_configs:
            input_dtype_val = dtypes[in_dtype_key]
            compute_dtype_val = dtypes[compute_dtype_key]

            if algo_mode == SPATIALLY_SPARSE_CONV_ALGO_MODE.IMPLICIT_GEMM:
                for fwd_bs, bwd_bs in implicit_gemm_block_config:
                    param_id = f"{algo_mode.value}_fwd{fwd_bs}_bwd{bwd_bs}_{in_dtype_key}inp_{compute_dtype_key}comp"
                    params.append(
                        pytest.param(
                            (
                                C_in,
                                C_out,
                                input_dtype_val,
                                compute_dtype_val,
                                algo_mode,
                                fwd_bs,
                                bwd_bs,
                            ),
                            id=param_id,
                        )
                    )
            else:  # EXPLICIT_GEMM
                # Block sizes are not applicable for explicit gemm, pass None
                param_id = f"{algo_mode.value}_{in_dtype_key}inp_{compute_dtype_key}comp"
                params.append(
                    pytest.param(
                        (C_in, C_out, input_dtype_val, compute_dtype_val, algo_mode, None, None),
                        id=param_id,
                    )
                )
    return params


# New fixture using these params
@pytest.fixture(params=_gen_algo_block_size_dtype_params())
def setup_sparse_conv_for_algo_bench(request, setup_voxel_data):
    C_in, C_out, input_dtype_config, compute_dtype_config, algo_mode, fwd_bs, bwd_bs = (
        request.param
    )
    device = "cuda:0"

    coords, base_features = setup_voxel_data  # base_features are fp32

    # Prepare features with the configured input_dtype_config
    # If input_dtype_config is None, features remain fp32. Otherwise, they are converted.
    if input_dtype_config is not None:
        features = [f[:, :C_in].to(dtype=input_dtype_config) for f in base_features]
    else:
        features = [f[:, :C_in] for f in base_features]  # Still slice to C_in

    input_voxels = Voxels(coords, features, device=device).unique()

    conv_params = {
        "in_channels": C_in,
        "out_channels": C_out,
        "kernel_size": 3,  # Consistent with existing benchmarks
        "stride": (2, 2, 2),  # Consistent
        "conv_algo": algo_mode,
        "stride_mode": STRIDED_CONV_MODE.STRIDE_ONLY,  # Consistent
        "out_code_backend": "morton",  # Consistent
        "compute_dtype": compute_dtype_config,  # This can be None
    }

    if algo_mode == SPATIALLY_SPARSE_CONV_ALGO_MODE.IMPLICIT_GEMM:
        # These will use SpatiallySparseConv defaults (32) if fwd_bs/bwd_bs are None
        # However, our _gen_algo_block_size_dtype_params provides non-None for IMPLICIT.
        if fwd_bs is not None:
            conv_params["implicit_matmul_fwd_block_size"] = fwd_bs
        if bwd_bs is not None:
            conv_params["implicit_matmul_bwd_block_size"] = bwd_bs

    conv = SpatiallySparseConv(**conv_params).to(device)

    return conv, input_voxels


# New benchmark function for forward-backward with algo and block size variations
@pytest.mark.benchmark(group="sparse_conv_fwd_bwd_algo_block_size")
def test_fwd_bwd_sparse_conv_algo_block_size(setup_sparse_conv_for_algo_bench, benchmark):
    conv, input_voxels = setup_sparse_conv_for_algo_bench

    use_amp = input_voxels.feature_tensor.dtype in [torch.float16, torch.bfloat16]
    amp_dtype = input_voxels.feature_tensor.dtype if use_amp else None

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    # Ensure optimizer is created for each benchmark instance if params are changing
    optimizer = torch.optim.Adam(conv.parameters())

    def run_forward_backward():
        # Ensure optimizer gradients are zeroed for each actual run, not just per benchmark call
        optimizer.zero_grad(set_to_none=True)

        # Use autocast for mixed precision if applicable
        autocast_context = (
            torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
        )

        with autocast_context:
            output = conv(input_voxels)
            # Ensure there are features to calculate loss, handle potential empty output
            loss = (
                output.features.mean()
                if output.features.numel() > 0
                else torch.tensor(0.0, device=output.device, dtype=output.features.dtype)
            )

        if scaler is not None:  # AMP path
            scaler.scale(loss).backward()
            # Unscale before clipping, if any grad clipping is added later
            # scaler.unscale_(optimizer) # Optional: if grad clipping needed
            scaler.step(optimizer)
            scaler.update()
        else:  # Non-AMP path
            loss.backward()
            optimizer.step()

        return output

    # The benchmark.pedantic function calls run_with_reset multiple times.
    # zero_grad is better placed inside run_forward_backward or run_with_reset.
    # Original benchmark had it in run_with_reset, let's keep that pattern
    # but ensure it's done correctly.
    # The current test_forward_backward_sparse_conv has run_with_reset that calls optimizer.zero_grad
    # My version of run_forward_backward now includes it.

    # The `run_with_reset` wrapper is good for things that need to happen once per benchmark call's internal iteration setup.
    # Since optimizer.zero_grad() should happen before *each* forward/backward pass pair if the same optimizer
    # instance is reused across multiple internal iterations of `pedantic`, it's fine in `run_forward_backward`.
    # benchmark.pedantic's `iterations` are the ones where zero_grad should apply each time.

    benchmark.pedantic(
        run_forward_backward,  # Directly use the function that has zero_grad
        iterations=10,  # Consistent with existing benchmarks
        rounds=3,
        warmup_rounds=1,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-columns=min,median"])
