#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: eager vs torch.compile inference on sparse conv models.

Usage:
    python tests/nn/bench_torch_compile.py
"""

import time
import torch
import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv


# ---------------------------------------------------------------------------
# Helper: create fresh voxel data each call (avoid caching artifacts)
# ---------------------------------------------------------------------------
def make_voxels(B=2, min_N=5_000, max_N=10_000, C=32, device="cuda", seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [(torch.rand((int(N.item()), 3)) / 0.02).int() for N in Ns]
    feats = [torch.rand((int(N.item()), C)) for N in Ns]
    return Voxels(coords, feats, device=device).unique()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ThreeLayerSparseConv(nn.Module):
    """Conv → BN → ReLU repeated three times."""

    def __init__(self, C, algo="explicit_gemm"):
        super().__init__()
        self.conv1 = SpatiallySparseConv(C, C, 3, fwd_algo=algo, dgrad_algo=algo, wgrad_algo=algo)
        self.bn1 = nn.BatchNorm1d(C)
        self.conv2 = SpatiallySparseConv(C, C, 3, fwd_algo=algo, dgrad_algo=algo, wgrad_algo=algo)
        self.bn2 = nn.BatchNorm1d(C)
        self.conv3 = SpatiallySparseConv(C, C, 3, fwd_algo=algo, dgrad_algo=algo, wgrad_algo=algo)
        self.bn3 = nn.BatchNorm1d(C)

    def forward(self, x: Voxels) -> Voxels:
        x = self.conv1(x)
        feats = torch.relu(self.bn1(x.feature_tensor))
        x = x.replace(batched_features=feats)

        x = self.conv2(x)
        feats = torch.relu(self.bn2(x.feature_tensor))
        x = x.replace(batched_features=feats)

        x = self.conv3(x)
        feats = torch.relu(self.bn3(x.feature_tensor))
        return x.replace(batched_features=feats)


class DownsampleModel(nn.Module):
    """s1 conv → s2 conv → s1 conv  (encoder-like)."""

    def __init__(self, C, algo="explicit_gemm"):
        super().__init__()
        kw = dict(fwd_algo=algo, dgrad_algo=algo, wgrad_algo=algo)
        self.conv1 = SpatiallySparseConv(C, C, 3, **kw)
        self.bn1 = nn.BatchNorm1d(C)
        self.conv2 = SpatiallySparseConv(C, 2 * C, 3, stride=2, **kw)
        self.bn2 = nn.BatchNorm1d(2 * C)
        self.conv3 = SpatiallySparseConv(2 * C, 2 * C, 3, **kw)
        self.bn3 = nn.BatchNorm1d(2 * C)

    def forward(self, x: Voxels) -> Voxels:
        x = self.conv1(x)
        feats = torch.relu(self.bn1(x.feature_tensor))
        x = x.replace(batched_features=feats)

        x = self.conv2(x)
        feats = torch.relu(self.bn2(x.feature_tensor))
        x = x.replace(batched_features=feats)

        x = self.conv3(x)
        feats = torch.relu(self.bn3(x.feature_tensor))
        return x.replace(batched_features=feats)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def benchmark_fn(fn, inputs, warmup=5, iters=20, label=""):
    """Time a function over pre-built inputs with CUDA sync."""
    # Warmup
    for i in range(warmup):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            fn(inputs[i % len(inputs)])
    torch.cuda.synchronize()

    # Timed iterations — each uses different data
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for i in range(iters):
        inp = inputs[(warmup + i) % len(inputs)]
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            fn(inp)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    med = sorted(times_ms)[len(times_ms) // 2]
    mean = sum(times_ms) / len(times_ms)
    lo = min(times_ms)
    hi = max(times_ms)
    print(
        f"  {label:30s}  median={med:7.2f} ms  mean={mean:7.2f} ms  "
        f"min={lo:7.2f} ms  max={hi:7.2f} ms"
    )
    return med


def run_benchmark(
    model_cls, model_name, C, N_range, B=2, algo="explicit_gemm", warmup=10, iters=30
):
    """Compare eager vs compiled for one model configuration."""
    print(f"\n{'='*72}")
    print(f"  {model_name}  (C={C}, N={N_range}, B={B}, algo={algo})")
    print(f"{'='*72}")

    model = model_cls(C, algo=algo).cuda().eval()

    # Pre-generate distinct inputs (different voxel data each iteration)
    total_inputs = warmup + iters
    inputs = [
        make_voxels(B=B, min_N=N_range[0], max_N=N_range[1], C=C, seed=42 + i)
        for i in range(total_inputs)
    ]

    # --- Eager ---
    eager_med = benchmark_fn(model, inputs, warmup=warmup, iters=iters, label="eager")

    # --- Compiled ---
    compiled = torch.compile(model, fullgraph=False)
    # Extra warmup for compilation
    compiled_med = benchmark_fn(
        compiled, inputs, warmup=warmup, iters=iters, label="torch.compile"
    )

    speedup = eager_med / compiled_med if compiled_med > 0 else float("inf")
    print(f"  {'speedup':30s}  {speedup:.3f}x")
    return eager_med, compiled_med


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.cuda.get_device_name()
    print(f"Device: {dev}")
    print(f"PyTorch: {torch.__version__}")

    configs = [
        # (model_cls, name, C, N_range, B, algo)
        (ThreeLayerSparseConv, "3-layer s1 conv (small)", 32, (2_000, 5_000), 2, "explicit_gemm"),
        (
            ThreeLayerSparseConv,
            "3-layer s1 conv (medium)",
            64,
            (10_000, 20_000),
            2,
            "explicit_gemm",
        ),
        (
            ThreeLayerSparseConv,
            "3-layer s1 conv (large)",
            64,
            (50_000, 80_000),
            2,
            "explicit_gemm",
        ),
        (DownsampleModel, "encoder 3-layer (small)", 32, (2_000, 5_000), 2, "explicit_gemm"),
        (DownsampleModel, "encoder 3-layer (medium)", 64, (10_000, 20_000), 2, "explicit_gemm"),
        (DownsampleModel, "encoder 3-layer (large)", 64, (50_000, 80_000), 2, "explicit_gemm"),
    ]

    results = []
    for model_cls, name, C, N_range, B, algo in configs:
        eager, compiled = run_benchmark(
            model_cls,
            name,
            C,
            N_range,
            B=B,
            algo=algo,
            warmup=10,
            iters=30,
        )
        results.append((name, eager, compiled))

    # Summary table
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Config':<35s} {'Eager':>9s} {'Compiled':>9s} {'Speedup':>8s}")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*8}")
    for name, eager, compiled in results:
        sp = eager / compiled if compiled > 0 else float("inf")
        print(f"  {name:<35s} {eager:8.2f}ms {compiled:8.2f}ms {sp:7.3f}x")


if __name__ == "__main__":
    main()
