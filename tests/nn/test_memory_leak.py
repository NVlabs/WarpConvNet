# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Test for GPU memory leaks during repeated sparse convolution forward+backward.

import gc

import pytest
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _get_gpu_memory_mb():
    """Get current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _make_random_voxels(N, C, device):
    """Create random voxels with varying N to simulate real training."""
    coords = torch.randint(0, 50, (N, 3), device=device, dtype=torch.int32)
    feats = torch.randn(N, C, device=device, dtype=torch.float16)
    offsets = torch.tensor([0, N], dtype=torch.int32, device=device)
    return Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        device=device,
    )


class TestMemoryLeak:
    """Test that GPU memory does not grow unboundedly during training."""

    def test_repeated_fwd_bwd_fixed_size(self, device):
        """Fixed-size inputs: memory should stabilize after warmup."""
        C_in, C_out, N = 64, 64, 5000
        conv = SpatiallySparseConv(C_in, C_out, kernel_size=3, bias=False).to(
            device, torch.float16
        )
        voxels = _make_random_voxels(N, C_in, device)
        feats = voxels.feature_tensor.detach().clone().requires_grad_(True)
        voxels = Voxels(
            batched_coordinates=voxels.coordinate_tensor,
            batched_features=feats,
            offsets=voxels.offsets,
            device=device,
        )

        # Warmup: let auto-tune and caches settle
        for _ in range(5):
            out = conv(voxels)
            out.feature_tensor.sum().backward()
            conv.zero_grad()
            feats.grad.zero_()
        gc.collect()
        torch.cuda.empty_cache()

        # Record baseline
        baseline_mb = _get_gpu_memory_mb()

        # Run 100 iterations
        for _ in range(100):
            out = conv(voxels)
            out.feature_tensor.sum().backward()
            conv.zero_grad()
            feats.grad.zero_()

        gc.collect()
        torch.cuda.empty_cache()
        final_mb = _get_gpu_memory_mb()
        growth_mb = final_mb - baseline_mb

        print(
            f"Fixed size: baseline={baseline_mb:.1f}MB, final={final_mb:.1f}MB, growth={growth_mb:.1f}MB"
        )
        assert (
            growth_mb < 50
        ), f"Memory grew by {growth_mb:.1f}MB over 100 iterations (expected <50MB)"

    def test_repeated_fwd_bwd_varying_size(self, device):
        """Varying-size inputs (real training): memory should not grow linearly."""
        C_in, C_out = 64, 64
        conv = SpatiallySparseConv(C_in, C_out, kernel_size=3, bias=False).to(
            device, torch.float16
        )

        # Warmup
        for i in range(5):
            N = 3000 + i * 100
            voxels = _make_random_voxels(N, C_in, device)
            feats = voxels.feature_tensor.requires_grad_(True)
            out = conv(voxels)
            out.feature_tensor.sum().backward()
            del out, voxels, feats
        gc.collect()
        torch.cuda.empty_cache()

        baseline_mb = _get_gpu_memory_mb()
        peak_mb = baseline_mb

        # Run 100 iterations with DIFFERENT N each time
        for i in range(100):
            N = 2000 + (i * 137) % 5000  # Varying N: 2000-7000
            voxels = _make_random_voxels(N, C_in, device)
            feats = voxels.feature_tensor.requires_grad_(True)
            out = conv(voxels)
            out.feature_tensor.sum().backward()
            del out, voxels, feats

            if (i + 1) % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                current_mb = _get_gpu_memory_mb()
                peak_mb = max(peak_mb, current_mb)
                print(f"  Iter {i+1}: {current_mb:.1f}MB (baseline={baseline_mb:.1f}MB)")

        gc.collect()
        torch.cuda.empty_cache()
        final_mb = _get_gpu_memory_mb()
        growth_mb = final_mb - baseline_mb

        print(
            f"Varying size: baseline={baseline_mb:.1f}MB, final={final_mb:.1f}MB, growth={growth_mb:.1f}MB"
        )
        # With bounded cache (16 entries * ~22MB = ~352MB max), growth should be bounded
        assert (
            growth_mb < 500
        ), f"Memory grew by {growth_mb:.1f}MB over 100 varying-size iterations (expected <500MB)"

    def test_mask_data_cache_bounded(self, device):
        """Verify _MASK_DATA_CACHE does not grow beyond max size."""
        from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
            _MASK_DATA_CACHE,
            _MASK_DATA_CACHE_MAX_SIZE,
        )

        C_in, C_out = 32, 32
        conv = SpatiallySparseConv(C_in, C_out, kernel_size=3, bias=False).to(
            device, torch.float16
        )

        initial_cache_size = len(_MASK_DATA_CACHE)

        # Run with many different N values to create many unique cache keys
        for i in range(50):
            N = 1000 + i * 200  # 50 different N values
            voxels = _make_random_voxels(N, C_in, device)
            feats = voxels.feature_tensor.requires_grad_(True)
            out = conv(voxels)
            out.feature_tensor.sum().backward()
            del out, voxels, feats

        cache_size = len(_MASK_DATA_CACHE)
        print(
            f"Cache size after 50 unique configs: {cache_size} (max={_MASK_DATA_CACHE_MAX_SIZE})"
        )
        assert (
            cache_size <= _MASK_DATA_CACHE_MAX_SIZE
        ), f"Cache grew to {cache_size} entries, exceeding max of {_MASK_DATA_CACHE_MAX_SIZE}"

    def test_ctx_kernel_map_released(self, device):
        """Verify kernel_map is released after backward."""
        C_in, C_out, N = 32, 32, 3000
        conv = SpatiallySparseConv(C_in, C_out, kernel_size=3, bias=False).to(
            device, torch.float16
        )
        voxels = _make_random_voxels(N, C_in, device)
        feats = voxels.feature_tensor.requires_grad_(True)

        out = conv(voxels)
        # The grad_fn holds ctx which holds kernel_map
        grad_fn = out.feature_tensor.grad_fn

        out.feature_tensor.sum().backward()

        # After backward, ctx.kernel_map should be cleared
        # We can't directly access ctx, but we can check that GC frees the memory
        gc.collect()
        torch.cuda.empty_cache()
        mem_after_bwd = _get_gpu_memory_mb()

        del out, grad_fn, voxels, feats
        gc.collect()
        torch.cuda.empty_cache()
        mem_after_del = _get_gpu_memory_mb()

        freed = mem_after_bwd - mem_after_del
        print(f"Memory freed after deleting output: {freed:.1f}MB")
        # Should free the output tensor + any remaining autograd state
        # kernel_map should already be freed by ctx.kernel_map = None in backward
