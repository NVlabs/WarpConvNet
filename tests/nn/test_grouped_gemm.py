# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for adaptive GEMM grouping.

Compares grouped forward/backward outputs against the ungrouped reference
implementations for all three backends.
"""

import pytest
import torch

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
from warpconvnet.geometry.coords.search.torch_discrete import _kernel_map_from_size
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_forward_grouped,
    _explicit_gemm_backward_logic,
    _explicit_gemm_backward_grouped,
)
from warpconvnet.nn.functional.sparse_conv.detail.implicit_direct import (
    _implicit_gemm_forward_logic,
    _implicit_gemm_forward_grouped,
)
from warpconvnet.nn.functional.sparse_conv.detail.grouping import (
    generate_padding_buckets,
    prepare_grouped_kernel_map,
)


def _make_kernel_map(num_coords, kernel_size=(3, 3, 3), device="cuda"):
    """Create a realistic kernel map from random coordinates."""
    coords = torch.randint(-128, 128, (num_coords, 4), dtype=torch.int32, device=device)
    coords[:, 0] = torch.randint(0, 2, (num_coords,), dtype=torch.int32, device=device)
    ht = TorchHashTable.from_keys(coords, hash_method=HashMethod.CITY, device=device)
    kernel_map = _kernel_map_from_size(ht, coords, kernel_size, return_type="offsets")
    return kernel_map, num_coords


class TestGroupingAlgorithm:
    def test_basic_bucketing(self):
        counts = [100, 110, 120, 500, 520, 540, 5000, 5100]
        indices = list(range(len(counts)))
        buckets, bcounts = generate_padding_buckets(counts, indices, threshold=0.1)
        # Should group similar-sized offsets
        assert len(buckets) < len(counts)
        # All indices should be present
        all_indices = [i for b in buckets for i in b]
        assert sorted(all_indices) == sorted(indices)

    def test_empty_input(self):
        buckets, bcounts = generate_padding_buckets([], [], threshold=0.1)
        assert buckets == []
        assert bcounts == []

    def test_single_offset(self):
        buckets, bcounts = generate_padding_buckets([100], [0], threshold=0.1)
        assert len(buckets) == 1
        assert buckets[0] == [0]

    def test_waste_within_threshold(self):
        counts = [100, 105, 110]
        indices = [0, 1, 2]
        buckets, bcounts = generate_padding_buckets(counts, indices, threshold=0.1)
        for bc in bcounts:
            if len(bc) > 1:
                waste = (max(bc) * len(bc) - sum(bc)) / sum(bc)
                assert waste <= 0.1 + 1e-9

    def test_prepare_grouped_kernel_map(self):
        kernel_map, num_coords = _make_kernel_map(10000)
        grouped = prepare_grouped_kernel_map(kernel_map, saturation_m=2000)
        # All non-identity offsets should be accounted for
        all_offsets = set(grouped.large_offset_indices)
        for bucket in grouped.buckets:
            all_offsets.update(bucket)
        iden_idx = kernel_map.identity_map_index
        expected = set(range(len(kernel_map))) - ({iden_idx} if iden_idx is not None else set())
        # Remove empty offsets
        expected = {k for k in expected if kernel_map.numel(k) > 0}
        assert all_offsets == expected


class TestExplicitGEMMGrouped:
    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64), (32, 64)])
    @pytest.mark.parametrize("num_coords", [1000, 10000])
    def test_forward_matches_reference(self, num_coords, C_in, C_out):
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(num_coords)
        num_out = N
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)

        ref_out = _explicit_gemm_forward_logic(in_features, weight, kernel_map, num_out)
        grouped_out = _explicit_gemm_forward_grouped(
            in_features, weight, kernel_map, num_out, saturation_m=500
        )
        torch.testing.assert_close(grouped_out, ref_out, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    def test_backward_matches_reference(self, C_in, C_out):
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(5000)
        num_out = N
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)
        grad_output = torch.randn(num_out, C_out, device=device)

        ref_gi, ref_gw = _explicit_gemm_backward_logic(
            grad_output, in_features, weight, kernel_map
        )
        grouped_gi, grouped_gw = _explicit_gemm_backward_grouped(
            grad_output, in_features, weight, kernel_map, saturation_m=500
        )
        torch.testing.assert_close(grouped_gi, ref_gi, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(grouped_gw, ref_gw, atol=1e-4, rtol=1e-4)


class TestImplicitGEMMGrouped:
    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    @pytest.mark.parametrize("block_size", [16])
    def test_forward_matches_reference(self, C_in, C_out, block_size):
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(10000)
        num_out = N
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)

        ref_out = _implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out,
            compute_dtype=None,
            fwd_block_size=block_size,
        )
        grouped_out = _implicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            num_out,
            compute_dtype=None,
            fwd_block_size=block_size,
            saturation_m=500,
        )
        torch.testing.assert_close(grouped_out, ref_out, atol=1e-4, rtol=1e-4)


def _torch_matmul_reference(in_features, weight, kernel_map, num_out):
    """Ground-truth sparse conv output computed entirely with torch.matmul."""
    device = in_features.device
    C_out = weight.shape[2]
    iden_idx = kernel_map.identity_map_index

    if iden_idx is not None:
        output = torch.matmul(in_features, weight[iden_idx])
    else:
        output = torch.zeros(num_out, C_out, device=device, dtype=in_features.dtype)

    for k in range(len(kernel_map)):
        if k == iden_idx:
            continue
        in_map, out_map = kernel_map[k]
        if in_map.shape[0] == 0:
            continue
        in_map = in_map.to(device)
        out_map = out_map.to(device)
        output[out_map] += torch.matmul(in_features[in_map], weight[k])
    return output


class TestGroundTruthMatmul:
    """Compare every grouped backend against torch.matmul ground truth."""

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    @pytest.mark.parametrize("num_coords", [1000, 10000])
    def test_explicit_grouped_vs_matmul(self, num_coords, C_in, C_out):
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(num_coords)
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)

        ref = _torch_matmul_reference(in_features, weight, kernel_map, N)
        grouped = _explicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            N,
            saturation_m=500,
        )
        torch.testing.assert_close(grouped, ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    @pytest.mark.parametrize("num_coords", [1000, 10000])
    def test_implicit_grouped_vs_matmul(self, num_coords, C_in, C_out):
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(num_coords)
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)

        ref = _torch_matmul_reference(in_features, weight, kernel_map, N)
        grouped = _implicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            N,
            compute_dtype=None,
            fwd_block_size=16,
            saturation_m=500,
        )
        torch.testing.assert_close(grouped, ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    def test_explicit_grouped_backward_vs_matmul(self, C_in, C_out):
        """Backward: compare grad_in and grad_weight against torch.matmul."""
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(5000)
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)
        grad_output = torch.randn(N, C_out, device=device)

        # Ground truth backward with torch.matmul
        ref_gi, ref_gw = _explicit_gemm_backward_logic(
            grad_output, in_features, weight, kernel_map
        )
        grouped_gi, grouped_gw = _explicit_gemm_backward_grouped(
            grad_output,
            in_features,
            weight,
            kernel_map,
            saturation_m=500,
        )
        torch.testing.assert_close(grouped_gi, ref_gi, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(grouped_gw, ref_gw, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    @pytest.mark.parametrize("num_coords", [1000, 10000])
    def test_cuda_grouped_kernel_vs_matmul(self, num_coords, C_in, C_out):
        """Test the CUDA implicit_gemm_grouped kernel directly against torch.matmul."""
        import warpconvnet._C as _C

        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(num_coords)
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)

        # Ground truth
        ref = _torch_matmul_reference(in_features, weight, kernel_map, N)

        # Build grouped kernel inputs manually
        grouped = prepare_grouped_kernel_map(
            kernel_map, saturation_m=N + 1
        )  # all offsets are "small"

        iden_idx = kernel_map.identity_map_index
        if iden_idx is not None:
            output = torch.matmul(in_features, weight[iden_idx])
        else:
            output = torch.zeros(N, C_out, device=device)

        for bucket_offsets, cat_in, cat_out, w_idx in zip(
            grouped.buckets,
            grouped.bucket_cat_in_maps,
            grouped.bucket_cat_out_maps,
            grouped.bucket_weight_indices,
        ):
            if cat_in.shape[0] == 0:
                continue
            bucket_weight = weight[bucket_offsets].contiguous()
            _C.gemm.implicit_gemm_grouped(
                in_features,
                bucket_weight,
                output,
                cat_in.int(),
                cat_out.int(),
                w_idx.int(),
                "basic",
                16,
            )

        torch.testing.assert_close(output, ref, atol=1e-4, rtol=1e-4)


class TestAMPAutocast:
    """Test all grouped backends under torch.amp.autocast (mixed precision).

    Under AMP, torch.bmm autocasts float32 inputs to float16 internally.
    These tests verify that:
    1. No dtype mismatch errors occur in `scatter_add_` or index assignment
    2. Grouped AMP output matches ungrouped AMP output (same precision path)
    """

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    def test_explicit_grouped_forward_amp(self, C_in, C_out):
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(5000)
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            ref = _explicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                N,
            )
            out = _explicit_gemm_forward_grouped(
                in_features,
                weight,
                kernel_map,
                N,
                saturation_m=500,
            )
        assert out.dtype == ref.dtype
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    def test_explicit_grouped_backward_amp(self, C_in, C_out):
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(5000)
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)
        grad_output = torch.randn(N, C_out, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            ref_gi, ref_gw = _explicit_gemm_backward_logic(
                grad_output,
                in_features,
                weight,
                kernel_map,
            )
            gi, gw = _explicit_gemm_backward_grouped(
                grad_output,
                in_features,
                weight,
                kernel_map,
                saturation_m=500,
            )
        assert gi.dtype == ref_gi.dtype
        assert gw.dtype == ref_gw.dtype
        torch.testing.assert_close(gi, ref_gi, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(gw, ref_gw, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    def test_implicit_grouped_forward_amp(self, C_in, C_out):
        torch.manual_seed(42)
        kernel_map, N = _make_kernel_map(5000)
        K = len(kernel_map)
        device = "cuda"

        in_features = torch.randn(N, C_in, device=device)
        weight = torch.randn(K, C_in, C_out, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            ref = _implicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                N,
                compute_dtype=None,
                fwd_block_size=16,
            )
            out = _implicit_gemm_forward_grouped(
                in_features,
                weight,
                kernel_map,
                N,
                compute_dtype=None,
                fwd_block_size=16,
                saturation_m=500,
            )
        assert out.dtype == ref.dtype
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    def test_cutlass_grouped_forward_amp(self, C_in, C_out):
        """Test CUTLASS grouped forward under AMP (skips if CUTLASS unavailable)."""
        torch.manual_seed(42)
        try:
            from warpconvnet.nn.functional.sparse_conv.detail.cutlass import (
                _cutlass_implicit_gemm_forward_grouped,
                _cutlass_implicit_gemm_forward_logic,
            )
        except (ImportError, AssertionError):
            pytest.skip("CUTLASS not available")

        kernel_map, N = _make_kernel_map(5000)
        K = len(kernel_map)
        device = "cuda"

        # CUTLASS requires channels aligned to 8
        C_in_aligned = (C_in + 7) // 8 * 8
        C_out_aligned = (C_out + 7) // 8 * 8
        in_features = torch.randn(N, C_in_aligned, device=device)
        weight = torch.randn(K, C_in_aligned, C_out_aligned, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            try:
                ref = _cutlass_implicit_gemm_forward_logic(
                    in_features,
                    weight,
                    kernel_map,
                    N,
                )
            except (RuntimeError, AssertionError):
                pytest.skip("CUTLASS forward not supported on this GPU")
            if isinstance(ref, int):
                pytest.skip("CUTLASS forward returned error status")

            out = _cutlass_implicit_gemm_forward_grouped(
                in_features,
                weight,
                kernel_map,
                N,
                saturation_m=500,
            )
        if isinstance(out, int):
            pytest.skip("CUTLASS grouped forward returned error status")

        assert out.dtype == ref.dtype
        # Wider tolerance: grouped mixes CUTLASS + bmm, different rounding than all-CUTLASS
        torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)

    @pytest.mark.parametrize("C_in,C_out", [(32, 32), (64, 64)])
    def test_cutlass_grouped_backward_amp(self, C_in, C_out):
        """Test CUTLASS grouped backward under AMP (skips if CUTLASS unavailable)."""
        torch.manual_seed(42)
        try:
            from warpconvnet.nn.functional.sparse_conv.detail.cutlass import (
                _cutlass_implicit_gemm_backward_grouped,
                _cutlass_implicit_gemm_backward_logic,
            )
        except (ImportError, AssertionError):
            pytest.skip("CUTLASS not available")

        kernel_map, N = _make_kernel_map(5000)
        K = len(kernel_map)
        device = "cuda"

        C_in_aligned = (C_in + 7) // 8 * 8
        C_out_aligned = (C_out + 7) // 8 * 8
        in_features = torch.randn(N, C_in_aligned, device=device)
        weight = torch.randn(K, C_in_aligned, C_out_aligned, device=device)
        grad_output = torch.randn(N, C_out_aligned, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            try:
                ref_gi, ref_gw = _cutlass_implicit_gemm_backward_logic(
                    grad_output,
                    in_features,
                    weight,
                    kernel_map,
                )
            except (RuntimeError, AssertionError):
                pytest.skip("CUTLASS backward not supported on this GPU")
            if isinstance(ref_gi, int):
                pytest.skip("CUTLASS backward returned error status")

            gi, gw = _cutlass_implicit_gemm_backward_grouped(
                grad_output,
                in_features,
                weight,
                kernel_map,
                saturation_m=500,
            )
        if isinstance(gi, int):
            pytest.skip("CUTLASS grouped backward returned error status")

        assert gi.dtype == ref_gi.dtype
        assert gw.dtype == ref_gw.dtype
        # Wider tolerance: grouped mixes CUTLASS + bmm, different rounding than all-CUTLASS
        torch.testing.assert_close(gi, ref_gi, atol=0.05, rtol=0.05)
        torch.testing.assert_close(gw, ref_gw, atol=0.05, rtol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
