"""Test mask-based implicit GEMM correctness against explicit_gemm reference."""

import os
import torch
import pytest

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
    _kernel_map_to_mask_data,
    _mask_implicit_gemm_forward_logic,
)
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_forward_logic,
)


def _make_test_data(num_voxels=500, C_in=32, batch_size=2, spatial_range=50):
    """Create random voxels and generate kernel map via SparseConv3d internals."""
    torch.manual_seed(42)
    coords_list = []
    feats_list = []
    for b in range(batch_size):
        n = num_voxels // batch_size
        coords = torch.randint(0, spatial_range, (n, 3), dtype=torch.int32)
        coords = torch.unique(coords, dim=0)
        feats = torch.randn(coords.shape[0], C_in, dtype=torch.float32)
        coords_list.append(coords)
        feats_list.append(feats)
    voxels = Voxels(coords_list, feats_list, device="cuda")
    return voxels


def _get_kernel_map(voxels, kernel_size=3):
    """Generate kernel map using warpconvnet internals."""
    from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map

    int_coords = voxels.batched_coordinates
    batch_coords = int_coords.batch_indexed_coordinates  # [N, 4] with batch dim

    kernel_map = generate_kernel_map(
        batch_indexed_in_coords=batch_coords,
        batch_indexed_out_coords=batch_coords,
        in_to_out_stride_ratio=(1, 1, 1),
        kernel_size=(kernel_size, kernel_size, kernel_size),
    )
    return kernel_map, batch_coords.shape[0]


@pytest.mark.parametrize("C_in,C_out", [(32, 32), (32, 64), (64, 128)])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mask_implicit_gemm_forward(C_in, C_out, kernel_size, dtype):
    """Test that mask_implicit_gemm produces the same output as explicit_gemm."""
    torch.manual_seed(42)
    voxels = _make_test_data(num_voxels=500, C_in=C_in, batch_size=2)
    voxels = voxels.replace(
        batched_features=voxels.feature_tensor.to(dtype=dtype)
    )

    K = kernel_size ** 3
    weight = torch.randn(K, C_in, C_out, device="cuda", dtype=dtype)

    kernel_map, num_out = _get_kernel_map(voxels, kernel_size)

    # Reference: explicit GEMM
    ref_output = _explicit_gemm_forward_logic(
        voxels.feature_tensor,
        weight,
        kernel_map,
        num_out,
        compute_dtype=None,
    )

    # Test: mask implicit GEMM
    test_output = _mask_implicit_gemm_forward_logic(
        voxels.feature_tensor,
        weight,
        kernel_map,
        num_out,
        compute_dtype=None,
        block_size=16,
    )

    if dtype == torch.float16:
        atol, rtol = 0.5, 0.1  # relaxed for fp16 SIMT kernel
    else:
        atol, rtol = 1e-3, 1e-3

    torch.testing.assert_close(test_output, ref_output, atol=atol, rtol=rtol)
    print(f"PASS: C_in={C_in}, C_out={C_out}, K={K}, dtype={dtype}, "
          f"max_diff={torch.max(torch.abs(test_output - ref_output)).item():.6f}")


def test_mask_data_generation():
    """Test that mask data is correctly generated from IntSearchResult."""
    torch.manual_seed(42)
    voxels = _make_test_data(num_voxels=200, C_in=32, batch_size=1)
    kernel_map, num_out = _get_kernel_map(voxels, kernel_size=3)

    pair_table, pair_mask, mask_argsort = _kernel_map_to_mask_data(
        kernel_map, num_out, torch.device("cuda")
    )

    K = len(kernel_map)
    assert pair_table.shape == (K * num_out,)
    assert pair_mask.shape == (num_out,)
    assert mask_argsort.shape == (num_out,)

    # Verify pair_table consistency
    pair_table_2d = pair_table.reshape(K, num_out)
    for k in range(K):
        in_map_k, out_map_k = kernel_map[k]
        if in_map_k.numel() > 0:
            for j in range(min(in_map_k.numel(), 10)):  # spot-check
                out_idx = out_map_k[j].item()
                in_idx = in_map_k[j].item()
                assert pair_table_2d[k, out_idx].item() == in_idx

    # Verify argsort is a valid permutation
    sorted_indices = mask_argsort.sort().values
    expected = torch.arange(num_out, device="cuda", dtype=torch.int32)
    assert torch.equal(sorted_indices, expected)

    print(f"PASS: mask data generation correct (N={num_out}, K={K})")


def benchmark_mask_vs_existing():
    """Quick benchmark comparing mask_implicit_gemm vs existing algorithms."""
    import time

    torch.manual_seed(42)

    for N in [10000, 50000, 100000]:
        for C in [32, 64, 96]:
            voxels = _make_test_data(num_voxels=N, C_in=C, batch_size=4, spatial_range=200)
            voxels = voxels.replace(
                batched_features=voxels.feature_tensor.to(dtype=torch.float16)
            )
            kernel_map, num_out = _get_kernel_map(voxels, kernel_size=3)
            K = 27
            weight = torch.randn(K, C, C, device="cuda", dtype=torch.float16)

            # Warmup
            for _ in range(3):
                _ = _mask_implicit_gemm_forward_logic(
                    voxels.feature_tensor, weight, kernel_map, num_out, block_size=16
                )
            torch.cuda.synchronize()

            # Benchmark mask_implicit_gemm
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                _ = _mask_implicit_gemm_forward_logic(
                    voxels.feature_tensor, weight, kernel_map, num_out, block_size=16
                )
            torch.cuda.synchronize()
            t_mask = (time.perf_counter() - t0) / 10

            # Benchmark explicit_gemm
            for _ in range(3):
                _ = _explicit_gemm_forward_logic(
                    voxels.feature_tensor, weight, kernel_map, num_out
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                _ = _explicit_gemm_forward_logic(
                    voxels.feature_tensor, weight, kernel_map, num_out
                )
            torch.cuda.synchronize()
            t_explicit = (time.perf_counter() - t0) / 10

            ratio = t_mask / t_explicit
            print(f"N={num_out:>7,} C={C:>3} | mask={t_mask*1000:.2f}ms  explicit={t_explicit*1000:.2f}ms  ratio={ratio:.2f}x")


if __name__ == "__main__":
    print("=== Test mask data generation ===")
    test_mask_data_generation()

    print("\n=== Test forward correctness ===")
    for dtype in [torch.float32, torch.float16]:
        for c_in, c_out in [(32, 32), (32, 64)]:
            test_mask_implicit_gemm_forward(c_in, c_out, 3, dtype)

    print("\n=== Benchmark ===")
    benchmark_mask_vs_existing()
