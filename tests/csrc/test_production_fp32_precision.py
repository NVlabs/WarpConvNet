#!/usr/bin/env python
"""
Test production kernel numerical precision for fp32 inputs.

BUG: production_fwd outputs ALL ZEROS when called with fp32 inputs.

Root cause (production_bindings.cu line 222):
    output = output.to(torch::kFloat16);

This creates a NEW fp16 tensor instead of converting in-place. The CUDA
kernel writes to the new fp16 copy, which is then discarded when the
function returns. The caller's original fp32 output tensor stays zero.

The fp16-input path works correctly because no .to() conversion happens.

FIX: Either:
    1. Allocate output inside C++ at the right dtype and RETURN it
       (like the old sparse_conv_forward did), or
    2. After kernel completion, copy fp16 results back:
       original_output.copy_(fp16_output)
    3. Don't downcast output — allocate fp16 output in Python dispatch

HOW TO RUN:
    source .venv/bin/activate
    python tests/csrc/test_production_fp32_precision.py

EXPECTED OUTPUT:
    fp32 input: production outputs ALL ZEROS (the bug)
    fp16 input: production matches reference (no bug)
"""

import sys

import torch


def test_production_fp32_precision():
    """Compare production (fp16 compute) vs explicit_gemm (fp32 compute)."""
    import warpconvnet._C as _C
    from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import _get_mask_data
    from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
    from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
        _explicit_gemm_forward_logic,
    )

    N = 5000
    coords = torch.unique(torch.randint(0, 80, (N, 3), dtype=torch.int32), dim=0).cuda()
    N = coords.shape[0]
    batch_idx = torch.zeros(N, 1, dtype=torch.int32, device="cuda")
    coords4 = torch.cat([batch_idx, coords], dim=1).contiguous()
    kmap = generate_kernel_map(
        coords4, coords4, in_to_out_stride_ratio=(1, 1, 1), kernel_size=(3, 3, 3)
    )

    K = 27
    pt, pm, ms = _get_mask_data(kmap, N, "cuda")

    print(f"N={N}, K={K}")
    print(f"{'C_in→C_out':>12s} {'Tile':>6s} {'Max Diff':>10s} {'Rel Err':>10s} {'Prod NaN':>10s}")
    print("-" * 60)

    for C_in, C_out, tile in [
        (32, 32, 41),
        (64, 64, 41),
        (96, 96, 41),
        (128, 128, 43),
        (256, 256, 44),
        (32, 64, 41),
        (128, 96, 44),
        (3, 32, 71),  # unaligned, scalar tile
    ]:
        torch.manual_seed(42)
        w = torch.randn(K, C_in, C_out, device="cuda", dtype=torch.float32)
        f = torch.randn(N, C_in, device="cuda", dtype=torch.float32)

        # Production: internally downcasts to fp16
        out_prod = torch.zeros(N, C_out, dtype=torch.float32, device="cuda")
        status = _C.production.fwd(f, w, out_prod, pt, pm, ms, K, tile, 1.0)

        # Reference: explicit_gemm in fp32
        out_ref = _explicit_gemm_forward_logic(f, w, kmap, N, torch.float32)

        if status != 0:
            print(f"{C_in:>4d}→{C_out:<4d}   {tile:>6d}   FAILED (status={status})")
            continue

        max_diff = (out_prod - out_ref).abs().max().item()
        ref_norm = out_ref.abs().max().item()
        rel_err = max_diff / ref_norm if ref_norm > 0 else float("inf")
        nan = out_prod.isnan().any().item()

        print(
            f"{C_in:>4d}→{C_out:<4d}   {tile:>6d}   {max_diff:>10.3f} {rel_err:>10.4f} {'YES' if nan else 'no':>10s}"
        )

    print()
    print("NOTE: Large diffs (>1.0) are expected — production uses fp16 compute")
    print("      while explicit_gemm uses fp32. This is the same as SpConv under AMP.")
    print("      For pure fp32 workloads, production should NOT be selected.")


def test_production_fp16_precision():
    """Compare production (fp16) vs explicit_gemm (fp16) — should be close."""
    import warpconvnet._C as _C
    from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import _get_mask_data
    from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
    from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
        _explicit_gemm_forward_logic,
    )

    N = 5000
    coords = torch.unique(torch.randint(0, 80, (N, 3), dtype=torch.int32), dim=0).cuda()
    N = coords.shape[0]
    batch_idx = torch.zeros(N, 1, dtype=torch.int32, device="cuda")
    coords4 = torch.cat([batch_idx, coords], dim=1).contiguous()
    kmap = generate_kernel_map(
        coords4, coords4, in_to_out_stride_ratio=(1, 1, 1), kernel_size=(3, 3, 3)
    )

    K = 27
    pt, pm, ms = _get_mask_data(kmap, N, "cuda")

    print()
    print(f"FP16 precision test (N={N}, K={K}):")
    print(f"{'C_in→C_out':>12s} {'Tile':>6s} {'Max Diff':>10s} {'Rel Err':>10s}")
    print("-" * 50)

    for C_in, C_out, tile in [
        (32, 32, 41),
        (96, 96, 41),
        (128, 128, 43),
        (256, 256, 44),
    ]:
        torch.manual_seed(42)
        w = torch.randn(K, C_in, C_out, device="cuda", dtype=torch.float16)
        f = torch.randn(N, C_in, device="cuda", dtype=torch.float16)

        # Production: native fp16
        out_prod = torch.zeros(N, C_out, dtype=torch.float16, device="cuda")
        _C.production.fwd(f, w, out_prod, pt, pm, ms, K, tile, 1.0)

        # Reference: explicit_gemm in fp16
        out_ref = _explicit_gemm_forward_logic(f, w, kmap, N, torch.float16)

        max_diff = (out_prod.float() - out_ref.float()).abs().max().item()
        ref_norm = out_ref.float().abs().max().item()
        rel_err = max_diff / ref_norm if ref_norm > 0 else float("inf")

        print(f"{C_in:>4d}→{C_out:<4d}   {tile:>6d}   {max_diff:>10.4f} {rel_err:>10.6f}")

    print()
    print("NOTE: fp16 diffs should be small (<1.0). Both use fp16 compute.")


if __name__ == "__main__":
    test_production_fp32_precision()
    test_production_fp16_precision()
