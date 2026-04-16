# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical correctness of the "production" sparse-conv kernel path.

Motivation. A ScanNet training run (wandb nvr-lpr/scannet-segmentation/y26vckgf)
plateaus at val/miou ~5% vs ~53% on historical runs. The asymmetry
(accuracy ~52% but mIoU ~5%) matches "majority-class collapse" — what a
numerical bug in fwd / dgrad / wgrad would produce once training degrades the
discriminative features.

These tests pin each of the three production kernels against the trusted
`_explicit_gemm_*_logic` reference (called directly to bypass autotune). They
exercise:
  - C_in != C_out (dgrad stride swap; commit 64a6e3db)
  - stride=1 (submanifold) and stride=2 (downsample)
  - fp32 and fp16 AMP autocast

If a check fails with a large rdiff, the failing (kernel, config) pair
pinpoints which code path is wrong.
"""

import pytest
import torch

from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
    spatially_sparse_conv,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


_RTOL_FP32 = 5e-3
_RTOL_FP16 = 3e-2


def _make_voxels(N=600, coord_range=12, C_in=16, batch_size=2, seed=0):
    torch.manual_seed(seed)
    coords_list, feats_list = [], []
    for _ in range(batch_size):
        c = torch.unique(torch.randint(0, coord_range, (N, 3), dtype=torch.int32), dim=0)
        coords_list.append(c)
        feats_list.append(torch.randn(c.shape[0], C_in))
    return Voxels(coords_list, feats_list).to("cuda")


def _rdiff(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-8)).item()


def _kmap_for(voxels: Voxels, kernel_size, stride, out_voxels=None):
    """Build a kernel map. For stride>1, pass `out_voxels` (typically the
    production's output Voxels) so the reference uses the same out-coord
    ordering — `spatially_sparse_conv` builds out_coords internally and the
    ordering does not match an external `stride_coords()` call.
    """
    in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    if stride == 1:
        out_coords = in_coords
    else:
        assert out_voxels is not None, "stride>1 requires out_voxels for coord alignment"
        out_coords = batch_indexed_coordinates(out_voxels.coordinate_tensor, out_voxels.offsets)
    kmap = generate_kernel_map(
        in_coords,
        out_coords,
        in_to_out_stride_ratio=(stride,) * len(kernel_size),
        kernel_size=kernel_size,
    )
    return kmap, out_coords.shape[0]


def _ref_fwd(voxels, weight, kernel_size=(3, 3, 3), stride=1, amp_dtype=None, out_voxels=None):
    kmap, num_out = _kmap_for(voxels, kernel_size, stride, out_voxels=out_voxels)
    ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else _null()
    with ctx:
        out = _explicit_gemm_forward_logic(
            voxels.feature_tensor,
            weight,
            kmap,
            num_out,
        )
    return out, kmap, num_out


def _ref_bwd(grad_out, voxels, weight, kmap, amp_dtype=None):
    ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else _null()
    with ctx:
        gi, gw = _explicit_gemm_backward_logic(
            grad_out,
            voxels.feature_tensor,
            weight,
            kmap,
        )
    return gi, gw


def _prod_fwd_bwd(voxels, weight, kernel_size=(3, 3, 3), stride=1, amp_dtype=None):
    """Run production conv; returns (out_voxels, feats_with_grad, weight_with_grad).

    For stride>1, callers should use `out_voxels.coordinate_tensor` as the
    reference's out_coords — see _kmap_for.
    """
    feats = voxels.feature_tensor.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    v = voxels.replace(batched_features=feats)
    ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype else _null()
    with ctx:
        out = spatially_sparse_conv(
            v,
            w,
            kernel_size=kernel_size,
            stride=stride,
            fwd_algo="production",
            dgrad_algo="production",
            wgrad_algo="production",
        )
    return out, feats, w


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C_in,C_out,stride",
    [
        (16, 32, 1),
        (32, 16, 1),
        (96, 128, 1),
        (128, 96, 1),  # MinkUNet18 upsample
        (16, 32, 2),
        (32, 64, 2),
    ],
    ids=["s1_16_32", "s1_32_16", "s1_96_128", "s1_128_96", "s2_16_32", "s2_32_64"],
)
def test_production_fwd(C_in, C_out, stride):
    voxels = _make_voxels(C_in=C_in, seed=1)
    weight = torch.randn(27, C_in, C_out, device="cuda")
    # Run production first; for stride>1 use its out_coords for the reference
    # so both paths emit results in the same row order.
    out_prod_v, _, _ = _prod_fwd_bwd(voxels, weight, stride=stride)
    out_ref, _, _ = _ref_fwd(voxels, weight, stride=stride, out_voxels=out_prod_v)
    r = _rdiff(out_prod_v.feature_tensor, out_ref)
    assert r < _RTOL_FP32, f"fwd rdiff={r:.4e} at C_in={C_in} C_out={C_out} stride={stride}"


# ---------------------------------------------------------------------------
# dgrad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C_in,C_out,stride",
    [
        (16, 32, 1),
        (32, 16, 1),
        (96, 128, 1),
        (128, 96, 1),  # the exact case commit 64a6e3db claims to fix
        (16, 32, 2),
        (32, 64, 2),
    ],
    ids=["s1_16_32", "s1_32_16", "s1_96_128", "s1_128_96", "s2_16_32", "s2_32_64"],
)
def test_production_dgrad(C_in, C_out, stride):
    voxels = _make_voxels(C_in=C_in, seed=2)
    weight = torch.randn(27, C_in, C_out, device="cuda")

    out_prod_v, feats, w = _prod_fwd_bwd(voxels, weight, stride=stride)
    out_ref, kmap, _ = _ref_fwd(voxels, weight, stride=stride, out_voxels=out_prod_v)
    torch.manual_seed(42)
    # g is in production's row order (same as out_ref via shared out_voxels)
    g = torch.randn_like(out_ref)
    grad_in_ref, _ = _ref_bwd(g, voxels, weight, kmap)

    out_prod_v.feature_tensor.backward(g)
    grad_in_prod = feats.grad

    r = _rdiff(grad_in_prod, grad_in_ref)
    assert r < _RTOL_FP32, f"dgrad rdiff={r:.4e} at C_in={C_in} C_out={C_out} stride={stride}"


# ---------------------------------------------------------------------------
# wgrad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C_in,C_out,stride",
    [
        (16, 32, 1),
        (32, 16, 1),
        (96, 128, 1),
        (16, 32, 2),
    ],
    ids=["s1_16_32", "s1_32_16", "s1_96_128", "s2_16_32"],
)
def test_production_wgrad(C_in, C_out, stride):
    voxels = _make_voxels(C_in=C_in, seed=3)
    weight = torch.randn(27, C_in, C_out, device="cuda")

    out_prod_v, _, w = _prod_fwd_bwd(voxels, weight, stride=stride)
    out_ref, kmap, _ = _ref_fwd(voxels, weight, stride=stride, out_voxels=out_prod_v)
    torch.manual_seed(42)
    g = torch.randn_like(out_ref)
    _, grad_w_ref = _ref_bwd(g, voxels, weight, kmap)

    out_prod_v.feature_tensor.backward(g)
    grad_w_prod = w.grad

    r = _rdiff(grad_w_prod, grad_w_ref)
    assert r < _RTOL_FP32, f"wgrad rdiff={r:.4e} at C_in={C_in} C_out={C_out} stride={stride}"


# ---------------------------------------------------------------------------
# AMP fp16 (what the failing ScanNet run uses)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "C_in,C_out,stride",
    [(32, 64, 1), (32, 64, 2)],
    ids=["s1", "s2"],
)
def test_production_amp_fwd(C_in, C_out, stride):
    voxels = _make_voxels(C_in=C_in, seed=5)
    weight = torch.randn(27, C_in, C_out, device="cuda")
    out_prod_v, _, _ = _prod_fwd_bwd(voxels, weight, stride=stride, amp_dtype=torch.float16)
    out_ref, _, _ = _ref_fwd(
        voxels, weight, stride=stride, amp_dtype=torch.float16, out_voxels=out_prod_v
    )
    r = _rdiff(out_prod_v.feature_tensor, out_ref)
    assert r < _RTOL_FP16, f"AMP fwd rdiff={r:.4e} stride={stride}"
