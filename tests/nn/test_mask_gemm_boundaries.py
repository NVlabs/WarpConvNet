# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask-word boundary validation matrix for the ``mask_gemm`` sparse-conv path.

Blackwell handoff §: pins forward / native dgrad / wgrad against the trusted
``explicit_gemm`` reference at every DISPATCH_MW template boundary. The kernel
allocates ``pair_mask[row * MW_stride + word]`` with ``MW_stride`` rounded up to
the physical tier ``{1, 2, 4, 8, 12}`` (see ``_dispatched_mask_words`` /
``mask_gemm_bindings.cu`` ``MW_stride``). An off-by-one in the logical→physical
rounding, or a kernel that reads the wrong stride, only shows up right at those
boundaries — so each volume V below straddles a tier edge.

Volumes are hit with non-cubic ``kernel_size`` tuples whose product is exactly V:

    V     kernel_size   MW_logical=ceil(V/32)   physical tier
    32    (1, 4, 8)     1                        1
    33    (1, 3, 11)    2                        2
    64    (1, 8, 8)     2                        2
    65    (1, 5, 13)    3                        4   <- logical 3 -> physical 4
    128   (2, 8, 8)     4                        4
    135   (3, 5, 9)     5                        8
    256   (4, 8, 8)     8                        8
    270   (5, 6, 9)     9                        12
    343   (7, 7, 7)     11                       12  <- logical 11 -> physical 12
    384   (6, 8, 8)     12                       12

The exact +1 boundary volumes 129 and 257 have no compact 3D factorization
(129 = 3*43, 257 prime); their elongated kernel shapes exceed the hierarchical
coordinate search's coarse-grid limit ("Coarse grid too large"), a pre-existing
searcher constraint unrelated to the GEMM path. 135 and 270 sit in the same
logical-word classes (5 and 9), which is what the mask-word boundary exercises.

V=65 and V=343 are the critical wgrad physical-stride cases (the mask is padded
past its logical word count, so the wgrad kernel must index with the physical
stride, not ceil(K/32)).

Kernels are driven through the direct dispatch path (``_execute_forward`` /
``_execute_backward`` with an explicit ``mask_gemm`` algo + tile_id) rather than
by monkey-patching autotune pools. For K>32 the dgrad uses the NATIVE route
(algo="mask_gemm", not fwd_as_dgrad — the dgrad_wt binding arm is MW1-only).
"""

import pytest
import torch

import warpconvnet._C as _C
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
)
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import (
    _execute_forward,
    _execute_backward,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# rdiff is a mean-relative error (see _rdiff), so these are generous. bf16 has
# ~3 fewer mantissa bits than fp16; wgrad reduces over all N_out rows so it
# carries the most accumulation error of the three kernels.
_TOL = {torch.float16: 4e-2, torch.bfloat16: 1.0e-1}
_TOL_WGRAD = {torch.float16: 6e-2, torch.bfloat16: 1.2e-1}


# (kernel_size, mask-word volume V). V == product(kernel_size) == K == len(kmap).
_VOLUMES = {
    32: (1, 4, 8),
    33: (1, 3, 11),
    64: (1, 8, 8),
    65: (1, 5, 13),
    128: (2, 8, 8),
    135: (3, 5, 9),
    256: (4, 8, 8),
    270: (5, 6, 9),
    343: (7, 7, 7),
    384: (6, 8, 8),
}


def _rdiff(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / (b.abs().mean() + 1e-8)).item()


def _make_voxels(N_req=4000, coord_range=10, C_in=32, batch_size=2, seed=0):
    """Dense-ish random voxel set. coord_range^3 near N_req keeps the grid
    dense so even large kernels see many valid neighbor maps. Features are fp32
    here and cast to the target dtype at call time."""
    torch.manual_seed(seed)
    coords_list, feats_list = [], []
    for _ in range(batch_size):
        c = torch.unique(torch.randint(0, coord_range, (N_req, 3), dtype=torch.int32), dim=0)
        coords_list.append(c)
        feats_list.append(torch.randn(c.shape[0], C_in))
    return Voxels(coords_list, feats_list).to("cuda")


def _kmap_stride1(voxels, kernel_size):
    """Stride-1 kernel map: out_coords == in_coords, so both the mask path and
    the reference emit rows in the same order (num_out == N_in)."""
    in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    kmap = generate_kernel_map(
        in_coords,
        in_coords,
        in_to_out_stride_ratio=(1,) * len(kernel_size),
        kernel_size=kernel_size,
    )
    return kmap, in_coords.shape[0]


def _make_weight(K, C_in, C_out, groups, dtype, device, seed=7):
    """Weight scaled by 0.1 to keep fp16 forward outputs well inside range even
    for large-K accumulation."""
    torch.manual_seed(seed)
    if groups == 1:
        return (torch.randn(K, C_in, C_out, device=device) * 0.1).to(dtype)
    return (torch.randn(K, groups, C_in // groups, C_out // groups, device=device) * 0.1).to(dtype)


def _ref_forward(x, w, kmap, num_out, groups, compute_dtype):
    if groups == 1:
        return _explicit_gemm_forward_logic(x, w, kmap, num_out, compute_dtype)
    C_in_g, C_out_g = w.shape[2], w.shape[3]
    outs = []
    for g in range(groups):
        xg = x[:, g * C_in_g : (g + 1) * C_in_g].contiguous()
        wg = w[:, g].contiguous()
        outs.append(_explicit_gemm_forward_logic(xg, wg, kmap, num_out, compute_dtype))
    return torch.cat(outs, dim=1)


def _ref_backward(g_out, x, w, kmap, groups):
    if groups == 1:
        return _explicit_gemm_backward_logic(g_out, x, w, kmap)
    C_in_g, C_out_g = w.shape[2], w.shape[3]
    gis, gws = [], []
    for g in range(groups):
        xg = x[:, g * C_in_g : (g + 1) * C_in_g].contiguous()
        wg = w[:, g].contiguous()
        gog = g_out[:, g * C_out_g : (g + 1) * C_out_g].contiguous()
        gi_g, gw_g = _explicit_gemm_backward_logic(gog, xg, wg, kmap)
        gis.append(gi_g)
        gws.append(gw_g)
    # mask_gemm returns grad_weight [K, groups, C_in_g, C_out_g] for groups>1.
    return torch.cat(gis, dim=1), torch.stack(gws, dim=1)


def _check_case(V, kernel_size, dtype, C_in, C_out, groups, seed):
    """Forward + native dgrad + wgrad vs explicit_gemm at one boundary volume."""
    voxels = _make_voxels(C_in=C_in, seed=seed)
    K = V
    device = voxels.device
    kmap, num_out = _kmap_stride1(voxels, kernel_size)
    assert len(kmap) == K, f"len(kmap)={len(kmap)} != V={K} for ks={kernel_size}"

    x = voxels.feature_tensor.to(dtype)
    w = _make_weight(K, C_in, C_out, groups, dtype, device)

    # --- forward -----------------------------------------------------------
    out_mask = _execute_forward(
        "mask_gemm",
        {"tile_id": 41},
        x,
        w,
        kmap,
        num_out,
        compute_dtype=dtype,
        fwd_block_size=None,
        groups=groups,
    )
    out_ref = _ref_forward(x, w, kmap, num_out, groups, dtype)
    r = _rdiff(out_mask, out_ref)
    assert r < _TOL[dtype], f"fwd rdiff={r:.4e} V={V} dt={dtype} groups={groups}"

    # --- native dgrad + wgrad ---------------------------------------------
    torch.manual_seed(1234)
    g = torch.randn_like(out_ref).to(dtype)
    gi_mask, gw_mask = _execute_backward(
        "mask_gemm",
        {},  # native dgrad tile selected internally; wgrad default tile_id=0
        g,
        x,
        w,
        kmap,
        num_out,
        compute_dtype=dtype,
        device=device,
        needs_input_grad=(True, True),
        groups=groups,
    )
    gi_ref, gw_ref = _ref_backward(g, x, w, kmap, groups)

    r = _rdiff(gi_mask, gi_ref)
    assert r < _TOL[dtype], f"dgrad rdiff={r:.4e} V={V} dt={dtype} groups={groups}"
    r = _rdiff(gw_mask, gw_ref)
    assert r < _TOL_WGRAD[dtype], f"wgrad rdiff={r:.4e} V={V} dt={dtype} groups={groups}"


# ---------------------------------------------------------------------------
# Aligned fp16 baseline at every boundary volume.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", list(_VOLUMES.keys()), ids=[f"V{v}" for v in _VOLUMES])
def test_boundary_fp16_aligned(V):
    _check_case(V, _VOLUMES[V], torch.float16, C_in=32, C_out=32, groups=1, seed=V)


# ---------------------------------------------------------------------------
# bf16 subset (looser tolerance) at the tier edges + the top physical stride.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [33, 65, 343], ids=["V33", "V65", "V343"])
def test_boundary_bf16_aligned(V):
    _check_case(V, _VOLUMES[V], torch.bfloat16, C_in=32, C_out=32, groups=1, seed=V + 1)


# ---------------------------------------------------------------------------
# Misaligned per-group channels (scalar tile path). fp16 vec_width = 8, so
# C_in=20 (20 % 8 == 4) and C_out=36 (36 % 8 == 4) are both unaligned and route
# to the scalar tile (70/71/72 fwd+dgrad, 73 wgrad). NB: the handoff note's
# example (24, 40) is actually 8-aligned; genuinely-misaligned values used here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [33, 65], ids=["V33", "V65"])
def test_boundary_fp16_misaligned(V):
    _check_case(V, _VOLUMES[V], torch.float16, C_in=20, C_out=36, groups=1, seed=V + 2)


# ---------------------------------------------------------------------------
# Group convolution (groups=2, per-group channels 16 -> 8-aligned).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("V", [33, 65], ids=["V33", "V65"])
def test_boundary_fp16_grouped(V):
    _check_case(V, _VOLUMES[V], torch.float16, C_in=32, C_out=32, groups=2, seed=V + 3)


# ---------------------------------------------------------------------------
# Empty-shape robustness.
# ---------------------------------------------------------------------------


def test_empty_wgrad_nout0_returns_zero():
    """N_out == 0: the wgrad binding takes its explicit empty branch (requires
    empty pair/reduced masks), zeroes grad_weight, and returns status 0."""
    K, C_in, C_out, N_in = 27, 32, 32, 128
    x = torch.randn(N_in, C_in, device="cuda", dtype=torch.float16)
    go = torch.zeros(0, C_out, device="cuda", dtype=torch.float16)
    gw = torch.ones(K, C_in, C_out, device="cuda", dtype=torch.float32)  # nonzero on purpose
    empty = torch.zeros(0, dtype=torch.int32, device="cuda")
    status = _C.mask_gemm.wgrad(x, go, gw, empty, empty, empty, empty, K, 0, 64, 1.0, 1)
    assert status == 0
    assert torch.count_nonzero(gw).item() == 0


def test_empty_forward_nin0_stays_zero():
    """N_in == 0 (empty input) with N_out > 0: an all-(-1) pair_table and an
    all-zero pair_mask mean no valid gather, so the forward must leave the
    zeroed output untouched and not read out of the empty input buffer."""
    K, C_in, C_out, N_out = 27, 32, 32, 128
    x = torch.zeros(0, C_in, device="cuda", dtype=torch.float16)
    w = (torch.randn(K, C_in, C_out, device="cuda") * 0.1).half()
    out = torch.zeros(N_out, C_out, device="cuda", dtype=torch.float16)
    pair_table = torch.full((K * N_out,), -1, dtype=torch.int32, device="cuda")
    pair_mask = torch.zeros(N_out, dtype=torch.int32, device="cuda")  # mask_words = 1
    mask_argsort = torch.arange(N_out, dtype=torch.int32, device="cuda")
    status = _C.mask_gemm.fwd(x, w, out, pair_table, pair_mask, mask_argsort, K, 41, 1, -1, 1.0, 1)
    assert status == 0
    assert torch.count_nonzero(out).item() == 0


# ---------------------------------------------------------------------------
# wgrad reduced-mask / physical-stride validation (negative path).
# ---------------------------------------------------------------------------


def test_wgrad_rejects_noncanonical_mask_stride():
    """A pair_mask allocated with the logical word count (ceil(65/32)=3) instead
    of the canonical physical tier (4) must be rejected by the binding, not read
    as a bad stride. stride=3 is not in {1,2,4,8,12}."""
    K, C_in, C_out, N = 65, 32, 32, 64
    x = torch.randn(N, C_in, device="cuda", dtype=torch.float16)
    go = torch.randn(N, C_out, device="cuda", dtype=torch.float16)
    gw = torch.zeros(K, C_in, C_out, device="cuda", dtype=torch.float32)
    wrong_stride = 3  # logical ceil(65/32); canonical tier would be 4
    pair_table = torch.full((K * N,), -1, dtype=torch.int32, device="cuda")
    pair_mask = torch.zeros(N * wrong_stride, dtype=torch.int32, device="cuda")
    mask_argsort = torch.arange(N, dtype=torch.int32, device="cuda")
    reduced_mask = torch.zeros(((N + 31) // 32) * wrong_stride, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="canonical tier|MW_stride"):
        _C.mask_gemm.wgrad(
            x,
            go,
            gw,
            pair_table,
            pair_mask,
            mask_argsort,
            reduced_mask,
            K,
            0,
            64,
            1.0,
            1,
        )
