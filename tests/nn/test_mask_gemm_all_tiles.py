# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-tile correctness pins for every mask_gemm kernel.

Exercises every (op, algo, tile_id[, split_k]) triple in the autotune
candidate pool and compares against the ``_explicit_gemm_*_logic`` reference.
A failure here fingerprints which specific kernel is numerically broken.

Coverage rationale: autotune picks one tile per shape at runtime, so a
correctness test that only runs the high-level nn module exercises a tiny
subset of the pool — exactly what let the v1.7.1 -> v1.7.2 fwd_as_dgrad
axis-swap bug reach ScanNet training (commit b6782be6). This test parameterises
directly on tile_id via ``_execute_forward`` / ``_execute_backward`` so every
tile runs on every CI GPU, with no autotune timing in the loop.

Structure:
  - test_fwd_all_tiles            — every _AB_MASK_GEMM entry
  - test_dgrad_fwd_as_dgrad_tiles — every _AB_MASK_GEMM_FWD_AS_DGRAD_* entry
  - test_dgrad_native_by_C        — dgrad tile chosen inside dispatch by C range
  - test_wgrad_all_tiles          — every _ATB_MASK_GEMM (tile_id, split_k) entry
"""

from contextlib import nullcontext

import pytest
import torch

from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates
from warpconvnet.geometry.coords.search.torch_discrete import generate_kernel_map
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv import (
    _explicit_gemm_backward_logic,
    _explicit_gemm_forward_logic,
)
from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
    _AB_MASK_GEMM,
    _AB_MASK_GEMM_FWD_AS_DGRAD,
    _ATB_MASK_GEMM,
)

# Canonical dgrad_wt aliases (900-911) map to underlying canonical fwd
# tile_ids. Used by tests below for skip-checks via the source fwd tile's
# alignment requirements.
_DGRAD_WT_TO_FWD_TILE = {
    900: 41,  # ex-83: 64x64 sa
    901: 3,  # ex-84: 64x128 3s
    902: 2,  # ex-85: 128x64
    903: 28,  # ex-86: 32x32 F16Accum
    904: 19,  # ex-87: 64x128 F16Accum
    905: 54,  # ex-88: Pcoff 64x64 flat (F16Accum)
    906: 55,  # ex-89: Pcoff 64x64 flat (F16K8)
    907: 56,  # ex-90: Pcoff 64x128 flat (F16K8)
    908: 57,  # ex-91: Pcoff 64x128 flat (F16Accum)
    909: 58,  # ex-92: Pcoff 64x64 3s
    910: 59,  # ex-93: Pcoff 64x64 WS
    911: 63,  # ex-94: Pcoff 64x128 WS
}
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import (
    _execute_backward,
    _execute_forward,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Tolerances — tuned against rdiff observed between explicit_gemm and
# mask_gemm under matching compute_dtype. F16Acc tiles accumulate loss with
# C_in, hence the looser bound.
# ---------------------------------------------------------------------------
_RTOL_F32ACC_FP32 = 5e-3
_RTOL_F32ACC_FP16 = 3e-2
_RTOL_F16ACC_FP16 = 1e-1
_RTOL_WGRAD_FP32 = 1e-2
_RTOL_WGRAD_FP16 = 5e-2

# F16-accumulator tile ids (both fwd and fwd_as_dgrad _wt variants). These
# tolerate larger rdiff because the MMA accumulates in fp16. tile_ids are
# canonical warpgemm IDs.
_F16ACC_FWD_TILES = {28, 19}  # ex-40, ex-42
_F16ACC_WT_TILES = {903, 904}  # ex-86, ex-87
# Pcoff tiles are also effectively f16acc (MW=1 only, f16 accumulate path).
_PCOFF_FWD_TILES = {54, 55, 56, 57, 58, 59, 63}
_PCOFF_WT_TILES = {905, 906, 907, 908, 909, 910, 911}  # ex-88..94


def _fwd_tile_tol(tile_id: int, dtype: torch.dtype) -> float:
    if tile_id in _F16ACC_FWD_TILES or tile_id in _PCOFF_FWD_TILES:
        return _RTOL_F16ACC_FP16
    return _RTOL_F32ACC_FP16 if dtype == torch.float16 else _RTOL_F32ACC_FP32


def _wt_tile_tol(tile_id: int, dtype: torch.dtype) -> float:
    if tile_id in _F16ACC_WT_TILES or tile_id in _PCOFF_WT_TILES:
        return _RTOL_F16ACC_FP16
    return _RTOL_F32ACC_FP16 if dtype == torch.float16 else _RTOL_F32ACC_FP32


def _wgrad_tol(dtype: torch.dtype) -> float:
    return _RTOL_WGRAD_FP16 if dtype == torch.float16 else _RTOL_WGRAD_FP32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_voxels(N=500, coord_range=12, C_in=32, batch_size=2, seed=0):
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


def _kernel_map(voxels: Voxels, kernel_size=(3, 3, 3), stride=1):
    in_coords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    out_coords = in_coords  # stride=1 covers the common case
    kmap = generate_kernel_map(
        in_coords,
        out_coords,
        in_to_out_stride_ratio=(stride,) * len(kernel_size),
        kernel_size=kernel_size,
    )
    return kmap, out_coords.shape[0]


def _skip_if_unsupported(tile_id: int, C_in: int, C_out: int, K: int, dtype: torch.dtype):
    """Skip tiles with constraints the shape doesn't satisfy."""
    # Vectorized tiles need C aligned to 16-byte boundary (8 elems fp16, 4 fp32).
    vec = 8 if dtype == torch.float16 else 4
    if tile_id not in (70, 71, 72, 73) and (C_in % vec or C_out % vec):
        pytest.skip(f"tile {tile_id} requires C aligned to {vec}")
    # F16Acc tiles require fp16 inputs.
    if tile_id in (_F16ACC_FWD_TILES | _F16ACC_WT_TILES) and dtype != torch.float16:
        pytest.skip(f"F16Acc tile {tile_id} requires fp16")
    # Pcoff tiles (MW=1 only) require K <= 32.
    if tile_id in (_PCOFF_FWD_TILES | _PCOFF_WT_TILES) and K > 32:
        pytest.skip(f"Pcoff tile {tile_id} is MW=1 only (K={K}>32)")


# ---------------------------------------------------------------------------
# Forward — every entry in _AB_MASK_GEMM
# ---------------------------------------------------------------------------

# Flatten (tile_id, split_k_or_none) to canonical parametrize ids.
_FWD_PARAMS = [(e[1]["tile_id"], e[1].get("split_k")) for e in _AB_MASK_GEMM]


@pytest.mark.parametrize("tile_id,split_k", _FWD_PARAMS, ids=[f"tile{t[0]}" for t in _FWD_PARAMS])
@pytest.mark.parametrize(
    "C_in,C_out",
    [(32, 32), (64, 64), (64, 128), (128, 64), (128, 128)],
    ids=["32_32", "64_64", "64_128", "128_64", "128_128"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_mask_gemm_fwd_all_tiles(tile_id, split_k, C_in, C_out, dtype):
    """Every mask_gemm fwd tile must match explicit_gemm on aligned symmetric shapes."""
    kernel_size = (3, 3, 3)
    K = 27
    _skip_if_unsupported(tile_id, C_in, C_out, K, dtype)

    voxels = _make_voxels(C_in=C_in, seed=1)
    weight = torch.randn(K, C_in, C_out, device="cuda", dtype=dtype)
    in_feats = voxels.feature_tensor.to(dtype)
    kmap, num_out = _kernel_map(voxels, kernel_size)

    params = {"tile_id": tile_id}
    if split_k is not None:
        params["split_k"] = split_k

    try:
        out_prod = _execute_forward(
            "mask_gemm", params, in_feats, weight, kmap, num_out, dtype, None
        )
    except RuntimeError as e:
        if "kErrorUnsupportedConfig" in str(e) or "Unsupported" in str(e):
            pytest.skip(f"tile {tile_id} unsupported at C_in={C_in} C_out={C_out}")
        raise

    out_ref = _explicit_gemm_forward_logic(in_feats, weight, kmap, num_out, dtype)

    tol = _fwd_tile_tol(tile_id, dtype)
    r = _rdiff(out_prod, out_ref)
    assert r < tol, f"fwd tile={tile_id} C={C_in},{C_out} dtype={dtype} rdiff={r:.3e} > {tol}"


# ---------------------------------------------------------------------------
# Dgrad via fwd kernel (mask_gemm_fwd_as_dgrad) — every _wt tile 83-94.
# These are the tiles that silently computed wrong gradients before b6782be6.
# ---------------------------------------------------------------------------

_WT_PARAMS = [e[1]["tile_id"] for e in _AB_MASK_GEMM_FWD_AS_DGRAD]


@pytest.mark.parametrize("tile_id", _WT_PARAMS, ids=[f"wt{t}" for t in _WT_PARAMS])
@pytest.mark.parametrize(
    "C_in,C_out",
    [(32, 32), (64, 64), (96, 96), (128, 128)],
    ids=["32_32", "64_64", "96_96", "128_128"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_mask_gemm_dgrad_fwd_as_dgrad_tiles(tile_id, C_in, C_out, dtype):
    """Every fwd_as_dgrad _wt tile must produce dgrad matching explicit_gemm."""
    kernel_size = (3, 3, 3)
    K = 27
    # Map dgrad_wt id back to the source fwd tile for alignment / dtype checks.
    base_tile = _DGRAD_WT_TO_FWD_TILE[tile_id]
    _skip_if_unsupported(base_tile, C_in, C_out, K, dtype)

    voxels = _make_voxels(C_in=C_in, seed=2)
    weight = torch.randn(K, C_in, C_out, device="cuda", dtype=dtype)
    in_feats = voxels.feature_tensor.to(dtype)
    kmap, num_out = _kernel_map(voxels, kernel_size)

    torch.manual_seed(42)
    grad_out = torch.randn(num_out, C_out, device="cuda", dtype=dtype)

    try:
        grad_in_prod, _ = _execute_backward(
            "mask_gemm_fwd_as_dgrad",
            {"tile_id": tile_id},
            grad_out,
            in_feats,
            weight,
            kmap,
            num_out,
            dtype,
            in_feats.device,
            needs_input_grad=(True, False),
        )
    except RuntimeError as e:
        if "kErrorUnsupportedConfig" in str(e) or "Unsupported" in str(e):
            pytest.skip(f"_wt tile {tile_id} unsupported at C_in={C_in} C_out={C_out}")
        raise

    grad_in_ref, _ = _explicit_gemm_backward_logic(
        grad_out, in_feats, weight, kmap, dtype, in_feats.device
    )

    tol = _wt_tile_tol(tile_id, dtype)
    r = _rdiff(grad_in_prod, grad_in_ref)
    assert (
        r < tol
    ), f"dgrad _wt tile={tile_id} (base {base_tile}) C={C_in},{C_out} dtype={dtype} rdiff={r:.3e} > {tol}"


# ---------------------------------------------------------------------------
# Dgrad native (mask_gemm algo). Dispatch picks tile 50/51/52/53/54 based on
# C_out_g and compute_dtype. Exercise each tile via a targeted shape.
# ---------------------------------------------------------------------------

# (C_in, C_out, dtype, expected_tile_bucket) — descriptive only; the dispatch
# actually chooses the tile internally. We run all shapes under mask_gemm algo
# and assert correctness; which internal tile fires is covered by code review +
# dispatch.py tile-selection matrix.
_NATIVE_DGRAD_SHAPES = [
    (32, 32, torch.float32),  # tile 50 (C<=48, fp32)
    (48, 48, torch.float32),  # tile 50 boundary
    (64, 64, torch.float32),  # tile 51 (48<C<=96, fp32)
    (96, 96, torch.float32),  # tile 51
    (96, 96, torch.float16),  # tile 53 (fp16 variant)
    (128, 128, torch.float32),  # tile 52 (C>96, fp32)
    (128, 128, torch.float16),  # tile 54 (fp16)
    (256, 256, torch.float16),  # tile 54
]


@pytest.mark.parametrize(
    "C_in,C_out,dtype",
    _NATIVE_DGRAD_SHAPES,
    ids=[f"{ci}x{co}_{str(d).split('.')[-1]}" for ci, co, d in _NATIVE_DGRAD_SHAPES],
)
def test_mask_gemm_dgrad_native_by_C(C_in, C_out, dtype):
    """Native mask_gemm dgrad path across C ranges that select different tiles."""
    kernel_size = (3, 3, 3)
    voxels = _make_voxels(C_in=C_in, seed=3)
    weight = torch.randn(27, C_in, C_out, device="cuda", dtype=dtype)
    in_feats = voxels.feature_tensor.to(dtype)
    kmap, num_out = _kernel_map(voxels, kernel_size)

    torch.manual_seed(42)
    grad_out = torch.randn(num_out, C_out, device="cuda", dtype=dtype)

    grad_in_prod, _ = _execute_backward(
        "mask_gemm",
        {},  # native dgrad picks tile inside dispatch, no tile_id needed
        grad_out,
        in_feats,
        weight,
        kmap,
        num_out,
        dtype,
        in_feats.device,
        needs_input_grad=(True, False),
    )
    grad_in_ref, _ = _explicit_gemm_backward_logic(
        grad_out, in_feats, weight, kmap, dtype, in_feats.device
    )

    tol = _RTOL_F32ACC_FP16 if dtype == torch.float16 else _RTOL_F32ACC_FP32
    r = _rdiff(grad_in_prod, grad_in_ref)
    assert r < tol, f"dgrad native C={C_in},{C_out} dtype={dtype} rdiff={r:.3e} > {tol}"


# ---------------------------------------------------------------------------
# Wgrad — every (tile_id, split_k) in _ATB_MASK_GEMM.
# ---------------------------------------------------------------------------

_WGRAD_PARAMS = [(e[1]["tile_id"], e[1]["split_k"]) for e in _ATB_MASK_GEMM]


@pytest.mark.parametrize(
    "tile_id,split_k", _WGRAD_PARAMS, ids=[f"tile{t}_sk{s}" for t, s in _WGRAD_PARAMS]
)
@pytest.mark.parametrize(
    "C_in,C_out",
    [(32, 32), (64, 64), (96, 96), (128, 128)],
    ids=["32_32", "64_64", "96_96", "128_128"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_mask_gemm_wgrad_all_tiles(tile_id, split_k, C_in, C_out, dtype):
    """Every wgrad tile x split_k must produce grad_weight matching explicit_gemm."""
    kernel_size = (3, 3, 3)
    K = 27
    _skip_if_unsupported(tile_id, C_in, C_out, K, dtype)

    voxels = _make_voxels(C_in=C_in, seed=4)
    weight = torch.randn(K, C_in, C_out, device="cuda", dtype=dtype)
    in_feats = voxels.feature_tensor.to(dtype)
    kmap, num_out = _kernel_map(voxels, kernel_size)

    torch.manual_seed(42)
    grad_out = torch.randn(num_out, C_out, device="cuda", dtype=dtype)

    try:
        _, grad_w_prod = _execute_backward(
            "mask_gemm",
            {"tile_id": tile_id, "split_k": split_k},
            grad_out,
            in_feats,
            weight,
            kmap,
            num_out,
            dtype,
            in_feats.device,
            needs_input_grad=(False, True),
        )
    except RuntimeError as e:
        if "kErrorUnsupportedConfig" in str(e) or "Unsupported" in str(e):
            pytest.skip(f"wgrad tile {tile_id} sk={split_k} unsupported at C={C_in},{C_out}")
        raise

    _, grad_w_ref = _explicit_gemm_backward_logic(
        grad_out, in_feats, weight, kmap, dtype, in_feats.device
    )

    tol = _wgrad_tol(dtype)
    r = _rdiff(grad_w_prod, grad_w_ref)
    assert (
        r < tol
    ), f"wgrad tile={tile_id} sk={split_k} C={C_in},{C_out} dtype={dtype} rdiff={r:.3e} > {tol}"


# ---------------------------------------------------------------------------
# Regression anchor: the exact shape class the v1.7.2 regression appeared on.
# Any future refactor of the dgrad autotune pool or fwd_as_dgrad dispatch must
# still pass this test or the ScanNet mIoU drop returns.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", [32, 64, 96, 128, 256])
def test_regression_fwd_as_dgrad_symmetric_fp16(C):
    """Post-b6782be6 anchor: fwd_as_dgrad on Cin==Cout symmetric shape, fp16.

    Before the weight-transpose fix, these tiles produced silently wrong
    gradients on symmetric shapes because the fwd kernel's B-loader reduced
    over the wrong channel axis. The kernel ran to completion with finite
    output — only end-to-end training exposed the issue.
    """
    kernel_size = (3, 3, 3)
    K = 27
    dtype = torch.float16
    voxels = _make_voxels(C_in=C, seed=5)
    weight = torch.randn(K, C, C, device="cuda", dtype=dtype)
    in_feats = voxels.feature_tensor.to(dtype)
    kmap, num_out = _kernel_map(voxels, kernel_size)

    torch.manual_seed(42)
    grad_out = torch.randn(num_out, C, device="cuda", dtype=dtype)

    # Use the canonical f32acc dgrad_wt tile (900 = ex-83, 64x64 sa) which is
    # what autotune picks for most MinkUNet layers.
    tile_id = 900
    _skip_if_unsupported(_DGRAD_WT_TO_FWD_TILE[tile_id], C, C, K, dtype)

    grad_in_prod, _ = _execute_backward(
        "mask_gemm_fwd_as_dgrad",
        {"tile_id": tile_id},
        grad_out,
        in_feats,
        weight,
        kmap,
        num_out,
        dtype,
        in_feats.device,
        needs_input_grad=(True, False),
    )
    grad_in_ref, _ = _explicit_gemm_backward_logic(
        grad_out, in_feats, weight, kmap, dtype, in_feats.device
    )

    tol = _RTOL_F32ACC_FP16
    r = _rdiff(grad_in_prod, grad_in_ref)
    assert r < tol, f"fwd_as_dgrad REGRESSION: C={C} rdiff={r:.3e} > {tol}"
