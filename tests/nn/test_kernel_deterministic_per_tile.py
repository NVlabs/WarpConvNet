# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-tile deterministic correctness sweep for mask_gemm kernels.
#
# For each algo (fwd, dgrad, wgrad, fwd_as_dgrad), iterates every tile_id
# the dispatch arms in mask_gemm_bindings.cu accept, runs against a fixed
# fp64 explicit_gemm reference, and asserts max error within fp16 tolerance.
#
# Tiles not compiled in / not dispatchable on this arch are skipped via
# RuntimeError pattern match — keeps the sweep portable across SM
# capabilities and WARPGEMM_TILES build modes.

from __future__ import annotations

import pytest
import torch

import warpconvnet._C as _C
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
)
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
)
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import (
    _execute_forward,
    _execute_backward,
)
from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
    _get_mask_data,
    _get_reverse_mask_data,
    _dispatched_mask_words,
)
from warpconvnet.csrc.mask_gemm.tile_metadata import (
    build_tile_metadata,
    filter_by_device,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(_C, "mask_gemm"),
    reason="needs CUDA + mask_gemm",
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KERNEL_SIZE = (3, 3, 3)
KERNEL_VOLUME = 27  # mask_words == 1 (fits all tile ladders)

# fp16 reference tolerance for matmul accumulation.
FP16_RTOL = 8e-3
FP16_ATOL = 5e-2


def _active_tile_ids(op: str) -> list:
    """Production-tier tile_ids compiled for the current SM arch."""
    if not torch.cuda.is_available():
        return []
    return sorted(
        t.tile_id for t in filter_by_device(build_tile_metadata(active_only=True, ops=(op,)))
    )


# Tile lists sourced from the warpgemm-emitted tile_metadata registry. This
# auto-tracks new tiles added by warpgemm codegen without manual range
# maintenance. Fwd op list also carries the 900-911 dgrad_wt aliases (fwd
# kernels invoked via the dgrad arm); split them off into the dedicated
# fwd_as_dgrad sweep.
_FWD_TILE_IDS_ALL = _active_tile_ids("forward")
_DGRAD_FWD_AS_DGRAD_IDS = [t for t in _FWD_TILE_IDS_ALL if 900 <= t <= 911]
_FWD_TILE_IDS = [t for t in _FWD_TILE_IDS_ALL if not (900 <= t <= 911)]
_DGRAD_NATIVE_TILE_IDS = _active_tile_ids("dgrad")
_WGRAD_TILE_IDS = _active_tile_ids("wgrad")
_WGRAD_SPLIT_KS = [1, 16, 64]


def _patterned_feats(N: int, C: int, dtype=torch.float64) -> torch.Tensor:
    col = torch.arange(C, device=DEVICE, dtype=dtype) / max(C, 1)
    row = (torch.arange(N, device=DEVICE, dtype=dtype) % 16) / 16.0
    return (row.unsqueeze(1) + col.unsqueeze(0)) / 2.0


def _patterned_weight(K: int, C_in: int, C_out: int, dtype=torch.float64) -> torch.Tensor:
    # row_in * 0.5 + col_out * 0.5, varying per kernel slice
    rc = torch.arange(K, device=DEVICE, dtype=dtype) / max(K, 1)
    cin = torch.arange(C_in, device=DEVICE, dtype=dtype) / max(C_in, 1)
    cout = torch.arange(C_out, device=DEVICE, dtype=dtype) / max(C_out, 1)
    base = cin.view(1, C_in, 1) + cout.view(1, 1, C_out)
    w = base * (1.0 + rc.view(K, 1, 1))
    return w.contiguous() / 4.0


def _patterned_grad_out(N: int, C: int, dtype=torch.float64) -> torch.Tensor:
    col = torch.arange(C, device=DEVICE, dtype=dtype) / max(C, 1)
    row = (torch.arange(N, device=DEVICE, dtype=dtype) % 8) / 8.0
    return (row.unsqueeze(1) + col.unsqueeze(0)) / 2.0


def _grid_coords(N: int) -> torch.Tensor:
    """Generate ~N unique 3D grid coords (single batch)."""
    extent = max(int(round((N * 8) ** (1.0 / 3.0))) + 4, 8)
    g = torch.Generator(device=DEVICE).manual_seed(0xC0DE ^ N)
    coords = torch.randint(
        0, extent, (int(N * 1.15), 3), device=DEVICE, dtype=torch.int32, generator=g
    )
    return coords


def _build_voxels(N: int):
    coords = _grid_coords(N)
    n_pad = coords.shape[0]
    feats = torch.zeros(n_pad, 1, device=DEVICE, dtype=torch.float32)
    offsets = torch.tensor([0, n_pad], dtype=torch.int32, device=DEVICE)
    voxels = Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        device=DEVICE,
    ).unique()
    actual_N = voxels.feature_tensor.shape[0]
    return voxels, actual_N


@pytest.fixture(scope="module", params=[(200, 64, 64), (200, 32, 32)], ids=["c64", "c32"])
def problem(request):
    N, C_in, C_out = request.param
    voxels, N_in = _build_voxels(N)
    out_coords, _, kmap = generate_output_coords_and_kernel_map(
        voxels,
        kernel_size=KERNEL_SIZE,
        kernel_dilation=(1, 1, 1),
        stride=(1, 1, 1),
        generative=False,
        transposed=False,
    )
    N_out = out_coords.shape[0]
    K = KERNEL_VOLUME

    in_feats_64 = _patterned_feats(N_in, C_in)
    weight_64 = _patterned_weight(K, C_in, C_out)
    grad_out_64 = _patterned_grad_out(N_out, C_out)

    fwd_ref = _explicit_gemm_forward_logic(in_feats_64, weight_64, kmap, N_out, torch.float64)
    gi_ref, gw_ref = _explicit_gemm_backward_logic(
        grad_out_64, in_feats_64, weight_64, kmap, torch.float64, DEVICE
    )

    return {
        "in_feats_16": in_feats_64.half().contiguous(),
        "weight_16": weight_64.half().contiguous(),
        "grad_out_16": grad_out_64.half().contiguous(),
        "fwd_ref": fwd_ref,
        "gi_ref": gi_ref,
        "gw_ref": gw_ref,
        "kmap": kmap,
        "N_in": N_in,
        "N_out": N_out,
        "C_in": C_in,
        "C_out": C_out,
        "K": K,
    }


_SKIP_PATTERNS = (
    "Unsupported",
    "mask_words",
    "Tile",
    "tile_id",
    "alignment",
    "not supported",
    "only supports",
    "not implemented",
    "f32 output",
    "fp32 out",
)


def _skip_if_dispatch_rejected(exc: BaseException) -> None:
    msg = str(exc)
    for pat in _SKIP_PATTERNS:
        if pat in msg:
            pytest.skip(f"tile not dispatchable here: {msg[:200]}")
    raise exc


def _assert_close(name, test, ref, rtol=FP16_RTOL, atol=FP16_ATOL):
    assert not test.isnan().any().item(), f"{name}: NaN"
    err = (test.float() - ref.float()).abs()
    ref_max = ref.float().abs().max().item()
    max_err = err.max().item()
    rel = max_err / (ref_max + 1e-12)
    assert max_err <= atol or rel <= rtol, (
        f"{name}: max_err={max_err:.4e} ref_max={ref_max:.4e} rel={rel:.4e} "
        f"(rtol={rtol}, atol={atol})"
    )


# ----------------------------------------------------------------------------
# Forward sweep: every FwdTile
# ----------------------------------------------------------------------------


class TestMaskGemmFwdPerTile:
    @pytest.mark.parametrize("tile_id", _FWD_TILE_IDS)
    def test_fwd(self, problem, tile_id):
        d = problem
        try:
            out = _execute_forward(
                algo="mask_gemm",
                params={"tile_id": tile_id},
                in_features=d["in_feats_16"],
                weight=d["weight_16"],
                kernel_map=d["kmap"],
                num_out_coords=d["N_out"],
                compute_dtype=torch.float16,
                fwd_block_size=None,
            )
        except (RuntimeError, AssertionError) as e:
            _skip_if_dispatch_rejected(e)
        _assert_close(f"mask_gemm fwd tile_id={tile_id} C={d['C_in']}", out, d["fwd_ref"])


# ----------------------------------------------------------------------------
# Native dgrad sweep: every DgradTile via direct C++ binding.
#
# Note: the high-level dispatch (_execute_backward) overrides params['tile_id']
# for native dgrad with a C-heuristic-driven choice (except for pcoff tiles
# 64-69). To actually sweep every tile_id we call _C.mask_gemm.dgrad directly
# with pre-built mask data + weight_T.
# ----------------------------------------------------------------------------


def _prepare_dgrad_inputs(d):
    """Build mask data + weight_T for direct _C.mask_gemm.dgrad call."""
    kmap = d["kmap"]
    K = d["K"]
    N_in = d["N_in"]
    N_out = d["N_out"]
    rev_pt, rev_pm, rev_as = _get_reverse_mask_data(kmap, N_in, N_out, torch.device(DEVICE))
    # Native dgrad reads W[K, G, Cin, Cout] with stride-transpose in smem.
    # For groups=1, weight_T arg is the original weight (no caller transpose).
    weight_T = d["weight_16"].contiguous()
    grad_input = torch.zeros((N_in, d["C_in"]), dtype=torch.float16, device=DEVICE)
    return rev_pt, rev_pm, rev_as, weight_T, grad_input


class TestMaskGemmDgradPerTile:
    @pytest.mark.parametrize("tile_id", _DGRAD_NATIVE_TILE_IDS)
    def test_dgrad(self, problem, tile_id):
        d = problem
        rev_pt, rev_pm, rev_as, weight_T, grad_input = _prepare_dgrad_inputs(d)
        K = d["K"]
        mask_words = _dispatched_mask_words(K)
        try:
            status = _C.mask_gemm.dgrad(
                d["grad_out_16"],
                weight_T,
                grad_input,
                rev_pt,
                rev_pm,
                rev_as,
                K,
                tile_id,
                mask_words,
                -1,  # identity_offset
                1.0,  # alpha
                1,  # groups
            )
        except (RuntimeError, AssertionError) as e:
            _skip_if_dispatch_rejected(e)
        if isinstance(status, int) and status != 0:
            pytest.skip(f"tile_id={tile_id} returned status={status}")
        _assert_close(f"mask_gemm dgrad tile_id={tile_id} C={d['C_in']}", grad_input, d["gi_ref"])


# ----------------------------------------------------------------------------
# fwd_as_dgrad alias sweep: DgradTile 900-911 (fwd kernel via weight transpose)
# ----------------------------------------------------------------------------


class TestMaskGemmFwdAsDgradPerTile:
    @pytest.mark.parametrize("tile_id", _DGRAD_FWD_AS_DGRAD_IDS)
    def test_fwd_as_dgrad(self, problem, tile_id):
        d = problem
        try:
            gi, _gw = _execute_backward(
                algo="mask_gemm_fwd_as_dgrad",
                params={"tile_id": tile_id},
                grad_output=d["grad_out_16"],
                in_features=d["in_feats_16"],
                weight=d["weight_16"],
                kernel_map=d["kmap"],
                num_out_coords=d["N_out"],
                compute_dtype=torch.float16,
                device=DEVICE,
                needs_input_grad=(True, False),
            )
        except (RuntimeError, AssertionError) as e:
            _skip_if_dispatch_rejected(e)
        _assert_close(f"mask_gemm_fwd_as_dgrad tile_id={tile_id} C={d['C_in']}", gi, d["gi_ref"])


# ----------------------------------------------------------------------------
# Wgrad sweep: every WgradTile × split_k
# ----------------------------------------------------------------------------


class TestMaskGemmWgradPerTile:
    @pytest.mark.parametrize("tile_id", _WGRAD_TILE_IDS)
    @pytest.mark.parametrize("split_k", _WGRAD_SPLIT_KS)
    def test_wgrad(self, problem, tile_id, split_k):
        d = problem
        try:
            _gi, gw = _execute_backward(
                algo="mask_gemm",
                params={"tile_id": tile_id, "split_k": split_k},
                grad_output=d["grad_out_16"],
                in_features=d["in_feats_16"],
                weight=d["weight_16"],
                kernel_map=d["kmap"],
                num_out_coords=d["N_out"],
                compute_dtype=torch.float16,
                device=DEVICE,
                needs_input_grad=(False, True),
            )
        except (RuntimeError, AssertionError) as e:
            _skip_if_dispatch_rejected(e)
        _assert_close(
            f"mask_gemm wgrad tile_id={tile_id} split_k={split_k} C={d['C_in']}",
            gw,
            d["gw_ref"],
        )
