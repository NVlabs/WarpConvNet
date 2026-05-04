# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deterministic correctness tests for sparse convolution kernels.
#
# Uses handcrafted weight patterns (ones, triu, tril, eye, single-k) and
# feature patterns (ones, range) so the expected output has analytic
# structure. fp64 explicit_gemm is the reference; backends are validated
# at tighter-than-random tolerances.
#
# Shape matrix exercises:
#   - same C_in == C_out (32, 64, 128)
#   - asymmetric C_in/C_out (smaller/larger)
#   - non-even channels (7, 13, 23)
#   - non-tile-divisible vs tile-divisible (33 vs 32; 65 vs 64)
#   - N from small (200) to ~2^20 (1048576) for cp.async tail / direct path

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
from warpconvnet.nn.functional.sparse_conv.detail.implicit_direct import (
    _implicit_gemm_forward_logic,
    _implicit_gemm_backward_logic,
    _implicit_gemm_forward_grouped,
    _implicit_gemm_backward_grouped,
)
from warpconvnet.nn.functional.sparse_conv.detail.cutlass import (
    _cutlass_implicit_gemm_forward_logic,
    _cutlass_implicit_gemm_backward_logic,
)
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import (
    _execute_forward,
    _execute_backward,
)
from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
    _HAS_CUTE_BACKEND,
    _HAS_CUTE_GROUPED,
    _HAS_CUTE_SM90,
    _HAS_CUTE_GROUPED_SM90,
)

if _HAS_CUTE_BACKEND:
    from warpconvnet.nn.functional.sparse_conv.detail.cute import (
        _cute_implicit_gemm_forward_logic,
        _cute_implicit_gemm_backward_logic,
    )
if _HAS_CUTE_GROUPED:
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped import (
        _cute_grouped_forward_logic,
        _cute_grouped_backward_logic,
    )
if _HAS_CUTE_SM90:
    from warpconvnet.nn.functional.sparse_conv.detail.cute_sm90 import (
        _cute_implicit_gemm_sm90_forward_logic,
        _cute_implicit_gemm_sm90_backward_logic,
    )
if _HAS_CUTE_GROUPED_SM90:
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped_sm90 import (
        _cute_grouped_sm90_forward_logic,
        _cute_grouped_sm90_backward_logic,
    )

DEVICE = torch.device("cuda")
KERNEL_SIZE = (3, 3, 3)
KERNEL_VOLUME = 27

# fp32 reference vs fp64 ground truth: machine-precision modulo accumulation
# order. Tolerances reflect the worst-case order-of-summation error for the
# largest tested shapes (K=27 neighbors × max C_in=128).
FP32_RTOL = 1e-4
FP32_ATOL = 1e-3
FP16_RTOL = 8e-3
FP16_ATOL = 5e-2


def _check_fp16_cublas_broken() -> bool:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; a=torch.ones(2,2,device='cuda',dtype=torch.float16); print((a@a).sum().item())",
        ],
        capture_output=True,
        timeout=30,
    )
    return result.returncode != 0


_FP16_CUBLAS_BROKEN = _check_fp16_cublas_broken()
_skip_fp16_cublas = pytest.mark.skipif(
    _FP16_CUBLAS_BROKEN,
    reason="torch.matmul fp16 broken (cuBLAS bug in this torch/CUDA version)",
)


# ----------------------------------------------------------------------------
# Deterministic data
# ----------------------------------------------------------------------------


def _grid_coords(N: int, device=DEVICE) -> torch.Tensor:
    """Generate N unique 3D grid coordinates with cubic-ish density.

    Uses a uniform grid with extent ~ 2 * N^(1/3) so post-unique density is
    ~1/8 — collisions are rare even at N=2^20.
    """
    extent = max(int(round((N * 8) ** (1.0 / 3.0))) + 4, 8)
    g = torch.Generator(device=device).manual_seed(0xC0DE ^ N)
    coords = torch.randint(
        0, extent, (int(N * 1.15), 3), device=device, dtype=torch.int32, generator=g
    )
    return coords


def _make_weight(K: int, C_in: int, C_out: int, pattern: str, dtype=torch.float64) -> torch.Tensor:
    """Construct deterministic weight [K, C_in, C_out]."""
    if pattern == "ones":
        return torch.ones(K, C_in, C_out, device=DEVICE, dtype=dtype)
    if pattern == "triu":
        # Upper triangle on the (C_in, C_out) plane, broadcast across K.
        m = torch.ones(C_in, C_out, device=DEVICE, dtype=dtype).triu()
        return m.unsqueeze(0).expand(K, -1, -1).contiguous()
    if pattern == "tril":
        m = torch.ones(C_in, C_out, device=DEVICE, dtype=dtype).tril()
        return m.unsqueeze(0).expand(K, -1, -1).contiguous()
    if pattern == "eye":
        # Identity over min(C_in, C_out), broadcast over K. Detects
        # channel-axis swaps and gather-scatter mis-indexing.
        d = min(C_in, C_out)
        m = torch.zeros(C_in, C_out, device=DEVICE, dtype=dtype)
        m[:d, :d] = torch.eye(d, device=DEVICE, dtype=dtype)
        return m.unsqueeze(0).expand(K, -1, -1).contiguous()
    if pattern == "center_eye":
        # Identity weight only at the kernel-center offset (k=13 for 3x3x3),
        # zero elsewhere. Sparse conv collapses to per-voxel pass-through
        # when output coords coincide with input coords.
        d = min(C_in, C_out)
        eye_block = torch.zeros(C_in, C_out, device=DEVICE, dtype=dtype)
        eye_block[:d, :d] = torch.eye(d, device=DEVICE, dtype=dtype)
        w = torch.zeros(K, C_in, C_out, device=DEVICE, dtype=dtype)
        w[K // 2] = eye_block
        return w
    raise ValueError(f"unknown weight pattern: {pattern}")


def _make_feats(N: int, C_in: int, pattern: str, dtype=torch.float64) -> torch.Tensor:
    """Construct deterministic features [N, C_in]."""
    if pattern == "ones":
        return torch.ones(N, C_in, device=DEVICE, dtype=dtype)
    if pattern == "range":
        # Small-magnitude range so fp16 doesn't overflow:
        # max accumulated value ≈ K * sum(range(C_in)/C_in) ≈ K * C_in / 2
        col = torch.arange(C_in, device=DEVICE, dtype=dtype) / max(C_in, 1)
        return col.unsqueeze(0).expand(N, -1).contiguous()
    if pattern == "row_index":
        # Row-varying values: detect index swaps in gather/scatter.
        col = torch.arange(C_in, device=DEVICE, dtype=dtype) / max(C_in, 1)
        row = (torch.arange(N, device=DEVICE, dtype=dtype) % 16) / 16.0
        return (row.unsqueeze(1) + col.unsqueeze(0)) / 2.0
    raise ValueError(f"unknown feat pattern: {pattern}")


def _make_grad_out(N_out: int, C_out: int, dtype=torch.float64) -> torch.Tensor:
    # Deterministic, non-uniform per-row pattern to expose dgrad/wgrad bugs.
    col = torch.arange(C_out, device=DEVICE, dtype=dtype) / max(C_out, 1)
    row = (torch.arange(N_out, device=DEVICE, dtype=dtype) % 8) / 8.0
    return (row.unsqueeze(1) + col.unsqueeze(0)) / 2.0


def _build_voxels(N: int):
    """Construct Voxels with N unique grid coords. Returns voxels + actual N_in."""
    coords = _grid_coords(N)
    n_pad = coords.shape[0]
    feats = torch.zeros(n_pad, 1, device=DEVICE, dtype=torch.float32)  # placeholder
    offsets = torch.tensor([0, n_pad], dtype=torch.int32, device=DEVICE)
    voxels = Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        device=DEVICE,
    ).unique()
    actual_N = voxels.feature_tensor.shape[0]
    if actual_N > N:
        # Trim deterministically.
        idx = torch.arange(N, device=DEVICE)
        voxels = Voxels(
            batched_coordinates=voxels.coordinate_tensor[idx].to(torch.int32),
            batched_features=voxels.feature_tensor[idx],
            offsets=torch.tensor([0, N], dtype=torch.int32, device=DEVICE),
            device=DEVICE,
        )
        actual_N = N
    return voxels, actual_N


# ----------------------------------------------------------------------------
# Reference computation
# ----------------------------------------------------------------------------


def _make_problem(N, C_in, C_out, weight_pattern, feat_pattern):
    """Build fp64 reference problem + fp32 inputs for kernels."""
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

    # fp64 ground truth.
    in_feats_64 = _make_feats(N_in, C_in, feat_pattern, dtype=torch.float64)
    weight_64 = _make_weight(K, C_in, C_out, weight_pattern, dtype=torch.float64)
    grad_out_64 = _make_grad_out(N_out, C_out, dtype=torch.float64)

    fwd_ref_64 = _explicit_gemm_forward_logic(in_feats_64, weight_64, kmap, N_out, torch.float64)
    gi_ref_64, gw_ref_64 = _explicit_gemm_backward_logic(
        grad_out_64, in_feats_64, weight_64, kmap, torch.float64, DEVICE
    )

    return {
        "in_feats_32": in_feats_64.float(),
        "in_feats_64": in_feats_64,
        "weight_32": weight_64.float(),
        "weight_64": weight_64,
        "grad_out_32": grad_out_64.float(),
        "fwd_ref_64": fwd_ref_64,
        "gi_ref_64": gi_ref_64,
        "gw_ref_64": gw_ref_64,
        "kmap": kmap,
        "N_in": N_in,
        "N_out": N_out,
        "K": K,
        "C_in": C_in,
        "C_out": C_out,
    }


def _assert_close(name, test, ref_64, rtol, atol):
    assert not test.isnan().any().item(), f"{name}: NaN"
    err = (test.float() - ref_64.float()).abs()
    ref_max = ref_64.float().abs().max().item()
    max_err = err.max().item()
    rel = max_err / (ref_max + 1e-12)
    assert max_err <= atol or rel <= rtol, (
        f"{name}: max_err={max_err:.6e} ref_max={ref_max:.6e} rel={rel:.6e} "
        f"(rtol={rtol}, atol={atol})"
    )


# ----------------------------------------------------------------------------
# Parametrization
# ----------------------------------------------------------------------------

# (N, C_in, C_out, label) — N small/medium for full coverage; large in slow tests.
_SHAPES = [
    # Same channels, divisible
    (200, 32, 32, "n200_c32_c32"),
    (200, 64, 64, "n200_c64_c64"),
    (200, 128, 128, "n200_c128_c128"),
    # Asymmetric C
    (200, 16, 64, "n200_c16_c64_grow"),
    (200, 64, 16, "n200_c64_c16_shrink"),
    (200, 8, 128, "n200_c8_c128_grow"),
    # Non-even channels (small primes)
    (200, 7, 13, "n200_c7_c13"),
    (200, 13, 7, "n200_c13_c7"),
    (200, 23, 23, "n200_c23_c23_prime"),
    # Tile-non-divisible (cp.async tail vs direct path)
    (200, 33, 65, "n200_c33_c65_nondiv"),
    (200, 65, 33, "n200_c65_c33_nondiv"),
    (200, 96, 64, "n200_c96_c64_div32"),
    # Medium N for cp.async sweep
    (5000, 32, 64, "n5k_c32_c64"),
    (5000, 33, 65, "n5k_c33_c65_nondiv"),
]

_LARGE_SHAPES = [
    (50000, 64, 128, "n50k_c64_c128"),
    (1 << 20, 32, 32, "n1M_c32_c32"),  # ~2^20 voxels
]

_PATTERNS = [
    ("ones", "ones", "ones_x_ones"),
    ("ones", "range", "ones_x_range"),
    ("triu", "range", "triu_x_range"),
    ("tril", "range", "tril_x_range"),
    ("eye", "row_index", "eye_x_rowidx"),
    ("center_eye", "row_index", "centereye_x_rowidx"),
]


@pytest.fixture(
    scope="module",
    params=_SHAPES,
    ids=lambda p: p[3],
)
def shape(request):
    N, C_in, C_out, _ = request.param
    return N, C_in, C_out


@pytest.fixture(
    scope="module",
    params=_PATTERNS,
    ids=lambda p: p[2],
)
def pattern(request):
    return request.param[0], request.param[1]


@pytest.fixture(scope="module")
def problem(shape, pattern):
    N, C_in, C_out = shape
    weight_pattern, feat_pattern = pattern
    # Skip eye for non-square C
    if weight_pattern in ("eye", "center_eye") and C_in != C_out:
        pytest.skip("eye pattern requires C_in == C_out for tightest semantics")
    return _make_problem(N, C_in, C_out, weight_pattern, feat_pattern)


# ----------------------------------------------------------------------------
# Tests: fp32 SIMT (covers all shapes including non-divisible)
# ----------------------------------------------------------------------------


class TestExplicitFp32:
    """Sanity check: explicit_gemm in fp32 must match fp64 reference tightly."""

    def test_forward(self, problem):
        d = problem
        out = _explicit_gemm_forward_logic(
            d["in_feats_32"], d["weight_32"], d["kmap"], d["N_out"], torch.float32
        )
        _assert_close("explicit_fp32 fwd", out, d["fwd_ref_64"], FP32_RTOL, FP32_ATOL)

    def test_backward(self, problem):
        d = problem
        gi, gw = _explicit_gemm_backward_logic(
            d["grad_out_32"], d["in_feats_32"], d["weight_32"], d["kmap"], torch.float32, DEVICE
        )
        _assert_close("explicit_fp32 dgrad", gi, d["gi_ref_64"], FP32_RTOL, FP32_ATOL)
        _assert_close("explicit_fp32 wgrad", gw, d["gw_ref_64"], FP32_RTOL, FP32_ATOL)


class TestImplicitFp32:
    def test_forward(self, problem):
        d = problem
        out = _implicit_gemm_forward_logic(
            d["in_feats_32"],
            d["weight_32"],
            d["kmap"],
            d["N_out"],
            torch.float32,
            fwd_block_size=16,
        )
        _assert_close("implicit_fp32 fwd", out, d["fwd_ref_64"], FP32_RTOL, FP32_ATOL)

    def test_backward(self, problem):
        d = problem
        gi, gw = _implicit_gemm_backward_logic(
            d["grad_out_32"],
            d["in_feats_32"],
            d["weight_32"],
            d["kmap"],
            d["N_out"],
            16,
            256,
            4,
            torch.float32,
        )
        _assert_close("implicit_fp32 dgrad", gi, d["gi_ref_64"], FP32_RTOL, FP32_ATOL)
        _assert_close("implicit_fp32 wgrad", gw, d["gw_ref_64"], FP32_RTOL, FP32_ATOL)


class TestImplicitGroupedFp32:
    def test_forward(self, problem):
        d = problem
        out = _implicit_gemm_forward_grouped(
            d["in_feats_32"],
            d["weight_32"],
            d["kmap"],
            d["N_out"],
            torch.float32,
            fwd_block_size=16,
            saturation_m=5000,
        )
        _assert_close("implicit_grouped_fp32 fwd", out, d["fwd_ref_64"], FP32_RTOL, FP32_ATOL)

    def test_backward(self, problem):
        d = problem
        gi, gw = _implicit_gemm_backward_grouped(
            d["grad_out_32"],
            d["in_feats_32"],
            d["weight_32"],
            d["kmap"],
            d["N_out"],
            16,
            256,
            4,
            torch.float32,
            saturation_m=5000,
        )
        _assert_close("implicit_grouped_fp32 dgrad", gi, d["gi_ref_64"], FP32_RTOL, FP32_ATOL)
        _assert_close("implicit_grouped_fp32 wgrad", gw, d["gw_ref_64"], FP32_RTOL, FP32_ATOL)


# ----------------------------------------------------------------------------
# Tests: fp16 tensor-core kernels (gated on alignment + cuBLAS health)
# ----------------------------------------------------------------------------


def _skip_if_unsupported(result, name):
    if isinstance(result, int):
        pytest.skip(f"{name} unsupported (status={result})")
    if isinstance(result, tuple) and isinstance(result[0], int):
        pytest.skip(f"{name} unsupported (status={result[0]})")


@_skip_fp16_cublas
class TestCutlassFp16:
    def test_forward(self, problem):
        d = problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("cutlass requires C_in/C_out aligned to 8")
        out = _cutlass_implicit_gemm_forward_logic(
            d["in_feats_32"].half(), d["weight_32"].half(), d["kmap"], d["N_out"]
        )
        _skip_if_unsupported(out, "cutlass")
        _assert_close("cutlass_fp16 fwd", out, d["fwd_ref_64"], FP16_RTOL, FP16_ATOL)

    def test_backward(self, problem):
        d = problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("cutlass requires C_in/C_out aligned to 8")
        gi, gw = _cutlass_implicit_gemm_backward_logic(
            d["grad_out_32"].half(),
            d["in_feats_32"].half(),
            d["weight_32"].half(),
            d["kmap"],
            device=DEVICE,
        )
        _skip_if_unsupported((gi, gw), "cutlass")
        _assert_close("cutlass_fp16 dgrad", gi, d["gi_ref_64"], FP16_RTOL, FP16_ATOL)
        _assert_close("cutlass_fp16 wgrad", gw, d["gw_ref_64"], FP16_RTOL, FP16_ATOL)


@_skip_fp16_cublas
@pytest.mark.skipif(not _HAS_CUTE_BACKEND, reason="CuTe backend not compiled")
class TestCuteFp16:
    def test_forward(self, problem):
        d = problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("cute requires C aligned to 8")
        out = _cute_implicit_gemm_forward_logic(
            d["in_feats_32"].half(), d["weight_32"].half(), d["kmap"], d["N_out"]
        )
        _skip_if_unsupported(out, "cute")
        _assert_close("cute_fp16 fwd", out, d["fwd_ref_64"], FP16_RTOL, FP16_ATOL)

    def test_backward(self, problem):
        d = problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("cute requires C aligned to 8")
        gi, gw = _cute_implicit_gemm_backward_logic(
            d["grad_out_32"].half(),
            d["in_feats_32"].half(),
            d["weight_32"].half(),
            d["kmap"],
            device=DEVICE,
        )
        _skip_if_unsupported((gi, gw), "cute")
        _assert_close("cute_fp16 dgrad", gi, d["gi_ref_64"], FP16_RTOL, FP16_ATOL)
        _assert_close("cute_fp16 wgrad", gw, d["gw_ref_64"], FP16_RTOL, FP16_ATOL)


@_skip_fp16_cublas
@pytest.mark.skipif(not _HAS_CUTE_GROUPED, reason="CuTe grouped not compiled")
class TestCuteGroupedFp16:
    @pytest.mark.parametrize("mma_tile", [0, 1, 2, 3])
    def test_forward(self, problem, mma_tile):
        d = problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("cute_grouped requires C aligned to 8")
        out = _cute_grouped_forward_logic(
            d["in_feats_32"].half(),
            d["weight_32"].half(),
            d["kmap"],
            d["N_out"],
            mma_tile=mma_tile,
        )
        _skip_if_unsupported(out, "cute_grouped")
        _assert_close(
            f"cute_grouped_fp16 fwd tile={mma_tile}", out, d["fwd_ref_64"], FP16_RTOL, FP16_ATOL
        )

    def test_backward(self, problem):
        d = problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("cute_grouped requires C aligned to 8")
        gi, gw = _cute_grouped_backward_logic(
            d["grad_out_32"].half(),
            d["in_feats_32"].half(),
            d["weight_32"].half(),
            d["kmap"],
            requires_grad=(True, True),
            device=DEVICE,
            mma_tile=3,
        )
        _skip_if_unsupported((gi, gw), "cute_grouped")
        _assert_close("cute_grouped_fp16 dgrad", gi, d["gi_ref_64"], FP16_RTOL, FP16_ATOL)
        _assert_close("cute_grouped_fp16 wgrad", gw, d["gw_ref_64"], FP16_RTOL, FP16_ATOL)


@_skip_fp16_cublas
@pytest.mark.skipif(not _HAS_CUTE_SM90, reason="CuTe SM90 not compiled")
class TestCuteSM90Fp16:
    def test_forward(self, problem):
        d = problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("sm90 requires C aligned to 8")
        out = _cute_implicit_gemm_sm90_forward_logic(
            d["in_feats_32"].half(),
            d["weight_32"].half(),
            d["kmap"],
            d["N_out"],
            mma_tile=100,
        )
        _skip_if_unsupported(out, "sm90")
        _assert_close("sm90_fp16 fwd", out, d["fwd_ref_64"], FP16_RTOL, FP16_ATOL)


@_skip_fp16_cublas
@pytest.mark.skipif(not _HAS_CUTE_GROUPED_SM90, reason="CuTe grouped SM90 not compiled")
class TestCuteGroupedSM90Fp16:
    def test_forward(self, problem):
        d = problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("sm90 grouped requires C aligned to 8")
        out = _cute_grouped_sm90_forward_logic(
            d["in_feats_32"].half(),
            d["weight_32"].half(),
            d["kmap"],
            d["N_out"],
            mma_tile=100,
        )
        _skip_if_unsupported(out, "sm90_grouped")
        _assert_close("sm90_grouped_fp16 fwd", out, d["fwd_ref_64"], FP16_RTOL, FP16_ATOL)


@pytest.mark.skipif(
    not hasattr(_C, "mask_gemm"),
    reason="mask_gemm (mask) kernels not compiled",
)
class TestMaskProductionFp16:
    """Mask-fused mask_gemm kernels via the dispatch wrapper.

    The wrapper picks a tile based on channel alignment + mask_words and
    handles fp32 ↔ fp16 casting. Test runs against fp16 inputs (the native
    kernel dtype) so non-aligned channels still execute (router falls back
    to scalar tile 70 / 71 / 72).
    """

    def test_forward(self, problem):
        d = problem
        out = _execute_forward(
            algo="mask_gemm",
            params={"tile_id": 41},
            in_features=d["in_feats_32"].half(),
            weight=d["weight_32"].half(),
            kernel_map=d["kmap"],
            num_out_coords=d["N_out"],
            compute_dtype=torch.float16,
            fwd_block_size=None,
        )
        _assert_close("mask_gemm fwd", out, d["fwd_ref_64"], FP16_RTOL, FP16_ATOL)

    def test_backward(self, problem):
        d = problem
        gi, gw = _execute_backward(
            algo="mask_gemm",
            params={"tile_id": 60, "split_k": 64},
            grad_output=d["grad_out_32"].half(),
            in_features=d["in_feats_32"].half(),
            weight=d["weight_32"].half(),
            kernel_map=d["kmap"],
            num_out_coords=d["N_out"],
            compute_dtype=torch.float16,
            device=DEVICE,
            needs_input_grad=(True, True),
        )
        _assert_close("mask_gemm dgrad", gi, d["gi_ref_64"], FP16_RTOL, FP16_ATOL)
        _assert_close("mask_gemm wgrad", gw, d["gw_ref_64"], FP16_RTOL, FP16_ATOL)


# ----------------------------------------------------------------------------
# Tests: large-N (~2^20) smoke (subset of backends, ones x ones only)
# ----------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=_LARGE_SHAPES,
    ids=lambda p: p[3],
)
def large_problem(request):
    N, C_in, C_out, _ = request.param
    return _make_problem(N, C_in, C_out, "ones", "ones")


@pytest.mark.slow
class TestLargeN:
    """Smoke test at N up to ~2^20 with ones x ones — exercises kernel-map
    pipelines and cp.async tile boundaries on real-scale inputs."""

    def test_explicit_fp32(self, large_problem):
        d = large_problem
        out = _explicit_gemm_forward_logic(
            d["in_feats_32"], d["weight_32"], d["kmap"], d["N_out"], torch.float32
        )
        _assert_close("explicit_fp32 fwd large", out, d["fwd_ref_64"], FP32_RTOL, FP32_ATOL)

    def test_implicit_fp32(self, large_problem):
        d = large_problem
        out = _implicit_gemm_forward_logic(
            d["in_feats_32"],
            d["weight_32"],
            d["kmap"],
            d["N_out"],
            torch.float32,
            fwd_block_size=16,
        )
        _assert_close("implicit_fp32 fwd large", out, d["fwd_ref_64"], FP32_RTOL, FP32_ATOL)

    @_skip_fp16_cublas
    @pytest.mark.skipif(not _HAS_CUTE_BACKEND, reason="CuTe backend not compiled")
    def test_cute_fp16(self, large_problem):
        d = large_problem
        if d["C_in"] % 8 != 0 or d["C_out"] % 8 != 0:
            pytest.skip("cute requires C aligned to 8")
        out = _cute_implicit_gemm_forward_logic(
            d["in_feats_32"].half(), d["weight_32"].half(), d["kmap"], d["N_out"]
        )
        _skip_if_unsupported(out, "cute")
        _assert_close("cute_fp16 fwd large", out, d["fwd_ref_64"], FP16_RTOL, FP16_ATOL)

    @pytest.mark.skipif(
        not hasattr(_C, "mask_gemm"),
        reason="mask_gemm (mask) kernels not compiled",
    )
    def test_mask_gemm_fp16(self, large_problem):
        d = large_problem
        out = _execute_forward(
            algo="mask_gemm",
            params={"tile_id": 41},
            in_features=d["in_feats_32"].half(),
            weight=d["weight_32"].half(),
            kernel_map=d["kmap"],
            num_out_coords=d["N_out"],
            compute_dtype=torch.float16,
            fwd_block_size=None,
        )
        _assert_close("mask_gemm fwd large", out, d["fwd_ref_64"], FP16_RTOL, FP16_ATOL)
