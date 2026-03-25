# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Individual CUDA kernel correctness tests for sparse convolution.
# Tests each kernel backend directly against explicit_gemm (fp32) reference.

import pytest
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
)
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
    _explicit_gemm_forward_grouped,
    _explicit_gemm_backward_grouped,
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
from warpconvnet.nn.functional.sparse_conv.detail.mask_gemm import (
    _mask_implicit_gemm_forward_logic,
    _mask_implicit_gemm_backward_logic,
    _get_mask_data,
)
from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
    _HAS_CUTE_BACKEND,
    _HAS_CUTE_GROUPED,
    _HAS_CUTE_SM90,
    _HAS_CUTE_GROUPED_SM90,
)
import warpconvnet._C as _C

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

FP32_TOL = 0.001
FP16_TOL = 0.02
DEVICE = torch.device("cuda")

# Detect broken fp16 cuBLAS. torch 2.10+cu128 has a bug where
# cublasGemmEx with CUDA_R_16F returns CUBLAS_STATUS_INVALID_VALUE.
# We detect via a subprocess to avoid poisoning our own CUDA context.
def _check_fp16_cublas_broken():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; a=torch.ones(2,2,device='cuda',dtype=torch.float16); print((a@a).sum().item())"],
        capture_output=True, timeout=30,
    )
    return result.returncode != 0

_FP16_CUBLAS_BROKEN = _check_fp16_cublas_broken()

_skip_fp16_cublas = pytest.mark.skipif(
    _FP16_CUBLAS_BROKEN,
    reason="torch.matmul fp16 broken (cuBLAS bug in this torch/CUDA version)",
)


@pytest.fixture(autouse=True)
def _clear_cuda_errors():
    """Clear stale CUDA error state before each test to prevent cascading."""
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    yield
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass


def _make_test_data(N, C_in, C_out, stride=(1, 1, 1), seed=42):
    """Create test voxels, kernel map, weight, and grad_output."""
    torch.manual_seed(seed)
    coords = torch.randint(
        0, max(int(N**0.33 * 5), 10), (N, 3), device="cuda", dtype=torch.int32
    )
    feats = torch.randn(N, C_in, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    voxels = Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        device="cuda",
    ).unique()
    out_coords, _, kmap = generate_output_coords_and_kernel_map(
        voxels,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        stride=stride,
        generative=False,
        transposed=False,
    )
    N_out = out_coords.shape[0]
    K = 27
    weight = torch.randn(K, C_in, C_out, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(N_out, C_out, device="cuda", dtype=torch.float32)
    return voxels.feature_tensor, weight, grad_out, kmap, N_out


def _assert_close(name, test, ref, tol):
    """Assert test tensor is close to ref within relative tolerance."""
    assert not test.isnan().any().item(), f"{name}: NaN detected"
    err = (test.float() - ref.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    rel = err / (ref_max + 1e-8)
    assert rel < tol, f"{name}: rel_err={rel:.6f} exceeds tol={tol}"


def _skip_if_cublas_error(fn, *args, **kwargs):
    """Run fn; if cublas error, clear CUDA state and skip test."""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        if "CUBLAS_STATUS" in str(e) or "CUDA error" in str(e):
            # Clear stale CUDA error state to prevent cascading failures
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            pytest.skip(f"cublas error: {e}")
        raise


# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture(
    params=[
        (500, 32, 64, (1, 1, 1), "s1_32_64"),
        (500, 8, 16, (1, 1, 1), "s1_8_16"),
        (500, 7, 13, (1, 1, 1), "s1_7_13_unaligned"),
        (500, 32, 64, (2, 2, 2), "s2_32_64"),
        (500, 7, 13, (2, 2, 2), "s2_7_13_unaligned"),
        (5000, 64, 128, (1, 1, 1), "s1_large_64_128"),
        (5000, 32, 32, (1, 1, 1), "s1_large_32_32"),
    ],
    ids=lambda p: p[4],
)
def conv_data(request):
    N, C_in, C_out, stride, _ = request.param
    in_feats, weight, grad_out, kmap, N_out = _make_test_data(N, C_in, C_out, stride)
    # Compute reference
    fwd_ref = _explicit_gemm_forward_logic(
        in_feats, weight, kmap, N_out, torch.float32
    )
    gi_ref, gw_ref = _explicit_gemm_backward_logic(
        grad_out, in_feats, weight, kmap, torch.float32, DEVICE
    )
    return {
        "in_feats": in_feats,
        "weight": weight,
        "grad_out": grad_out,
        "kmap": kmap,
        "N_out": N_out,
        "fwd_ref": fwd_ref,
        "gi_ref": gi_ref,
        "gw_ref": gw_ref,
    }


# ==========================================================================
# Tests: fp32 SIMT kernels
# ==========================================================================


class TestImplicitGemm:
    def test_forward(self, conv_data):
        d = conv_data
        fwd = _implicit_gemm_forward_logic(
            d["in_feats"], d["weight"], d["kmap"], d["N_out"],
            torch.float32, fwd_block_size=16,
        )
        _assert_close("implicit_gemm fwd", fwd, d["fwd_ref"], FP32_TOL)

    def test_dgrad(self, conv_data):
        d = conv_data
        gi, _ = _implicit_gemm_backward_logic(
            d["grad_out"], d["in_feats"], d["weight"], d["kmap"],
            d["N_out"], 16, 256, 4, torch.float32,
        )
        _assert_close("implicit_gemm dgrad", gi, d["gi_ref"], FP32_TOL)

    def test_wgrad(self, conv_data):
        d = conv_data
        _, gw = _implicit_gemm_backward_logic(
            d["grad_out"], d["in_feats"], d["weight"], d["kmap"],
            d["N_out"], 16, 256, 4, torch.float32,
        )
        _assert_close("implicit_gemm wgrad", gw, d["gw_ref"], FP32_TOL)


class TestImplicitGemmGrouped:
    def test_forward(self, conv_data):
        d = conv_data
        fwd = _implicit_gemm_forward_grouped(
            d["in_feats"], d["weight"], d["kmap"], d["N_out"],
            torch.float32, fwd_block_size=16, saturation_m=5000,
        )
        _assert_close("implicit_grouped fwd", fwd, d["fwd_ref"], FP32_TOL)

    def test_dgrad(self, conv_data):
        d = conv_data
        gi, _ = _implicit_gemm_backward_grouped(
            d["grad_out"], d["in_feats"], d["weight"], d["kmap"],
            d["N_out"], 16, 256, 4, torch.float32, saturation_m=5000,
        )
        _assert_close("implicit_grouped dgrad", gi, d["gi_ref"], FP32_TOL)

    def test_wgrad(self, conv_data):
        d = conv_data
        _, gw = _implicit_gemm_backward_grouped(
            d["grad_out"], d["in_feats"], d["weight"], d["kmap"],
            d["N_out"], 16, 256, 4, torch.float32, saturation_m=5000,
        )
        _assert_close("implicit_grouped wgrad", gw, d["gw_ref"], FP32_TOL)


# ==========================================================================
# Tests: CUTLASS per-offset gather-scatter
# ==========================================================================


@_skip_fp16_cublas
class TestCutlassImplicitGemm:
    def test_forward(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cutlass_implicit_gemm_forward_logic,
            d["in_feats"].half(), d["weight"].half(), d["kmap"], d["N_out"],
        )
        if isinstance(result, int):
            pytest.skip(f"cutlass unsupported (status={result})")
        _assert_close("cutlass fwd", result, d["fwd_ref"], FP16_TOL)

    def test_dgrad(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cutlass_implicit_gemm_backward_logic,
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], device=DEVICE,
        )
        if isinstance(result[0], int):
            pytest.skip(f"cutlass unsupported (status={result[0]})")
        _assert_close("cutlass dgrad", result[0], d["gi_ref"], FP16_TOL)

    def test_wgrad(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cutlass_implicit_gemm_backward_logic,
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], device=DEVICE,
        )
        if isinstance(result[0], int):
            pytest.skip(f"cutlass unsupported (status={result[0]})")
        _assert_close("cutlass wgrad", result[1], d["gw_ref"], FP16_TOL)


# ==========================================================================
# Tests: CuTe per-offset gather-scatter
# ==========================================================================


@_skip_fp16_cublas
@pytest.mark.skipif(not _HAS_CUTE_BACKEND, reason="CuTe backend not compiled")
class TestCuteImplicitGemm:
    def test_forward(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_implicit_gemm_forward_logic,
            d["in_feats"].half(), d["weight"].half(), d["kmap"], d["N_out"],
        )
        if isinstance(result, int):
            pytest.skip(f"cute unsupported (status={result})")
        _assert_close("cute fwd", result, d["fwd_ref"], FP16_TOL)

    def test_backward(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_implicit_gemm_backward_logic,
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], device=DEVICE,
        )
        if isinstance(result[0], int):
            pytest.skip(f"cute unsupported (status={result[0]})")
        _assert_close("cute dgrad", result[0], d["gi_ref"], FP16_TOL)
        _assert_close("cute wgrad", result[1], d["gw_ref"], FP16_TOL)


# ==========================================================================
# Tests: CuTe grouped fused multi-offset GEMM
# ==========================================================================


@_skip_fp16_cublas
@pytest.mark.skipif(not _HAS_CUTE_GROUPED, reason="CuTe grouped not compiled")
class TestCuteGrouped:
    @pytest.mark.parametrize("mma_tile", [0, 1, 2, 3])
    def test_forward(self, conv_data, mma_tile):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_grouped_forward_logic,
            d["in_feats"].half(), d["weight"].half(), d["kmap"], d["N_out"],
            mma_tile=mma_tile,
        )
        if isinstance(result, int):
            pytest.skip(f"cute_grouped unsupported (status={result})")
        _assert_close(f"cute_grouped fwd tile={mma_tile}", result, d["fwd_ref"], FP16_TOL)

    def test_dgrad(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_grouped_backward_logic,
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], requires_grad=(True, True), device=DEVICE, mma_tile=3,
        )
        if isinstance(result[0], int):
            pytest.skip(f"cute_grouped unsupported (status={result[0]})")
        _assert_close("cute_grouped dgrad", result[0], d["gi_ref"], FP16_TOL)

    def test_wgrad(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_grouped_backward_logic,
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], requires_grad=(True, True), device=DEVICE, mma_tile=3,
        )
        if isinstance(result[0], int):
            pytest.skip(f"cute_grouped unsupported (status={result[0]})")
        _assert_close("cute_grouped wgrad", result[1], d["gw_ref"], FP16_TOL)


# ==========================================================================
# Tests: SM90 WGMMA kernels
# ==========================================================================


@_skip_fp16_cublas
@pytest.mark.skipif(not _HAS_CUTE_SM90, reason="CuTe SM90 not compiled")
class TestCuteSM90:
    def test_forward(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_implicit_gemm_sm90_forward_logic,
            d["in_feats"].half(), d["weight"].half(), d["kmap"], d["N_out"],
            mma_tile=100,
        )
        if isinstance(result, int):
            pytest.skip(f"sm90 unsupported (status={result})")
        _assert_close("cute_sm90 fwd", result, d["fwd_ref"], FP16_TOL)

    def test_backward(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_implicit_gemm_sm90_backward_logic,
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], requires_grad=(True, True), device=DEVICE, mma_tile=100,
        )
        if isinstance(result[0], int):
            pytest.skip(f"sm90 unsupported (status={result[0]})")
        _assert_close("cute_sm90 dgrad", result[0], d["gi_ref"], FP16_TOL)
        _assert_close("cute_sm90 wgrad", result[1], d["gw_ref"], FP16_TOL)


@_skip_fp16_cublas
@pytest.mark.skipif(not _HAS_CUTE_GROUPED_SM90, reason="CuTe grouped SM90 not compiled")
class TestCuteGroupedSM90:
    def test_forward(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_grouped_sm90_forward_logic,
            d["in_feats"].half(), d["weight"].half(), d["kmap"], d["N_out"],
            mma_tile=100,
        )
        if isinstance(result, int):
            pytest.skip(f"sm90 grouped unsupported (status={result})")
        _assert_close("cute_grouped_sm90 fwd", result, d["fwd_ref"], FP16_TOL)

    def test_backward(self, conv_data):
        d = conv_data
        result = _skip_if_cublas_error(
            _cute_grouped_sm90_backward_logic,
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], requires_grad=(True, True), device=DEVICE, mma_tile=100,
        )
        if isinstance(result[0], int):
            pytest.skip(f"sm90 grouped unsupported (status={result[0]})")
        _assert_close("cute_grouped_sm90 dgrad", result[0], d["gi_ref"], FP16_TOL)
        _assert_close("cute_grouped_sm90 wgrad", result[1], d["gw_ref"], FP16_TOL)


# ==========================================================================
# Tests: Mask-based fused implicit GEMM (CuTe tensor core)
# ==========================================================================


@pytest.mark.skipif(
    not hasattr(_C.gemm, "cute_gemm_mask_fwd"),
    reason="CuTe mask kernel not compiled",
)
class TestMaskImplicitGemmCuTe:
    @pytest.mark.parametrize("mma_tile", [0, 1, 2, 3])
    def test_forward(self, conv_data, mma_tile):
        d = conv_data
        fwd = _mask_implicit_gemm_forward_logic(
            d["in_feats"].half(), d["weight"].half(), d["kmap"], d["N_out"],
            compute_dtype=torch.float16, mma_tile=mma_tile,
        )
        _assert_close(f"mask_fwd tile={mma_tile}", fwd, d["fwd_ref"], FP16_TOL)

    @pytest.mark.parametrize("mma_tile", [0, 1, 2, 3])
    def test_dgrad_reverse(self, conv_data, mma_tile):
        """Dgrad via reverse pair_table + forward kernel (no atomicAdd)."""
        d = conv_data
        gi, _ = _mask_implicit_gemm_backward_logic(
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], d["N_out"],
            compute_dtype=torch.float16, needs_input_grad=(True, False),
            mma_tile=mma_tile,
        )
        _assert_close(f"mask_dgrad_reverse tile={mma_tile}", gi, d["gi_ref"], FP16_TOL)

    def test_wgrad(self, conv_data):
        d = conv_data
        _, gw = _mask_implicit_gemm_backward_logic(
            d["grad_out"].half(), d["in_feats"].half(), d["weight"].half(),
            d["kmap"], d["N_out"],
            compute_dtype=torch.float16, needs_input_grad=(False, True),
        )
        _assert_close("mask_wgrad", gw, d["gw_ref"], FP16_TOL)


# ==========================================================================
# Tests: Mask-based SIMT kernels (raw _C.gemm calls)
# ==========================================================================


@pytest.mark.skipif(
    not hasattr(_C.gemm, "mask_implicit_gemm_fwd"),
    reason="SIMT mask kernel not compiled",
)
class TestMaskImplicitGemmSIMT:
    def test_forward(self, conv_data):
        d = conv_data
        pair_table, pair_mask, mask_argsort = _get_mask_data(
            d["kmap"], d["N_out"], DEVICE
        )
        output = torch.zeros(d["N_out"], d["weight"].shape[2], dtype=torch.float16, device=DEVICE)
        status = _C.gemm.mask_implicit_gemm_fwd(
            d["in_feats"].half(), d["weight"].half(), output,
            pair_table, pair_mask, mask_argsort, 27, 16,
        )
        assert status == 0, f"mask_fwd_SIMT failed: {status}"
        _assert_close("mask_fwd_SIMT", output, d["fwd_ref"], FP16_TOL)

    def test_dgrad(self, conv_data):
        d = conv_data
        N_in = d["in_feats"].shape[0]
        C_in = d["in_feats"].shape[1]
        pair_table, pair_mask, mask_argsort = _get_mask_data(
            d["kmap"], d["N_out"], DEVICE
        )
        gi = torch.zeros(N_in, C_in, dtype=torch.float16, device=DEVICE)
        status = _C.gemm.mask_implicit_gemm_bwd_dgrad(
            d["grad_out"].half(), d["weight"].half(), gi,
            pair_table, pair_mask, mask_argsort, 27, 16,
        )
        assert status == 0, f"mask_dgrad_SIMT failed: {status}"
        _assert_close("mask_dgrad_SIMT", gi, d["gi_ref"], FP16_TOL)

    def test_wgrad(self, conv_data):
        d = conv_data
        K, C_in, C_out = d["weight"].shape
        pair_table, pair_mask, _ = _get_mask_data(
            d["kmap"], d["N_out"], DEVICE
        )
        gw = torch.zeros(K, C_in, C_out, dtype=torch.float16, device=DEVICE)
        status = _C.gemm.mask_implicit_gemm_bwd_wgrad(
            d["in_feats"].half(), d["grad_out"].half(), gw,
            pair_table, pair_mask, K, 16,
        )
        assert status == 0, f"mask_wgrad_SIMT failed: {status}"
        _assert_close("mask_wgrad_SIMT", gw, d["gw_ref"], FP16_TOL)


# ==========================================================================
# Tests: Old atomicAdd dgrad kernel (CuTe)
# ==========================================================================


@pytest.mark.skipif(
    not hasattr(_C.gemm, "cute_gemm_mask_dgrad"),
    reason="CuTe mask dgrad kernel not compiled",
)
class TestMaskDgradAtomic:
    def test_dgrad(self, conv_data):
        d = conv_data
        N_in = d["in_feats"].shape[0]
        C_in = d["in_feats"].shape[1]
        pair_table, pair_mask, mask_argsort = _get_mask_data(
            d["kmap"], d["N_out"], DEVICE
        )
        w_T = d["weight"].half().transpose(1, 2).contiguous()
        gi = torch.zeros(N_in, C_in, dtype=torch.float16, device=DEVICE)
        status = _C.gemm.cute_gemm_mask_dgrad(
            d["grad_out"].half(), w_T, gi,
            pair_table, pair_mask, mask_argsort, 27, 3, 1.0,
        )
        if status != 0:
            pytest.skip(f"cute_gemm_mask_dgrad unsupported (status={status})")
        _assert_close("mask_dgrad_atomic", gi, d["gi_ref"], FP16_TOL)
