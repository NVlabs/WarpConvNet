# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the exact CUDA arch parser used by setup.py.

These tests import ``build_arch.py`` by file path so they run without torch,
without a GPU, and without installing warpconvnet.
"""

import importlib.util
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_ARCH_PATH = os.path.join(_REPO_ROOT, "build_arch.py")
_spec = importlib.util.spec_from_file_location("warpconvnet_build_arch_test", _BUILD_ARCH_PATH)
build_arch = importlib.util.module_from_spec(_spec)
# Register before exec: the frozen dataclass with `from __future__ import
# annotations` resolves its field types via sys.modules[cls.__module__].
sys.modules["warpconvnet_build_arch_test"] = build_arch
_spec.loader.exec_module(build_arch)

CudaArchTarget = build_arch.CudaArchTarget
CudaArchError = build_arch.CudaArchError
parse_cuda_arch_token = build_arch.parse_cuda_arch_token
parse_cuda_arch_list = build_arch.parse_cuda_arch_list
cuda_gencode_flags = build_arch.cuda_gencode_flags
cuda_feature_macros = build_arch.cuda_feature_macros


# --------------------------------------------------------------------------- #
# Token parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "token, major, minor, accelerated, emit_ptx",
    [
        ("10.0a", 10, 0, True, False),
        ("12.0", 12, 0, False, False),
        ("sm_120", 12, 0, False, False),
        ("sm_100a", 10, 0, True, False),
        ("compute_120", 12, 0, False, False),
        ("120", 12, 0, False, False),
        ("100a", 10, 0, True, False),
        ("8.9", 8, 9, False, False),
        ("12.0+PTX", 12, 0, False, True),
        ("SM_100A", 10, 0, True, False),
        ("12.0+ptx", 12, 0, False, True),
        ("compute_90a+PTX", 9, 0, True, True),
    ],
)
def test_parse_token_exact(token, major, minor, accelerated, emit_ptx):
    target = parse_cuda_arch_token(token)
    assert target.major == major
    assert target.minor == minor
    assert target.accelerated is accelerated
    assert target.emit_ptx is emit_ptx


def test_target_properties():
    t = parse_cuda_arch_token("10.0a")
    assert t.base_code == 100
    assert t.code == "100a"
    assert t.cubin_flag == "-gencode=arch=compute_100a,code=sm_100a"
    assert t.ptx_flag == "-gencode=arch=compute_100a,code=compute_100a"

    t = parse_cuda_arch_token("12.0")
    assert t.base_code == 120
    assert t.code == "120"
    assert t.cubin_flag == "-gencode=arch=compute_120,code=sm_120"


def test_accelerated_never_inferred():
    # A plain 9.0 request stays non-accelerated; the "a" marker is identity only.
    assert parse_cuda_arch_token("9.0").accelerated is False
    assert parse_cuda_arch_token("9.0a").accelerated is True


@pytest.mark.parametrize(
    "token",
    [
        "Blackwell",
        "Ampere",
        "Hopper",
        "10.5",
        "11.0",
        "sm_105",
        "garbage",
        "",
        "99",
    ],
)
def test_parse_token_rejected(token):
    with pytest.raises(CudaArchError):
        parse_cuda_arch_token(token)


def test_rejection_names_the_token():
    with pytest.raises(CudaArchError) as exc:
        parse_cuda_arch_token("10.5")
    assert "10.5" in str(exc.value)


# --------------------------------------------------------------------------- #
# GB-series build targets
#
# sm_103 (B300/GB300) and sm_121 (GB10) are certified build targets: nvcc 13.x
# emits them, and the runtime tile gate authorizes them via CUDA minor-version
# binary compatibility. Certifying a code must not make it alias onto another
# code's cubin — one token still maps to exactly one gencode target.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "token, code",
    [
        ("10.3", "103"),
        ("10.3a", "103a"),
        ("sm_103", "103"),
        ("103", "103"),
        ("12.1", "121"),
        ("12.1a", "121a"),
        ("sm_121", "121"),
        ("121", "121"),
    ],
)
def test_gb_series_tokens_accepted(token, code):
    assert parse_cuda_arch_token(token).code == code


def test_gb_series_targets_emit_exactly_one_cubin_each():
    targets = parse_cuda_arch_list("10.0;10.3;12.0;12.1")
    flags = cuda_gencode_flags(targets)
    assert flags == [
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_103,code=sm_103",
        "-gencode=arch=compute_120,code=sm_120",
        "-gencode=arch=compute_121,code=sm_121",
    ]


def test_gb_series_does_not_imply_accelerated_sm100_macro():
    # A non-accelerated Blackwell target must not turn on the SM100 (tcgen05)
    # backend — that still requires an explicit 10.0a. The wheel arch list is
    # non-accelerated on purpose so one cubin spans sm_100 and sm_103.
    macros = cuda_feature_macros(parse_cuda_arch_list("10.0;10.3"))
    assert "-DWARPCONVNET_SM80_ENABLED=1" in macros
    assert not any("SM100_ENABLED" in m for m in macros)


# --------------------------------------------------------------------------- #
# List parsing: order, dedupe, +PTX upgrade
# --------------------------------------------------------------------------- #
def test_parse_list_mixed_order_preserved():
    targets = parse_cuda_arch_list("8.9;12.0")
    assert [t.canonical_token for t in targets] == ["8.9", "12.0"]


def test_parse_list_dedupe():
    targets = parse_cuda_arch_list("12.0 12.0 sm_120")
    assert len(targets) == 1
    assert targets[0].base_code == 120
    assert targets[0].emit_ptx is False


def test_parse_list_ptx_upgrade_no_second_cubin():
    targets = parse_cuda_arch_list("12.0 12.0+PTX")
    assert len(targets) == 1
    assert targets[0].emit_ptx is True
    flags = cuda_gencode_flags(targets)
    # Exactly one cubin, one ptx.
    assert flags == [
        "-gencode=arch=compute_120,code=sm_120",
        "-gencode=arch=compute_120,code=compute_120",
    ]


def test_accelerated_and_nonaccelerated_are_distinct():
    targets = parse_cuda_arch_list("9.0 9.0a")
    assert len(targets) == 2


def test_parse_list_empty_rejected():
    with pytest.raises(CudaArchError):
        parse_cuda_arch_list("   ")


# --------------------------------------------------------------------------- #
# Gencode emission
# --------------------------------------------------------------------------- #
def test_sm120_only_gencode_is_exact():
    targets = parse_cuda_arch_list("12.0")
    flags = cuda_gencode_flags(targets)
    assert flags == ["-gencode=arch=compute_120,code=sm_120"]
    joined = " ".join(flags)
    assert "90a" not in joined
    assert "100a" not in joined
    assert "compute_120,code=compute_120" not in joined  # no auto PTX


def test_gencode_order_matches_request():
    targets = parse_cuda_arch_list("8.9 9.0a 12.0")
    flags = cuda_gencode_flags(targets)
    assert flags == [
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_120,code=sm_120",
    ]


# --------------------------------------------------------------------------- #
# Feature macros
# --------------------------------------------------------------------------- #
def test_macros_sm120_only():
    macros = cuda_feature_macros(parse_cuda_arch_list("12.0"))
    assert macros == ["-DWARPCONVNET_SM80_ENABLED=1"]


def test_macros_sm90_requires_accelerated():
    assert cuda_feature_macros(parse_cuda_arch_list("9.0")) == ["-DWARPCONVNET_SM80_ENABLED=1"]
    assert cuda_feature_macros(parse_cuda_arch_list("9.0a")) == [
        "-DWARPCONVNET_SM80_ENABLED=1",
        "-DWARPCONVNET_SM90_ENABLED=1",
    ]


def test_macros_sm100_requires_accelerated():
    assert "-DWARPCONVNET_SM100_ENABLED=1" not in cuda_feature_macros(parse_cuda_arch_list("10.0"))
    macros = cuda_feature_macros(parse_cuda_arch_list("10.0a"))
    assert "-DWARPCONVNET_SM100_ENABLED=1" in macros


def test_macros_sm7x_only_has_no_sm80():
    macros = cuda_feature_macros(parse_cuda_arch_list("7.0 7.5"))
    assert macros == []
