# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offset GEMM artifacts emitted by warpgemm.codegen.offset_gemm.

Per-tile .cu translation units, per-family .cuh explicit-instantiation
headers (under include/), and offset_gemm_dispatch_table.inc. The .cu /
.cuh / .inc files are consumed by the C++ extension build (see setup.py).
All files are tracked in git so the build does not require warpgemm; the
drift CI test (tests/csrc/test_warpgemm_drift.py) catches snapshots that
fall behind warpgemm emit.
"""
