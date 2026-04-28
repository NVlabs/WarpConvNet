# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored runtime constants for the offset GEMM dispatch table.

warpgemm.codegen.offset_gemm is a build-time codegen tool, not a runtime
dependency. This module mirrors the stable string identifiers and the
minimal record shape that runtime callers need. The full record schema
lives in warpgemm; only the fields surfaced by OFFSET_GEMM_KERNEL(...) entries
in csrc/offset_gemm/offset_gemm_dispatch_table.inc appear here.

A schema-major drift in warpgemm is detected by tests/csrc/test_warpgemm_drift.py.
"""

from __future__ import annotations

from dataclasses import dataclass

OP_AD_GATHER_SCATTER = "ad_gather_scatter"
OP_TRAB_GATHER = "trab_gather"
OP_WGRAD_SPARSE = "wgrad_sparse"

BACKEND_CUTE_SM80 = "cute_sm80"
BACKEND_CUTE_SM90 = "cute_sm90"
BACKEND_CUTE_GROUPED_SM80 = "cute_grouped_sm80"
BACKEND_CUTE_GROUPED_SM90 = "cute_grouped_sm90"
BACKEND_IMPLICIT = "implicit"
BACKEND_IMPLICIT_SPLIT_K = "implicit_split_k"


@dataclass(frozen=True)
class KernelVariant:
    op: str
    backend: str
    tile_id: int
    grouped: bool
    split_k: bool
    launch_symbol: str
    tile_tag: str
    # Discriminating fields surfaced by OFFSET_GEMM_KERNEL since schema 3.0.0.
    # Implicit family carries dtype/itype/block_size/kernel_variant; cute family
    # leaves block_size=0, kernel_variant="n/a" because INSTANTIATE_ALL_DTYPES
    # fans out at compile time.
    input_dtype: str
    output_dtype: str
    acc_dtype: str
    itype: str
    block_size: int
    kernel_variant: str
    mma_atom: str

    def key(self) -> tuple[str, str, int]:
        return (self.op, self.backend, self.tile_id)
