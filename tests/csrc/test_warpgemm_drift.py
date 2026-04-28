# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift detection between checked-in warpgemm-generated artifacts and
fresh emits from the installed warpgemm package.

warpgemm is a developer codegen tool, not a runtime or build dependency,
so these tests are skipped when warpgemm is not importable. CI runs that
have warpgemm available will catch:
  - warpgemm version drift relative to the checked-in snapshot
  - schema-major bumps that need integration work on the warpconvnet side
  - hand-edits to the tracked .cu / .cuh / .inc / .py files
"""

from __future__ import annotations

import filecmp
import tempfile
from pathlib import Path

import pytest

offset_gemm_codegen = pytest.importorskip(
    "warpgemm.codegen.offset_gemm",
    reason="warpgemm is a developer codegen tool; install it to run drift checks",
)
autotune = pytest.importorskip(
    "warpgemm.autotune",
    reason="warpgemm is a developer codegen tool; install it to run drift checks",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFSET_GEMM_DIR = REPO_ROOT / "warpconvnet" / "csrc" / "offset_gemm"
MASK_GEMM_DIR = REPO_ROOT / "warpconvnet" / "csrc" / "mask_gemm"
SUPPORTED_OFFSET_GEMM_SCHEMA_MAJOR = 3
SUPPORTED_TILE_METADATA_SCHEMA = 5


def test_offset_gemm_schema_major_pinned() -> None:
    major = int(offset_gemm_codegen.SCHEMA_VERSION.split(".", 1)[0])
    assert major == SUPPORTED_OFFSET_GEMM_SCHEMA_MAJOR, (
        f"warpgemm.codegen.offset_gemm.SCHEMA_VERSION={offset_gemm_codegen.SCHEMA_VERSION} "
        f"major={major} != {SUPPORTED_OFFSET_GEMM_SCHEMA_MAJOR}. A schema-major bump "
        "requires updating warpconvnet bindings, template headers, and this drift test."
    )


def test_tile_metadata_schema_pinned() -> None:
    schema = autotune.get_schema_version()
    assert schema == SUPPORTED_TILE_METADATA_SCHEMA, (
        f"warpgemm.autotune.get_schema_version()={schema} != "
        f"{SUPPORTED_TILE_METADATA_SCHEMA}. Schema bumps require reviewing the "
        "tile_metadata consumer in nn/.../sparse_conv/detail/tile_metadata.py."
    )


def _diff_emit(emitted_paths: list[Path], tmp: str, tracked_root: Path) -> list[str]:
    diffs: list[str] = []
    for emitted in emitted_paths:
        rel = emitted.relative_to(tmp)
        tracked = tracked_root / rel
        if not tracked.is_file():
            diffs.append(f"missing tracked file: {rel}")
            continue
        if not filecmp.cmp(str(emitted), str(tracked), shallow=False):
            diffs.append(f"drift: {rel}")
    return diffs


def test_offset_gemm_snapshot_matches_emit() -> None:
    """Every file warpgemm.codegen.offset_gemm.write_to() emits must equal the tracked copy."""

    with tempfile.TemporaryDirectory() as tmp:
        emitted = [Path(p) for p in offset_gemm_codegen.write_to(tmp)]
        diffs = _diff_emit(emitted, tmp, OFFSET_GEMM_DIR)
        assert not diffs, (
            "Tracked offset_gemm codegen snapshot drifted from warpgemm emit:\n  "
            + "\n  ".join(diffs)
            + "\nRefresh with `WARPGEMM_REGEN=1 pip install -e . --no-build-isolation` "
            "and commit the diff."
        )


def test_tile_metadata_snapshot_matches_emit() -> None:
    """tile_metadata.py emit must equal the tracked copy."""

    with tempfile.TemporaryDirectory() as tmp:
        emitted = [Path(autotune.write_tile_metadata_to(tmp))]
        diffs = _diff_emit(emitted, tmp, MASK_GEMM_DIR)
        assert not diffs, (
            "Tracked tile_metadata.py drifted from warpgemm emit:\n  "
            + "\n  ".join(diffs)
            + "\nRefresh with `WARPGEMM_REGEN=1 pip install -e . --no-build-isolation` "
            "and commit the diff."
        )
