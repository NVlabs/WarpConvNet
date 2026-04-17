# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile metadata adapter for warpgemm production tiles.

Single source of truth for per-tile correctness constraints and
performance hints. Used by ``dispatch.py`` to pick a tile given a
runtime problem shape (C_in, C_out, K, groups, dtypes) instead of
hardcoding rules per tile_id.

Backed by ``warpgemm.autotune.tile_metadata`` when warpgemm is
importable (always, today). Falls back to the committed JSON snapshot
at ``tile_metadata_snapshot.json`` if the Python package cannot be
imported (e.g., during docs build without warpgemm installed).

See ``../../../csrc/include/METADATA_USAGE.md`` (mirrored from
warpgemm) for the full API description and dispatcher-migration guide.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

_MIN_SCHEMA_VERSION = 3
_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "tile_metadata_snapshot.json")


@dataclass(frozen=True)
class _SnapshotTile:
    """Lightweight TileMetadata-compatible record parsed from the JSON snapshot.

    Mirrors the method surface used by ``candidate_tiles`` /
    ``rank_candidates`` so the warpgemm-imported path and the JSON-fallback
    path are interchangeable.
    """

    tile_id: int
    op: str
    kernel_struct: str
    tile_tag: str
    tier: str
    comment: str
    tile_m: int
    tile_n: int
    tile_k: int
    num_stages: int
    mainloop: str
    epilogue: str
    acc_dtype: str  # default accumulator (legacy, prefer supported_accumulator_dtypes)
    supported_accumulator_dtypes: tuple[str, ...]
    supported_input_dtypes: tuple[str, ...]
    supported_output_dtypes: tuple[str, ...]
    mask_words: int
    c_in_alignment: int
    c_out_alignment: int
    min_per_group_c_in: int
    min_per_group_c_out: int
    supports_groups: bool
    supports_identity_shortcut: bool
    supports_split_k: bool
    persistent: bool
    scalar_a: bool
    scalar_b: bool
    scalar_epi: bool

    def supports_mask_words(self, mw: int) -> bool:
        return self.mask_words >= mw

    def handles_c_in(self, c_in: int, groups: int = 1) -> bool:
        per_group = c_in // max(groups, 1)
        if self.op == "wgrad":
            # Wgrad binding requires strict per_group % kVec == 0
            # (or % 1 == 0 for scalar_a tiles). See warpgemm metadata v3.
            return self.min_per_group_c_in > 0 and per_group % self.min_per_group_c_in == 0
        padded = ((per_group + 7) // 8) * 8
        return padded >= self.min_per_group_c_in

    def handles_c_out(self, c_out: int, groups: int = 1) -> bool:
        per_group = c_out // max(groups, 1)
        if self.op == "wgrad":
            return self.min_per_group_c_out > 0 and per_group % self.min_per_group_c_out == 0
        padded = ((per_group + 7) // 8) * 8
        return padded >= self.min_per_group_c_out

    def prefers_vectorized_c_in(self, c_in: int, groups: int = 1) -> bool:
        if self.c_in_alignment == 1:
            return True
        return (c_in // max(groups, 1)) % self.c_in_alignment == 0

    def prefers_vectorized_c_out(self, c_out: int, groups: int = 1) -> bool:
        if self.c_out_alignment == 1:
            return True
        return (c_out // max(groups, 1)) % self.c_out_alignment == 0

    def handles_dtypes(self, input_dtype: str, output_dtype: str) -> bool:
        return (
            input_dtype in self.supported_input_dtypes
            and output_dtype in self.supported_output_dtypes
        )

    def handles_accumulator(self, acc_dtype: str) -> bool:
        """True if this tile can dispatch at the given accumulator dtype.

        Base tiles support both (``f16``, ``f32``), selected at runtime via
        ``use_fp16_accum`` on ``_C.mask_gemm_*``. Specialized tiles
        (``_F16Accum``/``_F16K8``/``_DgradK8``/``_F32K8``) support exactly one.
        """
        return acc_dtype in self.supported_accumulator_dtypes


def _load_snapshot() -> list[_SnapshotTile]:
    with open(_SNAPSHOT_PATH) as f:
        blob = json.load(f)
    if blob.get("schema_version", 0) < _MIN_SCHEMA_VERSION:
        raise RuntimeError(
            f"tile_metadata_snapshot.json schema_version "
            f"{blob.get('schema_version')} < required {_MIN_SCHEMA_VERSION}"
        )
    tiles = []
    for d in blob["tiles"]:
        tiles.append(
            _SnapshotTile(
                tile_id=d["tile_id"],
                op=d["op"],
                kernel_struct=d.get("kernel_struct", ""),
                tile_tag=d.get("tile_tag", ""),
                tier=d.get("tier", "production"),
                comment=d.get("comment", ""),
                tile_m=d.get("tile_m", 0),
                tile_n=d.get("tile_n", 0),
                tile_k=d.get("tile_k", 0),
                num_stages=d.get("num_stages", 1),
                mainloop=d.get("mainloop", ""),
                epilogue=d.get("epilogue", ""),
                acc_dtype=d.get("acc_dtype", "f32"),
                supported_accumulator_dtypes=tuple(
                    d.get("supported_accumulator_dtypes") or (d.get("acc_dtype", "f32"),)
                ),
                supported_input_dtypes=tuple(d.get("supported_input_dtypes", ())),
                supported_output_dtypes=tuple(d.get("supported_output_dtypes", ())),
                mask_words=d.get("mask_words", 1),
                c_in_alignment=d.get("c_in_alignment", 8),
                c_out_alignment=d.get("c_out_alignment", 8),
                min_per_group_c_in=d.get("min_per_group_c_in", 8),
                min_per_group_c_out=d.get("min_per_group_c_out", 8),
                supports_groups=d.get("supports_groups", True),
                supports_identity_shortcut=d.get("supports_identity_shortcut", False),
                supports_split_k=d.get("supports_split_k", False),
                persistent=d.get("persistent", False),
                scalar_a=d.get("scalar_a", False),
                scalar_b=d.get("scalar_b", False),
                scalar_epi=d.get("scalar_epi", False),
            )
        )
    return tiles


def _get_tiles(op: str):
    """Return tile records for ``op``. Prefer warpgemm import; fall back to snapshot."""
    try:
        from warpgemm.autotune import build_tile_metadata, get_schema_version

        if get_schema_version() < _MIN_SCHEMA_VERSION:
            raise RuntimeError(
                f"warpgemm tile_metadata schema_version "
                f"{get_schema_version()} < required {_MIN_SCHEMA_VERSION}"
            )
        return build_tile_metadata(active_only=True, ops=(op,))
    except ImportError:
        tiles = _load_snapshot()
        return [t for t in tiles if t.op == op]


# ---------------------------------------------------------------------------
# Filter + rank (same shape as warpgemm's tile_metadata_example.py)
# ---------------------------------------------------------------------------


def candidate_tiles(
    op: str,
    C_in: int,
    C_out: int,
    K: int,
    groups: int = 1,
    input_dtype: str = "f16",
    output_dtype: str = "f16",
    tier: str = "production",
    acc_dtype: str | None = None,
) -> list:
    """Return tiles that will accept the runtime shape (correctness gate).

    If ``acc_dtype`` is ``"f16"`` or ``"f32"``, hard-filter on accumulator
    support (useful to force fp32 for training stability or fp16 for
    inference throughput).
    """
    mw_needed = (K + 31) // 32
    tiles = _get_tiles(op)
    return [
        m
        for m in tiles
        if m.tier == tier
        and m.supports_mask_words(mw_needed)
        and m.handles_c_in(C_in, groups)
        and m.handles_c_out(C_out, groups)
        and m.handles_dtypes(input_dtype, output_dtype)
        and (acc_dtype is None or m.handles_accumulator(acc_dtype))
    ]


def rank_candidates(
    candidates: Iterable,
    C_in: int,
    C_out: int,
    groups: int = 1,
    prefer_fp16_accum: bool = False,
) -> list:
    """Rank candidates by perf preference. Best first."""
    pref_acc = "f16" if prefer_fp16_accum else "f32"

    def key(m):
        return (
            not m.prefers_vectorized_c_in(C_in, groups),
            not m.prefers_vectorized_c_out(C_out, groups),
            m.acc_dtype != pref_acc,
            -(m.tile_m * m.tile_n),
            m.tile_id,
        )

    return sorted(candidates, key=key)


def pick_tile(
    op: str,
    C_in: int,
    C_out: int,
    K: int,
    groups: int = 1,
    input_dtype: str = "f16",
    output_dtype: str = "f16",
    prefer_fp16_accum: bool = False,
    acc_dtype: str | None = None,
) -> object | None:
    """Filter + rank + top candidate. None if no tile fits.

    ``acc_dtype`` (``"f16"`` or ``"f32"``) hard-filters on accumulator; when
    unset, ``prefer_fp16_accum`` biases the ranker instead.
    """
    c = candidate_tiles(op, C_in, C_out, K, groups, input_dtype, output_dtype, acc_dtype=acc_dtype)
    if not c:
        return None
    return rank_candidates(c, C_in, C_out, groups, prefer_fp16_accum)[0]


def explain(
    op: str,
    C_in: int,
    C_out: int,
    K: int,
    groups: int = 1,
    input_dtype: str = "f16",
    output_dtype: str = "f16",
    acc_dtype: str | None = None,
) -> str:
    """Human-readable diagnostic for a dispatch lookup."""
    mw_needed = (K + 31) // 32
    tiles = _get_tiles(op)
    out = [
        f"Problem: op={op} C_in={C_in} C_out={C_out} K={K} "
        f"groups={groups} dtypes={input_dtype}->{output_dtype}",
        f"Active {op} tiles: {len(tiles)}",
    ]

    survivors = []
    for t in tiles:
        if t.tier != "production":
            out.append(f"  reject tile {t.tile_id:>3}: tier={t.tier}")
            continue
        if not t.supports_mask_words(mw_needed):
            out.append(f"  reject tile {t.tile_id:>3}: mask_words={t.mask_words} < {mw_needed}")
            continue
        if not t.handles_c_in(C_in, groups):
            out.append(
                f"  reject tile {t.tile_id:>3}: C_in={C_in}/g={groups} fails "
                f"padded >= {t.min_per_group_c_in}"
            )
            continue
        if not t.handles_c_out(C_out, groups):
            out.append(
                f"  reject tile {t.tile_id:>3}: C_out={C_out}/g={groups} fails "
                f"padded >= {t.min_per_group_c_out}"
            )
            continue
        if not t.handles_dtypes(input_dtype, output_dtype):
            out.append(
                f"  reject tile {t.tile_id:>3}: dtypes {input_dtype}->{output_dtype} "
                f"not in {t.supported_input_dtypes}->{t.supported_output_dtypes}"
            )
            continue
        if acc_dtype is not None and not t.handles_accumulator(acc_dtype):
            out.append(
                f"  reject tile {t.tile_id:>3}: acc={acc_dtype} "
                f"not in {t.supported_accumulator_dtypes}"
            )
            continue
        survivors.append(t)

    out.append(f"Survivors: {len(survivors)}")
    ranked = rank_candidates(survivors, C_in, C_out, groups)
    for i, t in enumerate(ranked[:5]):
        vec_in = t.prefers_vectorized_c_in(C_in, groups)
        vec_out = t.prefers_vectorized_c_out(C_out, groups)
        out.append(
            f"  [{i}] tile {t.tile_id:>3} {t.kernel_struct}  "
            f"mnk=({t.tile_m},{t.tile_n},{t.tile_k})  acc={t.acc_dtype}  "
            f"vec_in={vec_in} vec_out={vec_out}"
        )
    return "\n".join(out)
