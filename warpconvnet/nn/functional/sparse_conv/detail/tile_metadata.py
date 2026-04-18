# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile metadata adapter for warpgemm production tiles.

Thin wrapper over ``warpgemm.autotune.build_tile_metadata`` — the single
source of truth for per-tile correctness constraints and performance
hints. Used by ``dispatch.py`` to reason about which tile will accept a
runtime problem shape (C_in, C_out, K, groups, dtypes).

Warpgemm is a hard build dependency; no JSON-snapshot fallback. See
``../../../csrc/include/METADATA_USAGE.md`` (mirrored from warpgemm) for
the full API description and dispatcher-migration guide.
"""

from __future__ import annotations

from typing import Iterable

_MIN_SCHEMA_VERSION = 4


def _get_tiles(op: str):
    """Return active tile records for ``op``. Imports warpgemm directly."""
    from warpgemm.autotune import build_tile_metadata, get_schema_version

    if get_schema_version() < _MIN_SCHEMA_VERSION:
        raise RuntimeError(
            f"warpgemm tile_metadata schema_version "
            f"{get_schema_version()} < required {_MIN_SCHEMA_VERSION}"
        )
    return build_tile_metadata(active_only=True, ops=(op,))


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

    Uses warpgemm's v4 ``TileMetadata.accepts(...)`` when ``acc_dtype`` is
    set; otherwise composes the pre-v4 helpers so callers that don't care
    about accumulator can still filter.
    """
    mw_needed = (K + 31) // 32
    tiles = _get_tiles(op)
    acc = acc_dtype or "f32"
    return [
        m
        for m in tiles
        if m.tier == tier and m.accepts(C_in, C_out, K, groups, input_dtype, output_dtype, acc)
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
