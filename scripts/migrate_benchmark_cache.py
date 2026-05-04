#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Migrate an on-disk warpconvnet benchmark cache to the current schema version.

Reads ~/.cache/warpconvnet/benchmark_cache_generic.msgpack (or a custom path),
chains registered migrations in `warpconvnet.utils.benchmark_cache._CACHE_MIGRATIONS`
to the current `WARPCONVNET_BENCHMARK_CACHE_VERSION`, then atomically rewrites
the file. Falls back to no-op when the file is already current. Auto-loaded
processes also migrate on startup, but running this script up-front makes the
rewrite explicit and avoids racing with the first autotune run.

Usage:
    python scripts/migrate_benchmark_cache.py
    python scripts/migrate_benchmark_cache.py --cache-file /path/to/cache.msgpack
    python scripts/migrate_benchmark_cache.py --dry-run
    python scripts/migrate_benchmark_cache.py --backup
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import msgpack

from warpconvnet.constants import (
    WARPCONVNET_BENCHMARK_CACHE_DIR,
    WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE,
    WARPCONVNET_BENCHMARK_CACHE_VERSION,
)
from warpconvnet.utils.benchmark_cache import (
    _atomic_msgpack_replace,
    _from_msgpack,
    _namespace_from_msgpack,
    _namespace_to_msgpack,
    _parse_version_to_major_minor,
    _sanitize_for_pickle,
    _try_migrate_cache,
)


def _default_cache_file() -> Path:
    base = WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE or WARPCONVNET_BENCHMARK_CACHE_DIR
    return Path(base).expanduser().resolve() / "benchmark_cache_generic.msgpack"


def _load_raw(cache_file: Path):
    with open(cache_file, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)


def _deserialize_namespaces(raw_ns):
    result = {}
    for ns_name, pairs in raw_ns.items():
        if isinstance(pairs, list):
            result[ns_name] = _namespace_from_msgpack(pairs)
        elif isinstance(pairs, dict):
            result[ns_name] = {_from_msgpack(k): _from_msgpack(v) for k, v in pairs.items()}
    return result


def _serialize_namespaces(namespaces):
    out = {}
    for ns_name, ns_dict in namespaces.items():
        sanitized = {k: _sanitize_for_pickle(v) for k, v in ns_dict.items()}
        out[ns_name] = _namespace_to_msgpack(sanitized)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=_default_cache_file(),
        help="Path to the cache file (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be migrated without writing",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy cache file to <path>.bak.<timestamp> before rewriting",
    )
    args = parser.parse_args()

    cache_file: Path = args.cache_file
    if not cache_file.exists():
        print(f"No cache file at {cache_file}; nothing to migrate.")
        return 0

    try:
        cache_data = _load_raw(cache_file)
    except Exception as e:
        print(f"ERROR: failed to read {cache_file}: {e}", file=sys.stderr)
        return 1

    if not isinstance(cache_data, dict):
        print(f"ERROR: cache root is not a dict at {cache_file}", file=sys.stderr)
        return 1

    file_major, file_minor = _parse_version_to_major_minor(cache_data.get("version", "1.0"))
    expected_major, expected_minor = _parse_version_to_major_minor(
        WARPCONVNET_BENCHMARK_CACHE_VERSION
    )
    print(
        f"Cache file: {cache_file}\n"
        f"  on-disk version : v{file_major}.{file_minor}\n"
        f"  current version : v{expected_major}.{expected_minor}"
    )

    if file_major == expected_major:
        print("Already at current major version; no migration needed.")
        return 0

    raw_ns = cache_data.get("namespaces", {})
    if not isinstance(raw_ns, dict):
        print("ERROR: 'namespaces' is not a dict; refusing to migrate.", file=sys.stderr)
        return 1
    namespaces = _deserialize_namespaces(raw_ns)
    total_entries = sum(len(ns) for ns in namespaces.values())
    print(f"  loaded {len(namespaces)} namespaces, {total_entries} entries")

    migrated = _try_migrate_cache(namespaces, file_major, expected_major)
    if migrated is None:
        print(
            f"ERROR: no migration path v{file_major} → v{expected_major}. "
            f"Delete the cache file or add a migration step.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("Dry run: migration succeeded in memory. File not written.")
        return 0

    if args.backup:
        backup = cache_file.with_suffix(cache_file.suffix + f".bak.{int(time.time())}")
        shutil.copy2(cache_file, backup)
        print(f"  backup written to {backup}")

    new_data = {
        "namespaces": _serialize_namespaces(migrated),
        "timestamp": time.time(),
        "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
    }
    _atomic_msgpack_replace(cache_file, new_data)
    print(f"Wrote migrated cache to {cache_file} (v{expected_major}.{expected_minor}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
