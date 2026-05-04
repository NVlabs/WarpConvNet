<!--
created: 2026-05-04 16:25:00
edited:  2026-05-04 16:25:00
-->

# Migrating the Benchmark Cache

WarpConvNet writes autotune results to
`~/.cache/warpconvnet/benchmark_cache_generic.msgpack`. The file carries a
schema version (`WARPCONVNET_BENCHMARK_CACHE_VERSION`). When that version is
bumped — for example to rename an algorithm — older on-disk caches must be
rewritten so the values load against current code.

## Automatic migration on import

Migration runs automatically the first time `warpconvnet` instantiates the
benchmark cache. The flow:

1. `GenericBenchmarkCache.__init__` calls `load_cache()`.
2. `load_cache()` deserializes the file. If the file's major version differs
   from `WARPCONVNET_BENCHMARK_CACHE_VERSION`, it chains the registered
   migrations in `_CACHE_MIGRATIONS` to lift the data to the current version.
3. On success the cache is held in memory at the new schema and
   `pending_changes` is set, so the background saver flushes the rewritten
   file (and `atexit` flushes on shutdown).
4. If no migration path exists, the file is treated as incompatible and the
   cache resets to empty.

No user action is required. A successful migration emits one INFO log line:

```
Cache migration v8→v9 renamed N entries (production→mask_gemm)
Migrated generic benchmark cache v8.0 → v9.0
```

## Explicit migration

Use `scripts/migrate_benchmark_cache.py` when you want to migrate offline,
keep a backup, or run on a non-default cache file:

```bash
python scripts/migrate_benchmark_cache.py                           # migrate default cache
python scripts/migrate_benchmark_cache.py --backup                  # also write <path>.bak.<ts>
python scripts/migrate_benchmark_cache.py --cache-file /path/file.msgpack
python scripts/migrate_benchmark_cache.py --dry-run                 # report only
```

The script reuses the same `_try_migrate_cache` registry as the in-process
loader, so its result is byte-equivalent to what auto-migration would write.

Note: importing the `warpconvnet` package transitively triggers the
in-process auto-migration. Run the script in a fresh process if you want to
exercise it against an unmigrated file; otherwise it will report
`Already at current major version`.

## Adding a new migration step

When schema-level changes happen (renamed algorithms, new value layout):

1. Bump `WARPCONVNET_BENCHMARK_CACHE_VERSION` in `warpconvnet/constants.py`
   (major version increment).
2. Add a `_migrate_vN_vN+1(namespaces)` function in
   `warpconvnet/utils/benchmark_cache.py` that mutates the `{namespace: {key: value}}` dict in place and returns it.
3. Append `(N, N+1, _migrate_vN_vN+1)` to `_CACHE_MIGRATIONS`.

`_try_migrate_cache` chains steps automatically, so multi-version jumps work
without further plumbing.

## What is migrated

Migration touches namespace values only — keys (`SpatiallySparseConvConfig`
hashes) are left untouched. Each value is a list of `(algo, params, metric)`
tuples sorted best-first. A migration typically rewrites the `algo` string or
normalizes the `params` dict for the new schema.

## Bypassing migration

To force a fresh cache instead of migrating, delete the file:

```bash
rm ~/.cache/warpconvnet/benchmark_cache_generic.msgpack
```

Or point at a scratch directory:

```bash
WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE=/tmp/wcn-fresh python ...
```

## Migration history

| From | To  | Change                                                                                 |
| ---- | --- | -------------------------------------------------------------------------------------- |
| 8.0  | 9.0 | Rename `production`→`mask_gemm` and `production_fwd_as_dgrad`→`mask_gemm_fwd_as_dgrad` |
