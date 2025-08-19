# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import math
import os
import pickle
import threading
import time
import atexit
import enum
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Sequence, TypeVar, Generic, Callable, Iterable, List
from dataclasses import dataclass

import torch

from warpconvnet.constants import (
    WARPCONVNET_BENCHMARK_CACHE_DIR,
    WARPCONVNET_BENCHMARK_CACHE_VERSION,
    WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE,
)
from warpconvnet.utils.logger import get_logger
from warpconvnet.utils.dist import _get_current_rank, _is_rank_zero

logger = get_logger(__name__, rank_zero_only=False)


# Rank detection is now handled by the dist module


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    """Hash a sequence of ints into a single 32‑bit value."""
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = (x * 31 + hash(arr[i])) & 0xFFFFFFFF  # Keep it within 32-bit range
    return x


_SPARSE_CONV_CONFIG_DTYPE_TO_INT = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


# ----------------------
# Small shared utilities
# ----------------------


def _atomic_pickle_replace(target_path: Path, data: Any) -> None:
    """Atomically write pickle data to target by writing to a temp file then renaming."""
    temp_file = target_path.with_suffix(".tmp")
    with open(temp_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    temp_file.replace(target_path)


def _sanitize_for_pickle(value: Any) -> Any:
    """Recursively convert non-stable objects (Enums, dtypes, devices) to strings for safe pickling.

    Primary goal: avoid importing heavy modules (e.g., Enum classes defined in other modules)
    during unpickling that can break due to import-time side effects.
    """
    try:
        # Enums -> their name (fallback to str)
        if isinstance(value, enum.Enum):
            try:
                return str(value.name)
            except Exception:
                return str(value)

        # Common torch types to stringify
        try:
            import torch as _torch  # local import to avoid mandatory global dependency at module load

            if isinstance(value, _torch.dtype):
                return str(value)
            if isinstance(value, _torch.device):
                return str(value)
        except Exception:
            pass

        # Basic containers: preserve type where possible
        if isinstance(value, dict):
            return {_sanitize_for_pickle(k): _sanitize_for_pickle(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize_for_pickle(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_sanitize_for_pickle(v) for v in value)
        if isinstance(value, set):
            return {_sanitize_for_pickle(v) for v in value}
    except Exception:
        # Best-effort sanitization; fall back to string on any unexpected issue
        try:
            return str(value)
        except Exception:
            return None
    return value


def _parse_version_to_major_minor(version_value: Any) -> Tuple[int, int]:
    """Parse version like "v3.0" or 3.0 into (major, minor). Fallback to (1, 0)."""
    version_str = re.sub(r"[^0-9.]", "", str(version_value))
    parts = version_str.split(".")
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    if len(parts) == 1 and parts[0] != "":
        return int(parts[0]), 0
    return 1, 0


K = TypeVar("K")
V = TypeVar("V")


@dataclass
class SpatiallySparseConvConfig:
    log_num_in_coords: int
    log_num_out_coords: int
    in_channels: int
    out_channels: int
    kernel_volume: int
    in_dtype: torch.dtype
    # explicit_matmul_batch_size: Optional[int] = None # TODO: Add if supporting batched explicit

    def __init__(
        self,
        num_in_coords: int,
        num_out_coords: int,
        in_channels: int,
        out_channels: int,
        kernel_volume: int,
        in_dtype: torch.dtype,
        # explicit_matmul_batch_size: Optional[int] = None, # TODO
    ):
        self.log_num_in_coords = math.ceil(math.log2(num_in_coords)) if num_in_coords > 0 else 0
        self.log_num_out_coords = math.ceil(math.log2(num_out_coords)) if num_out_coords > 0 else 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_volume = kernel_volume
        assert in_dtype in _SPARSE_CONV_CONFIG_DTYPE_TO_INT, f"Unsupported in_dtype: {in_dtype}"
        self.in_dtype = in_dtype
        # self.explicit_matmul_batch_size = explicit_matmul_batch_size # TODO

    def __hash__(self):
        return _int_sequence_hash(
            [
                # self.log_num_in_coords,
                # self.log_num_out_coords,
                self.in_channels,
                self.out_channels,
                self.kernel_volume,
                _SPARSE_CONV_CONFIG_DTYPE_TO_INT[self.in_dtype],
            ]
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SpatiallySparseConvConfig):
            return False
        return (
            # self.log_num_in_coords == other.log_num_in_coords
            # and self.log_num_out_coords == other.log_num_out_coords and
            self.in_channels == other.in_channels
            and self.out_channels == other.out_channels
            and self.kernel_volume == other.kernel_volume
            and self.in_dtype == other.in_dtype
        )


# =============================
# Generic benchmark cache (K,V)
# =============================


class GenericBenchmarkCache(Generic[K, V]):
    """
    A generic on-disk benchmark cache that supports arbitrary namespaces and key/value types.

    - Thread-safe updates with a background saver on rank 0
    - Atomic writes using a temporary file rename
    - File format kept separate from sparse conv cache: benchmark_cache_generic.pkl

    Stored file schema (v3.0):
        {
            "namespaces": { str: { K: V, ... }, ... },
            "timestamp": float,
            "version": WARPCONVNET_BENCHMARK_CACHE_VERSION
        }
    """

    def __init__(self, cache_dir: str = WARPCONVNET_BENCHMARK_CACHE_DIR):
        # Use override cache directory if available (for debugging multi-GPU issues)
        if WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE:
            cache_dir = WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE
            logger.debug(f"Using override cache directory: {cache_dir}")

        self.cache_dir = Path(cache_dir).expanduser().resolve()  # Resolve symlinks
        self.cache_file = self.cache_dir / "benchmark_cache_generic.pkl"
        self.lock = threading.Lock()

        # Only log essential cache initialization info
        current_rank = _get_current_rank()
        if self.cache_file.exists():
            total_entries = 0
            try:
                with open(self.cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    if isinstance(cache_data, dict) and "namespaces" in cache_data:
                        namespaces = cache_data["namespaces"]
                        total_entries = sum(len(ns_dict) for ns_dict in namespaces.values())
                logger.info(
                    f"[Rank {current_rank}] Loaded benchmark cache: {total_entries} entries from {self.cache_file}"
                )
            except Exception:
                logger.info(
                    f"[Rank {current_rank}] Found cache file but failed to read: {self.cache_file}"
                )
        else:
            logger.info(
                f"[Rank {current_rank}] No existing cache found, will create: {self.cache_file}"
            )

        # In-memory accumulated results to be flushed by background saver
        # Preload existing cache to reduce risk of overwriting cross-process writes
        self._results: Dict[str, Dict[K, V]] = {}
        # Optional namespace -> validator function(value) that raises on invalid
        self._validators: Dict[str, Callable[[V], None]] = {}

        # Periodic save settings
        self.save_interval = 60.0
        self.last_save_time = 0.0
        self.pending_changes = False
        self._shutdown_requested = False

        # Background thread
        self._save_thread = None
        self._save_condition = threading.Condition(self.lock)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Preload from disk
        try:
            self._results = self.load_cache()
            total_entries = sum(len(ns_dict) for ns_dict in self._results.values())
            logger.debug(
                f"[Rank {current_rank}] Loaded cache: {len(self._results)} namespaces, {total_entries} total entries"
            )
        except Exception as e:
            logger.debug(f"[Rank {current_rank}] Failed to load cache: {e}")
            self._results = {}

        # Start background save thread only for rank 0
        if _is_rank_zero():
            self._start_background_saver()
            atexit.register(self._save_on_exit)
            logger.debug(f"[Rank {current_rank}] Started background saver (rank 0)")

    def register_value_validator(self, namespace: str, validator: Callable[[V], None]) -> None:
        """Register a validator for a namespace. Validator should raise ValueError/TypeError on invalid value."""
        if not callable(validator):
            raise TypeError("validator must be callable")
        self._validators[namespace] = validator

    def _validate_value(self, namespace: str, value: V) -> None:
        validator = self._validators.get(namespace)
        if validator is not None:
            validator(value)

    def _start_background_saver(self) -> None:
        if self._save_thread is not None:
            return
        self._save_thread = threading.Thread(
            target=self._background_save_worker, name="GenericBenchmarkCacheSaver", daemon=True
        )
        self._save_thread.start()
        logger.debug("Started background generic benchmark cache saver thread")

    def _background_save_worker(self) -> None:
        while not self._shutdown_requested:
            with self._save_condition:
                self._save_condition.wait(timeout=self.save_interval)
                if self._shutdown_requested:
                    break
                current_time = time.time()
                if (
                    self.pending_changes
                    and (current_time - self.last_save_time) >= self.save_interval
                ):
                    self._do_save()

    def _do_save(self) -> None:
        if not self.pending_changes:
            return
        try:
            current_time = time.time()
            # Merge with on-disk cache to avoid clobbering writes from other processes
            on_disk: Dict[str, Dict[K, V]] = {}
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, "rb") as f:
                        disk_data = pickle.load(f)
                        if isinstance(disk_data, dict):
                            on_disk = disk_data.get("namespaces", {}) or {}
                except Exception:
                    on_disk = {}

            merged: Dict[str, Dict[K, V]] = {}
            # Start with on-disk
            for ns, kv in on_disk.items():
                merged[ns] = dict(kv)
            # Overlay in-memory changes per-namespace
            for ns, kv in self._results.items():
                base = merged.get(ns, {})
                base.update(kv)
                merged[ns] = base

            # Keep in-memory in sync with what we are about to write
            self._results = merged

            # Sanitize values prior to pickling to avoid importing enum classes on load
            sanitized_namespaces: Dict[str, Dict[K, V]] = {}
            for ns, kv in merged.items():
                sanitized_ns: Dict[K, V] = {}
                for k, v in kv.items():
                    sanitized_ns[k] = _sanitize_for_pickle(v)  # type: ignore[assignment]
                sanitized_namespaces[ns] = sanitized_ns

            cache_data = {
                "namespaces": sanitized_namespaces,
                "timestamp": current_time,
                "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
            }
            _atomic_pickle_replace(self.cache_file, cache_data)
            self.last_save_time = current_time
            self.pending_changes = False
            total_entries = sum(len(ns_dict) for ns_dict in self._results.values())
            logger.info(
                f"Background saved generic benchmark cache: {len(self._results)} namespaces, {total_entries} total entries"
            )
        except Exception as e:
            logger.warning(f"Failed to save generic benchmark cache in background: {e}")

    def load_cache(self) -> Dict[str, Dict[K, V]]:
        if not self.cache_file.exists():
            logger.debug(f"No generic benchmark cache file found at {self.cache_file}")
            return {}
        try:
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            if not isinstance(cache_data, dict):
                logger.warning(
                    "Invalid generic cache file format, deleting cache and starting with empty cache"
                )
                return {}

            file_major, file_minor = _parse_version_to_major_minor(
                cache_data.get("version", "1.0")
            )
            expected_major, expected_minor = _parse_version_to_major_minor(
                WARPCONVNET_BENCHMARK_CACHE_VERSION
            )

            if int(file_major) == int(expected_major):
                namespaces = cache_data.get("namespaces", {})
                if not isinstance(namespaces, dict):
                    logger.warning(
                        "Generic cache 'namespaces' is not a dict; deleting cache and resetting to empty"
                    )
                    namespaces = {}
                return namespaces
            else:
                logger.warning(
                    f"Loaded generic benchmark cache v{file_major}.{file_minor}, but expected v{expected_major}.{expected_minor}. Deleting cache and resetting."
                )
                return {}
        except Exception as e:
            logger.warning(
                f"Failed to load generic benchmark cache: {e}. Deleting cache and starting with empty cache."
            )
            return {}

    def save_cache(self, cache_results: Dict[str, Dict[K, V]], force: bool = False) -> None:
        if not _is_rank_zero():
            return
        if not force:
            with self._save_condition:
                # Merge incoming results into in-memory results
                for ns, kv in cache_results.items():
                    # validate incoming values
                    for v in kv.values():
                        self._validate_value(ns, v)
                    current = self._results.get(ns, {})
                    current.update(kv)
                    self._results[ns] = current
                self.pending_changes = True
                self._save_condition.notify()
            return

        with self.lock:
            current_time = time.time()
            try:
                # Merge with on-disk cache
                on_disk: Dict[str, Dict[K, V]] = {}
                if self.cache_file.exists():
                    try:
                        with open(self.cache_file, "rb") as f:
                            disk_data = pickle.load(f)
                            if isinstance(disk_data, dict):
                                on_disk = disk_data.get("namespaces", {}) or {}
                    except Exception:
                        on_disk = {}

                merged: Dict[str, Dict[K, V]] = {}
                for ns, kv in on_disk.items():
                    merged[ns] = dict(kv)
                for ns, kv in cache_results.items():
                    for v in kv.values():
                        self._validate_value(ns, v)
                    base = merged.get(ns, {})
                    base.update(kv)
                    merged[ns] = base

                self._results = merged

                # Sanitize before writing
                sanitized_namespaces: Dict[str, Dict[K, V]] = {}
                for ns, kv in merged.items():
                    sanitized_ns: Dict[K, V] = {}
                    for k, v in kv.items():
                        sanitized_ns[k] = _sanitize_for_pickle(v)  # type: ignore[assignment]
                    sanitized_namespaces[ns] = sanitized_ns

                cache_data = {
                    "namespaces": sanitized_namespaces,
                    "timestamp": current_time,
                    "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
                }
                _atomic_pickle_replace(self.cache_file, cache_data)
                self.last_save_time = current_time
                self.pending_changes = False
                total_entries = sum(len(ns_dict) for ns_dict in self._results.values())
                logger.info(
                    f"Force saved generic benchmark cache v{WARPCONVNET_BENCHMARK_CACHE_VERSION}: {len(self._results)} namespaces, {total_entries} total entries"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to force save generic benchmark cache v{WARPCONVNET_BENCHMARK_CACHE_VERSION}: {e}"
                )

    def update_entry(self, namespace: str, key: K, value: V, force: bool = False) -> None:
        if not _is_rank_zero():
            return
        with self.lock:
            ns = self._results.setdefault(namespace, {})
            self._validate_value(namespace, value)
            # Sanitize on write to in-memory structure so later merges inherit sanitized values
            ns[key] = _sanitize_for_pickle(value)  # type: ignore[assignment]
        if force:
            self.save_cache(self._results, force=True)
        else:
            self.mark_dirty()

    def get_namespace(self, namespace: str) -> Dict[K, V]:
        # Fast path: if we have in-memory results for this namespace, return them immediately
        # to enable same-process cache hits right after warmup/benchmarking.
        try:
            with self.lock:
                ns_in_memory = self._results.get(namespace)
                if isinstance(ns_in_memory, dict) and len(ns_in_memory) > 0:
                    return dict(ns_in_memory)
        except Exception:
            pass

        # Fallback to disk for cross-process visibility when in-memory is empty
        logger.debug(f"Loading from disk for namespace '{namespace}'")
        namespaces = self.load_cache()
        result = namespaces.get(namespace, {})

        if result:
            logger.debug(f"Cache hit on disk for namespace '{namespace}': {len(result)} entries")
            # Update in-memory cache with disk results for future fast access
            with self.lock:
                if namespace not in self._results:
                    self._results[namespace] = {}
                self._results[namespace].update(result)
        else:
            logger.debug(f"Cache miss for namespace '{namespace}'")

        return result

    def mark_dirty(self) -> None:
        if not _is_rank_zero():
            return
        with self._save_condition:
            self.pending_changes = True
            self._save_condition.notify()

    def _save_on_exit(self) -> None:
        self._shutdown_requested = True
        with self._save_condition:
            self._save_condition.notify()
        if self._save_thread is not None:
            self._save_thread.join(timeout=5.0)
        if self.pending_changes:
            self.save_cache(self._results, force=True)


# Global generic cache instance
_generic_benchmark_cache: Optional[GenericBenchmarkCache[Any, Any]] = None


def get_generic_benchmark_cache() -> GenericBenchmarkCache[Any, Any]:
    global _generic_benchmark_cache
    if _generic_benchmark_cache is None:
        _generic_benchmark_cache = GenericBenchmarkCache()
    return _generic_benchmark_cache


def load_generic_benchmark_cache() -> Dict[str, Dict[Any, Any]]:
    cache = get_generic_benchmark_cache()
    return cache.load_cache()


def save_generic_benchmark_cache(
    cache_results: Dict[str, Dict[Any, Any]], force: bool = False
) -> None:
    cache = get_generic_benchmark_cache()
    cache.save_cache(cache_results, force=force)


def mark_generic_benchmark_cache_dirty() -> None:
    cache = get_generic_benchmark_cache()
    cache.mark_dirty()


def generic_benchmark_update_entry(
    namespace: str, key: Any, value: Any, force: bool = False
) -> None:
    cache = get_generic_benchmark_cache()
    cache.update_entry(namespace, key, value, force=force)


def generic_benchmark_get_namespace(namespace: str) -> Dict[Any, Any]:
    cache = get_generic_benchmark_cache()
    return cache.get_namespace(namespace)


def build_dict_schema_validator(
    required_keys_and_types: Dict[str, type],
) -> Callable[[Dict[str, Any]], None]:
    """
    Build a validator that enforces value to be a dict with specific key->type pairs.

    Example:
        validator = build_dict_schema_validator({"mma_tile": int, "split_k_slices": int})
        cache.register_value_validator("implicit_gemm", validator)
    """

    def _validator(value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise TypeError("Value must be a dict")
        for k, t in required_keys_and_types.items():
            if k not in value:
                raise ValueError(f"Missing required key: {k}")
            if not isinstance(value[k], t):
                raise TypeError(f"Key '{k}' must be of type {t.__name__}")

    return _validator


# =====================
# Autotuned op wrappers
# =====================


def _is_compiling_or_capturing() -> bool:
    """Best-effort check for torch.compile or CUDA graph capture contexts.

    We avoid importing torch._dynamo at module import time to prevent side effects.
    """
    try:
        import torch

        # Compile detection (Dynamo)
        is_compiling = False
        try:
            from torch._dynamo import is_compiling as _dyn_is_compiling  # type: ignore

            is_compiling = bool(_dyn_is_compiling())
        except Exception:
            is_compiling = False

        # CUDA graph capture detection
        is_capturing = False
        try:
            is_capturing = bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            is_capturing = False

        return is_compiling or is_capturing
    except Exception:
        return False


def make_autotuned_op(
    *,
    namespace: str,
    c_fn: Callable[..., Any],
    param_space: Iterable[Dict[str, Any]],
    key_fn: Callable[..., Any],
    run_and_time_fn: Optional[
        Callable[[Dict[str, Any], Tuple[Any, ...], Dict[str, Any]], float]
    ] = None,
    select_best: str = "min",  # or "max" if larger is better
    require_cached_when_compiling: bool = True,
    timing_reduction: str = "min",  # "min" or "avg" across iters when using default timing
    record_failures_as_inf: bool = True,
    timing_iters: int = 3,
) -> Callable[..., Any]:
    """
    Factory that wraps a low-level function (e.g., CUTLASS-bound op) with autotuning via GenericBenchmarkCache.

    - On cache hit: launches immediately with cached params.
    - On cache miss:
        - If compiling/capturing and require_cached_when_compiling=True: raises with a warmup hint.
        - Else: benchmarks all candidates in param_space using run_and_time_fn, caches best, and runs it.

    Args:
        namespace: Cache namespace, e.g. "implicit_gemm_ad_gather_scatter".
        c_fn: Callable that accepts original args/kwargs plus candidate params via **candidate.
        param_space: Iterable of dicts of candidate params.
        key_fn: Function mapping (args, kwargs) -> hashable key.
        run_and_time_fn: Optional custom timing function returning latency in ms. If None, a default torch.cuda timing runner is used.
        select_best: "min" to pick lowest latency, "max" to pick highest metric.
        require_cached_when_compiling: If True, disallow benchmarking during torch.compile or CUDA graph capture.
        timing_reduction: "min" or "avg" across iters when using default timing
        record_failures_as_inf: If True, record failures as inf and continue benchmarking.
        timing_iters: Number of timing iterations to run for each candidate.

    Returns:
        Callable with same signature as c_fn (no explicit config argument).
    """

    cache = get_generic_benchmark_cache()

    # Optional: register a light schema validator that accepts either:
    # - dict of kernel params (legacy single-best format), or
    # - list of {"params": dict, "metric": float} sorted best-first (new multi-result format)
    def _validator(v: Any) -> None:
        if isinstance(v, dict):
            return
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, dict):
                    raise TypeError("Each list item must be a dict with 'params' and 'metric'")
                if "params" not in item or "metric" not in item:
                    raise ValueError("Each list item must contain 'params' and 'metric' keys")
                if not isinstance(item["params"], dict):
                    raise TypeError("'params' must be a dict")
                # metric must be a real number
                if not isinstance(item["metric"], (int, float)):
                    raise TypeError("'metric' must be a number")
            return
        raise TypeError("Cached value must be a dict (legacy) or a list of benchmark results")

    cache.register_value_validator(namespace, _validator)

    # Materialize candidate space once to derive the set of allowed tunable keys
    try:
        _candidates_list: List[Dict[str, Any]] = list(param_space)
    except TypeError:
        # Some iterables can be consumed only once; fallback to a shallow copy approach
        _candidates_list = [c for c in param_space]
    _allowed_param_keys: set[str] = set()
    for _cand in _candidates_list:
        if isinstance(_cand, dict):
            _allowed_param_keys.update(_cand.keys())

    def _default_run_and_time(
        candidate: Dict[str, Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> float:
        runner = make_status_timing_runner(
            c_fn,
            output_positions=(),
            iters=timing_iters,
            reduction=timing_reduction,
        )
        if record_failures_as_inf:
            result = runner(candidate, args, kwargs)
            return float("inf") if isinstance(result, int) else float(result)
        # on_error='raise' behavior: attempt to surface underlying error
        try:
            result = runner(candidate, args, kwargs)
            if isinstance(result, int):
                logger.warning(
                    f"Candidate {candidate} returned status {result} during timing. Triggering error path."
                )
                _ = c_fn(*args, **kwargs, **candidate)
                return float("inf")
            return float(result)
        except Exception as e:
            logger.warning(f"Candidate {candidate} failed during timing: {e}")
            _ = c_fn(*args, **kwargs, **candidate)
            return float("inf")  # unreachable; placate type checkers

    timing_fn = run_and_time_fn or _default_run_and_time

    def _op(*args: Any, **kwargs: Any) -> Any:
        key = key_fn(*args, **kwargs)

        # Read the latest namespace map from disk for cross-process coherence
        ns_map = cache.get_namespace(namespace)
        cached = ns_map.get(key)
        if cached is not None:
            # Support both legacy dict and new list-of-results formats
            if isinstance(cached, list) and len(cached) > 0 and isinstance(cached[0], dict):
                best_params_raw = cached[0].get("params", {})
                if not isinstance(best_params_raw, dict):
                    raise TypeError(
                        "Invalid cache entry: first list element missing 'params' dict"
                    )
                # Filter to allowed tunable keys only
                best_params = {
                    k: v for k, v in best_params_raw.items() if k in _allowed_param_keys
                }
                return c_fn(*args, **kwargs, **best_params)
            elif isinstance(cached, dict):
                # Legacy single-dict format; filter to allowed tunable keys only
                filtered = {k: v for k, v in cached.items() if k in _allowed_param_keys}
                return c_fn(*args, **kwargs, **filtered)
            else:
                # Unexpected format; fall through to benchmarking to refresh entry
                logger.warning(
                    f"Cache entry has unexpected format, re-benchmarking: namespace='{namespace}'"
                )

        # Cache miss
        if require_cached_when_compiling and _is_compiling_or_capturing():
            raise RuntimeError(
                f"Autotune cache miss for '{namespace}' with key {key} during compile/capture. "
                "Call warmup_autotune(...) with representative inputs before torch.compile or CUDA graph capture."
            )

        # Benchmark all candidates
        best_candidate: Optional[Dict[str, Any]] = None
        best_metric: Optional[float] = None
        all_results: List[Dict[str, Any]] = []  # each entry: {"params": dict, "metric": float}

        # Use pre-materialized candidate list
        candidates: List[Dict[str, Any]] = _candidates_list
        if len(candidates) == 0:
            raise ValueError("param_space is empty for autotuned op")

        # Only log benchmarking for important operations
        logger.debug(f"Benchmarking {namespace} ({len(candidates)} candidates)")
        for cand in candidates:
            metric = timing_fn(cand, args, kwargs)
            if isinstance(metric, int):
                logger.warning(
                    f"Candidate {cand} returned status {metric} during timing. Skipping."
                )
                return metric

            all_results.append({"params": cand, "metric": float(metric)})
            if best_metric is None:
                best_metric = metric
                best_candidate = cand
            else:
                if (select_best == "min" and metric < best_metric) or (
                    select_best == "max" and metric > best_metric
                ):
                    best_metric = metric
                    best_candidate = cand

        if best_candidate is None:
            raise RuntimeError("No valid candidate found during autotune")

        # Sort all results best-first and persist full list to cache
        try:
            reverse = True if select_best == "max" else False
            all_results.sort(key=lambda r: r["metric"], reverse=reverse)
            cache.update_entry(namespace, key, all_results, force=False)
            logger.debug(
                f"Cached best result for {namespace}: {best_candidate} ({best_metric:.2f}ms)"
            )
        except Exception as e:
            logger.warning(f"Failed to update benchmark cache for {namespace}: {e}")

        return c_fn(*args, **kwargs, **best_candidate)

    return _op


def warmup_autotune(
    op: Callable[..., Any],
    sample_args_and_kwargs: Iterable[Tuple[Tuple[Any, ...], Dict[str, Any]]],
) -> None:
    """Run the provided op on a list of (args, kwargs) to populate the cache.

    This should be called before torch.compile or CUDA graph capture when using autotuned ops
    with require_cached_when_compiling=True.
    """

    for args, kwargs in sample_args_and_kwargs:
        try:
            _ = op(*args, **kwargs)
        except Exception as e:
            # Surface the first error, but continue attempting others
            logger.debug(f"Warmup invocation failed for autotuned op: {e}")


def make_status_timing_runner(
    c_fn: Callable[..., Any],
    *,
    output_positions: Tuple[int, ...],
    iters: int = 3,
    reduction: str = "min",
) -> Callable[[Dict[str, Any], Tuple[Any, ...], Dict[str, Any]], float | int]:
    """Timing runner for kernels that return an int status and write into output tensors.

    - Clones only the specified output args for each call
    - If a non-zero status is returned at any point, returns the status immediately
    - Otherwise returns latency in milliseconds (min or average)
    """

    def _clone_outputs_for_args(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        args_list = list(args)
        for pos in output_positions:
            if 0 <= pos < len(args_list) and isinstance(args_list[pos], torch.Tensor):
                args_list[pos] = args_list[pos].clone()
        return tuple(args_list)

    def _runner(
        candidate: Dict[str, Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> float | int:
        try:
            # Warmup using cloned outputs
            warm_args = _clone_outputs_for_args(args)
            status = c_fn(*warm_args, **kwargs, **candidate)
            if isinstance(status, int) and status != 0:
                return status

            # Choose timing backend
            use_cuda_timing = (
                any(
                    isinstance(arg, torch.Tensor) and getattr(arg, "is_cuda", False)
                    for arg in args
                )
                and torch.cuda.is_available()
            )

            times_ms: List[float] = []
            if use_cuda_timing:
                try:
                    torch.cuda.current_stream().synchronize()
                except Exception:
                    pass
                for _ in range(iters):
                    run_args = _clone_outputs_for_args(args)
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()
                    status = c_fn(*run_args, **kwargs, **candidate)
                    ender.record()
                    ender.synchronize()
                    if isinstance(status, int) and status != 0:
                        return status
                    times_ms.append(float(starter.elapsed_time(ender)))
            else:
                for _ in range(iters):
                    run_args = _clone_outputs_for_args(args)
                    t0 = time.perf_counter()
                    status = c_fn(*run_args, **kwargs, **candidate)
                    t1 = time.perf_counter()
                    if isinstance(status, int) and status != 0:
                        return status
                    times_ms.append(float((t1 - t0) * 1000.0))

            if not times_ms:
                return float("inf")
            if reduction == "avg":
                return float(sum(times_ms) / float(len(times_ms)))
            return float(min(times_ms))
        except Exception:
            return float("inf")

    return _runner
