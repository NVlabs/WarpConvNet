# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import math
import os
import pickle
import threading
import time
import atexit
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Sequence, TypeVar, Generic, Callable, Iterable, List
from dataclasses import dataclass

import torch

from warpconvnet.constants import (
    WARPCONVNET_BENCHMARK_CACHE_DIR,
    WARPCONVNET_BENCHMARK_CACHE_VERSION,
)
from warpconvnet.utils.logger import get_logger

logger = get_logger(__name__)


def _get_current_rank() -> int:
    """Get current process rank for distributed training."""
    # Check common distributed training environment variables
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    else:
        # Not in distributed mode or rank 0
        return 0


def _is_rank_zero() -> bool:
    """Check if current process is rank 0."""
    return _get_current_rank() == 0


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


class BenchmarkCache:
    """
    Manages saving and loading of benchmark results to/from disk.
    Only rank 0 process saves to avoid conflicts in distributed training.
    Uses a background thread for periodic saving to avoid blocking the main computation.

    Version 2.0 supports multiple benchmark result types:
    - sparse_conv_forward_results
    - sparse_conv_backward_results
    - sparse_conv_depthwise_forward_results
    - sparse_conv_depthwise_backward_results
    """

    def __init__(self, cache_dir: str = WARPCONVNET_BENCHMARK_CACHE_DIR):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_file = self.cache_dir / "benchmark_cache.pkl"
        self.lock = threading.Lock()

        # Periodic save settings
        self.save_interval = 60.0  # seconds - reduced from 300 to 60 for more frequent saves
        self.last_save_time = 0.0
        self.pending_changes = False
        self._shutdown_requested = False

        # Background thread for saving
        self._save_thread = None
        self._save_condition = threading.Condition(self.lock)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Start background save thread only for rank 0
        if _is_rank_zero():
            self._start_background_saver()
            atexit.register(self._save_on_exit)

    def _start_background_saver(self) -> None:
        """Start the background thread for periodic saving."""
        if self._save_thread is not None:
            return

        self._save_thread = threading.Thread(
            target=self._background_save_worker, name="BenchmarkCacheSaver", daemon=True
        )
        self._save_thread.start()
        logger.debug("Started background benchmark cache saver thread")

    def _background_save_worker(self) -> None:
        """Background worker thread that periodically saves the cache."""
        while not self._shutdown_requested:
            with self._save_condition:
                # Wait for either pending changes or timeout
                self._save_condition.wait(timeout=self.save_interval)

                if self._shutdown_requested:
                    break

                # Check if we need to save
                current_time = time.time()
                if (
                    self.pending_changes
                    and (current_time - self.last_save_time) >= self.save_interval
                ):
                    self._do_save()

    def _do_save(self) -> None:
        """Internal method to perform the actual save (assumes lock is held)."""
        if not self.pending_changes:
            return

        try:
            # Import here to avoid circular imports
            from warpconvnet.nn.functional.sparse_conv import (
                _BENCHMARK_FORWARD_RESULTS,
                _BENCHMARK_BACKWARD_RESULTS,
            )

            # Try to import depthwise results, but don't fail if not available
            try:
                from warpconvnet.nn.functional.sparse_conv_depth import (
                    _BENCHMARK_DEPTHWISE_FORWARD_RESULTS,
                    _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS,
                )
            except ImportError:
                _BENCHMARK_DEPTHWISE_FORWARD_RESULTS = {}
                _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS = {}

            current_time = time.time()

            # Prepare cache data in version 2.0 format
            cache_data = {
                "sparse_conv_forward_results": _BENCHMARK_FORWARD_RESULTS,
                "sparse_conv_backward_results": _BENCHMARK_BACKWARD_RESULTS,
                "sparse_conv_depthwise_forward_results": _BENCHMARK_DEPTHWISE_FORWARD_RESULTS,
                "sparse_conv_depthwise_backward_results": _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS,
                "timestamp": current_time,
                "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
            }

            # Atomic write: write to temp file then rename
            temp_file = self.cache_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic move
            temp_file.replace(self.cache_file)

            self.last_save_time = current_time
            self.pending_changes = False

            logger.debug(
                f"Background saved benchmark cache: {len(_BENCHMARK_FORWARD_RESULTS)} forward, "
                f"{len(_BENCHMARK_BACKWARD_RESULTS)} backward, "
                f"{len(_BENCHMARK_DEPTHWISE_FORWARD_RESULTS)} depthwise forward, "
                f"{len(_BENCHMARK_DEPTHWISE_BACKWARD_RESULTS)} depthwise backward configurations"
            )

        except Exception as e:
            logger.warning(f"Failed to save benchmark cache in background: {e}")

    def load_cache(self) -> Dict[str, Dict]:
        """
        Load benchmark results from cache file.

        If the loaded version is less than the latest version, the cache will be reset.

        Returns a dictionary with all cached benchmark results.
        """
        # Default empty results for all supported cache types
        default_results = {
            "sparse_conv_forward_results": {},
            "sparse_conv_backward_results": {},
            "sparse_conv_depthwise_forward_results": {},
            "sparse_conv_depthwise_backward_results": {},
        }

        if not self.cache_file.exists():
            logger.debug(f"No benchmark cache file found at {self.cache_file}")
            return default_results

        try:
            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            if not isinstance(cache_data, dict):
                logger.warning("Invalid cache file format, starting with empty cache")
                return default_results

            # Determine cache version and handle accordingly
            version = cache_data.get("version", "1.0")
            # Remove any string before and after major.minor
            version = re.sub(r"[^0-9.]", "", str(version))
            major_version, minor_version = version.split(".")

            if int(major_version) == 3:
                # Version 3.0 format - direct mapping
                result = {
                    "sparse_conv_forward_results": cache_data.get(
                        "sparse_conv_forward_results", {}
                    ),
                    "sparse_conv_backward_results": cache_data.get(
                        "sparse_conv_backward_results", {}
                    ),
                    "sparse_conv_depthwise_forward_results": cache_data.get(
                        "sparse_conv_depthwise_forward_results", {}
                    ),
                    "sparse_conv_depthwise_backward_results": cache_data.get(
                        "sparse_conv_depthwise_backward_results", {}
                    ),
                }

                total_configs = sum(len(results) for results in result.values())
                logger.info(f"Loaded benchmark cache v3.0: {total_configs} total configurations")

            else:
                logger.warning(
                    f"Loaded benchmark cache v{version}, but expected v3.0. Resetting cache."
                )
                return default_results

            return result

        except Exception as e:
            logger.warning(f"Failed to load benchmark cache: {e}. Starting with empty cache.")
            return default_results

    def save_cache(self, cache_results: Dict[str, Dict], force: bool = False) -> None:
        """
        Save benchmark results to cache file in version 2.0 format.

        Args:
            cache_results: Dictionary mapping cache types to their results. Current supported types are:
                - sparse_conv_forward_results
                - sparse_conv_backward_results
                - sparse_conv_depthwise_forward_results
                - sparse_conv_depthwise_backward_results
            force: If True, save immediately. If False, schedule for background save.
        """
        if not _is_rank_zero():
            return

        if not force:
            # For non-forced saves, just mark dirty and let background thread handle it
            self.mark_dirty()
            return

        with self.lock:
            current_time = time.time()

            try:
                # Prepare cache data in the latest version
                cache_data = {
                    "sparse_conv_forward_results": cache_results.get(
                        "sparse_conv_forward_results", {}
                    ),
                    "sparse_conv_backward_results": cache_results.get(
                        "sparse_conv_backward_results", {}
                    ),
                    "sparse_conv_depthwise_forward_results": cache_results.get(
                        "sparse_conv_depthwise_forward_results", {}
                    ),
                    "sparse_conv_depthwise_backward_results": cache_results.get(
                        "sparse_conv_depthwise_backward_results", {}
                    ),
                    "timestamp": current_time,
                    "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
                }

                # Atomic write: write to temp file then rename
                temp_file = self.cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Atomic move
                temp_file.replace(self.cache_file)

                self.last_save_time = current_time
                self.pending_changes = False

                total_configs = sum(len(results) for results in cache_results.values())
                logger.debug(
                    f"Force saved benchmark cache v{WARPCONVNET_BENCHMARK_CACHE_VERSION}: {total_configs} total configurations"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to force save benchmark cache v{WARPCONVNET_BENCHMARK_CACHE_VERSION}: {e}"
                )

    def mark_dirty(self) -> None:
        """Mark that cache has pending changes that should be saved."""
        if not _is_rank_zero():
            return

        with self._save_condition:
            self.pending_changes = True
            # Notify the background thread that there are changes
            self._save_condition.notify()

    def _save_on_exit(self) -> None:
        """Save cache on program exit if there are pending changes."""
        # Signal shutdown to background thread
        self._shutdown_requested = True

        # Wake up the background thread
        with self._save_condition:
            self._save_condition.notify()

        # Wait for background thread to finish (with timeout)
        if self._save_thread is not None:
            self._save_thread.join(timeout=5.0)

        # Perform final save if there are still pending changes
        if self.pending_changes:
            from warpconvnet.nn.functional.sparse_conv import (
                _BENCHMARK_FORWARD_RESULTS,
                _BENCHMARK_BACKWARD_RESULTS,
            )

            from warpconvnet.nn.functional.sparse_conv_depth import (
                _BENCHMARK_DEPTHWISE_FORWARD_RESULTS,
                _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS,
            )

            self.save_cache(
                {
                    "sparse_conv_forward_results": _BENCHMARK_FORWARD_RESULTS,
                    "sparse_conv_backward_results": _BENCHMARK_BACKWARD_RESULTS,
                    "sparse_conv_depthwise_forward_results": _BENCHMARK_DEPTHWISE_FORWARD_RESULTS,
                    "sparse_conv_depthwise_backward_results": _BENCHMARK_DEPTHWISE_BACKWARD_RESULTS,
                },
                force=True,
            )


# Global cache instance
_benchmark_cache: Optional[BenchmarkCache] = None


def get_benchmark_cache() -> BenchmarkCache:
    """Get the global benchmark cache instance."""
    global _benchmark_cache
    if _benchmark_cache is None:
        _benchmark_cache = BenchmarkCache()
    return _benchmark_cache


def load_sparse_conv_benchmark_cache() -> Tuple[Dict, Dict]:
    """
    Load benchmark cache and return (forward_results, backward_results) for backward compatibility.
    For full v2.0 results, use get_benchmark_cache().load_cache() directly.
    """
    cache = get_benchmark_cache()
    cache_results = cache.load_cache()

    # Return legacy format for backward compatibility
    forward_results = cache_results.get("sparse_conv_forward_results", {})
    backward_results = cache_results.get("sparse_conv_backward_results", {})

    return forward_results, backward_results


def load_dict_benchmark_cache() -> Dict[str, Dict]:
    """
    Load benchmark cache in version 2.0 format.
    Returns dictionary with all cached benchmark result types.
    """
    cache = get_benchmark_cache()
    return cache.load_cache()


def save_sparse_conv_benchmark_cache(
    forward_results: Dict, backward_results: Dict, force: bool = False
) -> None:
    """
    Save benchmark cache.

    Args:
        forward_results: Forward benchmark results
        backward_results: Backward benchmark results
        force: If True, save immediately. If False, schedule for background save.
    """
    cache = get_benchmark_cache()
    cache.save_cache(
        {
            "sparse_conv_forward_results": forward_results,
            "sparse_conv_backward_results": backward_results,
        },
        force=force,
    )


def save_dict_benchmark_cache(cache_results: Dict[str, Dict], force: bool = False) -> None:
    """
    Save benchmark cache in version 2.0 format.

    Args:
        cache_results: Dictionary mapping cache types to their results
        force: If True, save immediately. If False, schedule for background save.
    """
    cache = get_benchmark_cache()
    cache.save_cache(cache_results, force=force)


def mark_benchmark_cache_dirty() -> None:
    """Mark that benchmark cache has pending changes."""
    cache = get_benchmark_cache()
    cache.mark_dirty()


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
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_file = self.cache_dir / "benchmark_cache_generic.pkl"
        self.lock = threading.Lock()

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
        except Exception:
            self._results = {}

        # Start background save thread only for rank 0
        if _is_rank_zero():
            self._start_background_saver()
            atexit.register(self._save_on_exit)

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

            cache_data = {
                "namespaces": merged,
                "timestamp": current_time,
                "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
            }
            temp_file = self.cache_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.replace(self.cache_file)
            self.last_save_time = current_time
            self.pending_changes = False
            total_entries = sum(len(ns_dict) for ns_dict in self._results.values())
            logger.debug(
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
                logger.warning("Invalid generic cache file format, starting with empty cache")
                return {}

            version = cache_data.get("version", "1.0")
            version = re.sub(r"[^0-9.]", "", str(version))
            major_version, minor_version = version.split(".")

            if int(major_version) == 3:
                namespaces = cache_data.get("namespaces", {})
                if not isinstance(namespaces, dict):
                    logger.warning("Generic cache 'namespaces' is not a dict; resetting to empty")
                    namespaces = {}
                return namespaces
            else:
                logger.warning(
                    f"Loaded generic benchmark cache v{version}, but expected v3.0. Resetting cache."
                )
                return {}
        except Exception as e:
            logger.warning(
                f"Failed to load generic benchmark cache: {e}. Starting with empty cache."
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

                cache_data = {
                    "namespaces": merged,
                    "timestamp": current_time,
                    "version": WARPCONVNET_BENCHMARK_CACHE_VERSION,
                }
                temp_file = self.cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                temp_file.replace(self.cache_file)
                self.last_save_time = current_time
                self.pending_changes = False
                total_entries = sum(len(ns_dict) for ns_dict in self._results.values())
                logger.debug(
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
            ns[key] = value
        if force:
            self.save_cache(self._results, force=True)
        else:
            self.mark_dirty()

    def get_namespace(self, namespace: str) -> Dict[K, V]:
        # Prefer reading from disk for cross-process visibility
        namespaces = self.load_cache()
        return namespaces.get(namespace, {})

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

    def _default_run_and_time(
        candidate: Dict[str, Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> float:
        import torch

        # Use CUDA events for wall-time in milliseconds
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # Warmup single run to populate caches for this candidate
        c_fn(*args, **kwargs, **candidate)
        torch.cuda.synchronize()

        # Time a few runs to reduce noise
        iters = 3
        starter.record()
        for _ in range(iters):
            c_fn(*args, **kwargs, **candidate)
        ender.record()
        torch.cuda.synchronize()
        elapsed_ms: float = float(starter.elapsed_time(ender)) / iters
        return elapsed_ms

    timing_fn = run_and_time_fn or _default_run_and_time

    def _op(*args: Any, **kwargs: Any) -> Any:
        key = key_fn(*args, **kwargs)

        # Read the latest namespace map from disk for cross-process coherence
        ns_map = cache.get_namespace(namespace)
        cached = ns_map.get(key)
        if cached is not None:
            # Support both legacy dict and new list-of-results formats
            if isinstance(cached, list) and len(cached) > 0 and isinstance(cached[0], dict):
                best_params = cached[0].get("params", {})
                if not isinstance(best_params, dict):
                    raise TypeError(
                        "Invalid cache entry: first list element missing 'params' dict"
                    )
                return c_fn(*args, **kwargs, **best_params)
            elif isinstance(cached, dict):
                return c_fn(*args, **kwargs, **cached)
            else:
                # Unexpected format; fall through to benchmarking to refresh entry
                logger.debug(
                    f"Cache entry for namespace '{namespace}' key '{key}' has unexpected format; re-benchmarking."
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

        # Materialize param_space to a list to allow multi-pass
        candidates: List[Dict[str, Any]] = list(param_space)
        if len(candidates) == 0:
            raise ValueError("param_space is empty for autotuned op")

        for cand in candidates:
            try:
                metric = timing_fn(cand, args, kwargs)
            except Exception as e:
                # Skip invalid candidates, but continue searching
                logger.debug(f"Candidate {cand} failed during timing: {e}")
                continue

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
        except Exception as e:
            logger.debug(f"Failed to update generic benchmark cache for {namespace}: {e}")

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
