# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import pytest

from warpconvnet.utils.benchmark_cache import GenericBenchmarkCache, build_dict_schema_validator


@pytest.fixture
def tmp_cache_dir(tmp_path: Path):
    d = tmp_path / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clear_rank_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)


def test_save_and_load_roundtrip(tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_rank_env(monkeypatch)  # rank 0 by default

    cache = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    key = (10, 64, 128)
    value = {"mma_tile": 3, "split_k_slices": 8}

    # Force save to avoid waiting for background thread
    cache.update_entry("implicit_gemm", key, value, force=True)

    # Load via a fresh instance
    cache2 = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    ns = cache2.get_namespace("implicit_gemm")
    assert key in ns and ns[key] == value

    # Cleanup background thread
    cache._save_on_exit()
    cache2._save_on_exit()


def test_namespace_separation(tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_rank_env(monkeypatch)  # rank 0

    cache = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache.update_entry("ns1", (1, 2, 3), {"p": 1}, force=True)
    cache.update_entry("ns2", (1, 2, 3), {"p": 2}, force=True)

    cache3 = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    ns1 = cache3.get_namespace("ns1")
    ns2 = cache3.get_namespace("ns2")
    assert ns1 != ns2
    assert ns1[(1, 2, 3)] == {"p": 1}
    assert ns2[(1, 2, 3)] == {"p": 2}

    cache._save_on_exit()
    cache3._save_on_exit()


def test_merge_on_save_does_not_clobber(tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_rank_env(monkeypatch)  # rank 0

    # First writer
    cache_a = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache_a.update_entry("ns", (1, 1, 1), {"p": "a"}, force=True)

    # Simulate concurrent other process write (second writer)
    cache_b = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache_b.update_entry("ns", (2, 2, 2), {"p": "b"}, force=True)

    # Third writer saves only a new key; must not remove previous ones
    cache_c = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache_c.save_cache({"ns": {(3, 3, 3): {"p": "c"}}}, force=True)

    # Verify all entries present
    cache_r = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    ns = cache_r.get_namespace("ns")
    assert ns[(1, 1, 1)]["p"] == "a"
    assert ns[(2, 2, 2)]["p"] == "b"
    assert ns[(3, 3, 3)]["p"] == "c"

    # Cleanup
    cache_a._save_on_exit()
    cache_b._save_on_exit()
    cache_c._save_on_exit()
    cache_r._save_on_exit()


def test_value_validation_rejects_invalid(tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_rank_env(monkeypatch)  # rank 0

    cache = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache.register_value_validator(
        "implicit_gemm", build_dict_schema_validator({"mma_tile": int, "split_k_slices": int})
    )

    # Valid should pass
    cache.update_entry(
        "implicit_gemm", (1, 2, 3), {"mma_tile": 1, "split_k_slices": 8}, force=True
    )

    # Missing key should fail
    with pytest.raises(ValueError):
        cache.update_entry("implicit_gemm", (2, 2, 2), {"mma_tile": 1}, force=True)

    # Wrong type should fail
    with pytest.raises(TypeError):
        cache.update_entry(
            "implicit_gemm", (3, 3, 3), {"mma_tile": 1, "split_k_slices": "8"}, force=True
        )

    cache._save_on_exit()


def test_nonzero_rank_results_persist_and_merge(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Every rank's auto-tune results must reach the shared cache file.

    The multi-rank design parallelizes the tuning search: each rank records
    its own winners and the (file-locked) read-merge-write unions them, so
    ranks inherit each other's results on refresh. A rank gate on the write
    path silently discards every non-zero rank's work — this test pins the
    all-ranks-contribute contract by writing from a simulated rank 3.
    """
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")

    # "rank 3" records a winner and force-saves.
    cache_r3 = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache_r3.update_entry("ns", (7, 7, 7), {"p": "rank3"}, force=True)

    # "rank 0" (separate process in real training) records a different key.
    monkeypatch.setenv("RANK", "0")
    cache_r0 = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache_r0.update_entry("ns", (8, 8, 8), {"p": "rank0"}, force=True)

    # A fresh reader sees BOTH ranks' results merged.
    cache_r = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    ns = cache_r.get_namespace("ns")
    assert ns[(7, 7, 7)]["p"] == "rank3", "non-zero rank's result was dropped"
    assert ns[(8, 8, 8)]["p"] == "rank0"

    cache_r3._save_on_exit()
    cache_r0._save_on_exit()
    cache_r._save_on_exit()


def test_forced_save_fires_on_merge_callbacks(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Forced saves must refresh consumers exactly like the background saver.

    Consumers (autotune) mirror the cache into module-level dicts via
    on_merge callbacks; both save paths merge disk state into memory, so
    both must notify — otherwise entries absorbed from other ranks during a
    forced save would be invisible to consumers until the next periodic save.
    """
    _clear_rank_env(monkeypatch)

    # Another process left an entry on disk.
    cache_other = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache_other.update_entry("ns", (1, 1, 1), {"p": "other"}, force=True)

    cache = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    seen: dict = {}
    cache.register_on_merge_callback(lambda ns, d: seen.setdefault(ns, {}).update(d))

    # Forced save path must fire the callback with the merged (disk ∪ local) view.
    cache.update_entry("ns", (2, 2, 2), {"p": "local"}, force=True)

    assert (1, 1, 1) in seen.get("ns", {}), "disk entry missing from forced-save callback"
    assert (2, 2, 2) in seen.get("ns", {}), "local entry missing from forced-save callback"

    cache_other._save_on_exit()
    cache._save_on_exit()
