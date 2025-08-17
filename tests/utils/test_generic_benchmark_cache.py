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


def test_rank_gating_does_not_write_on_non_zero_rank(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    # Simulate non-zero rank
    monkeypatch.setenv("RANK", "1")

    cache = GenericBenchmarkCache(cache_dir=str(tmp_cache_dir))
    cache.update_entry("ns", (9, 9, 9), {"p": 9}, force=True)
    cache.save_cache({"ns": {(8, 8, 8): {"p": 8}}}, force=True)

    # No file should be created when rank != 0
    cache_file = tmp_cache_dir / "benchmark_cache_generic.pkl"
    assert not cache_file.exists()

    # Cleanup
    cache._save_on_exit()


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
