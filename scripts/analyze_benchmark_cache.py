#!/usr/bin/env python3
"""Analyze the WarpConvNet benchmark cache to measure kernel usefulness.

All analysis is weighted by absolute time (ms), not config count.
A win at N=1M saving 5ms matters far more than a win at N=64 saving 0.001ms.

Considers all axes: log2(N), in/out channels, kernel volume, and their
cross-products to derive data-driven auto-tuning policy recommendations.

Reads ~/.cache/warpconvnet/benchmark_cache_generic.msgpack and produces
a thorough report on which algorithms matter, by how much wall-clock time
they save, and under what conditions.

Usage:
    python scripts/analyze_benchmark_cache.py
    python scripts/analyze_benchmark_cache.py --cache /path/to/cache.msgpack
    python scripts/analyze_benchmark_cache.py --namespace sparse_conv_forward
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import msgpack


# ---------------------------------------------------------------------------
# Cache deserialization (standalone — no warpconvnet import needed)
# ---------------------------------------------------------------------------


def _from_msgpack(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, bytes)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return [_from_msgpack(v) for v in obj]
    if isinstance(obj, dict):
        tag = obj.get("__t__")
        if tag == "T":
            return tuple(_from_msgpack(v) for v in obj["d"])
        if tag == "S":
            return {_from_msgpack(v) for v in obj["d"]}
        if tag == "SCC":
            return SparseConvConfig(
                log_n_in=obj["li"],
                log_n_out=obj["lo"],
                c_in=obj["ci"],
                c_out=obj["co"],
                kv=obj["kv"],
                dtype=obj.get("dt", "float32"),
                sm=tuple(obj.get("sm", [0, 0])),
            )
        return {_from_msgpack(k): _from_msgpack(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class SparseConvConfig:
    log_n_in: int
    log_n_out: int
    c_in: int
    c_out: int
    kv: int
    dtype: str
    sm: tuple[int, int]

    @property
    def approx_n(self) -> int:
        return 2**self.log_n_in

    @property
    def max_ch(self) -> int:
        return max(self.c_in, self.c_out)

    @property
    def ch_bucket(self) -> str:
        m = self.max_ch
        if m <= 32:
            return "<=32"
        elif m <= 64:
            return "33-64"
        elif m <= 128:
            return "65-128"
        elif m <= 256:
            return "129-256"
        else:
            return ">256"

    @property
    def n_bucket(self) -> str:
        """Coarse N bucket for policy decisions."""
        if self.log_n_in <= 12:
            return "small (N<=4K)"
        elif self.log_n_in <= 16:
            return "medium (4K-64K)"
        elif self.log_n_in <= 19:
            return "large (64K-512K)"
        else:
            return "xlarge (N>512K)"

    def short_str(self) -> str:
        return f"logN={self.log_n_in:2d} C={self.c_in:3d}->{self.c_out:3d} K={self.kv:3d} {self.dtype}"


@dataclass
class BenchmarkEntry:
    """One (config, ranked_results) pair from a namespace."""

    config: Any
    results: list[tuple[str, dict, float]]  # [(algo, params, time_ms), ...] best-first

    @property
    def best_algo(self) -> str:
        return self.results[0][0]

    @property
    def best_params(self) -> dict:
        return self.results[0][1]

    @property
    def best_time(self) -> float:
        return self.results[0][2]

    def best_algo_with_params(self) -> str:
        p = self.best_params
        if p:
            ps = ",".join(f"{k}={v}" for k, v in p.items())
            return f"{self.best_algo}({ps})"
        return self.best_algo

    def algo_time(self, algo_name: str) -> float | None:
        """Return best time for a given algo (exact match), or None."""
        for a, _, t in self.results:
            if a == algo_name:
                return t
        return None

    def best_time_excluding(self, algo_name: str) -> float | None:
        """Return best time among results that are NOT algo_name."""
        for a, _, t in self.results:
            if a != algo_name:
                return t
        return None


def load_cache(path: str) -> dict[str, list[BenchmarkEntry]]:
    """Load and parse the benchmark cache msgpack file."""
    with open(path, "rb") as f:
        raw = msgpack.unpack(f, raw=False)

    namespaces: dict[str, list[BenchmarkEntry]] = {}
    ns_data = raw.get("namespaces", raw)

    for ns_name, pairs in ns_data.items():
        entries = []
        for config_raw, results_raw in pairs:
            config = _from_msgpack(config_raw)
            parsed_results = []
            for r in _from_msgpack(results_raw):
                if isinstance(r, tuple) and len(r) == 3:
                    algo, params, t = r
                    if isinstance(params, dict):
                        parsed_results.append((str(algo), params, float(t)))
                elif isinstance(r, dict) and "metric" in r:
                    params = r.get("params", {})
                    t = r["metric"]
                    algo_parts = []
                    for k, v in sorted(params.items()):
                        algo_parts.append(f"{k}={v}")
                    algo_name = ",".join(algo_parts) if algo_parts else "default"
                    parsed_results.append((algo_name, params, float(t)))
            if parsed_results:
                parsed_results.sort(key=lambda x: x[2])
                entries.append(BenchmarkEntry(config=config, results=parsed_results))
        namespaces[ns_name] = entries

    return namespaces


# ---------------------------------------------------------------------------
# Helper: print a 2D pivot table
# ---------------------------------------------------------------------------


def _print_pivot_table(
    entries: list[BenchmarkEntry],
    row_fn,  # entry -> row label
    col_algos: list[str],  # algo names for columns
    title: str,
    row_sort_key=None,
    show_pct: bool = True,
):
    """Print a pivot table: rows × algo columns, cells = ms won."""
    # Accumulate
    time_won: dict[Any, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_by_row: dict[Any, float] = defaultdict(float)

    for e in entries:
        if not isinstance(e.config, SparseConvConfig):
            continue
        row = row_fn(e)
        time_won[row][e.best_algo] += e.best_time
        total_by_row[row] += e.best_time

    if not time_won:
        return

    print(f"\n  --- {title} ---")
    col_w = 14
    header = f"  {'Row':<20s} | {'TotalMs':>8s} |"
    for algo in col_algos:
        header += f" {algo[:col_w]:>{col_w}s} |"
    print(header)
    print(f"  {'-'*20}-+-{'-'*8}-|" + "".join(f"-{'-'*col_w}-|" for _ in col_algos))

    rows = sorted(time_won.keys(), key=row_sort_key) if row_sort_key else sorted(time_won.keys())
    for row in rows:
        row_total = total_by_row[row]
        line = f"  {str(row):<20s} | {row_total:>7.1f}ms |"
        for algo in col_algos:
            t = time_won[row].get(algo, 0)
            if t > 0.001 and show_pct:
                pct = t / row_total * 100
                line += f" {t:>6.1f} ({pct:3.0f}%) |"
            elif t > 0.001:
                line += f" {t:>13.1f}ms |"
            else:
                line += f" {'':>{col_w}s} |"
        print(line)

    # Total row
    grand_total = sum(total_by_row.values())
    line = f"  {'ALL':<20s} | {grand_total:>7.1f}ms |"
    for algo in col_algos:
        t = sum(time_won[r].get(algo, 0) for r in rows)
        pct = t / grand_total * 100 if grand_total > 0 else 0
        line += f" {t:>6.1f} ({pct:3.0f}%) |"
    print(line)


# ---------------------------------------------------------------------------
# Helper: removal impact per axis bucket
# ---------------------------------------------------------------------------


def _print_removal_impact_by_bucket(
    entries: list[BenchmarkEntry],
    bucket_fn,  # entry -> bucket label
    algos: list[str],
    title: str,
):
    """For each bucket, show removal impact (ms lost) per algo."""
    # removal_impact[bucket][algo] = total ms lost if algo removed
    removal_impact: dict[Any, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_by_bucket: dict[Any, float] = defaultdict(float)

    for e in entries:
        if not isinstance(e.config, SparseConvConfig):
            continue
        bucket = bucket_fn(e)
        total_by_bucket[bucket] += e.best_time
        winner = e.best_algo
        next_best = e.best_time_excluding(winner)
        if next_best is not None:
            removal_impact[bucket][winner] += next_best - e.best_time

    if not removal_impact:
        return

    print(f"\n  --- {title} ---")
    col_w = 12
    header = f"  {'Bucket':<20s} | {'TotalMs':>8s} |"
    for algo in algos:
        header += f" {algo[:col_w]:>{col_w}s} |"
    print(header)
    print(f"  {'-'*20}-+-{'-'*8}-|" + "".join(f"-{'-'*col_w}-|" for _ in algos))

    for bucket in sorted(removal_impact.keys()):
        bt = total_by_bucket[bucket]
        line = f"  {str(bucket):<20s} | {bt:>7.1f}ms |"
        for algo in algos:
            v = removal_impact[bucket].get(algo, 0)
            if v > 0.001:
                line += f" {v:>11.3f}ms |"
            else:
                line += f" {'':>{col_w}s} |"
        print(line)


# ---------------------------------------------------------------------------
# Time-weighted analysis for sparse_conv namespaces
# ---------------------------------------------------------------------------


def analyze_sparse_conv_namespace(entries: list[BenchmarkEntry], ns_name: str) -> dict[str, Any]:
    """Analyze sparse_conv_forward or sparse_conv_backward, weighted by time."""
    if not entries:
        return {}

    total = len(entries)
    all_algos = sorted({a for e in entries for a, _, _ in e.results})

    # Metadata
    dtypes = sorted({e.config.dtype for e in entries if isinstance(e.config, SparseConvConfig)})
    kvs = sorted({e.config.kv for e in entries if isinstance(e.config, SparseConvConfig)})
    log_ns = sorted({e.config.log_n_in for e in entries if isinstance(e.config, SparseConvConfig)})
    ch_pairs = sorted(
        {
            (e.config.c_in, e.config.c_out)
            for e in entries
            if isinstance(e.config, SparseConvConfig)
        }
    )

    print(f"\n{'='*100}")
    print(f"  {ns_name}: {total} configurations")
    print(f"{'='*100}")
    print(f"  dtypes: {dtypes}")
    print(f"  kernel volumes: {kvs}")
    print(f"  log2(N) range: {log_ns}")
    print(f"  channel pairs (C_in->C_out): {len(ch_pairs)} unique")
    print(f"  algorithms benchmarked: {len(all_algos)}: {all_algos}")

    # Total optimal time across all configs (sum of best times)
    total_optimal_ms = sum(e.best_time for e in entries)
    print(f"  total optimal time (sum of all best): {total_optimal_ms:.2f} ms")

    # =====================================================================
    # 1. TIME-WEIGHTED WIN TABLE
    # =====================================================================
    print(f"\n  --- 1. Time-Weighted Algorithm Value ---")
    print(
        f"  {'Algorithm':<30s} | {'Wins':>5s} | {'WinTime':>9s} | {'SavedMs':>9s} |"
        f" {'AvgSave':>8s} | {'MaxSave':>8s} | {'Win%ms':>7s}"
    )
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*9}-+-{'-'*9}-+-" f"{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    algo_stats = []
    for algo in all_algos:
        wins = 0
        win_time_ms = 0.0
        total_saved_ms = 0.0
        max_saved_ms = 0.0
        for e in entries:
            if e.best_algo == algo:
                wins += 1
                win_time_ms += e.best_time
                t2 = e.best_time_excluding(algo)
                if t2 is not None:
                    saved = t2 - e.best_time
                    total_saved_ms += saved
                    max_saved_ms = max(max_saved_ms, saved)
        avg_saved = total_saved_ms / wins if wins > 0 else 0
        algo_stats.append((total_saved_ms, algo, wins, win_time_ms, avg_saved, max_saved_ms))

    algo_stats.sort(key=lambda x: -x[0])
    for total_saved, algo, wins, win_time, avg_saved, max_saved in algo_stats:
        if wins == 0:
            continue
        win_pct_ms = win_time / total_optimal_ms * 100
        print(
            f"  {algo:<30s} | {wins:>5d} | {win_time:>8.2f}ms | {total_saved:>8.2f}ms |"
            f" {avg_saved:>7.3f}ms | {max_saved:>7.3f}ms | {win_pct_ms:>6.1f}%"
        )

    # =====================================================================
    # 2. REMOVAL IMPACT
    # =====================================================================
    print(f"\n  --- 2. Removal Impact (total ms lost if algo is removed) ---")

    removal_stats = []
    for algo in all_algos:
        total_lost = 0.0
        max_lost = 0.0
        n_wins = 0
        for e in entries:
            if e.best_algo == algo:
                n_wins += 1
                next_best = e.best_time_excluding(algo)
                if next_best is not None:
                    lost = next_best - e.best_time
                    total_lost += lost
                    max_lost = max(max_lost, lost)
        if n_wins > 0:
            removal_stats.append((total_lost, algo, n_wins, max_lost))

    removal_stats.sort(key=lambda x: -x[0])
    print(
        f"  {'Algorithm':<30s} | {'Wins':>5s} | {'TotalLost':>10s} | {'MaxLost':>9s} | {'AvgLost':>9s}"
    )
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}")
    for total_lost, algo, n_wins, max_lost in removal_stats:
        avg_lost = total_lost / n_wins
        print(
            f"  {algo:<30s} | {n_wins:>5d} | {total_lost:>9.3f}ms | {max_lost:>8.3f}ms | {avg_lost:>8.3f}ms"
        )

    # =====================================================================
    # 3. GREEDY SET COVER BY TIME SAVED
    # =====================================================================
    print(f"\n  --- 3. Greedy Set Cover (by total ms saved) ---")

    current_best = [None] * total
    selected = []
    remaining_algos = set(all_algos)

    print(
        f"  {'Step':>4s} | {'Algorithm':<30s} | {'Saved':>10s} | {'CumulRegret':>12s} |"
        f" {'vs Optimal':>10s}"
    )
    print(f"  {'-'*4}-+-{'-'*30}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

    for step in range(len(all_algos)):
        best_algo = None
        best_savings = -1.0

        for algo in remaining_algos:
            savings = 0.0
            for i, e in enumerate(entries):
                t = e.algo_time(algo)
                if t is None:
                    continue
                if current_best[i] is None:
                    savings += 0
                elif t < current_best[i]:
                    savings += current_best[i] - t

            if step == 0:
                total_t = 0.0
                for e in entries:
                    t = e.algo_time(algo)
                    if t is not None:
                        total_t += t
                savings = -total_t

            if savings > best_savings or best_algo is None:
                best_savings = savings
                best_algo = algo

        if best_algo is None:
            break

        remaining_algos.remove(best_algo)
        selected.append(best_algo)

        for i, e in enumerate(entries):
            t = e.algo_time(best_algo)
            if t is not None:
                if current_best[i] is None or t < current_best[i]:
                    current_best[i] = t

        cumul_regret = 0.0
        for i, e in enumerate(entries):
            if current_best[i] is not None:
                cumul_regret += current_best[i] - e.best_time

        pct_vs_opt = cumul_regret / total_optimal_ms * 100
        if step == 0:
            saved_str = f"{'base':>10s}"
        else:
            saved_str = f"{best_savings:>9.2f}ms"

        print(
            f"  {step+1:>4d} | {best_algo:<30s} | {saved_str} | {cumul_regret:>11.2f}ms |"
            f" {pct_vs_opt:>9.1f}%"
        )

        if cumul_regret < 0.01:
            break

    # =====================================================================
    # 4. WIN RATE BY log2(N) — TIME-WEIGHTED
    # =====================================================================
    # Determine top algos by total time won
    total_time_won = defaultdict(float)
    for e in entries:
        total_time_won[e.best_algo] += e.best_time
    top_algos = [a for a, _ in sorted(total_time_won.items(), key=lambda x: -x[1])[:8]]

    _print_pivot_table(
        entries,
        row_fn=lambda e: f"logN={e.config.log_n_in:2d} (~{e.config.approx_n:>10,d})",
        col_algos=top_algos,
        title="4. Time Won by log2(N)",
        row_sort_key=lambda r: int(r.split("=")[1].split()[0]),
    )

    # =====================================================================
    # 5. WIN RATE BY CHANNEL RANGE — TIME-WEIGHTED
    # =====================================================================
    ch_order = {"<=32": 0, "33-64": 1, "65-128": 2, "129-256": 3, ">256": 4}
    _print_pivot_table(
        entries,
        row_fn=lambda e: e.config.ch_bucket,
        col_algos=top_algos,
        title="5. Time Won by Channel Range (max(C_in,C_out))",
        row_sort_key=lambda r: ch_order.get(r, 99),
    )

    # =====================================================================
    # 6. WIN RATE BY KERNEL VOLUME — TIME-WEIGHTED
    # =====================================================================
    _print_pivot_table(
        entries,
        row_fn=lambda e: f"kv={e.config.kv:3d}",
        col_algos=top_algos,
        title="6. Time Won by Kernel Volume",
        row_sort_key=lambda r: int(r.split("=")[1]),
    )

    # =====================================================================
    # 7. CROSS-AXIS: N bucket × Channel bucket — winner algo
    #    This is the key table for adaptive policy design.
    # =====================================================================
    print(f"\n  --- 7. Cross-Axis: N bucket x Channel bucket (winner + ms) ---")
    print(f"  Shows which algo wins most time in each (N, channel) region.")

    n_buckets_ordered = ["small (N<=4K)", "medium (4K-64K)", "large (64K-512K)", "xlarge (N>512K)"]
    ch_buckets_ordered = ["<=32", "33-64", "65-128", "129-256", ">256"]

    # Accumulate: (n_bucket, ch_bucket) -> {algo: ms_won}
    cross_time: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cross_total: dict[tuple[str, str], float] = defaultdict(float)
    cross_count: dict[tuple[str, str], int] = defaultdict(int)

    for e in entries:
        if not isinstance(e.config, SparseConvConfig):
            continue
        key = (e.config.n_bucket, e.config.ch_bucket)
        cross_time[key][e.best_algo] += e.best_time
        cross_total[key] += e.best_time
        cross_count[key] += 1

    # Print as a table
    col_w = 35
    n_ch_label = "N \\ Ch"
    header = f"  {n_ch_label:<20s}"
    for ch in ch_buckets_ordered:
        header += f" | {ch:^{col_w}s}"
    print(header)
    print(f"  {'-'*20}" + "".join(f"-+-{'-'*col_w}" for _ in ch_buckets_ordered))

    for nb in n_buckets_ordered:
        line = f"  {nb:<20s}"
        for cb in ch_buckets_ordered:
            key = (nb, cb)
            if key not in cross_time:
                line += f" | {'':^{col_w}s}"
                continue
            ct = cross_total[key]
            cc = cross_count[key]
            # Find top 2 algos
            ranked = sorted(cross_time[key].items(), key=lambda x: -x[1])
            if ranked:
                top1_algo, top1_ms = ranked[0]
                top1_pct = top1_ms / ct * 100
                cell = f"{top1_algo[:18]}:{top1_ms:.1f}ms({top1_pct:.0f}%)"
                if len(ranked) > 1:
                    top2_algo, top2_ms = ranked[1]
                    top2_pct = top2_ms / ct * 100
                    if top2_pct > 5:
                        cell += f" +{top2_algo[:8]}"
                cell += f" [{cc}c,{ct:.1f}ms]"
                line += f" | {cell:<{col_w}s}"
            else:
                line += f" | {'':^{col_w}s}"
        print(line)

    # =====================================================================
    # 8. CROSS-AXIS: N bucket × KV — winner algo
    # =====================================================================
    print(f"\n  --- 8. Cross-Axis: N bucket x Kernel Volume (winner + ms) ---")

    cross_nkv_time: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    cross_nkv_total: dict[tuple[str, int], float] = defaultdict(float)
    cross_nkv_count: dict[tuple[str, int], int] = defaultdict(int)

    for e in entries:
        if not isinstance(e.config, SparseConvConfig):
            continue
        key = (e.config.n_bucket, e.config.kv)
        cross_nkv_time[key][e.best_algo] += e.best_time
        cross_nkv_total[key] += e.best_time
        cross_nkv_count[key] += 1

    col_w = 40
    n_kv_label = "N \\ KV"
    header = f"  {n_kv_label:<20s}"
    for kv in kvs:
        header += f" | {'kv=' + str(kv):^{col_w}s}"
    print(header)
    print(f"  {'-'*20}" + "".join(f"-+-{'-'*col_w}" for _ in kvs))

    for nb in n_buckets_ordered:
        line = f"  {nb:<20s}"
        for kv in kvs:
            key = (nb, kv)
            if key not in cross_nkv_time:
                line += f" | {'':^{col_w}s}"
                continue
            ct = cross_nkv_total[key]
            cc = cross_nkv_count[key]
            ranked = sorted(cross_nkv_time[key].items(), key=lambda x: -x[1])
            if ranked:
                parts = []
                for algo, ms in ranked[:3]:
                    pct = ms / ct * 100
                    if pct > 5:
                        parts.append(f"{algo[:15]}:{pct:.0f}%")
                cell = " ".join(parts) + f" [{cc}c,{ct:.1f}ms]"
                line += f" | {cell:<{col_w}s}"
            else:
                line += f" | {'':^{col_w}s}"
        print(line)

    # =====================================================================
    # 9. REMOVAL IMPACT BY N BUCKET
    # =====================================================================
    removal_algos = [algo for _, algo, _, _ in removal_stats[:8]]
    _print_removal_impact_by_bucket(
        entries,
        bucket_fn=lambda e: e.config.n_bucket,
        algos=removal_algos,
        title="9. Removal Impact by N Bucket (ms lost if algo removed)",
    )

    # =====================================================================
    # 10. REMOVAL IMPACT BY CHANNEL BUCKET
    # =====================================================================
    _print_removal_impact_by_bucket(
        entries,
        bucket_fn=lambda e: e.config.ch_bucket,
        algos=removal_algos,
        title="10. Removal Impact by Channel Bucket",
    )

    # =====================================================================
    # 11. REMOVAL IMPACT BY KV
    # =====================================================================
    _print_removal_impact_by_bucket(
        entries,
        bucket_fn=lambda e: f"kv={e.config.kv}",
        algos=removal_algos,
        title="11. Removal Impact by Kernel Volume",
    )

    # =====================================================================
    # 12. PER-ALGO: where does it win? (logN x channel detail)
    #     Only for algos with non-trivial removal impact
    # =====================================================================
    significant_algos = [algo for total_lost, algo, _, _ in removal_stats if total_lost > 1.0]
    for algo in significant_algos:
        print(f"\n  --- 12-{algo}: Where does '{algo}' uniquely win? ---")
        print(f"  (Configs where removing it loses >0.01ms)")
        win_entries = []
        for e in entries:
            if not isinstance(e.config, SparseConvConfig):
                continue
            if e.best_algo == algo:
                next_best = e.best_time_excluding(algo)
                if next_best is not None:
                    delta = next_best - e.best_time
                    if delta > 0.01:
                        win_entries.append((delta, e))
        win_entries.sort(key=lambda x: -x[0])
        print(
            f"  {'Config':<45s} | {'Best':>8s} | {'2nd':>8s} | {'Saved':>8s} | {'2nd Algo':<25s}"
        )
        print(f"  {'-'*45}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*25}")
        for delta, e in win_entries[:15]:
            cfg_str = e.config.short_str()
            second = e.best_time_excluding(algo)
            second_algo = "?"
            for a, _, t in e.results:
                if a != algo:
                    second_algo = a
                    break
            print(
                f"  {cfg_str:<45s} | {e.best_time:>7.3f}ms | {second:>7.3f}ms | {delta:>7.3f}ms | {second_algo:<25s}"
            )

    # =====================================================================
    # 13. POLICY RECOMMENDATION
    #     Based on all the above, generate a concrete adaptive policy
    # =====================================================================
    print(f"\n  {'='*90}")
    print(f"  13. AUTO POLICY RECOMMENDATION for {ns_name}")
    print(f"  {'='*90}")

    # For each (n_bucket, ch_bucket), determine which algos have removal impact > threshold
    THRESHOLD_MS = 0.1  # algo must save at least this to be worth including
    print(f"\n  Threshold: algo included if removal impact > {THRESHOLD_MS}ms in that region")
    print(f"\n  {'N Bucket':<20s} | {'Ch Bucket':<12s} | {'TotalMs':>8s} | Recommended Algos")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*8}-+-{'-'*50}")

    # Compute removal impact per (n_bucket, ch_bucket, algo)
    region_removal: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    region_total: dict[tuple[str, str], float] = defaultdict(float)

    for e in entries:
        if not isinstance(e.config, SparseConvConfig):
            continue
        key = (e.config.n_bucket, e.config.ch_bucket)
        region_total[key] += e.best_time
        winner = e.best_algo
        next_best = e.best_time_excluding(winner)
        if next_best is not None:
            region_removal[key][winner] += next_best - e.best_time

    all_recommended = set()
    for nb in n_buckets_ordered:
        for cb in ch_buckets_ordered:
            key = (nb, cb)
            if key not in region_total:
                continue
            rt = region_total[key]
            rr = region_removal[key]
            # Filter: only algos with removal impact > threshold
            recommended = sorted(
                [(algo, impact) for algo, impact in rr.items() if impact > THRESHOLD_MS],
                key=lambda x: -x[1],
            )
            if not recommended:
                # Fall back to the single best algo
                best_algo = max(cross_time[key].items(), key=lambda x: x[1])[0]
                recommended = [(best_algo, 0.0)]
            algo_str = ", ".join(f"{a}({v:.1f}ms)" for a, v in recommended)
            all_recommended.update(a for a, _ in recommended)
            print(f"  {nb:<20s} | {cb:<12s} | {rt:>7.1f}ms | {algo_str}")

    print(f"\n  FULL RECOMMENDED SET: {sorted(all_recommended)}")
    print(f"  ({len(all_recommended)} algorithms)")

    # Also show which algos from the benchmarked set are NOT recommended
    never_needed = set(all_algos) - all_recommended
    if never_needed:
        print(f"  NEVER uniquely valuable (can exclude): {sorted(never_needed)}")

    # =====================================================================
    # 14. COMPACT ADAPTIVE POLICY TABLE
    #     Map (log_n_threshold, max_ch_threshold) -> algo set
    # =====================================================================
    print(f"\n  --- 14. Compact Adaptive Policy (for algo_params.py) ---")
    print(f"  Grouped by conditions that yield the same algo set:\n")

    # For each entry, determine which algos have removal impact > 0.01ms
    # Then group entries by their "needed algo set"
    entry_needed: dict[int, set] = {}  # entry index -> set of algos needed
    for i, e in enumerate(entries):
        if not isinstance(e.config, SparseConvConfig):
            continue
        needed = set()
        # The winner is always needed
        needed.add(e.best_algo)
        # Check if any other algo is within 5% of best — include those too
        for a, _, t in e.results:
            if t <= e.best_time * 1.05:
                needed.add(a)
        entry_needed[i] = needed

    # Determine which algos are needed across different N ranges
    n_ranges = [
        ("N <= 4K (logN<=12)", lambda e: e.config.log_n_in <= 12),
        ("4K < N <= 64K", lambda e: 12 < e.config.log_n_in <= 16),
        ("64K < N <= 512K", lambda e: 16 < e.config.log_n_in <= 19),
        ("N > 512K (logN>19)", lambda e: e.config.log_n_in > 19),
    ]
    ch_ranges = [
        ("ch <= 64", lambda e: e.config.max_ch <= 64),
        ("ch > 64", lambda e: e.config.max_ch > 64),
    ]

    print(
        f"  {'N Range':<22s} | {'Ch Range':<12s} | {'Configs':>7s} | {'TotalMs':>8s} | Algo Set (removal impact > 0.01ms)"
    )
    print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*7}-+-{'-'*8}-+-{'-'*50}")

    for n_label, n_pred in n_ranges:
        for ch_label, ch_pred in ch_ranges:
            filtered = [
                e
                for e in entries
                if isinstance(e.config, SparseConvConfig) and n_pred(e) and ch_pred(e)
            ]
            if not filtered:
                continue

            total_ms = sum(e.best_time for e in filtered)
            n_configs = len(filtered)

            # Compute removal impact for this subset
            sub_removal: dict[str, float] = defaultdict(float)
            for e in filtered:
                winner = e.best_algo
                next_best = e.best_time_excluding(winner)
                if next_best is not None:
                    sub_removal[winner] += next_best - e.best_time

            needed = sorted(
                [(a, v) for a, v in sub_removal.items() if v > 0.01], key=lambda x: -x[1]
            )
            algo_str = ", ".join(f"{a}({v:.2f}ms)" for a, v in needed)
            print(
                f"  {n_label:<22s} | {ch_label:<12s} | {n_configs:>7d} | {total_ms:>7.1f}ms | {algo_str}"
            )

    return {
        "win_counts": Counter({algo: wins for _, algo, wins, _, _, _ in algo_stats}),
        "total": total,
        "total_optimal_ms": total_optimal_ms,
        "removal_stats": removal_stats,
    }


def analyze_subgemm_namespace(entries: list[BenchmarkEntry], ns_name: str) -> dict[str, Any]:
    """Analyze a sub-GEMM namespace (AD_gather_scatter, trAB_gather), time-weighted."""
    if not entries:
        return {}

    total = len(entries)
    total_optimal_ms = sum(e.best_time for e in entries)

    print(f"\n{'='*100}")
    print(f"  {ns_name}: {total} configurations, total optimal: {total_optimal_ms:.2f}ms")
    print(f"{'='*100}")

    # Win by mma_tile, weighted by time
    time_by_mma: dict[str, float] = defaultdict(float)
    time_by_split_k: dict[str, float] = defaultdict(float)
    for e in entries:
        params = e.best_params
        mma_tile = params.get("mma_tile", "?")
        split_k = params.get("split_k_slices", "?")
        time_by_mma[f"mma_tile={mma_tile}"] += e.best_time
        if split_k != "?":
            time_by_split_k[f"split_k={split_k}"] += e.best_time

    print(f"\n  --- Time Won by mma_tile ---")
    for param, t in sorted(time_by_mma.items(), key=lambda x: -x[1]):
        pct = t / total_optimal_ms * 100
        print(f"  {param:<25s} {t:>8.2f}ms ({pct:5.1f}%)")

    if time_by_split_k:
        print(f"\n  --- Time Won by split_k_slices ---")
        for param, t in sorted(time_by_split_k.items(), key=lambda x: -x[1]):
            pct = t / total_optimal_ms * 100
            print(f"  {param:<25s} {t:>8.2f}ms ({pct:5.1f}%)")

    # Time won by channel size
    time_by_channels: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    total_by_channels: dict[tuple[int, int], float] = defaultdict(float)
    for e in entries:
        cfg = e.config
        if isinstance(cfg, tuple) and len(cfg) >= 8:
            K, N = cfg[4], cfg[5]
            mma_tile = e.best_params.get("mma_tile", "?")
            time_by_channels[(K, N)][f"mma_tile={mma_tile}"] += e.best_time
            total_by_channels[(K, N)] += e.best_time

    if time_by_channels:
        print(f"\n  --- Time Won by Channel Size (top 10 by total ms) ---")
        sorted_channels = sorted(total_by_channels.items(), key=lambda x: -x[1])[:10]
        for (K, N), ch_total in sorted_channels:
            winners = sorted(time_by_channels[(K, N)].items(), key=lambda x: -x[1])
            top = ", ".join(f"{p}: {t:.1f}ms" for p, t in winners[:3])
            print(f"  K={K:3d} N={N:3d} ({ch_total:>7.1f}ms): {top}")

    return {"total": total, "total_optimal_ms": total_optimal_ms}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(all_results: dict[str, dict[str, Any]]):
    print(f"\n{'='*100}")
    print(f"  CROSS-NAMESPACE SUMMARY (TIME-WEIGHTED)")
    print(f"{'='*100}")

    for ns_name in ["sparse_conv_forward", "sparse_conv_backward"]:
        r = all_results.get(ns_name)
        if not r:
            continue
        total_ms = r.get("total_optimal_ms", 0)
        print(f"\n  {ns_name} (total optimal: {total_ms:.1f}ms):")

        removal = r.get("removal_stats", [])
        if removal:
            print(f"  {'Algorithm':<30s} | {'RemovalCost':>11s} | {'Wins':>5s}")
            print(f"  {'-'*30}-+-{'-'*11}-+-{'-'*5}")
            for total_lost, algo, n_wins, _ in removal:
                print(f"  {algo:<30s} | {total_lost:>10.3f}ms | {n_wins:>5d}")

    fwd = all_results.get("sparse_conv_forward", {}).get("win_counts", Counter())
    bwd = all_results.get("sparse_conv_backward", {}).get("win_counts", Counter())
    fwd_only = set(fwd.keys()) - set(bwd.keys())
    bwd_only = set(bwd.keys()) - set(fwd.keys())
    if fwd_only:
        print(f"\n  Algos that win forward but NEVER backward: {fwd_only}")
    if bwd_only:
        print(f"\n  Algos that win backward but NEVER forward: {bwd_only}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze WarpConvNet benchmark cache (time-weighted)"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=os.path.expanduser("~/.cache/warpconvnet/benchmark_cache_generic.msgpack"),
        help="Path to benchmark cache msgpack file",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Analyze only this namespace (e.g., sparse_conv_forward)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        print(f"ERROR: Cache file not found: {args.cache}", file=sys.stderr)
        sys.exit(1)

    namespaces = load_cache(args.cache)

    print(f"Cache file: {args.cache}")
    print(f"  Size: {os.path.getsize(args.cache) / 1024:.1f} KB")
    for ns_name, entries in namespaces.items():
        print(f"    {ns_name}: {len(entries)} entries")

    all_results = {}
    for ns_name, entries in namespaces.items():
        if args.namespace and ns_name != args.namespace:
            continue
        if ns_name in ("sparse_conv_forward", "sparse_conv_backward"):
            r = analyze_sparse_conv_namespace(entries, ns_name)
        else:
            r = analyze_subgemm_namespace(entries, ns_name)
        all_results[ns_name] = r

    if not args.namespace:
        print_summary(all_results)


if __name__ == "__main__":
    main()
