# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run compute-sanitizer racecheck over every launchable mask_gemm tile.

Numeric validation cannot find a *latent* race — one that is real but that this
architecture's scheduling never turns into wrong output. On GB300 the forward
``2s_fused`` race did not corrupt a single result in 600 launches, while the
same kernels corrupt results on sm_120. racecheck finds it either way, because
it reports from the instrumented shared-memory accesses rather than from the
output.

So this is the complement to ``validate_tiles_on_device.py``, not a substitute:

    validate_tiles_on_device.py   catches wrong answers        (misses latent races)
    racecheck_tiles.py            catches shared-memory races  (misses logic errors)

Neither alone is a sufficient bring-up gate. Tile 41 is wrong on both arches and
racecheck-clean; tiles 2/19 are racecheck-dirty on both arches and numerically
clean on one of them.

Interpreting a hit: racecheck can false-positive on kernels whose named barriers
or ``mbarrier`` waits it cannot model, which is common in warp-specialized
kernels. Do not report a hit without a control — a known-good tile from the same
family that comes back clean. ``--controls`` prints which clean tiles were swept
alongside so a report carries its own control set.

    python scripts/racecheck_tiles.py --op forward
    python scripts/racecheck_tiles.py --op all --n 4000
    python scripts/racecheck_tiles.py --op forward --tiles 2,19,57
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "sanitize_tile.py"
# compute-sanitizer singularises: "(26 errors, 1 warning)" — an earlier version
# of this pattern required the plural "warnings" and so failed to match exactly
# the runs with one warning. Those are the interesting ones: a single warning is
# what "Maximum number of hazards reached" emits, i.e. the heavily-racing tiles.
# The mismatch silently downgraded a 26-error race to "SKIPPED (no summary)".
_SUMMARY = re.compile(r"RACECHECK SUMMARY: .*?\((\d+) errors?, (\d+) warnings?\)")
_KERNEL = re.compile(r"(MaskGemm_[a-z]+_[0-9x]+_[A-Za-z0-9_]+)")
# Racy tiles overflow the default hazard buffer, which truncates the report.
# Raising it keeps the error count meaningful rather than capped.
_MAX_HAZARDS = "1000000"


def executed_tiles(op, cin, cout, n):
    """Tile ids that actually launch for ``op`` — the binding refuses many."""
    code = f"""
import sys; sys.path.insert(0, {str(REPO)!r})
import warpconvnet
from scripts.validate_tiles_on_device import build, run_tile
from warpconvnet.nn.functional.sparse_conv.detail import tile_metadata as tm
p = build({n}, {cin}, {cout})
ok = []
for t in sorted(x.tile_id for x in tm._get_tiles({op!r}, filter_arch=True)):
    try:
        if run_tile({op!r}, t, p) is not None:
            ok.append(t)
    except RuntimeError:
        pass
print("TILES=" + ",".join(map(str, ok)))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    for line in r.stdout.splitlines():
        if line.startswith("TILES="):
            rest = line[6:]
            return [int(v) for v in rest.split(",") if v]
    return []


def racecheck_one(op, tile, cin, cout, n, timeout):
    cmd = [
        shutil.which("compute-sanitizer") or "compute-sanitizer",
        "--tool",
        "racecheck",
        "--print-limit",
        "1",
        sys.executable,
        str(DRIVER),
        "--op",
        op,
        "--tile",
        str(tile),
        "--cin",
        str(cin),
        "--cout",
        str(cout),
        "--n",
        str(n),
    ]
    env = dict(os.environ, NV_COMPUTE_SANITIZER_MAX_RACECHECK_HAZARDS=_MAX_HAZARDS)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO, env=env)
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"
    out = r.stdout + r.stderr
    if "\nOK " not in "\n" + out:
        return None, None, "did not launch"
    m = _SUMMARY.search(out)
    if not m:
        # Never treat an unparsed summary as clean: the tile ran, so a missing
        # count means this parser is wrong, not that the kernel is race-free.
        return None, None, "UNPARSED SUMMARY - treat as unknown, not clean"
    errors = int(m.group(1))
    k = _KERNEL.search(out)
    return errors, (k.group(1) if k else None), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="forward", choices=("forward", "dgrad", "wgrad", "all"))
    p.add_argument("--tiles", default="", help="comma-separated (default: all launchable)")
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--cout", type=int, default=64)
    # racecheck instruments every shared-memory access and is ~100x slower than
    # a bare run. A race is *detected* from the accesses, not from whether it
    # corrupted anything, so scale is not needed here (unlike the numeric sweep).
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--controls", action="store_true", help="list the clean tiles swept")
    args = p.parse_args()

    ops = ("forward", "dgrad", "wgrad") if args.op == "all" else (args.op,)
    if not shutil.which("compute-sanitizer"):
        print("compute-sanitizer not on PATH", file=sys.stderr)
        return 2

    dirty_total = 0
    for op in ops:
        tiles = (
            [int(t) for t in args.tiles.split(",") if t]
            if args.tiles
            else executed_tiles(op, args.cin, args.cout, 20000)
        )
        print(
            f"\n=== {op}: racecheck {len(tiles)} launchable tiles "
            f"at C={args.cin}->{args.cout}, N={args.n} ==="
        )
        dirty, clean, skipped = [], [], []
        for tile in tiles:
            errors, kernel, err = racecheck_one(
                op, tile, args.cin, args.cout, args.n, args.timeout
            )
            if err:
                skipped.append((tile, err))
                print(f"  tile {tile:>4}: SKIPPED ({err})")
            elif errors:
                dirty.append((tile, errors, kernel))
                print(f"  tile {tile:>4}: RACE  {errors:>4} errors   {kernel or ''}")
            else:
                clean.append(tile)
        dirty_total += len(dirty)
        print(f"  -- {len(dirty)} raced, {len(clean)} clean, {len(skipped)} skipped")
        if args.controls and clean:
            print(f"  -- clean controls: {clean}")
        if skipped:
            print(f"  -- skipped: {[t for t, _ in skipped]}")
    return 1 if dirty_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
