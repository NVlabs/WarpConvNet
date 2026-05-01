# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permutohedral d=6 hero bench (matches bpfilter standalone reproducer).

Workload:
  N = 196608 (= 512 * 384), d = 6
  Sigmas [16, 16, 12, 12, 12, 1]
  Features float32 [N, 3]
  3 warmup + 10 measured trials, cuda.synchronize between trials

Per-stage instrumentation via cuda.Event matches bpfilter's split.
"""
import torch

from warpconvnet.nn.functional.permutohedral import PermutohedralLattice


def _make_workload(N: int, d: int, device, seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    pos = torch.empty(N, d, device=device, dtype=torch.float32)
    # 2 spatial pixel-style + 3 color + 1 const-pad, all just random for bench
    pos[:, 0:2] = torch.randint(0, 512, (N, 2), generator=g, device=device).float()
    pos[:, 2:5] = torch.randint(0, 256, (N, 3), generator=g, device=device).float()
    pos[:, 5] = 0.0
    sigmas = torch.tensor([16, 16, 12, 12, 12, 1], device=device, dtype=torch.float32)
    feat = torch.randn(N, 3, device=device, dtype=torch.float32, generator=g)
    return pos, sigmas, feat


@torch.no_grad()
def run_bench(n_warmup: int = 3, n_trials: int = 10):
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    device = torch.device("cuda")

    N, d = 196608, 6
    pos, sigmas, feat = _make_workload(N, d, device)
    scaled = pos / sigmas

    # Warmup
    for _ in range(n_warmup):
        lat = PermutohedralLattice.build(scaled)
        _ = lat.filter(feat, normalize=True)
        torch.cuda.synchronize()

    # Per-stage timing via cuda.Event. Mirror bpfilter: measure build / splat
    # / blur / slice separately per trial.
    times = {"build": [], "splat": [], "blur": [], "slice": [], "total": []}

    for _ in range(n_trials):
        ev = {
            k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for k in times
        }
        torch.cuda.synchronize()

        ev["total"][0].record()

        ev["build"][0].record()
        lat = PermutohedralLattice.build(scaled)
        ev["build"][1].record()

        # Replicate filter() with normalize=True manually so we can split
        # splat/blur/slice events.
        ones = torch.ones((feat.shape[0], 1), dtype=feat.dtype, device=feat.device)
        f_ext = torch.cat([feat, ones], dim=-1)

        ev["splat"][0].record()
        lattice = lat._splat(f_ext)
        ev["splat"][1].record()

        ev["blur"][0].record()
        lattice = lat._blur(lattice)
        ev["blur"][1].record()

        ev["slice"][0].record()
        out = lat._slice(lattice, lat.inverse, lat.bary)
        ev["slice"][1].record()

        ev["total"][1].record()
        torch.cuda.synchronize()

        for k, (s, e) in ev.items():
            times[k].append(s.elapsed_time(e))

    print(f"N={N} d={d} sigmas={sigmas.tolist()}  (median of {n_trials} trials)")
    for k in ("build", "splat", "blur", "slice", "total"):
        vs = sorted(times[k])
        med = vs[len(vs) // 2]
        mn = min(vs)
        mx = max(vs)
        print(f"  {k:>6s}: {med:6.3f} ms  (min {mn:6.3f}, max {mx:6.3f})")


if __name__ == "__main__":
    run_bench()
