# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-algorithm sparse-conv timing, one algo per process (clean autotune cache).

python scripts/bench_algos_gb300.py --algo auto
python scripts/bench_algos_gb300.py --algo auto --spoof-arch 100
"""

import argparse
import os
import subprocess
import sys
import tempfile

ALGOS = [
    "explicit_gemm",
    "implicit_gemm",
    "cutlass_implicit_gemm",
    "cute_implicit_gemm",
    "mask_gemm",
    "auto",
]


def child(args):
    if args.spoof_arch:
        # Pretend the device is sm_<spoof> so the mask_gemm tile arch gate opens.
        # Kept for reproducing the pre-enablement comparison: the gate now admits
        # GB-series parts on its own, so this is only needed to measure what the
        # old exact-membership rule cost.
        import warpconvnet.nn.functional.sparse_conv.detail.tile_metadata as tm

        tm._DEVICE_ARCH = args.spoof_arch

    import torch
    import warpconvnet  # noqa: F401
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.nn.modules.sparse_conv import SparseConv3d

    dtype = getattr(torch, args.dtype)
    dev = "cuda"
    torch.manual_seed(0)
    coords, feats = [], []
    for _ in range(args.batch):
        c = torch.randint(0, args.extent, (int(args.n * 1.3), 3), device=dev, dtype=torch.int32)
        c = torch.unique(c, dim=0)[: args.n]
        coords.append(c)
        feats.append(torch.randn(c.shape[0], args.cin, device=dev, dtype=dtype))
    vox = Voxels(coords, feats).unique()

    conv = (
        SparseConv3d(
            args.cin,
            args.cout,
            kernel_size=args.k,
            stride=args.stride,
            bias=False,
            fwd_algo=args.algo,
            dgrad_algo=args.algo,
            wgrad_algo=args.algo,
        )
        .to(dev)
        .to(dtype)
    )

    def fwd():
        with torch.no_grad():
            conv(vox)

    def fwd_bwd():
        v = vox.replace(batched_features=vox.feature_tensor.detach().requires_grad_(True))
        conv(v).feature_tensor.sum().backward()

    def timeit(fn, warmup=10, iters=30):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    tf, tb = timeit(fwd), timeit(fwd_bwd)
    picked = ""
    if args.algo == "auto":
        from warpconvnet.nn.functional.sparse_conv.detail.autotune import _BENCHMARK_AB_RESULTS

        for v in _BENCHMARK_AB_RESULTS.values():
            best = v[0] if isinstance(v, list) else v
            picked = f" picked={best[0]}"
            break
    print(f"RESULT\t{args.algo}\t{tf:.3f}\t{tb:.3f}\t{vox.feature_tensor.shape[0]}{picked}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--extent", type=int, default=256)
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--cout", type=int, default=64)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--algo", default=None)
    p.add_argument("--spoof-arch", type=int, default=0)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()

    if args.child:
        child(args)
        return

    algos = [args.algo] if args.algo else ALGOS
    label = f"sm_{args.spoof_arch} (spoofed)" if args.spoof_arch else "real arch"
    print(
        f"N={args.n}x{args.batch} C={args.cin}->{args.cout} k={args.k} s={args.stride} "
        f"{args.dtype} | {label}"
    )
    print(f"{'algo':<24} {'fwd (ms)':>10} {'fwd+bwd (ms)':>14}  note")
    print("-" * 70)
    for algo in algos:
        with tempfile.TemporaryDirectory() as cache_dir:
            env = dict(os.environ, WARPCONVNET_BENCHMARK_CACHE_DIR=cache_dir)
            cmd = [sys.executable, __file__, "--child", "--algo", algo]
            for f in ("n", "batch", "extent", "cin", "cout", "k", "stride", "dtype"):
                cmd += [f"--{f}", str(getattr(args, f))]
            if args.spoof_arch:
                cmd += ["--spoof-arch", str(args.spoof_arch)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")), None)
            if line is None:
                err = (r.stderr.strip().splitlines() or ["(no output)"])[-1][:60]
                print(f"{algo:<24} {'-':>10} {'-':>14}  FAILED: {err}")
                continue
            _, a, tf, tb, nv, *rest = (line.split("\t") + [""])[:6]
            print(f"{a:<24} {float(tf):>10.3f} {float(tb):>14.3f}  {rest[0] if rest else ''}")


if __name__ == "__main__":
    main()
