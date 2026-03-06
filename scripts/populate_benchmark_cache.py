#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Populate the WarpConvNet benchmark cache by running sparse convolutions
across a grid of (num_voxels, in_channels, out_channels, kernel_size, dtype)
configurations.

The resulting cache file can be distributed to users so they skip first-run
auto-tuning entirely.

Usage
-----
# Run all default configurations (forward + backward):
python scripts/populate_benchmark_cache.py

# Forward only, exhaustive algo search:
python scripts/populate_benchmark_cache.py --forward-only --algo-mode all

# Specific channel pairs:
python scripts/populate_benchmark_cache.py --channels 32,32 64,128 128,256

# Specific voxel counts:
python scripts/populate_benchmark_cache.py --num-voxels 50000 200000 1000000

# Quick smoke test (small subset):
python scripts/populate_benchmark_cache.py --preset quick

# Dry run (list configs without running):
python scripts/populate_benchmark_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

# Channel pairs seen in common architectures:
#   MinkUNet18/34: 3->32, 32->32, 32->64, 64->64, 64->128, 128->128,
#                  128->256, 256->256, 256->128, 128->96, 96->96
#   MaxViT-UNet:   6->48, 48->96, 96->192, 192->384, 384->512
#   SparseConvUNet: 16->32->48->64->80->96->112->128
CHANNEL_PAIRS_COMMON = [
    # Initial layers (small C_in)
    (3, 32),
    (6, 32),
    (6, 48),
    # Encoder layers
    (32, 32),
    (32, 64),
    (48, 48),
    (48, 96),
    (64, 64),
    (64, 128),
    (96, 96),
    (96, 192),
    (128, 128),
    (128, 256),
    (192, 192),
    (192, 384),
    (256, 256),
    (256, 128),
    (384, 384),
    (384, 512),
    (512, 512),
    # Decoder / upsampling layers (reverse channel direction)
    (128, 96),
    (256, 128),
    (256, 256),
    (384, 256),
    (512, 384),
    # 1x1 projections (kernel_volume=1 handled separately)
    (96, 10),
    (96, 20),
    (128, 20),
]

# Voxel counts spanning typical indoor (ScanNet ~50K-500K) and outdoor
# (nuScenes/Waymo ~100K-2M) point clouds. Values are chosen so that
# ceil(log2(N)) covers the range [15..21] which is the log-N cache bucket space.
NUM_VOXELS_DEFAULT = [
    30_000,   # log2 ~ 15
    65_000,   # log2 ~ 16
    130_000,  # log2 ~ 17
    260_000,  # log2 ~ 18
    500_000,  # log2 ~ 19
    1_000_000,  # log2 ~ 20
    2_000_000,  # log2 ~ 21
]

KERNEL_SIZES_DEFAULT = [3, 5]  # 3^3=27, 5^3=125
DTYPES_DEFAULT = [torch.float16, torch.bfloat16]

# Quick preset: minimal grid for smoke testing
NUM_VOXELS_QUICK = [100_000, 500_000]
CHANNEL_PAIRS_QUICK = [
    (32, 32),
    (64, 128),
    (128, 256),
]
KERNEL_SIZES_QUICK = [3]
DTYPES_QUICK = [torch.float16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Populate WarpConvNet benchmark cache across a grid of configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--preset",
        choices=["default", "quick"],
        default="default",
        help="Configuration preset (default: %(default)s)",
    )
    p.add_argument(
        "--num-voxels",
        type=int,
        nargs="+",
        default=None,
        help="Override voxel counts (e.g. --num-voxels 100000 500000)",
    )
    p.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=None,
        help="Override channel pairs as C_in,C_out (e.g. --channels 32,64 128,256)",
    )
    p.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Override kernel sizes (e.g. --kernel-sizes 3 5)",
    )
    p.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Override dtypes",
    )
    p.add_argument(
        "--algo-mode",
        type=str,
        default="auto",
        help="Algorithm selection mode: auto, all, or specific algo name (default: auto)",
    )
    p.add_argument(
        "--forward-only",
        action="store_true",
        help="Only populate forward pass cache (skip backward)",
    )
    p.add_argument(
        "--backward-only",
        action="store_true",
        help="Only populate backward pass cache (skip forward)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of batches for voxel generation (default: 1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List configurations without running benchmarks",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs that already have a cache entry",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )
    return p.parse_args()


def _parse_dtype(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def _parse_channel_pair(s: str) -> Tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected C_in,C_out, got '{s}'")
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# Voxel generation
# ---------------------------------------------------------------------------

def make_voxels(
    num_voxels: int,
    num_channels: int,
    dtype: torch.dtype,
    device: str,
    batch_size: int = 1,
) -> "Voxels":
    """Create a Voxels object with approximately `num_voxels` unique voxels."""
    from warpconvnet.geometry.types.voxels import Voxels

    # Determine grid extent so that random sampling yields ~num_voxels unique
    # voxels per batch element after dedup.  Oversample by 1.3x to account for
    # collisions.
    n_per_batch = num_voxels // batch_size
    oversample = int(n_per_batch * 1.3)
    grid_extent = int(math.ceil(n_per_batch ** (1.0 / 3.0))) * 2

    coords_list = []
    feats_list = []
    for _ in range(batch_size):
        coords = torch.randint(0, grid_extent, (oversample, 3), dtype=torch.int32)
        feats = torch.randn(oversample, num_channels, dtype=dtype)
        coords_list.append(coords)
        feats_list.append(feats)

    voxels = Voxels(coords_list, feats_list, device=device).unique()
    return voxels


# ---------------------------------------------------------------------------
# Cache check helpers
# ---------------------------------------------------------------------------

def _config_is_cached(
    namespace: str,
    num_in: int,
    num_out: int,
    c_in: int,
    c_out: int,
    kv: int,
    dtype: torch.dtype,
) -> bool:
    """Check if a SpatiallySparseConvConfig already has a cache entry."""
    from warpconvnet.utils.benchmark_cache import (
        SpatiallySparseConvConfig,
        get_generic_benchmark_cache,
    )

    config = SpatiallySparseConvConfig(
        num_in_coords=num_in,
        num_out_coords=num_out,
        in_channels=c_in,
        out_channels=c_out,
        kernel_volume=kv,
        in_dtype=dtype,
    )
    cache = get_generic_benchmark_cache()
    ns = cache.get_namespace(namespace)
    return config in ns


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_single_config(
    num_voxels: int,
    c_in: int,
    c_out: int,
    kernel_size: int,
    dtype: torch.dtype,
    algo_mode: str,
    device: str,
    batch_size: int,
    do_forward: bool,
    do_backward: bool,
    resume: bool,
) -> Dict[str, Optional[float]]:
    """Run forward and/or backward for one configuration, returning timing info."""
    from warpconvnet.nn.modules.sparse_conv import SpatiallySparseConv

    kernel_volume = kernel_size ** 3
    result: Dict[str, Optional[float]] = {"forward_ms": None, "backward_ms": None}

    # Check cache for resume mode
    if resume:
        fwd_cached = _config_is_cached(
            "sparse_conv_forward", num_voxels, num_voxels, c_in, c_out, kernel_volume, dtype
        )
        bwd_cached = _config_is_cached(
            "sparse_conv_backward", num_voxels, num_voxels, c_in, c_out, kernel_volume, dtype
        )
        if (not do_forward or fwd_cached) and (not do_backward or bwd_cached):
            return result

    # Generate voxels with c_in features
    voxels = make_voxels(num_voxels, c_in, dtype, device, batch_size)

    # Build conv layer
    conv = SpatiallySparseConv(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel_size,
        bias=False,
        fwd_algo=algo_mode,
        bwd_algo=algo_mode,
    ).to(device=device, dtype=dtype)

    # Forward
    if do_forward:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = conv(voxels)
        torch.cuda.synchronize(device)
        result["forward_ms"] = (time.perf_counter() - t0) * 1000

    # Backward
    if do_backward and not (do_forward is False):
        if result["forward_ms"] is None:
            # Need to run forward first to get output for backward
            out = conv(voxels)
        loss = out.feature_tensor.sum()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        result["backward_ms"] = (time.perf_counter() - t0) * 1000

    return result


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.", file=sys.stderr)
        sys.exit(1)

    # Apply presets then overrides
    if args.preset == "quick":
        num_voxels_list = NUM_VOXELS_QUICK
        channel_pairs = CHANNEL_PAIRS_QUICK
        kernel_sizes = KERNEL_SIZES_QUICK
        dtypes = DTYPES_QUICK
    else:
        num_voxels_list = NUM_VOXELS_DEFAULT
        channel_pairs = CHANNEL_PAIRS_COMMON
        kernel_sizes = KERNEL_SIZES_DEFAULT
        dtypes = DTYPES_DEFAULT

    if args.num_voxels is not None:
        num_voxels_list = args.num_voxels
    if args.channels is not None:
        channel_pairs = [_parse_channel_pair(s) for s in args.channels]
    if args.kernel_sizes is not None:
        kernel_sizes = args.kernel_sizes
    if args.dtypes is not None:
        dtypes = [_parse_dtype(d) for d in args.dtypes]

    do_forward = not args.backward_only
    do_backward = not args.forward_only

    # Set algo mode via environment if not "auto"
    if args.algo_mode != "auto":
        os.environ["WARPCONVNET_FWD_ALGO_MODE"] = args.algo_mode
        os.environ["WARPCONVNET_BWD_ALGO_MODE"] = args.algo_mode

    # Build config grid
    configs: List[Tuple[int, int, int, int, torch.dtype]] = []
    for nv, (c_in, c_out), ks, dt in itertools.product(
        num_voxels_list, channel_pairs, kernel_sizes, dtypes
    ):
        configs.append((nv, c_in, c_out, ks, dt))

    # Deduplicate by cache key (log2(N), c_in, c_out, kv, dtype)
    seen = set()
    unique_configs = []
    for nv, c_in, c_out, ks, dt in configs:
        log_n = math.ceil(math.log2(nv)) if nv > 0 else 0
        kv = ks ** 3
        key = (log_n, c_in, c_out, kv, dt)
        if key not in seen:
            seen.add(key)
            unique_configs.append((nv, c_in, c_out, ks, dt))
    configs = unique_configs

    # Print summary
    dev = torch.device(args.device)
    sm = torch.cuda.get_device_capability(dev)
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"GPU: {gpu_name} (SM {sm[0]}.{sm[1]})")
    print(f"Algo mode: {args.algo_mode}")
    print(f"Directions: {'fwd' if do_forward else ''}{'+bwd' if do_backward else ''}")
    print(f"Configs: {len(configs)} unique (after log2-dedup)")
    print(f"  Voxel counts: {sorted(set(nv for nv, *_ in configs))}")
    print(f"  Channel pairs: {sorted(set((c, co) for _, c, co, *_ in configs))}")
    print(f"  Kernel sizes: {sorted(set(ks for *_, ks, _ in configs))}")
    print(f"  Dtypes: {sorted(set(str(dt) for *_, dt in configs))}")
    print()

    if args.dry_run:
        print("Dry run -- listing all configurations:")
        for i, (nv, c_in, c_out, ks, dt) in enumerate(configs, 1):
            log_n = math.ceil(math.log2(nv)) if nv > 0 else 0
            print(f"  [{i:4d}] N={nv:>9,d} (log2={log_n:2d})  "
                  f"C={c_in:3d}->{c_out:3d}  ks={ks}  dtype={dt}")
        return

    # Run benchmarks
    total = len(configs)
    num_done = 0
    num_skipped = 0
    start_time = time.time()

    for i, (nv, c_in, c_out, ks, dt) in enumerate(configs, 1):
        log_n = math.ceil(math.log2(nv)) if nv > 0 else 0
        tag = (f"[{i:4d}/{total}] N={nv:>9,d} (log2={log_n:2d}) "
               f"C={c_in:3d}->{c_out:3d} ks={ks} {dt}")

        try:
            result = run_single_config(
                num_voxels=nv,
                c_in=c_in,
                c_out=c_out,
                kernel_size=ks,
                dtype=dt,
                algo_mode=args.algo_mode,
                device=args.device,
                batch_size=args.batch_size,
                do_forward=do_forward,
                do_backward=do_backward,
                resume=args.resume,
            )

            if result["forward_ms"] is None and result["backward_ms"] is None:
                if args.resume:
                    num_skipped += 1
                    print(f"{tag}  [cached, skipped]")
                    continue

            parts = []
            if result["forward_ms"] is not None:
                parts.append(f"fwd={result['forward_ms']:8.2f}ms")
            if result["backward_ms"] is not None:
                parts.append(f"bwd={result['backward_ms']:8.2f}ms")
            print(f"{tag}  {'  '.join(parts)}")
            num_done += 1

        except Exception as e:
            print(f"{tag}  ERROR: {e}", file=sys.stderr)

        # Free GPU memory between configs
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print()
    print(f"Done: {num_done} benchmarked, {num_skipped} skipped "
          f"(resume={args.resume}) in {elapsed:.1f}s")

    # Print cache location
    from warpconvnet.constants import WARPCONVNET_BENCHMARK_CACHE_DIR
    cache_dir = os.path.expanduser(WARPCONVNET_BENCHMARK_CACHE_DIR)
    cache_file = os.path.join(cache_dir, "benchmark_cache_generic.msgpack")
    if os.path.exists(cache_file):
        size_kb = os.path.getsize(cache_file) / 1024
        print(f"Cache file: {cache_file} ({size_kb:.1f} KB)")
    else:
        print(f"Cache file: {cache_file} (not yet written -- background saver pending)")


if __name__ == "__main__":
    main()
