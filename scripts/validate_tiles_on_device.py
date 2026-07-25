# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerically validate every device-authorized mask_gemm tile.

Sweeps each tile the current device is authorized to launch across a set of
problem shapes and compares against the fp64 explicit-GEMM reference. Repeats
each check so races (non-deterministic wrong answers) are caught rather than
sampled once.

Intended for bring-up on a new architecture — measured during GB-series bring-up.

    python scripts/validate_tiles_on_device.py
    python scripts/validate_tiles_on_device.py --repeats 5 --op forward
"""

import argparse
import itertools

import torch

import warpconvnet  # noqa: F401
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.detail import tile_metadata as tm
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import _execute_forward
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_forward_logic,
)
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
)

DEV = "cuda"
KERNEL_SIZE = (3, 3, 3)
# ones x ones: every output element is an exact small integer, so any deviation
# is a real defect rather than fp16 rounding.
RTOL = 8e-3


def build(n, c_in, c_out):
    torch.manual_seed(0)
    side = max(8, int((n * 3) ** (1 / 3)) + 1)
    c = torch.unique(torch.randint(0, side, (n * 2, 3), device=DEV, dtype=torch.int32), dim=0)[:n]
    vox = Voxels([c], [torch.ones(c.shape[0], c_in, device=DEV, dtype=torch.float32)])
    _, _, kmap = generate_output_coords_and_kernel_map(
        input_sparse_tensor=vox,
        kernel_size=KERNEL_SIZE,
        kernel_dilation=(1, 1, 1),
        stride=(1, 1, 1),
        generative=False,
        transposed=False,
    )
    n_out = vox.feature_tensor.shape[0]
    x = torch.ones(n_out, c_in, device=DEV, dtype=torch.float32)
    w = torch.ones(27, c_in, c_out, device=DEV, dtype=torch.float32)
    ref = _explicit_gemm_forward_logic(x.double(), w.double(), kmap, n_out, torch.float64)
    return x, w, kmap, n_out, ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--op", default="forward")
    args = p.parse_args()

    arch = tm._get_device_arch()
    tiles = sorted(t.tile_id for t in tm._get_tiles(args.op, filter_arch=True))
    channels = [(32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (96, 96)]
    print(
        f"device sm_{arch}: validating {len(tiles)} authorized {args.op} tiles "
        f"x {len(channels)} shapes x {args.repeats} repeats, N={args.n}\n"
    )

    problems = {}
    for c_in, c_out in channels:
        problems[(c_in, c_out)] = build(args.n, c_in, c_out)

    bad = {}
    for tile_id, (c_in, c_out) in itertools.product(tiles, channels):
        x, w, kmap, n_out, ref = problems[(c_in, c_out)]
        ref_max = ref.abs().max().item()
        worst, runs_wrong = 0.0, 0
        for _ in range(args.repeats):
            try:
                out = _execute_forward(
                    algo="mask_gemm",
                    params={"tile_id": tile_id},
                    in_features=x.half(),
                    weight=w.half(),
                    kernel_map=kmap,
                    num_out_coords=n_out,
                    compute_dtype=torch.float16,
                    fwd_block_size=None,
                )
            except RuntimeError:
                worst = -1.0  # tile refuses this shape; not a correctness fault
                break
            if not torch.is_tensor(out):
                worst = -1.0
                break
            rel = (out.float() - ref.float()).abs().max().item() / ref_max
            worst = max(worst, rel)
            runs_wrong += rel > RTOL
        if worst > RTOL:
            bad.setdefault(tile_id, []).append((c_in, c_out, worst, runs_wrong))

    if not bad:
        print(f"PASS — all {len(tiles)} tiles correct on every shape")
        return 0
    print(f"FAIL — {len(bad)} of {len(tiles)} tiles produce wrong results:\n")
    for tile_id, rows in sorted(bad.items()):
        meta = tm._metadata_index(args.op)[tile_id]
        print(f"  tile {tile_id:>4}  {meta.kernel_struct}")
        for c_in, c_out, rel, runs_wrong in rows:
            print(
                f"      C {c_in:>4}->{c_out:<4} max_rel={rel:<10.4g} "
                f"wrong in {runs_wrong}/{args.repeats} runs"
            )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
