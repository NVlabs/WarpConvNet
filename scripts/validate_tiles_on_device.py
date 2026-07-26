# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerically validate every device-authorized mask_gemm tile.

Sweeps each tile the current device is authorized to launch across a set of
problem shapes and compares against an fp64 explicit-GEMM reference. Each check
is repeated so intermittent (racy) tiles are caught rather than sampled once.

This is the bring-up gate for a new architecture and for any new warpgemm
snapshot import. Two properties matter and are easy to lose:

* **Scale.** The corruption modes found on Blackwell need ~200k rows to appear;
  the repo's other numeric tests run at N~8k and see nothing. Do not lower
  ``--n`` to make this faster.
* **Repeats.** Some tiles fail 1 run in 3. A single capture proves nothing, and
  the autotuner's numeric self-check validates only one run — so a racy tile can
  pass autotune and corrupt later steps.

    python scripts/validate_tiles_on_device.py                    # forward
    python scripts/validate_tiles_on_device.py --op dgrad
    python scripts/validate_tiles_on_device.py --op all --repeats 10
"""

import argparse
import itertools

import torch

import warpconvnet  # noqa: F401
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.detail import tile_metadata as tm
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import (
    _execute_backward,
    _execute_forward,
)
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_backward_logic,
    _explicit_gemm_forward_logic,
)
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
)

DEV = "cuda"
KERNEL_SIZE = (3, 3, 3)
KERNEL_VOLUME = 27
OPS = ("forward", "dgrad", "wgrad")
# ones x ones inputs make every reference element an exact small integer, so any
# deviation is a real defect rather than fp16 rounding. Gradients use a
# row-varying pattern instead: a constant grad_output cannot detect a scatter
# that lands on the wrong row.
RTOL = 8e-3


def build(n, c_in, c_out):
    """Problem + fp64 references for all three ops at one (C_in, C_out)."""
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
    w = torch.ones(KERNEL_VOLUME, c_in, c_out, device=DEV, dtype=torch.float32)
    # Row-varying grad_output: a constant grad cannot detect a scatter landing on
    # the wrong row. Scaled by 1/n_out because wgrad accumulates one term per
    # matched row — at N=200k with unit inputs the true result is ~1e5, which
    # overflows fp16 (max 65504) in EVERY tile. That is correct arithmetic, not a
    # kernel defect, and without this scale the sweep reports inf for all 16
    # wgrad tiles and hides whatever real failures are in there.
    row = (torch.arange(n_out, device=DEV, dtype=torch.float32) % 8) / 8.0
    col = torch.arange(c_out, device=DEV, dtype=torch.float32) / max(c_out, 1)
    g = (row.unsqueeze(1) + col.unsqueeze(0)) / 2.0 * (4096.0 / max(n_out, 1))

    ref_fwd = _explicit_gemm_forward_logic(x.double(), w.double(), kmap, n_out, torch.float64)
    ref_din, ref_dw = _explicit_gemm_backward_logic(
        g.double(), x.double(), w.double(), kmap, torch.float64, torch.device(DEV)
    )
    return {
        "x": x,
        "w": w,
        "g": g,
        "kmap": kmap,
        "n_out": n_out,
        "forward": ref_fwd,
        "dgrad": ref_din,
        "wgrad": ref_dw,
    }


def run_tile(op, tile_id, p):
    """Execute one tile for ``op``. Returns the output tensor, or None if the
    tile refuses this shape (not a correctness fault)."""
    if op == "forward":
        out = _execute_forward(
            algo="mask_gemm",
            params={"tile_id": tile_id},
            in_features=p["x"].half(),
            weight=p["w"].half(),
            kernel_map=p["kmap"],
            num_out_coords=p["n_out"],
            compute_dtype=torch.float16,
            fwd_block_size=None,
        )
        return out if torch.is_tensor(out) else None

    needs = (True, False) if op == "dgrad" else (False, True)
    grad_in, grad_w = _execute_backward(
        algo="mask_gemm",
        params={"tile_id": tile_id},
        grad_output=p["g"].half(),
        in_features=p["x"].half(),
        weight=p["w"].half(),
        kernel_map=p["kmap"],
        num_out_coords=p["n_out"],
        compute_dtype=torch.float16,
        device=torch.device(DEV),
        needs_input_grad=needs,
    )
    out = grad_in if op == "dgrad" else grad_w
    return out if torch.is_tensor(out) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--op", default="forward", choices=(*OPS, "all"))
    p.add_argument("--channels", default="32:32,32:64,64:64,64:128,128:128,96:96")
    p.add_argument(
        "--tiles", default="", help="comma-separated tile_ids (default: all authorized)"
    )
    args = p.parse_args()

    ops = OPS if args.op == "all" else (args.op,)
    channels = [tuple(int(v) for v in pair.split(":")) for pair in args.channels.split(",")]
    only = {int(t) for t in args.tiles.split(",")} if args.tiles else None
    arch = tm._get_device_arch()

    problems = {cc: build(args.n, *cc) for cc in channels}

    failures = 0
    for op in ops:
        tiles = sorted(t.tile_id for t in tm._get_tiles(op, filter_arch=True))
        if only is not None:
            tiles = [t for t in tiles if t in only]
        print(
            f"\ndevice sm_{arch} | op={op}: {len(tiles)} authorized tiles "
            f"x {len(channels)} shapes x {args.repeats} repeats, N={args.n}"
        )
        bad = {}
        # A tile that refuses a shape is not a correctness fault, but it IS a
        # coverage hole: at C=64->64 on sm_103, 51 of 63 authorized forward
        # tiles raise "Unsupported tile_id" from the binding. Silently folding
        # those into a PASS makes this gate report far more coverage than it
        # actually has, so count and print them.
        checked = set()
        refused = set()
        for tile_id, cc in itertools.product(tiles, channels):
            prob = problems[cc]
            ref = prob[op]
            ref_max = ref.abs().max().item()
            if ref_max <= 0:
                continue
            worst, runs_wrong, ran = 0.0, 0, False
            for _ in range(args.repeats):
                try:
                    out = run_tile(op, tile_id, prob)
                except RuntimeError:
                    out = None
                if out is None:
                    break
                ran = True
                rel = (out.float() - ref.float()).abs().max().item() / ref_max
                worst = max(worst, rel)
                runs_wrong += rel > RTOL
            (checked if ran else refused).add(tile_id)
            if ran and worst > RTOL:
                bad.setdefault(tile_id, []).append((*cc, worst, runs_wrong))

        refused_only = sorted(refused - checked)
        cover = f"{len(checked)}/{len(tiles)} tiles actually executed"
        if refused_only:
            cover += (
                f"; {len(refused_only)} refused every shape (binding reports "
                f"unsupported tile_id — NOT validated): {refused_only}"
            )
        print(f"  coverage: {cover}")
        if not bad:
            print(f"  PASS — every executed {op} tile correct on every shape")
            continue
        failures += len(bad)
        print(f"  FAIL — {len(bad)} of {len(checked)} executed {op} tiles produce wrong results:")
        for tile_id, rows in sorted(bad.items()):
            meta = tm._metadata_index(op).get(tile_id)
            name = meta.kernel_struct if meta is not None else "<not in metadata>"
            # A tile wrong on every repeat is deterministic; wrong on some is a race.
            kind = "DETERMINISTIC" if all(r == args.repeats for *_, r in rows) else "INTERMITTENT"
            print(f"    tile {tile_id:>4}  {name}   [{kind}]")
            for c_in, c_out, rel, runs_wrong in rows:
                print(
                    f"        C {c_in:>4}->{c_out:<4} max_rel={rel:<10.4g} "
                    f"wrong in {runs_wrong}/{args.repeats} runs"
                )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
