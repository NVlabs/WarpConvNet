# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch exactly one mask_gemm tile, for running under compute-sanitizer.

Keeps the launch count minimal so `compute-sanitizer --tool racecheck` finishes
in reasonable time — racecheck instruments every shared-memory access and is
~100x slower than a bare run. Shared-memory hazards are reported from the
instrumented accesses themselves, so a race is flagged whether or not it
happened to corrupt the output on this run; that is the point of using it on a
tile whose numeric failure is intermittent.

    compute-sanitizer --tool racecheck --racecheck-detect-level info \
        python scripts/sanitize_tile.py --op forward --tile 2 --cin 64 --cout 64
"""

import argparse

import torch

import warpconvnet  # noqa: F401
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import (
    _execute_backward,
    _execute_forward,
)
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
)

DEV = "cuda"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="forward", choices=("forward", "dgrad", "wgrad"))
    p.add_argument("--tile", type=int, required=True)
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--cout", type=int, default=64)
    # Small by default: racecheck's slowdown makes 200k impractical, and a
    # shared-memory hazard does not need scale to be *detected*, only to be
    # *observed* in the output.
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--launches", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(0)
    side = max(8, int((args.n * 3) ** (1 / 3)) + 1)
    c = torch.unique(
        torch.randint(0, side, (args.n * 2, 3), device=DEV, dtype=torch.int32), dim=0
    )[: args.n]
    vox = Voxels([c], [torch.ones(c.shape[0], args.cin, device=DEV, dtype=torch.float32)])
    _, _, kmap = generate_output_coords_and_kernel_map(
        input_sparse_tensor=vox,
        kernel_size=(3, 3, 3),
        kernel_dilation=(1, 1, 1),
        stride=(1, 1, 1),
        generative=False,
        transposed=False,
    )
    n_out = vox.feature_tensor.shape[0]
    x = torch.ones(n_out, args.cin, device=DEV, dtype=torch.float16)
    w = torch.ones(27, args.cin, args.cout, device=DEV, dtype=torch.float16)
    row = (torch.arange(n_out, device=DEV, dtype=torch.float32) % 8) / 8.0
    col = torch.arange(args.cout, device=DEV, dtype=torch.float32) / max(args.cout, 1)
    g = ((row.unsqueeze(1) + col.unsqueeze(0)) / 2.0 * (4096.0 / max(n_out, 1))).half()

    torch.cuda.synchronize()
    for _ in range(args.launches):
        if args.op == "forward":
            _execute_forward(
                algo="mask_gemm",
                params={"tile_id": args.tile},
                in_features=x,
                weight=w,
                kernel_map=kmap,
                num_out_coords=n_out,
                compute_dtype=torch.float16,
                fwd_block_size=None,
            )
        else:
            needs = (True, False) if args.op == "dgrad" else (False, True)
            _execute_backward(
                algo="mask_gemm",
                params={"tile_id": args.tile},
                grad_output=g,
                in_features=x,
                weight=w,
                kernel_map=kmap,
                num_out_coords=n_out,
                compute_dtype=torch.float16,
                device=torch.device(DEV),
                needs_input_grad=needs,
            )
    torch.cuda.synchronize()
    print(f"OK {args.op} tile={args.tile} C={args.cin}->{args.cout} N_out={n_out}")


if __name__ == "__main__":
    main()
