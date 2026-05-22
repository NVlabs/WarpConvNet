# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-algorithm gradient-distribution sweep for sparse conv backward path.

For each algorithm and a set of MinkUNet-realistic shapes, runs fp16 fwd+bwd
with that algo pinned (via direct dispatch — bypasses autotune) and compares
grad_in / grad_w against a fp64 explicit_gemm reference.

Reports max/p99/mean relative diff per (algo, shape). Algos with anomalous
distribution at training-relevant shapes are candidates for regression cause.

Usage:
    cd /home/cchoy/projects/warpconvnet
    source .venv/bin/activate
    python scripts/per_algo_grad_sweep.py
"""

from __future__ import annotations

import argparse
import sys

import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
)
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
)
from warpconvnet.nn.functional.sparse_conv.detail.dispatch import (
    _execute_forward,
    _execute_backward,
)


DEVICE = "cuda"
KERNEL_SIZE = (3, 3, 3)
K_VOLUME = 27


# (N, C_in, C_out, label) — MinkUNet stem→encoder→bottleneck layer shapes.
SHAPES = [
    (2_000, 16, 16, "stem_c16"),
    (2_000, 32, 32, "enc_c32"),
    (12_000, 64, 64, "enc_c64"),
    (12_000, 128, 128, "mid_c128"),
    (48_000, 256, 256, "bot_c256"),
    (48_000, 512, 512, "bot_c512"),
]


# (algo, params, label) — pinned algos with explicit tile_ids where applicable.
# tile_id picks: 41 = mask_gemm baseline native fwd (Tile64x64x32_F16Accum)
# mma_tile=3 = cute_grouped Tile64x64x32; saturation_m default for cutlass_grouped
def algo_specs():
    yield ("explicit_gemm", {}, "explicit_gemm")
    yield ("implicit_gemm", {"fwd_block_size": 16}, "implicit_gemm_bs16")
    yield ("cutlass_implicit_gemm", {}, "cutlass_implicit_gemm")
    yield ("cutlass_grouped_hybrid", {"saturation_m": 5000}, "cutlass_grouped_hybrid")
    yield ("cute_grouped", {"mma_tile": 3}, "cute_grouped_64x64")
    yield ("cute_grouped", {"mma_tile": 0}, "cute_grouped_128x128")
    yield ("mask_gemm", {"tile_id": 41}, "mask_gemm_tile41_F16Acc")
    yield ("mask_gemm", {"tile_id": 3}, "mask_gemm_tile3_64x128_3s")
    yield ("mask_gemm", {"tile_id": 2}, "mask_gemm_tile2_128x64_2s")
    yield ("mask_gemm", {"tile_id": 54}, "mask_gemm_pcoff54_F16Acc")
    yield ("mask_gemm", {"tile_id": 56}, "mask_gemm_pcoff56_F16K8")
    yield ("mask_gemm", {"tile_id": 58}, "mask_gemm_pcoff58_F32")
    yield ("mask_gemm", {"tile_id": 63}, "mask_gemm_pcoff63_F32")
    yield ("mask_gemm_fwd_as_dgrad", {"tile_id": 901}, "mask_gemm_dgrad_wt_901_F32")
    yield ("mask_gemm_fwd_as_dgrad", {"tile_id": 905}, "mask_gemm_dgrad_wt_905_F16Acc")


# (wgrad-algo, params, label)
def wgrad_specs():
    yield ("explicit_gemm", {}, "explicit_gemm")
    yield ("cute_grouped", {"mma_tile": 3}, "cute_grouped_64x64")
    yield ("cute_grouped", {"mma_tile": 0}, "cute_grouped_128x128")
    yield ("mask_gemm", {"tile_id": 0, "split_k": 1}, "mask_gemm_wgrad_0_sk1")
    yield ("mask_gemm", {"tile_id": 4, "split_k": 32}, "mask_gemm_wgrad_4_sk32")
    yield ("mask_gemm", {"tile_id": 7, "split_k": 64}, "mask_gemm_wgrad_7_sk64")
    yield ("mask_gemm", {"tile_id": 1, "split_k": 16}, "mask_gemm_wgrad_1_sk16_ws")


def grid_coords(N: int, device=DEVICE) -> torch.Tensor:
    extent = max(int(round((N * 8) ** (1.0 / 3.0))) + 4, 8)
    g = torch.Generator(device=device).manual_seed(0xC0DE ^ N)
    return torch.randint(
        0, extent, (int(N * 1.15), 3), device=device, dtype=torch.int32, generator=g
    )


def build_voxels(N: int):
    coords = grid_coords(N)
    n = coords.shape[0]
    feats = torch.zeros(n, 1, device=DEVICE, dtype=torch.float32)
    offsets = torch.tensor([0, n], dtype=torch.int32, device=DEVICE)
    v = Voxels(
        batched_coordinates=coords,
        batched_features=feats,
        offsets=offsets,
        device=DEVICE,
    ).unique()
    return v, v.feature_tensor.shape[0]


def patterned_in(N: int, C: int, dtype) -> torch.Tensor:
    col = torch.arange(C, device=DEVICE, dtype=dtype) / max(C, 1)
    row = (torch.arange(N, device=DEVICE, dtype=dtype) % 16) / 16.0
    return (row.unsqueeze(1) + col.unsqueeze(0)) / 2.0


def patterned_w(K: int, C_in: int, C_out: int, dtype) -> torch.Tensor:
    rc = torch.arange(K, device=DEVICE, dtype=dtype) / max(K, 1)
    cin = torch.arange(C_in, device=DEVICE, dtype=dtype) / max(C_in, 1)
    cout = torch.arange(C_out, device=DEVICE, dtype=dtype) / max(C_out, 1)
    base = cin.view(1, C_in, 1) + cout.view(1, 1, C_out)
    return (base * (1.0 + rc.view(K, 1, 1))).contiguous() / 4.0


def patterned_go(N: int, C: int, dtype) -> torch.Tensor:
    col = torch.arange(C, device=DEVICE, dtype=dtype) / max(C, 1)
    row = (torch.arange(N, device=DEVICE, dtype=dtype) % 8) / 8.0
    return (row.unsqueeze(1) + col.unsqueeze(0)) / 2.0


def stats(test: torch.Tensor, ref: torch.Tensor) -> dict:
    """Per-element relative-diff stats."""
    diff = (test.float() - ref.float()).abs()
    ref_mag = ref.float().abs() + 1e-12
    rel = diff / ref_mag
    ref_max = ref.float().abs().max().item()
    flat_rel = rel.flatten()
    p99 = torch.quantile(flat_rel, 0.99).item()
    return {
        "max_abs": diff.max().item(),
        "ref_max_abs": ref_max,
        "max_rel": rel.max().item(),
        "p99_rel": p99,
        "mean_rel": rel.mean().item(),
        "has_nan": test.isnan().any().item(),
    }


def build_problem(N_target: int, C_in: int, C_out: int):
    v, N_in = build_voxels(N_target)
    out_coords, _, kmap = generate_output_coords_and_kernel_map(
        v,
        kernel_size=KERNEL_SIZE,
        kernel_dilation=(1, 1, 1),
        stride=(1, 1, 1),
        generative=False,
        transposed=False,
    )
    N_out = out_coords.shape[0]
    in_64 = patterned_in(N_in, C_in, torch.float64)
    w_64 = patterned_w(K_VOLUME, C_in, C_out, torch.float64)
    go_64 = patterned_go(N_out, C_out, torch.float64)
    fwd_ref = _explicit_gemm_forward_logic(in_64, w_64, kmap, N_out, torch.float64)
    gi_ref, gw_ref = _explicit_gemm_backward_logic(go_64, in_64, w_64, kmap, torch.float64, DEVICE)
    return {
        "in_16": in_64.half().contiguous(),
        "w_16": w_64.half().contiguous(),
        "go_16": go_64.half().contiguous(),
        "fwd_ref": fwd_ref,
        "gi_ref": gi_ref,
        "gw_ref": gw_ref,
        "kmap": kmap,
        "N_in": N_in,
        "N_out": N_out,
        "C_in": C_in,
        "C_out": C_out,
    }


def try_fwd(algo, params, d):
    try:
        out = _execute_forward(
            algo=algo,
            params=params,
            in_features=d["in_16"],
            weight=d["w_16"],
            kernel_map=d["kmap"],
            num_out_coords=d["N_out"],
            compute_dtype=torch.float16,
            fwd_block_size=params.get("fwd_block_size"),
        )
        return out, None
    except (RuntimeError, AssertionError, NotImplementedError) as e:
        return None, str(e)[:120]


def try_bwd(algo, params, d):
    try:
        gi, gw = _execute_backward(
            algo=algo,
            params=params,
            grad_output=d["go_16"],
            in_features=d["in_16"],
            weight=d["w_16"],
            kernel_map=d["kmap"],
            num_out_coords=d["N_out"],
            compute_dtype=torch.float16,
            device=DEVICE,
            needs_input_grad=(True, True),
        )
        return gi, gw, None
    except (RuntimeError, AssertionError, NotImplementedError) as e:
        return None, None, str(e)[:120]


def format_stats(s: dict) -> str:
    if s is None:
        return "  SKIP"
    nan = " NAN!" if s.get("has_nan") else ""
    return f"max_rel={s['max_rel']:.2e} p99={s['p99_rel']:.2e} " f"mean={s['mean_rel']:.2e}{nan}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fwd_bwd", "wgrad"], default="fwd_bwd")
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Comma-separated shape labels (default: all)",
    )
    args = parser.parse_args()

    shapes = SHAPES
    if args.shapes:
        wanted = set(args.shapes.split(","))
        shapes = [s for s in SHAPES if s[3] in wanted]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"K={K_VOLUME} (3x3x3)  fp16 inputs vs fp64 explicit_gemm reference")
    print()

    if args.mode == "fwd_bwd":
        print(f"{'algo_label':<38} {'shape':<14} {'kind':<8} {'stats':<60}")
        print("-" * 122)

        for N_tgt, C_in, C_out, label in shapes:
            print(f"# building {label} (N~={N_tgt}, C_in={C_in}, C_out={C_out})", file=sys.stderr)
            d = build_problem(N_tgt, C_in, C_out)
            shape_label = f"{label}_N={d['N_in']}"

            for algo, params, algo_label in algo_specs():
                if algo == "mask_gemm_fwd_as_dgrad":
                    # No fwd analog; only dgrad makes sense.
                    out, err = None, "no-fwd-path"
                else:
                    out, err = try_fwd(algo, params, d)
                if out is not None:
                    s = stats(out, d["fwd_ref"])
                    print(f"{algo_label:<38} {shape_label:<14} {'fwd':<8} {format_stats(s)}")
                else:
                    print(f"{algo_label:<38} {shape_label:<14} {'fwd':<8} SKIP: {err}")

                gi, gw, err = try_bwd(algo, params, d)
                if gi is not None:
                    sgi = stats(gi, d["gi_ref"])
                    sgw = stats(gw, d["gw_ref"])
                    print(f"{'':38} {'':14} {'grad_in':<8} {format_stats(sgi)}")
                    print(f"{'':38} {'':14} {'grad_w':<8} {format_stats(sgw)}")
                else:
                    print(f"{'':38} {'':14} {'grad':<8} SKIP: {err}")
            print()

    elif args.mode == "wgrad":
        print(f"{'wgrad_label':<38} {'shape':<14} {'stats':<60}")
        print("-" * 112)
        for N_tgt, C_in, C_out, label in shapes:
            print(f"# building {label}", file=sys.stderr)
            d = build_problem(N_tgt, C_in, C_out)
            shape_label = f"{label}_N={d['N_in']}"
            for algo, params, algo_label in wgrad_specs():
                try:
                    gi, gw = _execute_backward(
                        algo=algo,
                        params=params,
                        grad_output=d["go_16"],
                        in_features=d["in_16"],
                        weight=d["w_16"],
                        kernel_map=d["kmap"],
                        num_out_coords=d["N_out"],
                        compute_dtype=torch.float16,
                        device=DEVICE,
                        needs_input_grad=(False, True),
                    )
                    s = stats(gw, d["gw_ref"])
                    print(f"{algo_label:<38} {shape_label:<14} {format_stats(s)}")
                except (RuntimeError, AssertionError, NotImplementedError) as e:
                    print(f"{algo_label:<38} {shape_label:<14} SKIP: {str(e)[:80]}")
            print()


if __name__ == "__main__":
    main()
