# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive per-tile correctness sweep for sparse-conv kernels.

Enumerates every (algo, tile_id, split_k) entry in the autotune candidate pools
(_AB_MASK_GEMM, _AB_MASK_GEMM_STRIDED_F32ACC, _AB_MASK_GEMM_FWD_AS_DGRAD,
_AB_MASK_GEMM_DGRAD_PCOFF, _ATB_MASK_GEMM, plus cutlass / cute_grouped /
explicit), runs fp16 fwd+bwd at MinkUNet-realistic shapes, and compares
against the fp64 explicit_gemm reference.

Surfaces silently-wrong kernels that pass the loose 8e-3 tolerance in
test_mask_gemm_all_tiles. Per-tile rel_diff threshold tightened to 3e-3
for F32-accumulator tiles and 1.5e-1 for F16-accumulator / pcoff tiles.

Coverage knobs:
  --kernel-size 3 (default 3x3x3, K=27)  --kernel-size 2 (2x2x2, K=8)
  --stride 1 (submanifold)  --stride 2 (native downsample, fires 300-307)
  --shapes <subset>           --large (bump N x 4)
  --algos fwd|dgrad|wgrad|all

Usage:
    source .venv/bin/activate
    python scripts/exhaustive_kernel_correctness.py --stride 2 --kernel-size 2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.sparse_conv.helper import (
    generate_output_coords_and_kernel_map,
)
from warpconvnet.nn.functional.sparse_conv.detail.algo_params import (
    _AB_MASK_GEMM_F32ACC,
    _AB_MASK_GEMM_F16ACC,
    _AB_MASK_GEMM_PCOFF_F32ACC,
    _AB_MASK_GEMM_PCOFF_F16ACC,
    _AB_MASK_GEMM_STRIDED_F32ACC,
    _AB_MASK_GEMM_FWD_AS_DGRAD_F32ACC,
    _AB_MASK_GEMM_FWD_AS_DGRAD_F16ACC,
    _AB_MASK_GEMM_FWD_AS_DGRAD_PCOFF_F32ACC,
    _AB_MASK_GEMM_FWD_AS_DGRAD_PCOFF_F16ACC,
    _AB_MASK_GEMM_DGRAD_PCOFF_F32ACC,
    _AB_MASK_GEMM_DGRAD_PCOFF_F16ACC,
    _ATB_MASK_GEMM,
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


# (N_target, C_in, C_out, label). Default N values keep the sweep cheap
# (~30s on RTX 6000 Ada with default cache); `--realistic-N` bumps the
# narrow-channel shapes to ScanNet200-realistic counts that surface the
# F16-accumulator pcoff failure mode at (C=32, K=27) — see
# `tests/nn/test_pcoff_f16acc_regression.py`.
# Memory cost is dominated by the fp64 reference; the realistic preset still
# fits comfortably in a 24 GB GPU.
DEFAULT_SHAPES = [
    (2_000, 16, 16, "stem_c16"),
    (2_000, 32, 32, "enc_c32"),
    (12_000, 64, 64, "enc_c64"),
    (12_000, 128, 128, "mid_c128"),
    (48_000, 256, 256, "bot_c256"),
    (48_000, 512, 512, "bot_c512"),
]

REALISTIC_SHAPES = [
    (250_000, 16, 16, "stem_c16"),
    (250_000, 32, 32, "enc_c32"),
    (200_000, 64, 64, "enc_c64"),
    (100_000, 128, 128, "mid_c128"),
    (50_000, 256, 256, "bot_c256"),
    (25_000, 512, 512, "bot_c512"),
]


# Tile families and their tolerances. F16Acc / pcoff tiles have known fp16
# accumulator drift; F32Acc tiles must clear the tighter bound.
_F16ACC_FWD_TILES = frozenset({19, 28})
_PCOFF_FWD_TILES = frozenset({54, 55, 56, 57, 58, 59, 63})
_F16ACC_PCOFF_TILES = frozenset({54, 55, 56, 57})  # F16-accum pcoff
_STRIDED_FWD_TILES = frozenset(range(300, 308))
_F16ACC_WT_TILES = frozenset({903, 904})
_PCOFF_WT_TILES = frozenset({905, 906, 907, 908, 909, 910, 911})
_F16ACC_PCOFF_WT_TILES = frozenset({905, 906, 907, 908})
_DGRAD_PCOFF_TILES = frozenset({64, 65, 66, 67, 68, 69})
_F16ACC_DGRAD_PCOFF_TILES = frozenset({64, 65, 66, 67})


# Per-tile-family pass thresholds (max relative diff vs fp64 ref).
TOL_F32ACC = 3e-3
TOL_F16ACC = 1.5e-1
TOL_WGRAD_F32ACC = 5e-3


def fwd_tol(tile_id: int | None) -> float:
    if tile_id is None:
        return TOL_F32ACC
    if tile_id in _F16ACC_FWD_TILES:
        return TOL_F16ACC
    if tile_id in _F16ACC_PCOFF_TILES:
        return TOL_F16ACC
    if tile_id in _PCOFF_FWD_TILES:
        return TOL_F32ACC  # F32-accum pcoff (58/59/63)
    return TOL_F32ACC


def dgrad_wt_tol(tile_id: int) -> float:
    if tile_id in _F16ACC_WT_TILES or tile_id in _F16ACC_PCOFF_WT_TILES:
        return TOL_F16ACC
    return TOL_F32ACC


def dgrad_native_tol(tile_id: int) -> float:
    if tile_id in _F16ACC_DGRAD_PCOFF_TILES:
        return TOL_F16ACC
    return TOL_F32ACC


def wgrad_tol(tile_id: int) -> float:
    return TOL_WGRAD_F32ACC


@dataclass
class Spec:
    algo: str
    params: dict[str, Any]
    label: str
    tol_fn: Any  # callable(tile_id) -> float


def fwd_specs(stride2: bool) -> Iterable[Spec]:
    yield Spec("explicit_gemm", {}, "explicit_gemm", lambda _: TOL_F32ACC)
    yield Spec("cutlass_implicit_gemm", {}, "cutlass_implicit_gemm", lambda _: TOL_F32ACC)
    yield Spec("cute_grouped", {"mma_tile": 0}, "cute_grouped_128x128", lambda _: TOL_F32ACC)
    yield Spec("cute_grouped", {"mma_tile": 3}, "cute_grouped_64x64", lambda _: TOL_F32ACC)
    for algo, params in _AB_MASK_GEMM_F32ACC:
        yield Spec(algo, params, f"mask_gemm_{params['tile_id']}_F32Acc", fwd_tol)
    for algo, params in _AB_MASK_GEMM_F16ACC:
        yield Spec(algo, params, f"mask_gemm_{params['tile_id']}_F16Acc", fwd_tol)
    for algo, params in _AB_MASK_GEMM_PCOFF_F32ACC:
        yield Spec(algo, params, f"mask_gemm_pcoff_{params['tile_id']}_F32Acc", fwd_tol)
    for algo, params in _AB_MASK_GEMM_PCOFF_F16ACC:
        yield Spec(algo, params, f"mask_gemm_pcoff_{params['tile_id']}_F16Acc", fwd_tol)
    if stride2:
        for algo, params in _AB_MASK_GEMM_STRIDED_F32ACC:
            yield Spec(
                algo, params, f"mask_gemm_strided_{params['tile_id']}", lambda _: TOL_F32ACC
            )


def dgrad_specs() -> Iterable[Spec]:
    yield Spec("explicit_gemm", {}, "explicit_gemm", lambda _: TOL_F32ACC)
    yield Spec("cutlass_implicit_gemm", {}, "cutlass_implicit_gemm", lambda _: TOL_F32ACC)
    # fwd_as_dgrad (canonical 900-911)
    for algo, params in _AB_MASK_GEMM_FWD_AS_DGRAD_F32ACC:
        yield Spec(algo, params, f"fwd_as_dgrad_{params['tile_id']}_F32Acc", dgrad_wt_tol)
    for algo, params in _AB_MASK_GEMM_FWD_AS_DGRAD_F16ACC:
        yield Spec(algo, params, f"fwd_as_dgrad_{params['tile_id']}_F16Acc", dgrad_wt_tol)
    for algo, params in _AB_MASK_GEMM_FWD_AS_DGRAD_PCOFF_F32ACC:
        yield Spec(algo, params, f"fwd_as_dgrad_pcoff_{params['tile_id']}_F32Acc", dgrad_wt_tol)
    for algo, params in _AB_MASK_GEMM_FWD_AS_DGRAD_PCOFF_F16ACC:
        yield Spec(algo, params, f"fwd_as_dgrad_pcoff_{params['tile_id']}_F16Acc", dgrad_wt_tol)
    # Native dgrad pcoff (64-69)
    for algo, params in _AB_MASK_GEMM_DGRAD_PCOFF_F32ACC:
        yield Spec(
            algo, params, f"native_dgrad_pcoff_{params['tile_id']}_F32Acc", dgrad_native_tol
        )
    for algo, params in _AB_MASK_GEMM_DGRAD_PCOFF_F16ACC:
        yield Spec(
            algo, params, f"native_dgrad_pcoff_{params['tile_id']}_F16Acc", dgrad_native_tol
        )


def wgrad_specs() -> Iterable[Spec]:
    yield Spec("explicit_gemm", {}, "explicit_gemm", lambda _: TOL_WGRAD_F32ACC)
    yield Spec("cute_grouped", {"mma_tile": 0}, "cute_grouped_128x128", lambda _: TOL_WGRAD_F32ACC)
    yield Spec("cute_grouped", {"mma_tile": 3}, "cute_grouped_64x64", lambda _: TOL_WGRAD_F32ACC)
    for algo, params in _ATB_MASK_GEMM:
        sk = params.get("split_k", 1)
        yield Spec(algo, params, f"wgrad_{params['tile_id']}_sk{sk}", wgrad_tol)


def grid_coords(N: int, seed: int = 0xC0DE) -> torch.Tensor:
    extent = max(int(round((N * 8) ** (1.0 / 3.0))) + 4, 8)
    g = torch.Generator(device=DEVICE).manual_seed(seed ^ N)
    return torch.randint(
        0, extent, (int(N * 1.15), 3), device=DEVICE, dtype=torch.int32, generator=g
    )


def build_voxels(N: int) -> tuple[Voxels, int]:
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


def stats(test: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (test.float() - ref.float()).abs()
    ref_mag = ref.float().abs() + 1e-12
    rel = diff / ref_mag
    flat_rel = rel.flatten()
    finite_mask = torch.isfinite(flat_rel)
    finite_rel = flat_rel[finite_mask]
    # torch.quantile caps at 16M elements; subsample for larger tensors.
    if finite_rel.numel() > 10_000_000:
        idx = torch.randint(0, finite_rel.numel(), (10_000_000,), device=finite_rel.device)
        finite_rel = finite_rel[idx]
    return {
        "max_rel": finite_rel.max().item() if finite_rel.numel() else float("nan"),
        "p99_rel": torch.quantile(finite_rel, 0.99).item() if finite_rel.numel() else float("nan"),
        "mean_rel": finite_rel.mean().item() if finite_rel.numel() else float("nan"),
        "has_nan": bool(test.isnan().any().item()),
        "has_inf": bool((~torch.isfinite(test)).any().item() and not test.isnan().any().item()),
        "n_nonfinite": int((~finite_mask).sum().item()),
    }


def build_problem(
    N_target: int,
    C_in: int,
    C_out: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
) -> dict[str, Any]:
    v, N_in = build_voxels(N_target)
    out_coords, _, kmap = generate_output_coords_and_kernel_map(
        v,
        kernel_size=kernel_size,
        kernel_dilation=(1,) * len(kernel_size),
        stride=stride,
        generative=False,
        transposed=False,
    )
    N_out = out_coords.shape[0]
    K = 1
    for k in kernel_size:
        K *= k
    in_64 = patterned_in(N_in, C_in, torch.float64)
    w_64 = patterned_w(K, C_in, C_out, torch.float64)
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
        "K": K,
    }


def try_fwd(spec: Spec, d: dict[str, Any]):
    try:
        out = _execute_forward(
            algo=spec.algo,
            params=spec.params,
            in_features=d["in_16"],
            weight=d["w_16"],
            kernel_map=d["kmap"],
            num_out_coords=d["N_out"],
            compute_dtype=torch.float16,
            fwd_block_size=spec.params.get("fwd_block_size"),
        )
        return out, None
    except (RuntimeError, AssertionError, NotImplementedError) as e:
        return None, str(e)[:140]


def try_bwd(spec: Spec, d: dict[str, Any], needs):
    try:
        gi, gw = _execute_backward(
            algo=spec.algo,
            params=spec.params,
            grad_output=d["go_16"],
            in_features=d["in_16"],
            weight=d["w_16"],
            kernel_map=d["kmap"],
            num_out_coords=d["N_out"],
            compute_dtype=torch.float16,
            device=DEVICE,
            needs_input_grad=needs,
        )
        return gi, gw, None
    except (RuntimeError, AssertionError, NotImplementedError) as e:
        return None, None, str(e)[:140]


@dataclass
class SuspectRow:
    shape_label: str
    spec_label: str
    kind: str
    max_rel: float
    p99_rel: float
    tol: float
    nan: bool
    err: str | None = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel-size", type=int, default=3, help="cubic kernel (2 → K=8, 3 → K=27)"
    )
    parser.add_argument("--stride", type=int, default=1, help="conv stride (1 or 2)")
    parser.add_argument("--algos", choices=["fwd", "dgrad", "wgrad", "all"], default="all")
    parser.add_argument("--shapes", type=str, default=None, help="csv subset of shape labels")
    parser.add_argument("--large", action="store_true", help="bump N x 4 for stress")
    parser.add_argument(
        "--realistic-N",
        action="store_true",
        help=(
            "Use ScanNet200-realistic per-shape N counts (e.g. enc_c32 N=250k). "
            "Required to surface the F16-accumulator pcoff saturation failure mode. "
            "Higher memory; runs in ~3 min on a 24 GB GPU."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print every row, not just suspects"
    )
    parser.add_argument(
        "--max-rel-cap", type=float, default=None, help="override pass tolerance globally"
    )
    args = parser.parse_args()

    ks = (args.kernel_size,) * 3
    st = (args.stride,) * 3
    K = ks[0] ** 3

    shapes = REALISTIC_SHAPES if args.realistic_N else DEFAULT_SHAPES
    if args.shapes:
        wanted = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[3] in wanted]
    if args.large:
        shapes = [(N * 4, ci, co, lab) for (N, ci, co, lab) in shapes]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"kernel_size={ks} (K={K}) stride={st}  fp16 vs fp64 explicit_gemm reference")
    print(
        f"tolerances: F32Acc={TOL_F32ACC:.0e}  F16Acc/pcoff_f16={TOL_F16ACC:.0e}  wgrad={TOL_WGRAD_F32ACC:.0e}"
    )
    print()

    suspects: list[SuspectRow] = []
    total = 0
    skipped = 0
    passed = 0

    for N_tgt, C_in, C_out, label in shapes:
        print(f"# building {label} (N~={N_tgt}, C_in={C_in}, C_out={C_out})", file=sys.stderr)
        d = build_problem(N_tgt, C_in, C_out, ks, st)
        shape_label = f"{label}_Nin={d['N_in']}_Nout={d['N_out']}"
        print(f"## {shape_label}")

        # ---- fwd ----
        if args.algos in ("fwd", "all"):
            for spec in fwd_specs(stride2=(args.stride > 1)):
                total += 1
                out, err = try_fwd(spec, d)
                tile_id = spec.params.get("tile_id")
                tol = args.max_rel_cap if args.max_rel_cap else spec.tol_fn(tile_id)
                if out is None:
                    skipped += 1
                    if args.verbose:
                        print(f"  fwd   {spec.label:<40} SKIP {err}")
                    continue
                s = stats(out, d["fwd_ref"])
                row = SuspectRow(
                    shape_label, spec.label, "fwd", s["max_rel"], s["p99_rel"], tol, s["has_nan"]
                )
                # Isolated inf at F16Acc large-C is legit accumulator saturation;
                # NaN or sustained drift (p99) is a real bug.
                bad = s["has_nan"] or s["max_rel"] > tol or s["p99_rel"] > tol
                if bad:
                    suspects.append(row)
                    print(
                        f"  fwd   {spec.label:<40} FAIL max={s['max_rel']:.2e} p99={s['p99_rel']:.2e} (tol={tol:.0e}){' NAN' if s['has_nan'] else ''}"
                    )
                else:
                    passed += 1
                    if args.verbose:
                        print(
                            f"  fwd   {spec.label:<40} ok   max={s['max_rel']:.2e} p99={s['p99_rel']:.2e}"
                        )

        # ---- dgrad ----
        if args.algos in ("dgrad", "all"):
            for spec in dgrad_specs():
                total += 1
                gi, _gw, err = try_bwd(spec, d, needs=(True, False))
                tile_id = spec.params.get("tile_id")
                tol = args.max_rel_cap if args.max_rel_cap else spec.tol_fn(tile_id)
                if gi is None:
                    skipped += 1
                    if args.verbose:
                        print(f"  dgrad {spec.label:<40} SKIP {err}")
                    continue
                s = stats(gi, d["gi_ref"])
                row = SuspectRow(
                    shape_label, spec.label, "dgrad", s["max_rel"], s["p99_rel"], tol, s["has_nan"]
                )
                bad = s["has_nan"] or s["max_rel"] > tol or s["p99_rel"] > tol
                if bad:
                    suspects.append(row)
                    print(
                        f"  dgrad {spec.label:<40} FAIL max={s['max_rel']:.2e} p99={s['p99_rel']:.2e} (tol={tol:.0e}){' NAN' if s['has_nan'] else ''}"
                    )
                else:
                    passed += 1
                    if args.verbose:
                        print(
                            f"  dgrad {spec.label:<40} ok   max={s['max_rel']:.2e} p99={s['p99_rel']:.2e}"
                        )

        # ---- wgrad ----
        if args.algos in ("wgrad", "all"):
            for spec in wgrad_specs():
                total += 1
                _gi, gw, err = try_bwd(spec, d, needs=(False, True))
                tile_id = spec.params.get("tile_id")
                tol = args.max_rel_cap if args.max_rel_cap else spec.tol_fn(tile_id)
                if gw is None:
                    skipped += 1
                    if args.verbose:
                        print(f"  wgrad {spec.label:<40} SKIP {err}")
                    continue
                s = stats(gw, d["gw_ref"])
                row = SuspectRow(
                    shape_label, spec.label, "wgrad", s["max_rel"], s["p99_rel"], tol, s["has_nan"]
                )
                bad = s["has_nan"] or s["max_rel"] > tol or s["p99_rel"] > tol
                if bad:
                    suspects.append(row)
                    print(
                        f"  wgrad {spec.label:<40} FAIL max={s['max_rel']:.2e} p99={s['p99_rel']:.2e} (tol={tol:.0e}){' NAN' if s['has_nan'] else ''}"
                    )
                else:
                    passed += 1
                    if args.verbose:
                        print(
                            f"  wgrad {spec.label:<40} ok   max={s['max_rel']:.2e} p99={s['p99_rel']:.2e}"
                        )
        print()

    print("=" * 80)
    print(f"Summary: {passed}/{total} passed, {skipped} skipped, {len(suspects)} suspect")
    if suspects:
        print()
        print("Suspect tiles:")
        for r in suspects:
            print(
                f"  {r.shape_label:<32} {r.kind:<6} {r.spec_label:<40} max={r.max_rel:.2e} p99={r.p99_rel:.2e} tol={r.tol:.0e}{' NAN' if r.nan else ''}"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
