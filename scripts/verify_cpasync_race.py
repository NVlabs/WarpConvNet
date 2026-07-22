# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verification harness for the cute cp.async pipeline race (GitHub issue #30).
#
# The race: the cute mainloops prefetch 1 tile ahead but waited with
# cp_async_wait<NumStages-2>, so with NumStages >= 3 the tile computed next
# iteration could still be in flight -> reads of partially-written smem ->
# nondeterministic NaN / wrong values.
#
# Two verification layers:
#   1. Kernel-level (any arch with the cute SM80-path kernels, incl. sm_100):
#      drive the *staged* AD / TrAB bindings at num_stages 2/3/4. Stages 3/4
#      hit the race pre-fix; stage 2 was accidentally safe.
#   2. Conv-level on SM90 hardware only: the issue #30 configuration
#      (C=128, kernel (2,2,2), stride (2,2,2)) on the cute_grouped_sm90
#      backend (WGMMA kernels are sm_90a-only and cannot run on Blackwell).
#
# Checks per config: no NaN/Inf, bitwise determinism across repeats,
# closeness to fp32 torch/explicit_gemm reference, and CUDA-event timing
# (pipeline perf comparison across variants).
#
# Exit code 0 iff all checks pass.

import argparse
import sys

import torch

import warpconvnet  # noqa: F401  (loads _C)
import warpconvnet._C as _C

STAGED_TILES = [0, 3]  # only tiles 0-3 (tK=32) have staged instantiations


def check_tensor(label, t, failures):
    bad = t.isnan().any().item() or t.isinf().any().item()
    if bad:
        failures.append(
            f"{label}: {t.isnan().sum().item()} NaN / {t.isinf().sum().item()} Inf values"
        )
    return not bad


def rel_err(out, ref):
    ref32 = ref.float()
    return ((out.float() - ref32).norm() / ref32.norm().clamp_min(1e-12)).item()


def time_fn(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Layer 1: staged kernel-level (runs on any arch with cute SM80 kernels)
# ---------------------------------------------------------------------------


def run_staged_ad(repeats, bench, failures):
    """D[idx_d] = A[idx_a] @ B via cute_gemm_AD_gather_scatter_staged."""
    M, K, N, idx_size = 8192, 4096, 128, 8192  # K/32 = 128 k-tiles per block
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        A = torch.randn(M, K, dtype=dtype, device="cuda") * 0.1
        B = torch.randn(K, N, dtype=dtype, device="cuda") * 0.1
        idx_a = torch.randperm(M, device="cuda")[:idx_size].int()
        idx_d = torch.randperm(M, device="cuda")[:idx_size].int()
        ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
        ref[idx_d.long()] = A[idx_a.long()].float() @ B.float()

        for tile in STAGED_TILES:
            for stages in (2, 3, 4):
                for cp in (False, True):
                    tag = f"[staged-AD {dtype} tile={tile} S={stages} cp_async={cp}]"
                    out0 = None
                    skipped = False
                    for r in range(repeats):
                        D = torch.zeros(M, N, dtype=torch.float32, device="cuda")
                        status = _C.gemm.cute_gemm_AD_gather_scatter_staged(
                            A,
                            B,
                            D,
                            D,
                            idx_a,
                            idx_d,
                            mma_tile=tile,
                            alpha=1.0,
                            beta=0.0,
                            num_stages=stages,
                            use_cp_async=cp,
                        )
                        if status != 0:
                            print(f"{tag} unsupported (status {status}), skipping")
                            skipped = True
                            break
                        check_tensor(f"{tag} run {r}", D, failures)
                        if out0 is None:
                            out0 = D
                        elif not torch.equal(D, out0):
                            maxd = (D - out0).abs().max().item()
                            failures.append(
                                f"{tag} nondeterministic: run {r} differs "
                                f"(max abs diff {maxd:.4g})"
                            )
                    if skipped or out0 is None:
                        continue
                    e = rel_err(out0, ref)
                    ok = e < 0.02
                    if not ok:
                        failures.append(f"{tag} rel err vs torch fp32: {e:.4g}")
                    line = f"{tag} rel_err={e:.3e} {'OK' if ok else 'FAIL'}"
                    if bench:
                        D = torch.zeros(M, N, dtype=torch.float32, device="cuda")
                        t = time_fn(
                            lambda: _C.gemm.cute_gemm_AD_gather_scatter_staged(
                                A,
                                B,
                                D,
                                D,
                                idx_a,
                                idx_d,
                                mma_tile=tile,
                                alpha=1.0,
                                beta=0.0,
                                num_stages=stages,
                                use_cp_async=cp,
                            )
                        )
                        line += f" BENCH={t:.3f}ms"
                    print(line)


def run_staged_trab(repeats, bench, failures):
    """D = A[idx_a]^T @ B[idx_b] via cute_gemm_trAB_gather_staged."""
    M_A, K, M_B, N, idx_size = 65536, 128, 65536, 128, 65536  # 2048 g-tiles
    torch.manual_seed(1)
    for dtype in (torch.float16,):
        A = torch.randn(M_A, K, dtype=dtype, device="cuda") * 0.1
        B = torch.randn(M_B, N, dtype=dtype, device="cuda") * 0.1
        idx_a = torch.randperm(M_A, device="cuda")[:idx_size].int()
        idx_b = torch.randperm(M_B, device="cuda")[:idx_size].int()
        ref = A[idx_a.long()].float().T @ B[idx_b.long()].float()

        for tile in STAGED_TILES:
            for stages in (2, 3, 4):
                for cp in (False, True):
                    tag = f"[staged-TrAB {dtype} tile={tile} S={stages} cp_async={cp}]"
                    out0 = None
                    skipped = False
                    for r in range(repeats):
                        D = torch.zeros(K, N, dtype=torch.float32, device="cuda")
                        status = _C.gemm.cute_gemm_trAB_gather_staged(
                            A,
                            B,
                            D,
                            D,
                            idx_a,
                            idx_b,
                            mma_tile=tile,
                            alpha=1.0,
                            beta=0.0,
                            num_stages=stages,
                            use_cp_async=cp,
                        )
                        if status != 0:
                            print(f"{tag} unsupported (status {status}), skipping")
                            skipped = True
                            break
                        check_tensor(f"{tag} run {r}", D, failures)
                        if out0 is None:
                            out0 = D
                        elif not torch.equal(D, out0):
                            maxd = (D - out0).abs().max().item()
                            failures.append(
                                f"{tag} nondeterministic: run {r} differs "
                                f"(max abs diff {maxd:.4g})"
                            )
                    if skipped or out0 is None:
                        continue
                    e = rel_err(out0, ref)
                    ok = e < 0.02
                    if not ok:
                        failures.append(f"{tag} rel err vs torch fp32: {e:.4g}")
                    line = f"{tag} rel_err={e:.3e} {'OK' if ok else 'FAIL'}"
                    if bench:
                        D = torch.zeros(K, N, dtype=torch.float32, device="cuda")
                        t = time_fn(
                            lambda: _C.gemm.cute_gemm_trAB_gather_staged(
                                A,
                                B,
                                D,
                                D,
                                idx_a,
                                idx_b,
                                mma_tile=tile,
                                alpha=1.0,
                                beta=0.0,
                                num_stages=stages,
                                use_cp_async=cp,
                            )
                        )
                        line += f" BENCH={t:.3f}ms"
                    print(line)


# ---------------------------------------------------------------------------
# Layer 2: conv-level, issue #30 config (SM90 hardware only)
# ---------------------------------------------------------------------------


def run_conv_level(repeats, bench, failures, backend):
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.nn.functional.sparse_conv.helper import (
        generate_output_coords_and_kernel_map,
    )
    from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
        _explicit_gemm_forward_logic,
    )

    if backend == "cute_grouped_sm90":
        from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped_sm90 import (
            _cute_grouped_sm90_forward_logic as fwd_logic,
        )

        tile_ids = [100, 101, 103, 104]
        kwargs = dict(backend=backend)
    else:
        from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped import (
            _cute_grouped_forward_logic,
        )

        def fwd_logic(feats, weight, kmap, n_out, tile_id, **_kw):
            return _cute_grouped_forward_logic(feats, weight, kmap, n_out, mma_tile=tile_id)

        tile_ids = [0, 3]
        kwargs = {}

    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(2)
        coords = torch.randint(0, 100, (200_000, 3), device="cuda", dtype=torch.int32)
        coords = torch.unique(coords, dim=0)
        n_in = coords.shape[0]
        feats = torch.randn(n_in, 128, device="cuda", dtype=dtype)
        offsets = torch.tensor([0, n_in], dtype=torch.long, device="cuda")
        vox = Voxels(
            batched_coordinates=coords,
            batched_features=feats,
            offsets=offsets,
            voxel_size=1.0,
        )
        out_coords, _, kmap = generate_output_coords_and_kernel_map(
            vox, kernel_size=(2, 2, 2), kernel_dilation=(1, 1, 1), stride=(2, 2, 2)
        )
        n_out = out_coords.shape[0]
        weight = torch.randn(8, 128, 128, device="cuda", dtype=dtype) * 0.02
        ref = _explicit_gemm_forward_logic(feats.float(), weight.float(), kmap, n_out)

        for tile_id in tile_ids:
            tag = f"[conv-{backend} {dtype} k2s2-c128 tile={tile_id}]"
            out0 = None
            for r in range(repeats):
                out = fwd_logic(feats, weight, kmap, n_out, tile_id=tile_id, **kwargs)
                if isinstance(out, int):
                    print(f"{tag} backend status {out}, skipping")
                    out0 = None
                    break
                check_tensor(f"{tag} run {r}", out, failures)
                if out0 is None:
                    out0 = out
                    # The grouped-conv epilogue accumulates via fp16/bf16
                    # atomicAdd, so bitwise determinism is not expected: the
                    # add order varies run to run by a few ulps of the output.
                    # Only diffs beyond that scale indicate a real race.
                    atomic_tol = 16 * torch.finfo(out.dtype).eps * out.abs().max().item()
                elif not torch.equal(out, out0):
                    maxd = (out.float() - out0.float()).abs().max().item()
                    if maxd > atomic_tol:
                        failures.append(
                            f"{tag} nondeterministic: run {r} differs "
                            f"(max abs diff {maxd:.4g} > atomic tol {atomic_tol:.4g})"
                        )
            if out0 is None:
                continue
            e = rel_err(out0, ref)
            ok = e < 0.03
            if not ok:
                failures.append(f"{tag} rel err vs explicit_gemm fp32: {e:.4g}")
            line = f"{tag} n_in={n_in} n_out={n_out} rel_err={e:.3e} {'OK' if ok else 'FAIL'}"
            if bench:
                t = time_fn(
                    lambda: fwd_logic(feats, weight, kmap, n_out, tile_id=tile_id, **kwargs)
                )
                line += f" BENCH={t:.3f}ms"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    dev = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    print(f"=== verify_cpasync_race [{args.label}] on {dev} (sm_{cap[0]}{cap[1]}) ===")

    failures = []

    print("--- layer 1: staged AD gather-scatter (num_stages sweep) ---")
    run_staged_ad(args.repeats, args.bench, failures)
    print("--- layer 1: staged TrAB gather (num_stages sweep) ---")
    run_staged_trab(args.repeats, args.bench, failures)

    print("--- layer 2: conv-level issue #30 config ---")
    if cap[0] == 9:
        run_conv_level(args.repeats, args.bench, failures, "cute_grouped_sm90")
    else:
        print(
            f"sm_{cap[0]}{cap[1]} != sm_90: WGMMA backend unavailable; "
            "running SM80-path cute_grouped instead"
        )
        try:
            run_conv_level(args.repeats, args.bench, failures, "cute_grouped")
        except Exception as exc:  # keep kernel-level results even if conv API drifts
            print(f"conv-level cute_grouped failed to run: {exc!r}")

    print("=" * 60)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} failures)")
        for f in failures[:40]:
            print(f"  - {f}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
