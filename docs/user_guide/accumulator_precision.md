# Accumulator Precision

**Created**: 2026-04-15 14:00:00
**Edited**: 2026-05-26 14:30:00

WarpConvNet's mask_gemm kernels use tensor core MMA instructions that support two accumulator modes:

| Mode                 | MMA Atom                       | Throughput | Precision | Default |
| -------------------- | ------------------------------ | ---------- | --------- | ------- |
| **fp32 accumulator** | `SM80_16x8x16_F32F16F16F32_TN` | 1x         | Higher    | Yes     |
| **fp16 accumulator** | `SM80_16x8x16_F16F16F16F16_TN` | 2x         | Lower     | No      |

The fp16 accumulator doubles tensor core throughput by accumulating partial sums in fp16 instead of fp32. This gives ~15% end-to-end speedup at large channels (C >= 64) where the computation is tensor-core-bound. The precision loss can be significant at wide channels (measured `rd` vs fp32: 1.4e-5 @ C=8, 2.4e-4 @ C=32, 5.3e-4 @ C=128, 7.5e-4 @ C=256) and training convergence on sensitive models can trail the fp32-accumulator baseline over many epochs. Default is off for that reason; enable explicitly for inference or after verifying training stability.

## Configuration

### Environment Variable (no code changes)

```bash
# Enable fp16 accumulator globally
export WARPCONVNET_USE_FP16_ACCUM=true

# Disable (default)
export WARPCONVNET_USE_FP16_ACCUM=false
```

### Runtime API (no code changes)

```python
import warpconvnet

# Enable fp16 accumulator for all subsequent convolutions
warpconvnet.set_fp16_accum(True)

# Query current setting
print(warpconvnet.get_fp16_accum())  # True

# Disable
warpconvnet.set_fp16_accum(False)
```

### Per-Module Override

Individual convolution layers can override the global setting:

```python
from warpconvnet.nn.modules.sparse_conv import SparseConv3d

# This layer always uses fp16 accumulator, regardless of global setting
conv_fast = SparseConv3d(64, 128, kernel_size=3, use_fp16_accum=True)

# This layer always uses fp32 accumulator
conv_precise = SparseConv3d(64, 128, kernel_size=3, use_fp16_accum=False)

# This layer follows the global setting (default)
conv_default = SparseConv3d(64, 128, kernel_size=3)  # use_fp16_accum=None
```

### Functional API

```python
from warpconvnet.nn.functional.sparse_conv import spatially_sparse_conv

output = spatially_sparse_conv(
    input_voxels,
    weight,
    kernel_size=3,
    use_fp16_accum=True,  # or None for global setting
)
```

## Precedence

1. **Per-module** `use_fp16_accum=True/False` -- highest priority
2. **Global runtime** `warpconvnet.set_fp16_accum(True)` -- used when module is `None`
3. **Environment variable** `WARPCONVNET_USE_FP16_ACCUM=true` -- sets initial global value
4. **Default** `False` (fp32 accumulator)

## Small-Channel F16-Accum Pcoff Allowance

The pcoff (E1 offset-precompute) tiles 54/55/56/57 use F16Accum / F16K8 base
configs. Default is **off** (`WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL=0`) —
F16-accum pcoff tiles require explicit opt-in via
`WARPCONVNET_USE_FP16_ACCUM=true`.

The previous default (`32`) was reverted after a regression at
training-realistic sparse convolution sizes. At `(C=32, K=3x3x3, N>=200k)` —
an early-encoder shape in large 3D backbones — tiles 54/55/56 saturate
isolated output cells with `max_rel` reaching several hundred against the fp64
reference (verified on both sm_80 A100 and sm_89 Ada). `p99` stays at the AMP
noise floor (~1e-3), so per-tile sweeps that report only `p99` or mean drift
miss the failure mode entirely; only `max_rel` surfaces it. The training
symptom is the classic outlier pattern: train/loss descends normally, val mAP
collapses to ~0 because instance-level metrics are sensitive to per-cell
corruption while the mean-reduced loss tolerates outliers.

```bash
# Default (disabled) — F16-accum pcoff requires WARPCONVNET_USE_FP16_ACCUM=true
export WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL=0

# Opt back into prior behavior (validate val metric on your workload first)
export WARPCONVNET_PCOFF_F16ACC_SMALL_CH_CEIL=32
```

The F32-accum pcoff tiles (58, 59, 63 fwd; 68, 69 native dgrad; 909, 910,
911 fwd_as_dgrad) are unaffected — they remain in the pool by default
and capture the pcoff E1-offset-hoist speed win without the accumulator
drift, because the hoist is independent of accumulator precision. Users
who set a non-zero ceiling are responsible for validating training
stability + val metrics; the regression test in
`tests/nn/test_pcoff_f16acc_regression.py` keeps the gate load-bearing.

## What the Flag Does

When `use_fp16_accum=True` is resolved (per-module, global, or env):

1. **mask_gemm pool**: F16Acc tiles (40/42 forward, equivalent dgrad variants) are **added** to the autotune candidate pool. When the flag is unset, only F32Acc mask_gemm tiles are benchmarked.
2. **CUTLASS pool**: Every CUTLASS entry's `accumulator_type` parameter is rewritten to `torch.float16`. Without the flag, CUTLASS runs with its default fp32 accumulator.

Both the adaptive (`auto`) and trimmed pools default to F32Acc only. Setting `WARPCONVNET_FWD_ALGO_MODE=all` / `WARPCONVNET_DGRAD_ALGO_MODE=all`, or explicitly naming `mask_gemm` via the env config, includes F16Acc tiles regardless of the flag; the flag is the normal gate for adaptive/trimmed pools.

## Which Tiles Use FP16 Accumulator

The autotune system selects between F16Accum and F32 tile variants based on the `use_fp16_accum` setting:

| Tile ID | Name                          | Accumulator | Use Case                     | In adaptive/trimmed pool?       |
| ------- | ----------------------------- | ----------- | ---------------------------- | ------------------------------- |
| 40      | `Prod_Fwd_32x32x32_F16Acc`    | fp16        | Forward, C \<= 48, fp16 only | Only when `use_fp16_accum=True` |
| 42      | `Prod_Fwd_64x128x32_F16Acc`   | fp16        | Forward, C >= 128            | Only when `use_fp16_accum=True` |
| 53      | `Prod_Dgrad_64x64x32_F16Acc`  | fp16        | Dgrad, C = 64                | Only when `use_fp16_accum=True` |
| 54      | `Prod_Dgrad_64x128x32_F16Acc` | fp16        | Dgrad, C >= 128              | Only when `use_fp16_accum=True` |
| 41      | `Prod_Fwd_64x64x32`           | fp32        | Forward, C = 64              | Yes                             |
| 43      | `Prod_Fwd_64x128x32_3s`       | fp32        | Forward, C >= 128            | Yes                             |
| 44      | `Prod_Fwd_128x64x32`          | fp32        | Forward, C = 64              | Yes                             |
| 51      | `Prod_Dgrad_64x64x32`         | fp32        | Dgrad, C = 64                | Yes                             |
| 52      | `Prod_Dgrad_64x128x32`        | fp32        | Dgrad, C >= 128              | Yes                             |

Wgrad always uses fp32 accumulator regardless of this setting, since weight gradient precision directly affects training convergence.

## Recommendations

- **Training**: Use fp32 accumulator (default). Switch to fp16 only after verifying convergence is unaffected on your model. The F16Acc tiles' per-step relative difference (up to ~7.5e-4 at C=256) accumulates across epochs and has been observed to slow convergence on ScanNet MinkUNet-style models.
- **Inference**: fp16 accumulator is safe and recommended for maximum throughput.
- **Large channels (C >= 128)**: Largest speedup from fp16 accumulator (~15%).
- **Small channels (C \<= 32)**: Minimal benefit since the computation is memory-bound, not compute-bound.
- **After switching**: clear the cache (`rm ~/.cache/warpconvnet/benchmark_cache_generic.*`) so the pool change triggers a fresh autotune pass.
