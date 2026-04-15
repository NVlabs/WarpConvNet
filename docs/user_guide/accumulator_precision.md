# Accumulator Precision

Created: 2026-04-15 14:00:00
Edited: 2026-04-15 14:00:00

WarpConvNet's production mask GEMM kernels use tensor core MMA instructions that support two accumulator modes:

| Mode                 | MMA Atom                       | Throughput | Precision | Default |
| -------------------- | ------------------------------ | ---------- | --------- | ------- |
| **fp32 accumulator** | `SM80_16x8x16_F32F16F16F32_TN` | 1x         | Higher    | Yes     |
| **fp16 accumulator** | `SM80_16x8x16_F16F16F16F16_TN` | 2x         | Lower     | No      |

The fp16 accumulator doubles tensor core throughput by accumulating partial sums in fp16 instead of fp32. This gives ~15% end-to-end speedup at large channels (C >= 64) where the computation is tensor-core-bound. The precision loss is small (~0.03% relative difference vs fp32) but may affect training stability for sensitive models.

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

## Which Tiles Use FP16 Accumulator

The autotune system selects between F16Accum and F32 tile variants based on the `use_fp16_accum` setting:

| Tile ID | Name                          | Accumulator | Use Case                     |
| ------- | ----------------------------- | ----------- | ---------------------------- |
| 40      | `Prod_Fwd_32x32x32_F16Acc`    | fp16        | Forward, C \<= 48, fp16 only |
| 42      | `Prod_Fwd_64x128x32_F16Acc`   | fp16        | Forward, C >= 128            |
| 53      | `Prod_Dgrad_64x64x32_F16Acc`  | fp16        | Dgrad, C = 64                |
| 54      | `Prod_Dgrad_64x128x32_F16Acc` | fp16        | Dgrad, C >= 128              |
| 41      | `Prod_Fwd_64x64x32`           | fp32        | Forward, C = 64              |
| 43      | `Prod_Fwd_64x128x32_3s`       | fp32        | Forward, C >= 128            |
| 51      | `Prod_Dgrad_64x64x32`         | fp32        | Dgrad, C = 64                |
| 52      | `Prod_Dgrad_64x128x32`        | fp32        | Dgrad, C >= 128              |

Wgrad always uses fp32 accumulator regardless of this setting, since weight gradient precision directly affects training convergence.

## Recommendations

- **Training**: Use fp32 accumulator (default). Switch to fp16 only after verifying convergence is unaffected on your model.
- **Inference**: fp16 accumulator is safe and recommended for maximum throughput.
- **Large channels (C >= 128)**: Largest speedup from fp16 accumulator (~15%).
- **Small channels (C \<= 32)**: Minimal benefit since the computation is memory-bound, not compute-bound.
