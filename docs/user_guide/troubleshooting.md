# Troubleshooting

**Created**: 2026-04-15 14:00:00
**Edited**: 2026-04-18 16:35:57

## cuBLAS Version Mismatch: `CUBLAS_STATUS_INVALID_VALUE` with fp16/bf16

## Symptom

`torch.matmul` on `float16` or `bfloat16` tensors fails with:

```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling
`cublasGemmEx( handle, opa, opb, m, n, k, alpha_ptr, a, CUDA_R_16F, lda,
b, CUDA_R_16F, ldb, beta_ptr, c, ..., compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
```

`float32` matmul works fine. The error is not specific to WarpConvNet — it affects any PyTorch fp16/bf16 matmul.

## Root Cause

**Two incompatible cuBLAS libraries are loaded simultaneously** when `LD_LIBRARY_PATH` points to a system CUDA installation whose version differs from the CUDA bundled with your pip-installed PyTorch.

For example, with `torch==2.10.0+cu128` (bundles CUDA 12.8 libs) and system CUDA 12.9:

| Library             | Source                                              | Version |
| ------------------- | --------------------------------------------------- | ------- |
| `libcublas.so.12`   | pip `nvidia-cublas-cu12` (bundled with torch)       | 12.8    |
| `libcublasLt.so.12` | `/usr/local/cuda-12.9/lib64/` via `LD_LIBRARY_PATH` | 12.9    |

cuBLAS delegates tensor-core GEMM (fp16/bf16) to cuBLASLt internally. When `libcublas.so` 12.8 calls into `libcublasLt.so` 12.9, the internal ABI is incompatible, producing `CUBLAS_STATUS_INVALID_VALUE`. Float32 matmul uses a code path that avoids this cross-library call, which is why it still works.

### How to confirm

```python
# This will fail:
import torch
x = torch.randn(10, 10, device='cuda', dtype=torch.float16)
torch.matmul(x, x)

# This will work:
x = torch.randn(10, 10, device='cuda', dtype=torch.float32)
torch.matmul(x, x)
```

You can also inspect loaded libraries to see the mismatch:

```bash
python -c "
import torch; torch.randn(2,2,device='cuda')
import os
with open(f'/proc/{os.getpid()}/maps') as f:
    for line in f:
        if 'cublas' in line.lower():
            print(line.strip())
"
```

If you see `libcublas.so` from one path and `libcublasLt.so` from a different CUDA version, that's the mismatch.

## Fix

### Option 1: Clear `LD_LIBRARY_PATH` (recommended for pip-installed torch)

Pip-installed PyTorch bundles its own CUDA libraries. The system CUDA in `LD_LIBRARY_PATH` is unnecessary and causes conflicts.

```bash
# Per-command
LD_LIBRARY_PATH="" python your_script.py

# Or unset in your shell/activate script
unset LD_LIBRARY_PATH
```

### Option 2: Point `LD_LIBRARY_PATH` to matching CUDA version

If you need system CUDA libraries (e.g., for nvcc compilation), ensure the version matches torch's bundled CUDA:

```bash
# torch+cu128 → use system CUDA 12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

### Option 3: Install torch built for your system CUDA version

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu129
```

## Affected Configurations

- **Triggers**: `LD_LIBRARY_PATH` includes a system CUDA `lib64` path with a different minor version than torch's bundled CUDA
- **Affected ops**: Any fp16 or bf16 matmul (`torch.matmul`, `torch.mm`, `F.linear`, etc.)
- **Not affected**: float32 matmul, custom CUTLASS/CuTe kernels (they don't use cuBLAS)
- **Observed with**: torch 2.10.0+cu128 + system CUDA 12.9, but applies to any version mismatch

## AMP fp16 NaN on Large-Kernel Group Convolutions

## Symptom

Training a network that wraps a large-`K` group convolution (e.g. `kernel_size=7`, `groups=1` with wide channels) under `precision=16-mixed` (AMP fp16) shows:

- Loss value starts at a reasonable number on the first forward pass (~2.2).
- Gradients overflow on the very first backward pass — `torch.amp.GradScaler` skips the optimizer step.
- The training loop continues but parameters never update; loss stays pinned at its initial value for many steps.

Switching the same run to `precision=32-true` (fp32) is stable and loss descends normally.

## Root Cause

This is **not** a kernel-dispatch bug. The forward pass produces numerically correct outputs (measured relative error 1.6e-4 vs `explicit_gemm` at `K=343`, `C=16` — correct within fp16 precision).

The issue is that a single kxkxk group convolution with a large kernel and wide fan-in (e.g. `K=343`, `C=16-128`) produces per-output gradient magnitudes that can saturate fp16 during the AMP backward, especially on the very first step before the gradient scaler has had a chance to adapt. This is a general AMP model-recipe problem rather than anything specific to sparse vs dense convolutions.

## Workarounds

Pick whichever works for your recipe:

1. **Train in fp32** — `precision=32-true`. Simplest option if the loss of throughput is acceptable.
2. **Tighter `GradScaler`** — lower initial scale, longer `growth_interval`, or explicit `init_scale` tuning:
   ```python
   scaler = torch.amp.GradScaler("cuda", init_scale=2**10, growth_interval=2000)
   ```
3. **Force fp32 compute on the wide group conv only** — autocast-exempt the offending layer:
   ```python
   with torch.amp.autocast("cuda", enabled=False):
       x_fp32 = x.float()
       x_fp32 = wide_group_conv(x_fp32)
       x = x_fp32.to(dtype)
   ```
4. **Gradient clipping** — add `torch.nn.utils.clip_grad_norm_` before the optimizer step so that overflow-adjacent gradients do not propagate.

## Scope

This affects **very wide per-group fan-in** in fp16 autocast and is not unique to any kernel backend. Narrower kernels (e.g. `K=27` with standard channel widths) train fine under AMP fp16. There is no warpconvnet-side fix planned because the root cause is in the model's gradient magnitudes, not the GEMM kernel.
