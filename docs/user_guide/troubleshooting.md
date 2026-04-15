# Troubleshooting

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
