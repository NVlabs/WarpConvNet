// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// cuhash - Optimized CUDA Hash Table Library
// cuda_check.cuh - CUDA error-checking macro for kernel launches
#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

// Check whether the most recent CUDA kernel launch reported an error.
// Call this immediately after every `kernel<<<...>>>(...)`. Uses
// `cudaGetLastError` (non-synchronous, effectively free when no error).
// Throws via TORCH_CHECK so the failure surfaces as a Python exception.
#define CUHASH_CHECK_CUDA_LAUNCH()                                                    \
  do {                                                                                \
    cudaError_t err = cudaGetLastError();                                             \
    TORCH_CHECK(err == cudaSuccess, "CUDA launch failed: ", cudaGetErrorString(err)); \
  } while (0)
