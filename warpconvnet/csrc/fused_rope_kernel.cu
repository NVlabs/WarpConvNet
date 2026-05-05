// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused RoPE + QKV reshape: replaces qkv.chunk(3) -> rope(Q) -> rope(K) ->
// cat([Q,K,V]) -> reshape(M, 3, H, D) with a single CUDA kernel. Forward and
// backward share one kernel; backward passes ``conjugate=1`` to flip the sin
// sign so the rotation is inverted.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

namespace warpconvnet {
namespace fused_rope {

template <typename T>
__global__ void fused_rope_qkv_kernel(const T* __restrict__ qkv,         // [M, 3, C]
                                      const float* __restrict__ coords,  // [M, 3]
                                      const float* __restrict__ theta,   // [theta_len]
                                      T* __restrict__ out,               // [M, 3, H, D]
                                      int M,
                                      int C,
                                      int head_dim,
                                      int rope_dim,
                                      int theta_len,
                                      int num_heads,
                                      float coord_min_x,
                                      float coord_min_y,
                                      float coord_min_z,
                                      int conjugate) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = blockIdx.y;
  const int qkv_idx = blockIdx.z;
  if (m >= M) return;

  const int half_rope = rope_dim / 2;

  // Per-thread input/output base pointers for (m, qkv_idx, h).
  const T* in_ptr = qkv + (static_cast<size_t>(m) * 3 + qkv_idx) * C + h * head_dim;
  T* out_ptr = out + ((static_cast<size_t>(m) * 3 + qkv_idx) * num_heads + h) * head_dim;

  const bool apply_rope = (qkv_idx < 2) && (rope_dim > 0);

  if (apply_rope) {
    const float cx = coords[m * 3 + 0] - coord_min_x + 1.0f;
    const float cy = coords[m * 3 + 1] - coord_min_y + 1.0f;
    const float cz = coords[m * 3 + 2] - coord_min_z + 1.0f;

    for (int hr = 0; hr < half_rope; ++hr) {
      const int axis = hr / theta_len;
      const int t_idx = hr % theta_len;
      const float coord_axis = (axis == 0 ? cx : (axis == 1 ? cy : cz));
      const float angle = coord_axis * theta[t_idx];
      float cos_v = cosf(angle);
      float sin_v = sinf(angle);
      if (conjugate) sin_v = -sin_v;

      const float real_in = static_cast<float>(in_ptr[hr * 2]);
      const float imag_in = static_cast<float>(in_ptr[hr * 2 + 1]);
      const float new_real = real_in * cos_v - imag_in * sin_v;
      const float new_imag = real_in * sin_v + imag_in * cos_v;
      out_ptr[hr * 2] = static_cast<T>(new_real);
      out_ptr[hr * 2 + 1] = static_cast<T>(new_imag);
    }

    // Pass-through dims (no RoPE applied).
    for (int d = rope_dim; d < head_dim; ++d) {
      out_ptr[d] = in_ptr[d];
    }
  } else {
    // V (qkv_idx == 2) or rope_dim == 0: straight copy.
    for (int d = 0; d < head_dim; ++d) {
      out_ptr[d] = in_ptr[d];
    }
  }
}

// Host-side launcher. Allocates nothing — caller owns ``out``.
void run_fused_rope_qkv(const at::Tensor& qkv,
                        const at::Tensor& coords,
                        const at::Tensor& theta,
                        at::Tensor& out,
                        int num_heads,
                        int rope_dim,
                        int conjugate) {
  TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");
  TORCH_CHECK(coords.is_cuda(), "coords must be CUDA");
  TORCH_CHECK(theta.is_cuda(), "theta must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(qkv.dim() == 3, "qkv must be [M, 3, C]");
  TORCH_CHECK(qkv.size(1) == 3, "qkv.size(1) must be 3");
  TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 3, "coords must be [M, 3]");
  // Coords are required as float32 even though voxel-grid coordinates are
  // typically int32 in the upstream pipeline. Caller pays a host-side
  // .float().contiguous() allocation (M*3*4 bytes per call). To skip that,
  // template the kernel on coord dtype (int32_t / float) and dispatch via
  // AT_DISPATCH_* — cast happens inside the kernel, no extra alloc.
  TORCH_CHECK(coords.scalar_type() == at::kFloat, "coords must be float32");
  TORCH_CHECK(theta.scalar_type() == at::kFloat, "theta must be float32");
  TORCH_CHECK(out.scalar_type() == qkv.scalar_type(), "out dtype must match qkv");

  const int M = qkv.size(0);
  const int C = qkv.size(2);
  const int head_dim = C / num_heads;
  const int theta_len = rope_dim / 6;

  TORCH_CHECK(C % num_heads == 0, "C must be divisible by num_heads");
  TORCH_CHECK(out.dim() == 4, "out must be [M, 3, H, D]");
  TORCH_CHECK(out.size(0) == M && out.size(1) == 3, "out shape mismatch");
  TORCH_CHECK(out.size(2) == num_heads && out.size(3) == head_dim, "out [M,3,H,D] shape mismatch");

  if (M == 0) return;

  // coord_min computed on host (single sync, tiny tensor).
  auto coord_min_t = std::get<0>(coords.min(/*dim=*/0)).cpu();
  const float coord_min_x = coord_min_t[0].item<float>();
  const float coord_min_y = coord_min_t[1].item<float>();
  const float coord_min_z = coord_min_t[2].item<float>();

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(qkv));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int BLOCK_M = 128;
  const dim3 block(BLOCK_M);
  const dim3 grid((M + BLOCK_M - 1) / BLOCK_M, num_heads, 3);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, qkv.scalar_type(), "fused_rope_qkv", [&] {
        fused_rope_qkv_kernel<scalar_t><<<grid, block, 0, stream>>>(qkv.data_ptr<scalar_t>(),
                                                                    coords.data_ptr<float>(),
                                                                    theta.data_ptr<float>(),
                                                                    out.data_ptr<scalar_t>(),
                                                                    M,
                                                                    C,
                                                                    head_dim,
                                                                    rope_dim,
                                                                    theta_len,
                                                                    num_heads,
                                                                    coord_min_x,
                                                                    coord_min_y,
                                                                    coord_min_z,
                                                                    conjugate);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace fused_rope
}  // namespace warpconvnet
