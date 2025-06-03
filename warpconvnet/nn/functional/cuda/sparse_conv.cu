// From https://github.com/NVIDIA/MinkowskiEngine/blob/master/src/convolution_kernel.cu

/*
 * Copyright (c) 2020 NVIDIA Corporation.
 * Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 * Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 * of the code.
 */

#include <cuda_fp16.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void
matmul(const Dtype *__restrict__ A, const int wA, const int hA, //
       const Dtype *__restrict__ B, const int wB, const int hB, //
       Dtype *__restrict__ C,                                   //
       const Itype *__restrict__ in_map, const Itype *__restrict__ out_map) {
  // Use in_feat as A and kernel as B

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  Dtype Csub = 0;

  const Itype in_row = y < hA ? in_map[y] : 0;
  const Itype out_row = y < hA ? out_map[y] : 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ Dtype Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < wA && y < hA) ? A[wA * in_row + s + tx] : Dtype(0);
    Bs[ty][tx] = ((s + ty) < hB && x < wB) ? B[wB * (s + ty) + x] : Dtype(0);

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (y < hA && x < wB)
    atomicAdd(&C[wB * out_row + x], Csub);
  // C[wB * out_row + x] += Csub;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B^T, E = D^T * A
 * wA is A's width and wB is B's width
 *
 *                +---+
 *                |B^T|
 *            +-------+
 *            |   |   |
 *            | A | C |
 *            |   |   |
 *            |   |   |
 * +------------------+
 * |    D^T   | E |
 * +----------+---+
 *
 */
template <typename Dtype, typename Itype, int BLOCK_SIZE>
__global__ void
matmul2(const Dtype *__restrict__ A, const int wA, const int hA, //
        const Dtype *__restrict__ B, const int wB, const int hB, //
        const Dtype *__restrict__ D, const int wD, const int hD, //
        Dtype *__restrict__ C, Dtype *__restrict__ E,
        const Itype *__restrict__ in_map, const Itype *__restrict__ out_map) {
  // Use grad_out_feat as A, transposed kernel weight as B, and in_feat as D

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. y is for rows, x is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  const Itype in_row = y < hA ? in_map[y] : 0;
  const Itype out_row = y < hA ? out_map[y] : 0;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  Dtype Csub = 0;
  Dtype Esub = 0;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ Dtype As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ Dtype BTs[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Ds used to
  // store the sub-matrix of D
  __shared__ Dtype DTs[BLOCK_SIZE][BLOCK_SIZE];

  // For Ds = D^T[...:..., ...:...], use the transposed grid dimension for A
  DTs[ty][tx] = (x < wD && y < hD) ? D[wD * in_row + x] : Dtype(0);

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < wA; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < wA && y < hA) ? A[wA * out_row + s + tx] : Dtype(0);

    // Transposed kernel
    BTs[ty][tx] = ((s + ty) < wB && x < hB) ? B[wB * x + s + ty] : Dtype(0);

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * BTs[k][tx];
    }

    // For Esub, reset to 0
    Esub = 0;
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Esub += DTs[k][ty] * As[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();

    // For the E matrix which requires accmulation of multiple blocks, use
    // atomic addition. This can be replaced with a more sophisticaed
    // reduction algorithm.
    if ((bx * BLOCK_SIZE + ty) < wD && (s + tx) < wA)
      atomicAdd(&E[wA * (bx * BLOCK_SIZE + ty) + (s + tx)], Esub);
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (y < hA && x < hB)
    atomicAdd(&C[hB * in_row + x], Csub);
}


// Explicit instantiations
// TODO(cchoy): half types

// For Dtype = float, Itype = int, BLOCK_SIZE = 8

extern "C" __global__ void matmul_float_int_b8(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    float *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<float, int, 8>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_float_int_b8(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    const float *__restrict__ D, const int wD, const int hD,
    float *__restrict__ C, float *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<float, int, 8>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}


// For Dtype = float, Itype = int, BLOCK_SIZE = 16

extern "C" __global__ void matmul_float_int_b16(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    float *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<float, int, 16>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_float_int_b16(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    const float *__restrict__ D, const int wD, const int hD,
    float *__restrict__ C, float *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<float, int, 16>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}


// For Dtype = float, Itype = int, BLOCK_SIZE = 24

extern "C" __global__ void matmul_float_int_b24(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    float *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<float, int, 24>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_float_int_b24(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    const float *__restrict__ D, const int wD, const int hD,
    float *__restrict__ C, float *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<float, int, 24>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}

// For Dtype = float, Itype = int, BLOCK_SIZE = 32

extern "C" __global__ void matmul_float_int_b32(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    float *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<float, int, 32>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_float_int_b32(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    const float *__restrict__ D, const int wD, const int hD,
    float *__restrict__ C, float *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<float, int, 32>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}


// For Dtype = float, Itype = int, BLOCK_SIZE = 48

extern "C" __global__ void matmul_float_int_b48(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    float *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<float, int, 48>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_float_int_b48(
    const float *__restrict__ A, const int wA, const int hA,
    const float *__restrict__ B, const int wB, const int hB,
    const float *__restrict__ D, const int wD, const int hD,
    float *__restrict__ C, float *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<float, int, 48>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}


// For Dtype = float, Itype = int, BLOCK_SIZE = 64

// extern "C" __global__ void matmul_float_int_b64(
//     const float *__restrict__ A, const int wA, const int hA,
//     const float *__restrict__ B, const int wB, const int hB,
//     float *__restrict__ C,
//     const int *__restrict__ in_map, const int *__restrict__ out_map) {
//   matmul<float, int, 64>(A, wA, hA, B, wB, hB, C, in_map, out_map);
// }

// extern "C" __global__ void matmul2_float_int_b64(
//     const float *__restrict__ A, const int wA, const int hA,
//     const float *__restrict__ B, const int wB, const int hB,
//     const float *__restrict__ D, const int wD, const int hD,
//     float *__restrict__ C, float *__restrict__ E,
//     const int *__restrict__ in_map, const int *__restrict__ out_map) {
//   matmul2<float, int, 64>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
// }

// Double types
// For Dtype = double, Itype = int, BLOCK_SIZE = 8

extern "C" __global__ void matmul_double_int_b8(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    double *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<double, int, 8>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_double_int_b8(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    const double *__restrict__ D, const int wD, const int hD,
    double *__restrict__ C, double *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<double, int, 8>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}


// For Dtype = double, Itype = int, BLOCK_SIZE = 16

extern "C" __global__ void matmul_double_int_b16(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    double *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<double, int, 16>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_double_int_b16(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    const double *__restrict__ D, const int wD, const int hD,
    double *__restrict__ C, double *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<double, int, 16>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}


// For Dtype = double, Itype = int, BLOCK_SIZE = 24

extern "C" __global__ void matmul_double_int_b24(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    double *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<double, int, 24>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_double_int_b24(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    const double *__restrict__ D, const int wD, const int hD,
    double *__restrict__ C, double *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<double, int, 24>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}


// For Dtype = double, Itype = int, BLOCK_SIZE = 32

extern "C" __global__ void matmul_double_int_b32(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    double *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<double, int, 32>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_double_int_b32(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    const double *__restrict__ D, const int wD, const int hD,
    double *__restrict__ C, double *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<double, int, 32>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}


// For Dtype = double, Itype = int, BLOCK_SIZE = 48

extern "C" __global__ void matmul_double_int_b48(
    const double *__restrict__ A, const int wA, const int hA,
    const double *__restrict__ B, const int wB, const int hB,
    double *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<double, int, 48>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

// extern "C" __global__ void matmul2_double_int_b48(
//     const double *__restrict__ A, const int wA, const int hA,
//     const double *__restrict__ B, const int wB, const int hB,
//     const double *__restrict__ D, const int wD, const int hD,
//     double *__restrict__ C, double *__restrict__ E,
//     const int *__restrict__ in_map, const int *__restrict__ out_map) {
//   matmul2<double, int, 48>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
// }

// For Dtype = double, Itype = int, BLOCK_SIZE = 64

// extern "C" __global__ void matmul_double_int_b64(
//     const double *__restrict__ A, const int wA, const int hA,
//     const double *__restrict__ B, const int wB, const int hB,
//     double *__restrict__ C,
//     const int *__restrict__ in_map, const int *__restrict__ out_map) {
//   matmul<double, int, 64>(A, wA, hA, B, wB, hB, C, in_map, out_map);
// }

// extern "C" __global__ void matmul2_double_int_b64(
//     const double *__restrict__ A, const int wA, const int hA,
//     const double *__restrict__ B, const int wB, const int hB,
//     const double *__restrict__ D, const int wD, const int hD,
//     double *__restrict__ C, double *__restrict__ E,
//     const int *__restrict__ in_map, const int *__restrict__ out_map) {
//   matmul2<double, int, 64>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
// }

// Half precision types
// For Dtype = __half, Itype = int, BLOCK_SIZE = 8

extern "C" __global__ void matmul_half_int_b8(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    __half *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<__half, int, 8>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_half_int_b8(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    const __half *__restrict__ D, const int wD, const int hD,
    __half *__restrict__ C, __half *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<__half, int, 8>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}

// For Dtype = __half, Itype = int, BLOCK_SIZE = 16

extern "C" __global__ void matmul_half_int_b16(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    __half *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<__half, int, 16>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_half_int_b16(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    const __half *__restrict__ D, const int wD, const int hD,
    __half *__restrict__ C, __half *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<__half, int, 16>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}

// For Dtype = __half, Itype = int, BLOCK_SIZE = 24

extern "C" __global__ void matmul_half_int_b24(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    __half *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<__half, int, 24>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_half_int_b24(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    const __half *__restrict__ D, const int wD, const int hD,
    __half *__restrict__ C, __half *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<__half, int, 24>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}

// For Dtype = __half, Itype = int, BLOCK_SIZE = 32

extern "C" __global__ void matmul_half_int_b32(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    __half *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<__half, int, 32>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

extern "C" __global__ void matmul2_half_int_b32(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    const __half *__restrict__ D, const int wD, const int hD,
    __half *__restrict__ C, __half *__restrict__ E,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul2<__half, int, 32>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
}

// For Dtype = __half, Itype = int, BLOCK_SIZE = 48

extern "C" __global__ void matmul_half_int_b48(
    const __half *__restrict__ A, const int wA, const int hA,
    const __half *__restrict__ B, const int wB, const int hB,
    __half *__restrict__ C,
    const int *__restrict__ in_map, const int *__restrict__ out_map) {
  matmul<__half, int, 48>(A, wA, hA, B, wB, hB, C, in_map, out_map);
}

// Shared memory limit
// extern "C" __global__ void matmul2_half_int_b48(
//     const __half *__restrict__ A, const int wA, const int hA,
//     const __half *__restrict__ B, const int wB, const int hB,
//     const __half *__restrict__ D, const int wD, const int hD,
//     __half *__restrict__ C, __half *__restrict__ E,
//     const int *__restrict__ in_map, const int *__restrict__ out_map) {
//   matmul2<__half, int, 48>(A, wA, hA, B, wB, hB, D, wD, hD, C, E, in_map, out_map);
// }
