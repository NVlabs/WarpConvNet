// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// __device__ int32_t part1by2(int32_t n) {
//     n = (n ^ (n << 16)) & 0xFF0000FF;
//     n = (n ^ (n << 8))  & 0x0300F00F;
//     n = (n ^ (n << 4))  & 0x030C30C3;
//     n = (n ^ (n << 2))  & 0x09249249;
//     return n;
// }

__device__ int64_t part1by2_long(int64_t n) {
  n = (n ^ (n << 32)) & 0xFFFF00000000FFFFLL;
  n = (n ^ (n << 16)) & 0x00FF0000FF0000FFLL;
  n = (n ^ (n << 8)) & 0xF00F00F00F00F00FLL;
  n = (n ^ (n << 4)) & 0x30C30C30C30C30C3LL;
  n = (n ^ (n << 2)) & 0x9249249249249249LL;
  return n;
}

// Corresponds to _morton_code in Warp, for 16-bit coordinates per dimension + batch index
// bcoord_point is an array for a single point: [batch_idx, x, y, z]
// Coords (x,y,z) are assumed to be in the range [0, 2^16 - 1]
// Batch index should be less than 2^15
__device__ int64_t morton_code_16bit_device(const int* bcoord_point) {
  int64_t batch_idx = (int64_t)bcoord_point[0];
  int64_t ux = (int64_t)bcoord_point[1];  // x
  int64_t uy = (int64_t)bcoord_point[2];  // y
  int64_t uz = (int64_t)bcoord_point[3];  // z

  int64_t morton = (part1by2_long(uz) << 2) | (part1by2_long(uy) << 1) | part1by2_long(ux);

  // Erase the first 16 bits (from the original 64-bit space for 3*21 bits)
  // of the Morton code to make space for the batch index.
  // Z-order for 3 coords up to 21 bits each would be 63 bits.
  // Here, coords are 16-bit, so 3*16 = 48 bits.
  morton &= 0x0000FFFFFFFFFFFFLL;  // Mask to keep lower 48 bits

  // Combine the batch index with the Morton order
  // Shift batch_idx by 48 bits to place it in the most significant bits
  return (batch_idx << 48) | morton;
}

// Corresponds to the 20-bit coordinate path in Warp's _assign_order_discrete_20bit
// Coords (x,y,z) are assumed to be in a range that fits into 20 bits effectively, [0, 2^20 -1]
__device__ int64_t morton_code_20bit_device(int coord_x, int coord_y, int coord_z) {
  int64_t ux = (int64_t)coord_x;
  int64_t uy = (int64_t)coord_y;
  int64_t uz = (int64_t)coord_z;

  // Calculate the Morton order for 3 coordinates (max 21 bits each for 63-bit total)
  return (part1by2_long(uz) << 2) | (part1by2_long(uy) << 1) | part1by2_long(ux);
}

extern "C" __global__ void assign_order_discrete_16bit_kernel(
    const int* __restrict__ bcoords_data,  // Flattened 2D array (num_points, 4), [batch_idx, x, y,
                                           // z]
    int num_points,
    int64_t* __restrict__ result_order) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_points) {
    // Each point has 4 components: batch_idx, x, y, z
    const int* current_bcoord_point = bcoords_data + tid * 4;
    result_order[tid] = morton_code_16bit_device(current_bcoord_point);
  }
}

extern "C" __global__ void assign_order_discrete_20bit_kernel_4points(
    const int* __restrict__ coords_data,  // Flattened 2D array (num_points, 3), [x, y, z]
    int num_points,
    int64_t* __restrict__ result_order) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Process 4 points per thread for better memory throughput
  constexpr int NUM_POINTS_PER_THREAD = 4;
  int base_idx = tid * NUM_POINTS_PER_THREAD;

  if (base_idx < num_points) {
    // Vectorized load: 4 threads cooperatively load 12 ints (48 bytes)
    // This can be more efficient than individual loads

#pragma unroll
    for (int i = 0; i < NUM_POINTS_PER_THREAD && (base_idx + i) < num_points; i++) {
      int point_idx = base_idx + i;
      int coord_x = coords_data[point_idx * 3 + 0];
      int coord_y = coords_data[point_idx * 3 + 1];
      int coord_z = coords_data[point_idx * 3 + 2];
      result_order[point_idx] = morton_code_20bit_device(coord_x, coord_y, coord_z);
    }
  }
}
