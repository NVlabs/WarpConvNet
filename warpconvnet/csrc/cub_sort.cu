// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

namespace py = pybind11;

// Helper function to check CUDA errors
#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t error = call;                                                            \
    if (error != cudaSuccess) {                                                          \
      throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
    }                                                                                    \
  } while (0)

// Unified segmented sort function with optional values
py::object cub_segmented_sort(const torch::Tensor& keys,
                              const torch::Tensor& segment_offsets,
                              const py::object& values = py::none(),
                              bool descending = false,
                              bool return_indices = false) {
  TORCH_CHECK(keys.is_cuda(), "Input keys must be on CUDA device");
  TORCH_CHECK(segment_offsets.is_cuda(), "Segment offsets must be on CUDA device");
  TORCH_CHECK(keys.dtype() == torch::kInt64, "Keys must be int64");

  // Convert segment_offsets to int32 if needed (CUB requires int32 offsets)
  torch::Tensor offsets_int32 = segment_offsets;
  if (segment_offsets.dtype() != torch::kInt32) {
    offsets_int32 = segment_offsets.to(torch::kInt32);
  }

  c10::cuda::CUDAGuard device_guard(keys.device());
  // Run cub sorts on the caller's current PyTorch stream, not the default
  // stream 0 — otherwise this sort and its current-stream producers/consumers
  // race under a non-default stream (DeepSpeed / autograd). See the cute/cutlass
  // gather-scatter stream fix for the same bug class.
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int num_items = keys.numel();
  const int num_segments = offsets_int32.numel() - 1;

  bool has_values = !values.is_none();
  torch::Tensor values_tensor;
  if (has_values) {
    values_tensor = values.cast<torch::Tensor>();
    TORCH_CHECK(values_tensor.is_cuda(), "Values must be on CUDA device");
    TORCH_CHECK(values_tensor.size(0) == num_items, "Values must have same length as keys");
  }

  if (!has_values && !return_indices) {
    // Case 1: Keys-only sort, return sorted keys
    auto sorted_keys = torch::empty_like(keys);
    int64_t* keys_ptr = keys.data_ptr<int64_t>();
    int64_t* sorted_keys_ptr = sorted_keys.data_ptr<int64_t>();
    int* offsets_ptr = offsets_int32.data_ptr<int>();

    // Query temp storage size
    size_t temp_storage_bytes = 0;
    if (descending) {
      CUDA_CHECK(cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
                                                              temp_storage_bytes,
                                                              keys_ptr,
                                                              sorted_keys_ptr,
                                                              num_items,
                                                              num_segments,
                                                              offsets_ptr,
                                                              offsets_ptr + 1,
                                                              stream));
    } else {
      CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                    temp_storage_bytes,
                                                    keys_ptr,
                                                    sorted_keys_ptr,
                                                    num_items,
                                                    num_segments,
                                                    offsets_ptr,
                                                    offsets_ptr + 1,
                                                    stream));
    }

    // Allocate temp storage and sort
    auto temp_storage = torch::empty({(int64_t)temp_storage_bytes},
                                     torch::dtype(torch::kUInt8).device(keys.device()));
    void* temp_storage_ptr = temp_storage.data_ptr();

    if (descending) {
      CUDA_CHECK(cub::DeviceSegmentedSort::SortKeysDescending(temp_storage_ptr,
                                                              temp_storage_bytes,
                                                              keys_ptr,
                                                              sorted_keys_ptr,
                                                              num_items,
                                                              num_segments,
                                                              offsets_ptr,
                                                              offsets_ptr + 1,
                                                              stream));
    } else {
      CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(temp_storage_ptr,
                                                    temp_storage_bytes,
                                                    keys_ptr,
                                                    sorted_keys_ptr,
                                                    num_items,
                                                    num_segments,
                                                    offsets_ptr,
                                                    offsets_ptr + 1,
                                                    stream));
    }

    return py::cast(sorted_keys);
  } else {
    // Case 2: Need indices (either for return_indices=True or for values sorting)
    auto sorted_keys = torch::empty_like(keys);
    auto indices = torch::arange(num_items, torch::dtype(torch::kInt32).device(keys.device()));
    auto sorted_indices = torch::empty_like(indices);

    int64_t* keys_ptr = keys.data_ptr<int64_t>();
    int64_t* sorted_keys_ptr = sorted_keys.data_ptr<int64_t>();
    int* indices_ptr = indices.data_ptr<int>();
    int* sorted_indices_ptr = sorted_indices.data_ptr<int>();
    int* offsets_ptr = offsets_int32.data_ptr<int>();

    // Query temp storage size
    size_t temp_storage_bytes = 0;
    if (descending) {
      CUDA_CHECK(cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                               temp_storage_bytes,
                                                               keys_ptr,
                                                               sorted_keys_ptr,
                                                               indices_ptr,
                                                               sorted_indices_ptr,
                                                               num_items,
                                                               num_segments,
                                                               offsets_ptr,
                                                               offsets_ptr + 1,
                                                               stream));
    } else {
      CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                     temp_storage_bytes,
                                                     keys_ptr,
                                                     sorted_keys_ptr,
                                                     indices_ptr,
                                                     sorted_indices_ptr,
                                                     num_items,
                                                     num_segments,
                                                     offsets_ptr,
                                                     offsets_ptr + 1,
                                                     stream));
    }

    // Allocate temp storage and sort
    auto temp_storage = torch::empty({(int64_t)temp_storage_bytes},
                                     torch::dtype(torch::kUInt8).device(keys.device()));
    void* temp_storage_ptr = temp_storage.data_ptr();

    if (descending) {
      CUDA_CHECK(cub::DeviceSegmentedSort::SortPairsDescending(temp_storage_ptr,
                                                               temp_storage_bytes,
                                                               keys_ptr,
                                                               sorted_keys_ptr,
                                                               indices_ptr,
                                                               sorted_indices_ptr,
                                                               num_items,
                                                               num_segments,
                                                               offsets_ptr,
                                                               offsets_ptr + 1,
                                                               stream));
    } else {
      CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(temp_storage_ptr,
                                                     temp_storage_bytes,
                                                     keys_ptr,
                                                     sorted_keys_ptr,
                                                     indices_ptr,
                                                     sorted_indices_ptr,
                                                     num_items,
                                                     num_segments,
                                                     offsets_ptr,
                                                     offsets_ptr + 1,
                                                     stream));
    }

    auto perm = sorted_indices.to(torch::kInt64);

    if (has_values && return_indices) {
      // Return (sorted_values, sorted_keys, indices)
      auto sorted_values = values_tensor.index_select(0, perm);
      return py::make_tuple(sorted_values, sorted_keys, perm);
    } else if (has_values) {
      // Return (sorted_values, sorted_keys)
      auto sorted_values = values_tensor.index_select(0, perm);
      return py::make_tuple(sorted_values, sorted_keys);
    } else {
      // return_indices=True, no values: Return (indices, sorted_keys)
      return py::make_tuple(perm, sorted_keys);
    }
  }
}
