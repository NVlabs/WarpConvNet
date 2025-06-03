// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <vector_types.h> // For int4

// --- Hash Helper Functions (Copied from hashmap_kernels.cu for simplicity) ---
typedef unsigned int uint32_t;

__device__ inline uint32_t murmur_32_scramble(uint32_t k) {
    k *= 0xCC9E2D51;
    k = (k << 15) | (k >> 17);
    k *= 0x1B873593;
    return k;
}

__device__ inline uint32_t _hash_murmur_impl(uint32_t h, uint32_t k) {
    h ^= murmur_32_scramble(k);
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xE6546B64;
    return h;
}

__device__ inline uint32_t _hash_murmur_finalize(uint32_t h, int length_bytes) {
    h ^= length_bytes;
    h ^= h >> 16;
    h *= 0x85EBCA6B;
    h ^= h >> 13;
    h *= 0xC2B2AE35;
    h ^= h >> 16;
    return h;
}

__device__ inline uint32_t _hash_fnv1a_impl(uint32_t hash_val, uint32_t key) {
    hash_val ^= key;
    hash_val *= 16777619; // FNV prime
    return hash_val;
}


__device__ inline uint32_t _hash_city_impl(uint32_t hash_val, uint32_t key) {
    hash_val += key * 0x9E3779B9;
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85EBCA6B;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xC2B2AE35;
    hash_val ^= hash_val >> 16;
    return hash_val;
}

struct FNV1AHash {
    __device__ inline static int hash(const int* key, int key_dim, int capacity) {
        uint32_t hash_val = 2166136261u;
        for (int i = 0; i < key_dim; ++i) {
            hash_val = _hash_fnv1a_impl(hash_val, (uint32_t)key[i]);
        }
        int signed_hash = (int)hash_val;
        return ((signed_hash % capacity) + capacity) % capacity;
    }
};

struct CityHash {
     __device__ inline static int hash(const int* key, int key_dim, int capacity) {
        uint32_t hash_val = 0;
        for (int i = 0; i < key_dim; ++i) {
            hash_val = _hash_city_impl(hash_val, (uint32_t)key[i]);
        }
        int signed_hash = (int)hash_val;
        return ((signed_hash % capacity) + capacity) % capacity;
    }
};

struct MurmurHash {
     __device__ inline static int hash(const int* key, int key_dim, int capacity) {
        uint32_t h = 0x9747B28Cu;
        for (int i = 0; i < key_dim; ++i) {
            h = _hash_murmur_impl(h, (uint32_t)key[i]);
        }
        h = _hash_murmur_finalize(h, key_dim * 4);
        int signed_hash = (int)h;
        return ((signed_hash % capacity) + capacity) % capacity;
    }
};

// --- Vector Comparison (Copied from hashmap_kernels.cu) ---
__device__ inline bool vec_equal(const int* a, const int* b, int dim) {
    for (int i = 0; i < dim; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// --- Device Function for Hash Table Search (Copied from hashmap_kernels.cu) ---
template<typename HashFuncT>
__device__ inline int search_hash_table(
    const int* __restrict__ table_kvs,
    const int* __restrict__ vector_keys,
    const int* __restrict__ query_key,
    int key_dim,
    int table_capacity) {

    int slot = HashFuncT::hash(query_key, key_dim, table_capacity);
    int initial_slot = slot;
    int attempts = 0;

    while (attempts < table_capacity) {
        int slot_marker = table_kvs[slot * 2 + 0];
        if (slot_marker == -1) { return -1; }

        int vector_index = table_kvs[slot * 2 + 1];
        if (vector_index != -1) {
            const int* candidate_key = &vector_keys[vector_index * key_dim];
            if (vec_equal(candidate_key, query_key, key_dim)) {
                return vector_index;
            }
        }
        slot = (slot + 1) % table_capacity;
        if (slot == initial_slot) { return -1;}
        attempts++;
    }
    return -1;
}


// --- Kernel Implementations ---

// Equivalent of conv_kernel_map_arr / conv_kernel_map_vec4i (combined for array input)
template<typename HashFuncT>
__global__ void kernel_map_offset_templated(
    const int* __restrict__ table_kvs,            // Hash table key-value store (capacity, 2)
    const int* __restrict__ vector_keys,          // Original stored keys (num_in_keys, key_dim)
    const int* __restrict__ query_coords,         // Coordinates to query (num_query_coords, key_dim)
    const int* __restrict__ kernel_offsets,       // Offsets to apply (num_kernel_offsets, key_dim)
    int* __restrict__ found_in_coord_index, // Output array (num_kernel_offsets, num_query_coords)
    int num_query_coords,
    int key_dim,
    int num_kernel_offsets,
    int table_capacity)
{
    // Thread ID corresponds to the query coordinate index (x dimension)
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread ID corresponds to the kernel offset index (y dimension)
    int kernel_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (query_idx >= num_query_coords) {
        return;
    }

    if (kernel_idx >= num_kernel_offsets) {
        return;
    }

    // Temporary storage for the calculated query coordinate + offset
    // Using stack allocation assuming key_dim is small (e.g., 4)
    // For larger key_dim, consider shared memory or scratchpad global memory if needed.
    int temp_coord[16]; // Max key_dim assumed <= 16; adjust if necessary

    const int* base_query_coord = &query_coords[query_idx * key_dim];
    const int* offset = &kernel_offsets[kernel_idx * key_dim];

    // Calculate query_coord + offset
    for (int dim = 0; dim < key_dim; ++dim) {
        temp_coord[dim] = base_query_coord[dim] + offset[dim];
    }

    // Search for the calculated coordinate in the hash table
    int found_index = search_hash_table<HashFuncT>(
        table_kvs,
        vector_keys,
        temp_coord,
        key_dim,
        table_capacity
    );

    // Store the result in the output array [kernel_idx, query_idx]
    // Note: CuPy/PyTorch tensors are row-major by default.
    // Accessing found_in_coord_index[kernel_idx][query_idx] corresponds to index kernel_idx * num_query_coords + query_idx
    found_in_coord_index[kernel_idx * num_query_coords + query_idx] = found_index;
}

// Kernel to map found indices to flattened in/out maps and offsets
__global__ void map_found_indices_to_maps_kernel(
    const int* __restrict__ found_in_coord_index, // Input: Found indices (num_kernel_offsets, num_query_coords)
    const int* __restrict__ mapped_indices,       // Input: Cumulative sum (-1) per row (num_kernel_offsets, num_query_coords)
    const int* __restrict__ offsets,              // Input: Offsets per kernel (num_kernel_offsets + 1)
    int* __restrict__ out_in_maps,            // Output: Flattened input indices (num_total_maps)
    int* __restrict__ out_out_maps,           // Output: Flattened output indices (num_total_maps)
    int num_kernel_offsets,
    int num_query_coords)
{
    // Global thread ID covering the entire found_in_coord_index matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_kernel_offsets * num_query_coords;

    if (idx >= total_elements) {
        return;
    }

    // Decompose global index into kernel index (k) and query index (m)
    int k = idx / num_query_coords;
    int m = idx % num_query_coords;

    int found_index = found_in_coord_index[idx]; // Direct access using global index

    if (found_index >= 0) {
        // Calculate the output position in the flattened maps
        int output_map_idx = mapped_indices[idx] + offsets[k];

        out_in_maps[output_map_idx] = found_index; // Input map index (from hash table search)
        out_out_maps[output_map_idx] = m;          // Output map index (query coordinate index)
    }
}


// --- Extern "C" Wrappers ---

// kernel_map_offset wrappers
extern "C" __global__ void kernel_map_offset_fnv1a(
    const int* table_kvs, const int* vector_keys, const int* query_coords,
    const int* kernel_offsets, int* found_in_coord_index,
    int num_query_coords, int key_dim, int num_kernel_offsets, int table_capacity)
{
    kernel_map_offset_templated<FNV1AHash>(
        table_kvs, vector_keys, query_coords, kernel_offsets, found_in_coord_index,
        num_query_coords, key_dim, num_kernel_offsets, table_capacity);
}

extern "C" __global__ void kernel_map_offset_city(
    const int* table_kvs, const int* vector_keys, const int* query_coords,
    const int* kernel_offsets, int* found_in_coord_index,
    int num_query_coords, int key_dim, int num_kernel_offsets, int table_capacity)
{
    kernel_map_offset_templated<CityHash>(
        table_kvs, vector_keys, query_coords, kernel_offsets, found_in_coord_index,
        num_query_coords, key_dim, num_kernel_offsets, table_capacity);
}

extern "C" __global__ void kernel_map_offset_murmur(
    const int* table_kvs, const int* vector_keys, const int* query_coords,
    const int* kernel_offsets, int* found_in_coord_index,
    int num_query_coords, int key_dim, int num_kernel_offsets, int table_capacity)
{
    kernel_map_offset_templated<MurmurHash>(
        table_kvs, vector_keys, query_coords, kernel_offsets, found_in_coord_index,
        num_query_coords, key_dim, num_kernel_offsets, table_capacity);
}

// map_found_indices_to_maps wrapper (no template needed)
extern "C" __global__ void map_found_indices_to_maps_cuda(
    const int* found_in_coord_index, const int* mapped_indices, const int* offsets,
    int* out_in_maps, int* out_out_maps,
    int num_kernel_offsets, int num_query_coords)
{
    map_found_indices_to_maps_kernel(
        found_in_coord_index, mapped_indices, offsets,
        out_in_maps, out_out_maps,
        num_kernel_offsets, num_query_coords);
}

// --- Specialized Kernel for 4D Coordinates and Kernel Size ---

// Optimized kernel for 4D coordinates (batch, x, y, z) when kernel is defined by size
template<typename HashFuncT>
__global__ void kernel_map_size_4d_templated(
    const int* __restrict__ table_kvs,            // Hash table key-value store (capacity, 2)
    const int* __restrict__ vector_keys,          // Original stored keys (num_in_keys, 4)
    const int* __restrict__ query_coords,         // Coordinates to query (num_query_coords, 4)
    const int* __restrict__ kernel_sizes,                           // Kernel dimensions (kx, ky, kz)
    int* __restrict__ found_in_coord_index, // Output array (kx*ky*kz, num_query_coords)
    int num_query_coords,
    int table_capacity,
    bool skip_symmetric_kernel_map)  // Added parameter
{
    // Thread ID corresponds to the query coordinate index (x dimension)
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread ID corresponds to the kernel position index (y dimension)
    int kernel_map_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (query_idx >= num_query_coords) {
        return;
    }

    const int key_dim = 4;
    const int* base_query_coord_ptr = &query_coords[query_idx * key_dim];

    // Calculate center offset for kernel (handle even sizes like Warp)
    // If even, center is 0, effectively shifting the kernel relative to the query point.
    int3 center;
    center.x = (kernel_sizes[0] % 2 != 0) ? kernel_sizes[0] / 2 : 0;
    center.y = (kernel_sizes[1] % 2 != 0) ? kernel_sizes[1] / 2 : 0;
    center.z = (kernel_sizes[2] % 2 != 0) ? kernel_sizes[2] / 2 : 0;

    // Calculate total number of kernel positions and effective limit for symmetric skipping
    int total_kernel_positions = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
    int effective_kernel_limit = skip_symmetric_kernel_map ? total_kernel_positions / 2 : total_kernel_positions;

    // Check if this kernel position should be processed
    if (kernel_map_idx >= effective_kernel_limit) {
        return;
    }

    // Convert linear kernel_map_idx back to 3D indices (i, j, k)
    int k = kernel_map_idx % kernel_sizes[2];
    int j = (kernel_map_idx / kernel_sizes[2]) % kernel_sizes[1];
    int i = kernel_map_idx / (kernel_sizes[2] * kernel_sizes[1]);

    int temp_coord[4]; // Stack allocation for the temporary coordinate

    // Set batch index (doesn't change)
    temp_coord[0] = base_query_coord_ptr[0];

    // Calculate the coordinate to search for
    temp_coord[1] = base_query_coord_ptr[1] + i - center.x;
    temp_coord[2] = base_query_coord_ptr[2] + j - center.y;
    temp_coord[3] = base_query_coord_ptr[3] + k - center.z;

    // Search for the calculated coordinate in the hash table
    int found_index = search_hash_table<HashFuncT>(
        table_kvs,
        vector_keys,
        temp_coord,
        key_dim, // key_dim is 4
        table_capacity
    );

    // Store the result in the output array [kernel_map_idx, query_idx]
    // Output is K * M, index = kernel_map_idx * M + query_idx
    found_in_coord_index[kernel_map_idx * num_query_coords + query_idx] = found_index;
}

// --- Extern "C" Wrappers ---

// kernel_map_size_4d wrappers
// kernel_map_size_4d wrappers with skip_symmetric_kernel_map
extern "C" __global__ void kernel_map_size_4d_fnv1a(
    const int* table_kvs, const int* vector_keys, const int* query_coords,
    const int* kernel_sizes, int* found_in_coord_index,
    int num_query_coords, int table_capacity, bool skip_symmetric_kernel_map)
{
    kernel_map_size_4d_templated<FNV1AHash>(
        table_kvs, vector_keys, query_coords, kernel_sizes, found_in_coord_index,
        num_query_coords, table_capacity, skip_symmetric_kernel_map);
}

extern "C" __global__ void kernel_map_size_4d_city(
    const int* table_kvs, const int* vector_keys, const int* query_coords,
    const int* kernel_sizes, int* found_in_coord_index,
    int num_query_coords, int table_capacity, bool skip_symmetric_kernel_map)
{
    kernel_map_size_4d_templated<CityHash>(
        table_kvs, vector_keys, query_coords, kernel_sizes, found_in_coord_index,
        num_query_coords, table_capacity, skip_symmetric_kernel_map);
}

extern "C" __global__ void kernel_map_size_4d_murmur(
    const int* table_kvs, const int* vector_keys, const int* query_coords,
    const int* kernel_sizes, int* found_in_coord_index,
    int num_query_coords, int table_capacity, bool skip_symmetric_kernel_map)
{
    kernel_map_size_4d_templated<MurmurHash>(
        table_kvs, vector_keys, query_coords, kernel_sizes, found_in_coord_index,
        num_query_coords, table_capacity, skip_symmetric_kernel_map);
}
