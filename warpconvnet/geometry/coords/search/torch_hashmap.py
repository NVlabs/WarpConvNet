# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum
import math
import os
from typing import Union, Optional

import cupy as cp
import numpy as np
import torch
from torch import Tensor

from warpconvnet.utils.cuda_utils import load_kernel


class HashMethod(enum.Enum):
    """Hash method enumeration for the vector hash table.

    Attributes:
        FNV1A: FNV-1a hash algorithm
        CITY: CityHash algorithm
        MURMUR: MurmurHash algorithm
    """

    FNV1A = 0
    CITY = 1
    MURMUR = 2

    def kernel_suffix(self) -> str:
        """Return the suffix used for templated kernel names."""
        return self.name.lower()  # fnv1a, city, murmur


class TorchHashTable:
    """
    A hash table implementation using CuPy RawKernels for vector key storage and lookup,
    operating on pytorch Tensors. Uses templated CUDA kernels.
    """

    _capacity: int
    _hash_method_enum: HashMethod
    _table_kvs: Tensor = None  # Shape: (capacity, 2), dtype=torch.int32
    _vector_keys: Tensor = None  # Shape: (num_keys, key_dim), dtype=torch.int32
    _key_dim: int = -1
    _device: torch.device = None

    _prepare_kernel: cp.RawKernel
    _insert_kernel: cp.RawKernel  # Will hold the specific insert kernel (e.g., _fnv1a)
    _search_kernel: cp.RawKernel  # Will hold the specific search kernel (e.g., _fnv1a)

    def __init__(
        self,
        capacity: int,
        hash_method: HashMethod = HashMethod.CITY,
        device: Union[str, torch.device] = "cuda",
    ):
        """Initialize the hash table using PyTorch tensors.

        Args:
            capacity: Maximum number of entries the table can store (number of slots)
            hash_method: HashMethod enum value (default: CITY)
            device: The torch device to allocate tensors on (e.g., 'cuda', 'cuda:0').
        """
        if not isinstance(hash_method, HashMethod):
            raise TypeError(
                f"hash_method must be a HashMethod enum member, got {type(hash_method)}"
            )
        assert capacity > 0

        self._capacity = capacity
        self._hash_method_enum = hash_method
        self._device = torch.device(device)

        # Load CUDA kernels using cupy RawKernel loader
        kernel_dir = os.path.dirname(__file__)
        kernel_file_path = os.path.join(kernel_dir, "cuda", "hashmap_kernels.cu")
        self._prepare_kernel = load_kernel("prepare_key_value_pairs_kernel", kernel_file_path)

        # Load the specific insert/search kernels based on the chosen hash method
        suffix = hash_method.kernel_suffix()
        self._insert_kernel = load_kernel(f"insert_kernel_{suffix}", kernel_file_path)
        self._search_kernel = load_kernel(f"search_kernel_{suffix}", kernel_file_path)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def hash_method(self) -> HashMethod:
        return self._hash_method_enum

    @property
    def device(self) -> torch.device:
        if self._table_kvs is not None:
            return self._table_kvs.device
        return self._device  # Return the device specified during init

    @property
    def key_dim(self) -> int:
        return self._key_dim

    def insert(self, vec_keys: Tensor, threads_per_block: int = 256):
        """Insert vector keys (PyTorch Tensor) into the hash table.

        Args:
            vec_keys: PyTorch tensor of int32 vector keys, shape (num_keys, key_dim), on CUDA device.

        Raises:
            AssertionError: If capacity is invalid or number of keys exceeds capacity/2.
            TypeError: If input is not a Torch Tensor or has wrong dtype.
            ValueError: If input is not 2D or not on a CUDA device.
        """
        if not isinstance(vec_keys, torch.Tensor):
            raise TypeError(f"Input vec_keys must be a PyTorch Tensor, got {type(vec_keys)}")
        if not vec_keys.is_cuda:
            raise ValueError(f"Input vec_keys must be on a CUDA device, got {vec_keys.device}")
        if vec_keys.ndim != 2:
            raise ValueError(f"Input vec_keys must be 2D, got {vec_keys.ndim} dimensions")

        # Ensure correct device and dtype
        if vec_keys.device != self.device:
            vec_keys = vec_keys.to(self.device)

        if vec_keys.dtype != torch.int32:
            if vec_keys.dtype == torch.int64:
                # print(f"Warning: Input vec_keys dtype is {vec_keys.dtype}, casting to torch.int32.")
                vec_keys = vec_keys.to(dtype=torch.int32)
            else:
                raise TypeError(
                    f"Input vec_keys must have dtype torch.int32, got {vec_keys.dtype}"
                )

        num_keys, key_dim = vec_keys.shape
        self._key_dim = key_dim

        assert self._capacity > 0
        assert (
            num_keys <= self._capacity / 2
        ), f"Number of keys {num_keys} exceeds recommended capacity/2 ({self._capacity / 2}) for table size {self._capacity}"

        # Allocate table on the target device
        self._table_kvs = torch.empty((self._capacity, 2), dtype=torch.int32, device=self.device)
        # Store reference to original keys (already checked for device/dtype)
        self._vector_keys = vec_keys

        # --- Launch Prepare Kernel ---
        grid_size_prep = math.ceil(self._capacity / threads_per_block)
        self._prepare_kernel(
            (grid_size_prep,),
            (threads_per_block,),
            (self._table_kvs.data_ptr(), self._capacity),  # Pass data pointer
        )
        torch.cuda.synchronize(self.device)  # Synchronize on the correct device

        # --- Launch Insert Kernel (Templated Version) ---
        grid_size_ins = math.ceil(num_keys / threads_per_block)
        self._insert_kernel(
            (grid_size_ins,),
            (threads_per_block,),
            (
                self._table_kvs.data_ptr(),  # ptr table_kvs
                self._vector_keys.data_ptr(),  # ptr vector_keys
                num_keys,  # int num_keys
                self._key_dim,  # int key_dim
                self._capacity,
            ),  # int table_capacity
        )

        # TODO(cchoy): Use dlpack to skip explicit synchronization
        torch.cuda.synchronize(self.device)  # Synchronize

    @classmethod
    def from_keys(
        cls,
        vec_keys: Union[Tensor, np.ndarray],
        hash_method: HashMethod = HashMethod.CITY,
        device: Union[str, torch.device] = "cuda",
    ):
        """Create a hash table from a set of vector keys.

        Args:
            vec_keys: Vector keys (PyTorch Tensor or NumPy array). If NumPy, moved to `device`.
            hash_method: HashMethod enum value to use (default: CITY)
            device: The torch device to use.

        Returns:
            TorchHashTable: New hash table instance containing the keys.
        """
        target_device = torch.device(device)

        if not isinstance(vec_keys, torch.Tensor):
            # If NumPy or other array-like, convert to Tensor on the target device
            try:
                vec_keys = torch.as_tensor(vec_keys, device=target_device)
            except Exception as e:
                raise TypeError(
                    f"Could not convert input vec_keys to a Torch Tensor on device {target_device}: {e}"
                )

        # Ensure correct device and dtype
        if vec_keys.device != target_device:
            vec_keys = vec_keys.to(target_device)

        if vec_keys.dtype != torch.int32:
            if vec_keys.dtype == torch.int64:
                # print(f"Warning: Input vec_keys dtype is {vec_keys.dtype}, casting to torch.int32 for from_keys.")
                vec_keys = vec_keys.to(dtype=torch.int32)
            else:
                raise TypeError(
                    f"Input vec_keys for from_keys must have dtype torch.int32 or compatible, got {vec_keys.dtype}"
                )

        if vec_keys.ndim != 2:
            raise ValueError(
                f"Input vec_keys for from_keys must be 2D, got {vec_keys.ndim} dimensions"
            )

        num_keys = len(vec_keys)
        capacity = max(16, int(num_keys * 2))
        # Pass the hash_method and device to the constructor
        obj = cls(capacity=capacity, hash_method=hash_method, device=target_device)
        obj.insert(vec_keys)
        return obj

    def search(
        self, search_keys: Union[Tensor, np.ndarray], threads_per_block: int = 256
    ) -> Tensor:
        """Search for keys (PyTorch Tensor) in the hash table.

        Args:
            search_keys: Keys to search for (Torch Tensor or NumPy array). If NumPy, moved to table's device.
                         Shape (num_search, key_dim).

        Returns:
            torch.Tensor: Array of indices (int32) where keys were found in the original
                          `vector_keys` tensor. -1 if not found. On the same device as the table.
        Raises:
            RuntimeError: If insert() has not been called yet.
            ValueError: If search_keys dimensions don't match inserted keys.
            TypeError: If search_keys cannot be converted to a Tensor or have wrong dtype.
        """
        if self._table_kvs is None or self._vector_keys is None:
            raise RuntimeError(
                "Hash table has not been populated. Call insert() or from_keys() first."
            )

        table_device = self.device

        if not isinstance(search_keys, torch.Tensor):
            try:
                # Convert NumPy or other to Tensor on the hash table's device
                search_keys = torch.as_tensor(search_keys, device=table_device)
            except Exception as e:
                raise TypeError(
                    f"Could not convert input search_keys to a Torch Tensor on device {table_device}: {e}"
                )

        # Ensure correct device and dtype
        if search_keys.device != table_device:
            search_keys = search_keys.to(table_device)

        if search_keys.ndim != 2:
            raise ValueError(f"Input search_keys must be 2D, got {search_keys.ndim} dimensions")
        if search_keys.shape[1] != self._key_dim:
            raise ValueError(
                f"Search keys dimension ({search_keys.shape[1]}) must match "
                f"inserted keys dimension ({self._key_dim})"
            )

        if search_keys.dtype != torch.int32:
            if search_keys.dtype == torch.int64:
                # print(f"Warning: Input search_keys dtype is {search_keys.dtype}, casting to torch.int32.")
                search_keys = search_keys.to(dtype=torch.int32)
            else:
                raise TypeError(
                    f"Input search_keys must have dtype torch.int32 or compatible, got {search_keys.dtype}"
                )

        num_search_keys = len(search_keys)
        # Allocate results tensor on the correct device
        results = torch.empty(num_search_keys, dtype=torch.int32, device=table_device)

        # --- Launch Search Kernel (Templated Version) ---
        grid_size_search = math.ceil(num_search_keys / threads_per_block)

        self._search_kernel(
            (grid_size_search,),
            (threads_per_block,),
            (
                self._table_kvs.data_ptr(),  # ptr table_kvs
                self._vector_keys.data_ptr(),  # ptr vector_keys
                search_keys.data_ptr(),  # ptr search_keys
                results.data_ptr(),  # ptr results
                num_search_keys,  # int num_search_keys
                self._key_dim,  # int key_dim
                self._capacity,
            ),  # int table_capacity
        )
        torch.cuda.synchronize(table_device)  # Synchronize

        return results

    @property
    def unique_index(self) -> Tensor:
        """Get sorted unique indices from the hash table.

        Returns:
            torch.Tensor: Sorted tensor of unique indices (int32) corresponding
                          to the originally inserted keys. On the same device as the table.
        Raises:
            RuntimeError: If insert() has not been called yet.
        """
        if self._vector_keys is None:
            raise RuntimeError(
                "Hash table has not been populated. Call insert() or from_keys() first."
            )

        indices = self.search(self._vector_keys)
        valid_indices = indices[indices != -1]
        # torch.unique returns sorted unique values
        unique_indices = torch.unique(valid_indices)
        return unique_indices

    @property
    def vector_keys(self) -> Tensor:
        """Return the 2D vector keys tensor used to build the hash table."""
        if self._vector_keys is None:
            raise RuntimeError(
                "Hash table has not been populated. Call insert() or from_keys() first."
            )
        return self._vector_keys

    @property
    def unique_vector_keys(self) -> Tensor:
        """Return the unique 2D vector keys present in the hash table, sorted by index.

        Returns:
            torch.Tensor: Tensor shape (num_unique_keys, key_dim), dtype int32.
                          On the same device as the table.
        Raises:
            RuntimeError: If insert() has not been called yet.
        """
        unique_idx = self.unique_index
        if unique_idx.numel() == 0:  # Use numel() for checking empty tensors
            # Return an empty tensor with the correct shape, dtype, and device
            return torch.empty((0, self.key_dim), dtype=torch.int32, device=self.device)
        return self._vector_keys[unique_idx]

    def to_dict(self) -> dict:
        """Serializes the data arrays and metadata (transfers tensors to CPU as NumPy arrays)."""
        if self._table_kvs is None or self._vector_keys is None:
            table_kvs_np = None
            vec_keys_np = None
        else:
            # Ensure tensors are on CPU before converting to numpy
            table_kvs_np = self._table_kvs.cpu().numpy()
            vec_keys_np = self._vector_keys.cpu().numpy()

        return {
            "table_kvs": table_kvs_np,
            "vec_keys": vec_keys_np,
            "hash_method_value": self._hash_method_enum.value,
            "capacity": self._capacity,
            "key_dim": self._key_dim,
            "device": str(self.device),  # Store device as string
        }

    def from_dict(self, data: dict):
        """
        Loads data from a dict and re-initializes the hash table state.
        Assumes data arrays are numpy arrays. Tensors are created on the specified device.
        Requires re-compilation of kernels implicitly via __init__.
        """
        required_keys = {
            "capacity",
            "hash_method_value",
            "key_dim",
            "table_kvs",
            "vec_keys",
            "device",
        }
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"Data dictionary missing required keys. Need: {required_keys}")

        capacity = data["capacity"]
        hash_method_value = data["hash_method_value"]
        key_dim = data["key_dim"]
        table_kvs_np = data["table_kvs"]
        vec_keys_np = data["vec_keys"]
        device_str = data["device"]
        target_device = torch.device(device_str)

        # Re-initialize the object with the correct capacity, hash method, and device
        self.__init__(
            capacity=capacity, hash_method=HashMethod(hash_method_value), device=target_device
        )

        self._key_dim = key_dim

        # Load data if present, creating tensors on the target device
        if table_kvs_np is not None:
            self._table_kvs = torch.as_tensor(
                table_kvs_np, dtype=torch.int32, device=target_device
            )
            assert self._table_kvs.shape == (self._capacity, 2)
            assert self._table_kvs.dtype == torch.int32
        else:
            self._table_kvs = None

        if vec_keys_np is not None:
            self._vector_keys = torch.as_tensor(
                vec_keys_np, dtype=torch.int32, device=target_device
            )
            assert self._vector_keys.ndim == 2
            # If key_dim wasn't -1 initially, check consistency
            if self._key_dim != -1:
                assert self._vector_keys.shape[1] == self._key_dim
            else:  # Infer key_dim if it wasn't set (e.g., loading into a fresh object)
                self._key_dim = self._vector_keys.shape[1]
            assert self._vector_keys.dtype == torch.int32
        else:
            self._vector_keys = None

        # State is loaded, kernels are implicitly ready from __init__.
        return self
