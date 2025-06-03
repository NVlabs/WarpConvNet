# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Optional
from jaxtyping import Int

from dataclasses import dataclass

from torch import Tensor

from warpconvnet.geometry.coords.search.torch_discrete import _int_sequence_hash, string_hash
from warpconvnet.geometry.coords.search.utils import _int_tensor_hash

from warpconvnet.geometry.coords.search.search_results import IntSearchResult, RealSearchResult
from warpconvnet.geometry.coords.search.search_configs import RealSearchConfig


@dataclass
class RealSearchCacheKey:

    search_args: RealSearchConfig
    ref_offsets: Int[Tensor, "B+1"]  # noqa: F821
    query_offsets: Int[Tensor, "B+1"]  # noqa: F821

    def __init__(self, search_args: RealSearchConfig, ref_offsets: Tensor, query_offsets: Tensor):
        self.search_args = search_args
        self.ref_offsets = ref_offsets.detach().cpu().int()
        self.query_offsets = query_offsets.detach().cpu().int()

    def __hash__(self):
        return int(
            hash(self.search_args)
            ^ _int_tensor_hash(self.ref_offsets)
            ^ _int_tensor_hash(self.query_offsets)
        )

    def __eq__(self, other: "RealSearchCacheKey"):
        if not isinstance(other, RealSearchCacheKey):
            return False
        return (
            self.search_args == other.search_args
            and self.ref_offsets.equal(other.ref_offsets)
            and self.query_offsets.equal(other.query_offsets)
        )

    def __repr__(self):
        return f"CacheKey(search_args={self.search_args}, ref_offsets={self.ref_offsets}, query_offsets={self.query_offsets})"


@dataclass
class RealSearchCache:

    _search_cache: dict[RealSearchCacheKey, RealSearchResult]

    def __init__(self):
        self._search_cache = {}

    def get(self, search_args: RealSearchConfig, ref_offsets: Tensor, query_offsets: Tensor):
        key = RealSearchCacheKey(search_args, ref_offsets, query_offsets)
        return self._search_cache.get(key)

    def put(
        self,
        search_args: RealSearchConfig,
        ref_offsets: Tensor,
        query_offsets: Tensor,
        result: RealSearchResult,
    ):
        key = RealSearchCacheKey(search_args, ref_offsets, query_offsets)
        self._search_cache[key] = result

    def __getstate__(self):
        # Exclude the cache from being pickled
        return None

    def __setstate__(self, state):
        # Restore the cache as an empty dictionary
        self._search_cache = {}

    def __repr__(self):
        return f"Cache({len(self._search_cache)} keys)"


@dataclass
class IntSearchCacheKey:

    kernel_size: Tuple[int, ...]
    kernel_dilation: Tuple[int, ...]
    transposed: bool
    generative: bool
    stride_mode: str
    skip_symmetric_kernel_map: bool
    in_offsets: Int[Tensor, "B+1"]  # noqa: F821
    out_offsets: Int[Tensor, "B+1"]  # noqa: F821

    def __init__(
        self,
        kernel_size,
        kernel_dilation,
        transposed,
        generative,
        stride_mode,
        skip_symmetric_kernel_map,
        in_offsets,
        out_offsets,
    ):
        self.kernel_size = kernel_size
        self.kernel_dilation = kernel_dilation
        self.transposed = transposed
        self.generative = generative
        self.stride_mode = stride_mode
        self.skip_symmetric_kernel_map = skip_symmetric_kernel_map
        self.in_offsets = in_offsets.detach().cpu().int()
        self.out_offsets = out_offsets.detach().cpu().int()

    def __hash__(self):
        return int(
            _int_sequence_hash(self.kernel_size)
            ^ _int_sequence_hash(self.kernel_dilation)
            ^ hash(self.transposed)
            ^ hash(self.generative)
            ^ string_hash(self.stride_mode)  # Use string_hash for stride_mode
            ^ hash(self.skip_symmetric_kernel_map)
            ^ _int_sequence_hash(self.in_offsets.tolist())
            ^ _int_sequence_hash(self.out_offsets.tolist())
        )

    def __eq__(self, other: "IntSearchCacheKey"):
        return (
            self.kernel_size == other.kernel_size
            and self.kernel_dilation == other.kernel_dilation
            and self.transposed == other.transposed
            and self.generative == other.generative
            and self.stride_mode == other.stride_mode
            and self.skip_symmetric_kernel_map == other.skip_symmetric_kernel_map
            and self.in_offsets.equal(other.in_offsets)
            and self.out_offsets.equal(other.out_offsets)
        )

    def __repr__(self):
        return f"IntSearchCacheKey(kernel_size={self.kernel_size}, kernel_dilation={self.kernel_dilation}, transposed={self.transposed}, generative={self.generative}, stride_mode={self.stride_mode}, skip_symmetric_kernel_map={self.skip_symmetric_kernel_map}, in_offsets={self.in_offsets}, out_offsets={self.out_offsets})"


@dataclass
class IntSearchCache:
    cache: dict[IntSearchCacheKey, IntSearchResult]

    def __init__(self):
        self.cache = {}

    def get(self, key: IntSearchCacheKey) -> Optional[IntSearchResult]:
        return self.cache.get(key, None)

    def put(self, key: IntSearchCacheKey, value: IntSearchResult):
        self.cache[key] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.cache)} keys)"

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        self.cache = {}
