# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch.compile support for WarpConvNet.

This module registers custom types with pytree and marks autograd Functions
and data-dependent functions so that ``torch.compile(model)`` works correctly.

Imported automatically by ``warpconvnet.__init__``.
"""

import torch
import torch._dynamo
from torch.utils._pytree import register_pytree_node


# ---------------------------------------------------------------------------
# 1. Register IntSearchResult with pytree
# ---------------------------------------------------------------------------
# IntSearchResult holds GPU tensors (in_maps, out_maps) and a CPU tensor
# (offsets) plus scalar metadata.  Registering it as a pytree leaf lets
# TorchDynamo flatten / unflatten it so that its tensor children
# participate in tracing.
# ---------------------------------------------------------------------------

from warpconvnet.geometry.coords.search.search_results import IntSearchResult


def _isr_flatten(obj: IntSearchResult):
    """Return (children_tensors, context) for pytree."""
    return [obj.in_maps, obj.out_maps, obj.offsets], {
        "identity_map_index": obj.identity_map_index,
    }


def _isr_unflatten(tensors, ctx):
    """Reconstruct IntSearchResult from pytree parts.

    Bypasses the normal constructor to avoid the ``offsets[-1].item()``
    assertion which would cause a graph break under tracing.
    """
    result = object.__new__(IntSearchResult)
    result.in_maps = tensors[0]
    result.out_maps = tensors[1]
    result.offsets = tensors[2]
    result.identity_map_index = ctx["identity_map_index"]
    # Reset lazy caches — they will be rebuilt on first use.
    result._mask_data = None
    result._reverse_mask_data = None
    result._reduced_mask = None
    result._grouped_params_cache = {}
    result._hashtable = None
    result._kernel_size = None
    return result


register_pytree_node(IntSearchResult, _isr_flatten, _isr_unflatten)


# ---------------------------------------------------------------------------
# 2. Mark autograd.Function subclasses with allow_in_graph
# ---------------------------------------------------------------------------
# allow_in_graph tells TorchDynamo to treat the Function as an opaque node
# in the FX graph rather than trying to trace into it.  The forward and
# backward methods run in eager mode inside the compiled graph.
# ---------------------------------------------------------------------------

from warpconvnet.nn.functional.normalizations import (
    SegmentedRangeNormFunction,
    SegmentedLayerNormFunction,
)
from warpconvnet.nn.functional.segmented_arithmetics import SegmentedArithmeticFunction
from warpconvnet.nn.functional.sparse_conv.detail.unified import (
    UnifiedSpatiallySparseConvFunction,
)
from warpconvnet.nn.functional.sparse_conv.detail.explicit import (
    SpatiallySparseConvExplicitGEMMFunction,
)
from warpconvnet.nn.functional.sparse_conv.detail.implicit_direct import (
    SpatiallySparseConvImplicitGEMMFunction,
)
from warpconvnet.nn.functional.sparse_conv.detail.cutlass import (
    SpatiallySparseConvCutlassImplicitGEMMFunction,
)

_AUTOGRAD_FUNCTIONS = [
    SegmentedRangeNormFunction,
    SegmentedLayerNormFunction,
    SegmentedArithmeticFunction,
    UnifiedSpatiallySparseConvFunction,
    SpatiallySparseConvExplicitGEMMFunction,
    SpatiallySparseConvImplicitGEMMFunction,
    SpatiallySparseConvCutlassImplicitGEMMFunction,
]

try:
    from warpconvnet.nn.functional.sparse_conv_depth import (
        UnifiedSpatiallySparseDepthwiseConvFunction,
    )

    _AUTOGRAD_FUNCTIONS.append(UnifiedSpatiallySparseDepthwiseConvFunction)
except ImportError:
    pass

# Optional backends — only register if available.
try:
    from warpconvnet.nn.functional.sparse_conv.detail.cute import (
        SpatiallySparseConvCuteImplicitGEMMFunction,
    )

    _AUTOGRAD_FUNCTIONS.append(SpatiallySparseConvCuteImplicitGEMMFunction)
except ImportError:
    pass

try:
    from warpconvnet.nn.functional.sparse_conv.detail.cute_grouped import (
        CuteGroupedSpatiallySparseConvFunction,
    )

    _AUTOGRAD_FUNCTIONS.append(CuteGroupedSpatiallySparseConvFunction)
except ImportError:
    pass

for fn_cls in _AUTOGRAD_FUNCTIONS:
    torch._dynamo.allow_in_graph(fn_cls)


# ---------------------------------------------------------------------------
# 3. Note on torch.compiler.disable decorators
# ---------------------------------------------------------------------------
# The following functions have @torch.compiler.disable applied directly in
# their source files (not monkey-patched here) because other modules import
# them before _compile.py runs:
#
#   - spatially_sparse_conv          (helper.py)
#   - generate_output_coords_and_kernel_map  (helper.py)
#   - generate_kernel_map            (torch_discrete.py)
#
# These functions contain .item() calls, data-dependent control flow, or
# operate on custom Python objects (Voxels, IntCoords) that Dynamo cannot
# trace.  The @torch.compiler.disable decorator inserts clean graph breaks
# so that surrounding code can still be compiled.
# ---------------------------------------------------------------------------
