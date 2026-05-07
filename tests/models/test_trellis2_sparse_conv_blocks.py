# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 9 tests: SparseConv3d + SparseConvNeXtBlock3d.

Sparse conv requires CUDA + a built warpconvnet kernel; non-CUDA hosts
skip forward tests but keep state_dict / shape sanity coverage.
"""
import importlib
import os
import sys

import pytest
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.trellis2.sparse_conv_blocks import (
    SparseConv3d,
    SparseConvNeXtBlock3d,
)
from warpconvnet.models.trellis2.sparse_ops import from_feats_coords

_HAS_CUDA = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
_REF_SPARSE_CONV_BACKEND = "flex" + "_gemm"

_TRELLIS2_PATH = os.environ.get("TRELLIS2_PATH")
_HAS_REF = False
if _TRELLIS2_PATH and os.path.isdir(_TRELLIS2_PATH):
    os.environ["ATTN_BACKEND"] = "flash_attn"
    os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
    os.environ.setdefault("SPARSE_CONV_BACKEND", _REF_SPARSE_CONV_BACKEND)
    if _TRELLIS2_PATH not in sys.path:
        sys.path.insert(0, _TRELLIS2_PATH)
    try:
        from trellis2.modules.sparse.basic import SparseTensor as RefSparseTensor
        from trellis2.modules.sparse.conv import SparseConv3d as RefSparseConv3d

        _HAS_REF = importlib.util.find_spec(_REF_SPARSE_CONV_BACKEND) is not None
    except Exception:  # noqa: BLE001
        _HAS_REF = False


def _make_voxels(B: int = 2, N_per: int = 32, C: int = 16, R: int = 8, seed: int = 0) -> Voxels:
    g = torch.Generator().manual_seed(seed)
    coords_list, feats_list = [], []
    for b in range(B):
        flat = torch.randperm(R**3, generator=g)[:N_per]
        x = flat // (R * R)
        y = (flat // R) % R
        z = flat % R
        coords_list.append(torch.stack([torch.full_like(x, b), x, y, z], dim=-1).int())
        feats_list.append(torch.randn(N_per, C, generator=g))
    return from_feats_coords(torch.cat(feats_list, dim=0), torch.cat(coords_list, dim=0))


# -----------------------------------------------------------------------------
# State_dict layout
# -----------------------------------------------------------------------------
def test_sparse_conv3d_weight_layout():
    """Weight uses native WarpConvNet ``(K^3, Cin, Cout)`` layout."""
    m = SparseConv3d(in_channels=8, out_channels=16, kernel_size=3)
    assert m.weight.shape == (27, 8, 16)
    assert m.bias.shape == (16,)


def test_convnext_block_state_dict_keys():
    blk = SparseConvNeXtBlock3d(channels=32, mlp_ratio=4.0)
    keys = set(blk.state_dict().keys())
    assert {"norm.weight", "norm.bias"} <= keys
    assert "conv.weight" in keys and "conv.bias" in keys
    assert {"mlp.0.weight", "mlp.2.weight"} <= keys


def test_sparse_conv3d_is_native_warpconvnet_module():
    down = SparseConv3d(8, 16, 3, stride=2)
    assert down.stride == (2, 2, 2)


# -----------------------------------------------------------------------------
# CUDA forward
# -----------------------------------------------------------------------------
@_skip_no_cuda
def test_sparse_conv3d_forward_shape():
    torch.manual_seed(0)
    m = SparseConv3d(in_channels=16, out_channels=32, kernel_size=3).cuda()
    v = _make_voxels(B=2, N_per=32, C=16, R=8).to("cuda")
    out = m(v)
    assert isinstance(out, Voxels)
    assert out.feats.shape == (v.feats.shape[0], 32)


@_skip_no_cuda
def test_convnext_block_forward_shape():
    torch.manual_seed(0)
    blk = SparseConvNeXtBlock3d(channels=32, mlp_ratio=4.0).cuda()
    v = _make_voxels(B=2, N_per=32, C=32, R=8).to("cuda")
    out = blk(v)
    assert out.feats.shape == v.feats.shape


# -----------------------------------------------------------------------------
# Reference parity
# -----------------------------------------------------------------------------
@_skip_no_cuda
def test_sparse_conv3d_matches_dense_conv3d_on_full_grid():
    """Cross-check kernel ordering against `torch.nn.Conv3d` on a fully-occupied
    grid. Validates that native WarpConvNet sparse-conv weights map to the same
    kernel offsets as dense `torch.nn.Conv3d`.
    """
    torch.manual_seed(0)
    Cin, Cout, R = 4, 6, 4
    # Reference dense Conv3d (kernel=3, padding=1 ⇒ same output size).
    ref = torch.nn.Conv3d(Cin, Cout, kernel_size=3, padding=1).cuda()

    # Build a fully-occupied R^3 grid Voxels.
    coords = torch.stack(
        torch.meshgrid(torch.arange(R), torch.arange(R), torch.arange(R), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    coords = torch.cat([torch.zeros(coords.shape[0], 1).int(), coords.int()], dim=-1)
    feats = torch.randn(coords.shape[0], Cin)
    v = from_feats_coords(feats, coords).to("cuda")

    # Pin to a full-precision path: the default autotuner may choose mask_gemm,
    # which uses fp16 inputs internally and is intentionally less bit-exact.
    ours = SparseConv3d(Cin, Cout, kernel_size=3, bias=True, fwd_algo="explicit_gemm").cuda()
    # Copy reference weights into WarpConvNet layout: (Cout, Cin, Kd, Kh, Kw)
    # -> (Kd*Kh*Kw, Cin, Cout).
    ours.weight.data.copy_(ref.weight.data.permute(2, 3, 4, 1, 0).reshape(27, Cin, Cout))
    ours.bias.data.copy_(ref.bias.data)

    # Dense reference forward.
    dense_in = feats.permute(1, 0).reshape(1, Cin, R, R, R).cuda()
    dense_out = ref(dense_in)  # (1, Cout, R, R, R)
    expected = dense_out.reshape(Cout, -1).permute(1, 0)  # (R^3, Cout)

    got = ours(v).feats
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


@_skip_no_cuda
@pytest.mark.skipif(not _HAS_REF, reason="upstream trellis2 sparse conv backend not available")
def test_sparse_conv3d_matches_reference():
    """Bit-parity vs upstream submanifold conv on identical weights + coords."""
    torch.manual_seed(0)
    ours = SparseConv3d(in_channels=16, out_channels=32, kernel_size=3).cuda()
    ref = RefSparseConv3d(16, 32, kernel_size=3).cuda()
    # Convert native WarpConvNet layout to upstream reference layout.
    ref.weight.data.copy_(ours.weight.data.reshape(3, 3, 3, 16, 32).permute(4, 0, 1, 2, 3))
    ref.bias.data.copy_(ours.bias.data)

    v_ours = _make_voxels(B=1, N_per=64, C=16, R=8).to("cuda")
    v_ref = RefSparseTensor(v_ours.feats.clone(), v_ours.coords.clone())
    o = ours(v_ours).feats
    r = ref(v_ref).feats
    torch.testing.assert_close(o, r, rtol=5e-3, atol=5e-3)
