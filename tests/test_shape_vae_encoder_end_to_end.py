# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test: ``mesh_to_fdg`` -> frozen TRELLIS.2 shape encoder.

Joins two pieces we already verified independently:

* ``streaming_ovoxel.mesh_to_fdg`` produces ``(coords, dual_v_world, intersected)``
  for a watertight mesh.
* ``FlexiDualGridVaeEncoder`` loads the published 709 MB safetensors with
  ``missing=0 unexpected=0`` (covered by
  ``test_shape_vae_encoder_structure.py``).

This test converts the world-frame dual vertices to per-voxel ``[0, 1]``
offsets (the form the encoder expects), wraps both arrays as ``Voxels``
sharing the same integer coordinates, runs a forward pass with loaded
weights, and asserts the latent is finite and non-degenerate.

Gated: skipped cleanly if either ``streaming_ovoxel`` is not importable
or the safetensors file is not at the expected cache path. Requires CUDA.
"""

from __future__ import annotations

import os
from typing import Tuple

import pytest
import torch

# Deeper import path to dodge ``models/__init__`` aggregate re-exports, same
# as the structural test.
from warpconvnet.models.trellis2.shape_vae import FlexiDualGridVaeEncoder
from warpconvnet.geometry.types.voxels import Voxels


# Config copied verbatim from the structural test (which itself mirrors
# ``TRELLIS.2/configs/scvae/shape_vae_next_dc_f16c32_fp16.json``).
_SHAPE_ENC_CONFIG = dict(
    model_channels=[64, 128, 256, 512, 1024],
    latent_channels=32,
    num_blocks=[0, 4, 8, 16, 4],
    block_type=["SparseConvNeXtBlock3d"] * 5,
    down_block_type=["SparseResBlockS2C3d"] * 4,
    block_args=[
        {"use_checkpoint": True},
        {"use_checkpoint": True},
        {"use_checkpoint": False},
        {"use_checkpoint": False},
        {"use_checkpoint": False},
    ],
    use_fp16=True,
)

_SAFETENSORS_PATH = os.environ.get(
    "TRELLIS2_SHAPE_ENC",
    "/lustre/fsw/portfolios/nvr/projects/nvr_lpr_nvgptvision"
    "/huggingface_cache/trellis2_shape_enc/shape_enc_next_dc_f16c32_fp16.safetensors",
)


def _derive_mesh_to_fdg_aabb_min(
    vertices: torch.Tensor, voxel_size: float
) -> torch.Tensor:
    """Replicate the auto-AABB derivation in ``mesh_to_fdg``.

    Mirrors ``streaming_ovoxel/mesh_to_fdg.py:174-185``: per-axis padding
    of ``ceil((vmax - vmin) / vs) * vs - (vmax - vmin)`` is split evenly
    on either side, giving the origin ``aabb[0] = vmin - padding * 0.5``.
    """
    vmin = vertices.min(dim=0).values
    vmax = vertices.max(dim=0).values
    vs = torch.full_like(vmin, float(voxel_size))
    padding = torch.ceil((vmax - vmin) / vs) * vs - (vmax - vmin)
    return vmin - padding * 0.5


def test_encoder_accepts_mesh_to_fdg_output() -> None:
    """Round-trip: icosphere -> mesh_to_fdg -> Voxels -> encoder -> latent.

    Asserts the latent is finite, has the expected channel width, and has
    non-trivial variance (a frozen encoder fed real geometry should never
    collapse to constant output).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for mesh_to_fdg and the shape encoder.")

    try:
        import trimesh  # noqa: F401
    except ImportError:
        pytest.skip("trimesh not available.")

    try:
        from streaming_ovoxel.mesh_to_fdg import mesh_to_fdg
    except ImportError:
        pytest.skip("streaming_ovoxel.mesh_to_fdg not importable.")

    if not os.path.exists(_SAFETENSORS_PATH):
        pytest.skip(f"Shape-encoder safetensors not found at {_SAFETENSORS_PATH}.")

    from safetensors.torch import load_file

    import trimesh

    # 1. Build an icosphere on CUDA.
    mesh = trimesh.creation.icosphere(subdivisions=3)
    verts = torch.as_tensor(mesh.vertices, dtype=torch.float32, device="cuda")
    faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device="cuda")

    # 2. Voxelize via mesh_to_fdg -> (coords, dual_v_world, intersected).
    voxel_size = 0.05
    coords, dual_v_world, intersected = mesh_to_fdg(
        verts, faces, voxel_size=voxel_size
    )
    assert coords.shape[0] > 0, "icosphere produced zero occupied voxels"

    # 3. World-frame dual vertices -> per-voxel [0, 1] offsets. The origin
    #    must match mesh_to_fdg's internal aabb[0] (see helper above).
    origin = _derive_mesh_to_fdg_aabb_min(verts, voxel_size)
    offsets_world_local: torch.Tensor = (
        (dual_v_world - origin[None, :]) / voxel_size - coords.float()
    ).clamp(0.0, 1.0)

    # 4. Wrap as two Voxels sharing coords. The encoder concatenates
    #    ``vertices.feats`` (3-ch fp32 offsets) and ``intersected.feats``
    #    (3-ch bool->fp32) along the feature dim into a 6-ch input.
    coords_i32 = coords.to(torch.int32).contiguous()
    batch_offsets = torch.tensor([0, coords_i32.shape[0]], dtype=torch.int64)
    vertices_voxels = Voxels(
        coords_i32,
        offsets_world_local.contiguous(),
        batch_offsets,
        voxel_size=voxel_size,
        device="cuda",
    )
    intersected_voxels = Voxels(
        coords_i32,
        intersected.to(torch.float32).contiguous(),
        batch_offsets,
        voxel_size=voxel_size,
        device="cuda",
    )

    # 5. Build encoder and load published weights.
    enc = FlexiDualGridVaeEncoder(**_SHAPE_ENC_CONFIG)
    raw_sd = load_file(_SAFETENSORS_PATH)
    enc.load_trellis2_state_dict(raw_sd, strict=False)
    enc = enc.cuda().eval()

    # 6. Forward.
    with torch.no_grad():
        z = enc(vertices_voxels, intersected_voxels)

    # 7. Assertions on the latent.
    z_feats: torch.Tensor = z.feats
    assert torch.isfinite(z_feats).all(), "latent contains NaN/Inf"
    assert z_feats.shape[-1] == _SHAPE_ENC_CONFIG["latent_channels"], (
        f"expected latent_channels="
        f"{_SHAPE_ENC_CONFIG['latent_channels']}, got {z_feats.shape[-1]}"
    )
    std: float = z_feats.float().std().item()
    assert std > 1e-3, f"latent has near-zero variance ({std:.2e}); encoder likely broken"

    print(
        f"[e2e] coords={tuple(coords_i32.shape)} "
        f"latent={tuple(z_feats.shape)} std={std:.4f}"
    )


if __name__ == "__main__":
    test_encoder_accepts_mesh_to_fdg_output()
