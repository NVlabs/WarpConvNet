# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 mesh extraction (FlexiDualGrid → triangle mesh).

Wraps `o_voxel.convert.flexible_dual_grid_to_mesh` so the rest of the port
doesn't import the heavy o-voxel package directly. The o-voxel CUDA extension
must be installed separately (``pip install -e /path/to/TRELLIS.2/o-voxel``);
that package is the upstream's mesh-extraction backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.trellis2.sparse_ops import sparse_unbind


__all__ = ["MeshOut", "extract_meshes"]


@dataclass
class MeshOut:
    """Per-batch mesh extracted from FlexiDualGridVaeDecoder output.

    `vertices` and `faces` are CPU/GPU tensors compatible with downstream
    renderers (e.g. trimesh, nvdiffrast). `coords`, `attrs`, `voxel_size`,
    and `aabb` are kept so the caller can run o-voxel's `to_glb` /
    decimation step without re-deriving them.
    """

    vertices: torch.Tensor  # (V, 3)
    faces: torch.Tensor  # (F, 3) int
    coords: torch.Tensor  # (N, 3) input voxel coords
    voxel_size: float
    aabb: list[list[float]]


def _maybe_import_o_voxel():
    try:
        from o_voxel.convert import flexible_dual_grid_to_mesh
    except ImportError as e:
        raise ImportError(
            "TRELLIS.2 mesh extraction requires the `o_voxel` package. "
            "Build it via: pip install -e /path/to/TRELLIS.2/o-voxel "
            "(after `git submodule update --init` to populate eigen)."
        ) from e
    return flexible_dual_grid_to_mesh


def _fill_holes_inplace(
    vertices: torch.Tensor, faces: torch.Tensor, max_hole_perimeter: float = 3e-2
):
    """Patch boundary loops in a triangle mesh using cumesh (matches upstream)."""
    try:
        import cumesh
    except ImportError:
        return vertices, faces
    if vertices.device.type != "cuda":
        vertices = vertices.cuda()
        faces = faces.cuda()
    m = cumesh.CuMesh()
    m.init(vertices, faces)
    m.get_edges()
    m.get_boundary_info()
    if m.num_boundaries == 0:
        return vertices, faces
    m.get_vertex_edge_adjacency()
    m.get_vertex_boundary_adjacency()
    m.get_manifold_boundary_adjacency()
    m.read_manifold_boundary_adjacency()
    m.get_boundary_connected_components()
    m.get_boundary_loops()
    if m.num_boundary_loops == 0:
        return vertices, faces
    m.fill_holes(max_hole_perimeter=max_hole_perimeter)
    return m.read()


def extract_meshes(
    vertices: Voxels,
    intersected: Voxels,
    quad_lerp: Voxels | None,
    grid_size: int,
    aabb: list | tuple = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    train: bool = False,
    fill_holes: bool = True,
) -> list[MeshOut]:
    """Run FDG mesh extraction per-batch.

    `vertices`, `intersected`, `quad_lerp` come from
    `FlexiDualGridVaeDecoder.forward` and share coords / offsets. Each batch
    element is unbound and passed independently to o-voxel's CUDA kernel.
    """
    fdg_to_mesh = _maybe_import_o_voxel()

    v_list = sparse_unbind(vertices, dim=0)
    i_list = sparse_unbind(intersected, dim=0)
    q_list = sparse_unbind(quad_lerp, dim=0) if quad_lerp is not None else [None] * len(v_list)

    aabb_list = [list(a) for a in aabb]
    voxel_size = (aabb_list[1][0] - aabb_list[0][0]) / grid_size

    out: list[MeshOut] = []
    for v, i, q in zip(v_list, i_list, q_list):
        coords = v.coords[:, 1:]  # drop batch col → (N, 3)
        verts, faces = fdg_to_mesh(
            coords=coords,
            dual_vertices=v.feats,
            intersected_flag=i.feats,
            split_weight=q.feats if q is not None else None,
            aabb=aabb_list,
            grid_size=grid_size,
            train=train,
        )
        if fill_holes:
            verts, faces = _fill_holes_inplace(verts, faces)
        out.append(
            MeshOut(
                vertices=verts,
                faces=faces,
                coords=coords,
                voxel_size=float(voxel_size),
                aabb=aabb_list,
            )
        )
    return out
