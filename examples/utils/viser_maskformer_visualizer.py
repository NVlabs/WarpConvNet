# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live viser visualizer for MaskFormer instance-segmentation training.

Spins up a viser server and renders three side-by-side voxel scenes that
refresh from the training loop every ``interval_seconds``:

    [ input RGB ]   [ ground-truth instances ]   [ predicted instances ]

Each voxel is drawn as an axis-aligned cube. Instance ids are mapped to a
hash-based color palette so adjacent instances are visually distinct.

Usage from the training loop::

    viz = MaskFormerViserVisualizer(voxel_size=0.04, port=8080,
                                    interval_seconds=10.0)
    # ... after a forward pass:
    viz.maybe_update(
        coords=batch[0]["coords"],
        colors=batch[0]["colors"],
        gt_instance=batch[0]["instance"],
        pred_logits=logits[0],   # (Q, C+1)
        pred_masks=masks[0],     # (Q, N)
        epoch=epoch, step=step,
    )
"""

from __future__ import annotations

import threading
import time

import numpy as np
import torch

try:
    import trimesh
    import viser
except ImportError as exc:  # pragma: no cover - optional dep
    raise ImportError("viser visualization requires `pip install viser trimesh`") from exc

# Voxel that has no instance (gt -1 or pred had no query above threshold).
NO_INSTANCE_COLOR = np.array([60, 60, 60], dtype=np.uint8)

# Reuse the cube primitive used by the ScanNet visualizer.
_CUBE_VERTS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.float32,
)
_CUBE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def _build_voxel_cube_mesh(
    voxel_xyz: np.ndarray, voxel_rgb: np.ndarray, voxel_size: float
) -> trimesh.Trimesh:
    n = voxel_xyz.shape[0]
    if n == 0:
        return trimesh.Trimesh()
    verts = (voxel_xyz[:, None, :].astype(np.float32) + _CUBE_VERTS[None, :, :]) * voxel_size
    faces = _CUBE_FACES[None, :, :] + (np.arange(n, dtype=np.int32)[:, None, None] * 8)
    face_colors = np.repeat(voxel_rgb.astype(np.uint8), _CUBE_FACES.shape[0], axis=0)
    return trimesh.Trimesh(
        vertices=verts.reshape(-1, 3),
        faces=faces.reshape(-1, 3),
        face_colors=face_colors,
        process=False,
    )


def _instance_to_color(instance_ids: np.ndarray) -> np.ndarray:
    """Hash each instance id to a stable RGB. id == -1 → NO_INSTANCE_COLOR."""
    out = np.tile(NO_INSTANCE_COLOR, (instance_ids.shape[0], 1))
    valid = instance_ids >= 0
    if not np.any(valid):
        return out
    ids = instance_ids[valid].astype(np.int64)
    # Hash via golden-ratio-ish multiplier + bit shuffle, then split into RGB.
    h = (ids * np.int64(2654435761)) & np.int64(0xFFFFFFFF)
    r = ((h >> 16) & 0xFF).astype(np.uint8)
    g = ((h >> 8) & 0xFF).astype(np.uint8)
    b = (h & 0xFF).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=1)
    # Lift dim values so colors stay readable.
    rgb = np.clip(rgb.astype(np.int32) + 60, 0, 255).astype(np.uint8)
    out[valid] = rgb
    return out


def _voxelize_with_instances(
    coords: np.ndarray,
    colors: np.ndarray,
    gt_instance: np.ndarray,
    pred_instance: np.ndarray,
    voxel_size: float,
):
    """Bucket points into voxels. Returns
    (voxel_xyz, mean_rgb, gt_inst_mode, pred_inst_mode)."""
    voxel_xyz = np.floor(coords / voxel_size).astype(np.int64)
    unique_xyz, inverse = np.unique(voxel_xyz, axis=0, return_inverse=True)
    n_vox = unique_xyz.shape[0]

    rgb_sum = np.zeros((n_vox, 3), dtype=np.float64)
    counts = np.zeros(n_vox, dtype=np.int64)
    np.add.at(rgb_sum, inverse, colors.astype(np.float64))
    np.add.at(counts, inverse, 1)
    rgb = (rgb_sum / np.maximum(counts[:, None], 1)).astype(np.uint8)

    def _mode(labels: np.ndarray) -> np.ndarray:
        labels = labels.astype(np.int64)
        max_id = int(labels.max(initial=-1))
        if max_id < 0:
            return np.full(n_vox, -1, dtype=np.int64)
        # Histogram per voxel × instance-id (offset by +1 so -1 lands at column 0).
        K = max_id + 2
        hist = np.zeros((n_vox, K), dtype=np.int64)
        np.add.at(hist, (inverse, labels + 1), 1)
        # Ignore the "no instance" column when picking mode unless it dominates.
        valid_hist = hist[:, 1:]
        valid_argmax = valid_hist.argmax(axis=1)
        valid_max = valid_hist.max(axis=1)
        mode = valid_argmax  # ids are already offset back by indexing 1:
        no_valid = valid_max == 0
        mode[no_valid] = -1
        return mode

    return unique_xyz, rgb, _mode(gt_instance), _mode(pred_instance)


def predict_instances_from_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    score_thresh: float = 0.5,
    mask_thresh: float = 0.0,
) -> np.ndarray:
    """Convert (Q, C+1) logits + (Q, N) mask logits to per-point instance ids.

    A point is assigned to the highest-scoring foreground query whose mask
    logit is above ``mask_thresh`` and whose class probability (excluding
    "no object") is above ``score_thresh``. Otherwise the point gets ``-1``.
    """
    Q = logits.shape[0]
    num_classes = logits.shape[1] - 1  # last column is "no object"

    cls_prob = logits.softmax(-1)[:, :num_classes]  # (Q, C)
    query_score, _ = cls_prob.max(dim=-1)  # (Q,)
    keep = query_score > score_thresh
    if not torch.any(keep):
        return np.full(masks.shape[1], -1, dtype=np.int64)

    kept_idx = torch.nonzero(keep, as_tuple=False).flatten()
    kept_masks = masks[kept_idx]  # (Qk, N)
    kept_scores = query_score[kept_idx]  # (Qk,)

    # Score-weight each query mask, then argmax across queries.
    weighted = kept_masks * kept_scores[:, None]
    best_q = weighted.argmax(dim=0)  # (N,) index into kept_idx
    best_logit = kept_masks.gather(0, best_q[None, :]).squeeze(0)
    valid = best_logit > mask_thresh
    instance = torch.full((masks.shape[1],), -1, dtype=torch.long, device=masks.device)
    instance[valid] = kept_idx[best_q[valid]]
    return instance.detach().cpu().numpy().astype(np.int64)


class MaskFormerViserVisualizer:
    """Throttled viser visualizer for MaskFormer training.

    Build once outside the training loop, then call ``maybe_update``
    per step. Updates rebuild at most once every ``interval_seconds``.
    """

    def __init__(
        self,
        voxel_size: float,
        port: int = 8080,
        interval_seconds: float = 10.0,
        panel_gap_voxels: int = 8,
        color_range: tuple[float, float] | str = "auto",
        score_thresh: float = 0.5,
        mask_thresh: float = 0.0,
    ):
        """
        score_thresh, mask_thresh: thresholds used when converting MaskFormer
            outputs to per-point instance ids; see
            ``predict_instances_from_logits``.
        """
        self.voxel_size = float(voxel_size)
        self.interval = float(interval_seconds)
        self.panel_gap = int(panel_gap_voxels)
        self.color_range = color_range
        self.score_thresh = float(score_thresh)
        self.mask_thresh = float(mask_thresh)

        self.server = viser.ViserServer(port=port)
        self._lock = threading.Lock()
        self._last_push = 0.0
        self._mesh_handles: dict[str, object] = {}
        self._setup_gui()

    def _setup_gui(self) -> None:
        self.server.scene.set_up_direction("+z")
        with self.server.gui.add_folder("MaskFormer visualizer"):
            self._info_text = self.server.gui.add_markdown("_waiting for first batch_")
            self._epoch_text = self.server.gui.add_markdown("epoch: —  step: —")
            with self.server.gui.add_folder("Layout"):
                self.server.gui.add_markdown(
                    "Left: **input RGB** · Middle: **GT instances** · Right: **predicted instances**"
                )
            with self.server.gui.add_folder("Decoding"):
                self.server.gui.add_markdown(
                    f"score_thresh = `{self.score_thresh}` · mask_thresh = `{self.mask_thresh}`\n\n"
                    "Dark gray = no instance assigned. Each foreground query gets a "
                    "stable hashed color, so the same query keeps the same color across updates."
                )

    def maybe_update(
        self,
        coords: torch.Tensor,
        colors: torch.Tensor,
        gt_instance: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """Push one scene if the interval has elapsed."""
        now = time.time()
        if now - self._last_push < self.interval:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._last_push = now
            self._update(
                coords, colors, gt_instance, pred_logits, pred_masks, epoch=epoch, step=step
            )
        finally:
            self._lock.release()

    def _normalize_colors(self, col: np.ndarray) -> np.ndarray:
        if not col.size:
            return col.astype(np.uint8)
        if isinstance(self.color_range, tuple):
            lo, hi = float(self.color_range[0]), float(self.color_range[1])
            col = (col - lo) / max(hi - lo, 1e-6) * 255.0
        else:
            cmin, cmax = float(col.min()), float(col.max())
            if cmin < 0.0:
                col = (col + 1.0) * 0.5
                col = np.clip(col, 0.0, 1.0) * 255.0
            elif cmax <= 1.5:
                col = np.clip(col, 0.0, 1.0) * 255.0
        return np.clip(col, 0.0, 255.0).astype(np.uint8)

    def _update(
        self,
        coords: torch.Tensor,
        colors: torch.Tensor,
        gt_instance: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        epoch: int | None,
        step: int | None,
    ) -> None:
        c_np = coords.detach().cpu().to(torch.float32).numpy()
        col = self._normalize_colors(colors.detach().cpu().to(torch.float32).numpy())
        gt = gt_instance.detach().cpu().to(torch.int64).numpy()
        pred = predict_instances_from_logits(
            pred_logits.detach(),
            pred_masks.detach(),
            score_thresh=self.score_thresh,
            mask_thresh=self.mask_thresh,
        )

        unique_xyz, rgb, gt_mode, pred_mode = _voxelize_with_instances(
            c_np, col, gt, pred, voxel_size=self.voxel_size
        )
        if unique_xyz.shape[0] == 0:
            return

        center = np.round(unique_xyz.mean(axis=0)).astype(np.int64)
        centered = unique_xyz - center
        bbox_x = int(centered[:, 0].max() - centered[:, 0].min() + 1)
        offset_step = bbox_x + self.panel_gap

        panels = {
            "input": (centered, rgb),
            "gt": (
                centered + np.array([offset_step, 0, 0], dtype=np.int64),
                _instance_to_color(gt_mode),
            ),
            "pred": (
                centered + np.array([2 * offset_step, 0, 0], dtype=np.int64),
                _instance_to_color(pred_mode),
            ),
        }
        for name, (xyz, color) in panels.items():
            old = self._mesh_handles.pop(name, None)
            if old is not None:
                old.remove()
            mesh = _build_voxel_cube_mesh(xyz, color, self.voxel_size)
            self._mesh_handles[name] = self.server.scene.add_mesh_trimesh(f"/{name}", mesh)

        n_gt = int((np.unique(gt_mode) >= 0).sum())
        n_pred = int((np.unique(pred_mode) >= 0).sum())
        coverage_pred = float((pred_mode >= 0).mean()) * 100.0
        coverage_gt = float((gt_mode >= 0).mean()) * 100.0
        self._info_text.content = (
            f"voxels: **{unique_xyz.shape[0]:,}** · voxel size: {self.voxel_size:.3f} m\n\n"
            f"GT instances: **{n_gt}** ({coverage_gt:.1f}% labeled)\n\n"
            f"Pred instances: **{n_pred}** ({coverage_pred:.1f}% covered)"
        )
        self._epoch_text.content = (
            f"epoch: **{epoch if epoch is not None else '—'}**  "
            f"step: **{step if step is not None else '—'}**"
        )
