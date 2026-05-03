# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minecraft-style viser visualizer for ScanNet training.

Spins up a viser server and renders three side-by-side voxel scenes that
refresh from training every ``interval_seconds``:

    [ input RGB ]   [ ground-truth seg ]   [ predicted seg ]

Each voxel is drawn as an axis-aligned cube (Minecraft-style) so the
discrete grid structure is visible. Open the URL printed at startup
(default ``http://localhost:8080``) in a browser.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import trimesh
    import viser
except ImportError as exc:  # pragma: no cover - optional dep
    raise ImportError("viser visualization requires `pip install viser trimesh`") from exc


# ScanNet 20-class palette (RGB 0..255), order matches ScanNet benchmark.
SCANNET20_PALETTE = np.array(
    [
        (174, 199, 232),  # 0  wall
        (152, 223, 138),  # 1  floor
        (31, 119, 180),  # 2  cabinet
        (255, 187, 120),  # 3  bed
        (188, 189, 34),  # 4  chair
        (140, 86, 75),  # 5  sofa
        (255, 152, 150),  # 6  table
        (214, 39, 40),  # 7  door
        (197, 176, 213),  # 8  window
        (148, 103, 189),  # 9  bookshelf
        (196, 156, 148),  # 10 picture
        (23, 190, 207),  # 11 counter
        (247, 182, 210),  # 12 desk
        (219, 219, 141),  # 13 curtain
        (255, 127, 14),  # 14 refrigerator
        (158, 218, 229),  # 15 showercurtain
        (44, 160, 44),  # 16 toilet
        (112, 128, 144),  # 17 sink
        (227, 119, 194),  # 18 bathtub
        (82, 84, 163),  # 19 otherfurniture
    ],
    dtype=np.uint8,
)

SCANNET20_NAMES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "showercurtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]

# Color used for voxels whose majority label is `ignore_index` (255) — these
# are voxels where every source point in the cell was unlabeled in ScanNet.
# Bright magenta makes them obvious and gives the legend a distinct entry.
IGNORE_COLOR = np.array([255, 0, 255], dtype=np.uint8)
IGNORE_NAME = "ignore / unlabeled"


def _build_legend_image(width_px: int = 320, row_h: int = 22) -> np.ndarray:
    """Render the SCANNET20 class legend as an RGB image (no matplotlib dep at import)."""
    # 20 valid classes + 1 "ignore / unlabeled" row.
    rows = [
        (rgb, f"{i:>2}  {name}")
        for i, (rgb, name) in enumerate(zip(SCANNET20_PALETTE, SCANNET20_NAMES))
    ]
    rows.append((IGNORE_COLOR, f"--  {IGNORE_NAME}"))
    n = len(rows)
    h = n * row_h + 8
    img = np.full((h, width_px, 3), 255, dtype=np.uint8)

    swatch_w = 28
    swatch_h = row_h - 8
    swatch_x0 = 8
    text_x0 = swatch_x0 + swatch_w + 10

    # Pre-rasterized 5×7 bitmap font would be heavy; instead, fall back to PIL
    # if available (it ships with matplotlib's deps). If not, skip text — the
    # caller can provide a markdown caption alongside.
    try:
        from PIL import Image, ImageDraw, ImageFont

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except OSError:
            font = ImageFont.load_default()
        for i, (rgb, label) in enumerate(rows):
            y0 = 4 + i * row_h
            draw.rectangle(
                [swatch_x0, y0, swatch_x0 + swatch_w, y0 + swatch_h],
                fill=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
                outline=(0, 0, 0),
            )
            draw.text(
                (text_x0, y0 - 1),
                label,
                fill=(20, 20, 20),
                font=font,
            )
        return np.asarray(pil)
    except ImportError:
        # PIL not available — return color-bar only (palette + ignore).
        all_rgb = np.concatenate([SCANNET20_PALETTE, IGNORE_COLOR[None, :]], axis=0)
        for i, rgb in enumerate(all_rgb):
            y0 = 4 + i * row_h
            img[y0 : y0 + swatch_h, swatch_x0 : swatch_x0 + swatch_w] = rgb
        return img


# Unit cube template (8 verts, 12 triangles) — front-face winding outward.
_CUBE_VERTS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.float32,
)
_CUBE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],  # z=0
        [4, 5, 6],
        [4, 6, 7],  # z=1
        [0, 1, 5],
        [0, 5, 4],  # y=0
        [1, 2, 6],
        [1, 6, 5],  # x=1
        [2, 3, 7],
        [2, 7, 6],  # y=1
        [3, 0, 4],
        [3, 4, 7],  # x=0
    ],
    dtype=np.int32,
)


def _build_voxel_cube_mesh(
    voxel_xyz: np.ndarray, voxel_rgb: np.ndarray, voxel_size: float
) -> trimesh.Trimesh:
    """Concatenate one cube per voxel into a single trimesh."""
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


def _voxelize(
    coords: np.ndarray,
    colors: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    voxel_size: float,
    num_classes: int,
    ignore_index: int,
):
    """Bucket points into voxels. Returns (voxel_xyz, mean_rgb, gt_mode, pred_mode)."""
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
        valid = (labels >= 0) & (labels < num_classes)
        hist = np.zeros((n_vox, num_classes), dtype=np.int64)
        np.add.at(hist, (inverse[valid], labels[valid]), 1)
        mode = hist.argmax(axis=1)
        no_valid = hist.sum(axis=1) == 0
        mode[no_valid] = ignore_index
        return mode

    return unique_xyz, rgb, _mode(gt_labels), _mode(pred_labels)


def _label_to_color(labels: np.ndarray, ignore_index: int) -> np.ndarray:
    # Default = bright magenta = "ignore / unlabeled" (voxels with no valid
    # source label after majority pooling). Listed in the GUI legend.
    out = np.tile(IGNORE_COLOR, (labels.shape[0], 1))
    valid = (labels != ignore_index) & (labels >= 0) & (labels < SCANNET20_PALETTE.shape[0])
    out[valid] = SCANNET20_PALETTE[labels[valid]]
    return out


class ScanNetViserVisualizer:
    """Throttled, thread-safe viser visualizer for ScanNet training.

    Call ``maybe_update`` from the training loop with a single scan
    (xyz, rgb, gt labels, predicted labels). At most one rebuild every
    ``interval_seconds`` seconds; calls in between return immediately.
    """

    def __init__(
        self,
        voxel_size: float,
        num_classes: int = 20,
        ignore_index: int = 255,
        port: int = 8080,
        interval_seconds: float = 10.0,
        panel_gap_voxels: int = 8,
        color_range: tuple[float, float] | str = "auto",
    ):
        """
        color_range: explicit (min, max) of input color values, or "auto" to
            detect from the first batch. Auto-detection breaks when the entire
            batch has constant colors (e.g. ChromaticDrop set everything to a
            single value), so prefer passing the canonical range explicitly:
            ``(-1.0, 1.0)`` for OpenScene-normalized RGB, ``(0.0, 255.0)`` for
            raw 8-bit RGB, ``(0.0, 1.0)`` for [0,1] floats.
        """
        self.voxel_size = float(voxel_size)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.interval = float(interval_seconds)
        self.panel_gap = int(panel_gap_voxels)
        self.color_range = color_range

        self.server = viser.ViserServer(port=port)
        self._lock = threading.Lock()
        self._last_push = 0.0
        self._mesh_handles: dict[str, object] = {}
        self._setup_gui()

    def _setup_gui(self) -> None:
        self.server.scene.set_up_direction("+z")
        with self.server.gui.add_folder("ScanNet visualizer"):
            self._info_text = self.server.gui.add_markdown("_waiting for first batch_")
            self._epoch_text = self.server.gui.add_markdown("epoch: —  step: —")
            with self.server.gui.add_folder("Layout"):
                self.server.gui.add_markdown(
                    "Left: **input RGB** · Middle: **ground truth** · Right: **prediction**"
                )
            with self.server.gui.add_folder("Class legend"):
                # viser markdown drops inline HTML and resizes embedded images
                # to full width, so render the legend once as a single bitmap
                # and ship it via gui.add_image.
                self.server.gui.add_image(
                    _build_legend_image(),
                    label=None,
                    format="png",
                )

    def maybe_update(
        self,
        coords: torch.Tensor,
        colors: torch.Tensor,
        gt_labels: torch.Tensor,
        pred_labels: torch.Tensor,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """Push a sample if at least ``interval`` seconds passed since last push.

        ``coords`` is (M, 3) float meters; ``colors`` is (M, 3) RGB in
        either [0, 255] or [0, 1] or normalized [-1, 1]; labels are (M,) int.
        """
        now = time.time()
        if now - self._last_push < self.interval:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._last_push = now
            self._update(coords, colors, gt_labels, pred_labels, epoch=epoch, step=step)
        finally:
            self._lock.release()

    def _update(
        self,
        coords: torch.Tensor,
        colors: torch.Tensor,
        gt_labels: torch.Tensor,
        pred_labels: torch.Tensor,
        epoch: int | None,
        step: int | None,
    ) -> None:
        c_np = coords.detach().cpu().to(torch.float32).numpy()
        col = colors.detach().cpu().to(torch.float32).numpy()
        gt = gt_labels.detach().cpu().to(torch.int64).numpy()
        pr = pred_labels.detach().cpu().to(torch.int64).numpy()

        # Normalize colors to uint8 [0, 255] RGB. If the caller passed an
        # explicit (lo, hi) range, use that; otherwise fall back to per-batch
        # min/max heuristics. Heuristics break when colors are constant
        # (e.g. ChromaticDrop set every voxel to the same value), so an
        # explicit range is strongly preferred.
        if col.size:
            if isinstance(self.color_range, tuple):
                lo, hi = float(self.color_range[0]), float(self.color_range[1])
                col = (col - lo) / max(hi - lo, 1e-6) * 255.0
            else:
                cmin, cmax = float(col.min()), float(col.max())
                if cmin < 0.0:
                    col = (col + 1.0) * 0.5  # assume [-1, 1] -> [0, 1]
                    col = np.clip(col, 0.0, 1.0) * 255.0
                elif cmax <= 1.5:
                    col = np.clip(col, 0.0, 1.0) * 255.0
        col = np.clip(col, 0.0, 255.0).astype(np.uint8)

        unique_xyz, rgb, gt_mode, pred_mode = _voxelize(
            c_np,
            col,
            gt,
            pr,
            voxel_size=self.voxel_size,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        if unique_xyz.shape[0] == 0:
            return

        # Center scene at origin (in voxel units).
        center = np.round(unique_xyz.mean(axis=0)).astype(np.int64)
        centered = unique_xyz - center

        # Layout 3 panels side-by-side along +x.
        bbox_x = int(centered[:, 0].max() - centered[:, 0].min() + 1)
        offset_step = bbox_x + self.panel_gap

        panels = {
            "input": (centered, rgb),
            "gt": (
                centered + np.array([offset_step, 0, 0], dtype=np.int64),
                _label_to_color(gt_mode, self.ignore_index),
            ),
            "pred": (
                centered + np.array([2 * offset_step, 0, 0], dtype=np.int64),
                _label_to_color(pred_mode, self.ignore_index),
            ),
        }

        for name, (xyz, color) in panels.items():
            old = self._mesh_handles.pop(name, None)
            if old is not None:
                old.remove()
            mesh = _build_voxel_cube_mesh(xyz, color, self.voxel_size)
            self._mesh_handles[name] = self.server.scene.add_mesh_trimesh(f"/{name}", mesh)

        valid = (pred_mode != self.ignore_index) & (gt_mode != self.ignore_index)
        agree = int(((pred_mode == gt_mode) & valid).sum())
        denom = max(int(valid.sum()), 1)
        self._info_text.content = (
            f"voxels: **{unique_xyz.shape[0]:,}** · "
            f"voxel size: {self.voxel_size:.3f} m · "
            f"voxel-acc: **{agree / denom * 100:.1f}%**"
        )
        self._epoch_text.content = (
            f"epoch: **{epoch if epoch is not None else '—'}**  "
            f"step: **{step if step is not None else '—'}**"
        )
