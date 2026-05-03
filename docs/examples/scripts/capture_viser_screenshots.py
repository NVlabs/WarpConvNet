# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture viser screenshots for the ScanNet docs page.

Spins up the ScanNet viser visualizer with a hand-crafted synthetic mini-room
(no GPU, no real ScanNet data, no model checkpoint), then drives a headless
Chromium via playwright to snapshot the WebGL scene.

Run from repo root:

    source .venv/bin/activate
    pip install viser trimesh playwright
    playwright install chromium
    python docs/examples/scripts/capture_viser_screenshots.py

Outputs PNGs to docs/examples/img/scannet_viser/.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import numpy as np
import torch

# Make the visualizer module importable without packaging gymnastics.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples"))

from viser_scannet_visualizer import (  # type: ignore
    ScanNetViserVisualizer,
    SCANNET20_PALETTE,
)

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "img", "scannet_viser")
)
os.makedirs(OUT_DIR, exist_ok=True)

VOXEL_SIZE = 0.10  # bigger cubes so the Minecraft look reads at screenshot scale
PORT = 8091

# Class indices we'll use (subset of SCANNET20):
WALL = 0
FLOOR = 1
BED = 3
CHAIR = 4
SOFA = 5
TABLE = 6
DOOR = 7
WINDOW = 8


# ---------------------------------------------------------------- #
# Build a synthetic mini-room of points + class labels
# ---------------------------------------------------------------- #
def _box_surface_points(
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    spacing: float = 0.04,
) -> np.ndarray:
    """Sample points on the surface of an axis-aligned box."""
    pts = []
    nx = max(int(sx / spacing), 2)
    ny = max(int(sy / spacing), 2)
    nz = max(int(sz / spacing), 2)
    xs = np.linspace(cx - sx / 2, cx + sx / 2, nx)
    ys = np.linspace(cy - sy / 2, cy + sy / 2, ny)
    zs = np.linspace(cz - sz / 2, cz + sz / 2, nz)

    # Top + bottom faces
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, cz - sz / 2)], axis=1))
    pts.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, cz + sz / 2)], axis=1))
    # Front + back
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pts.append(np.stack([X.ravel(), np.full(X.size, cy - sy / 2), Z.ravel()], axis=1))
    pts.append(np.stack([X.ravel(), np.full(X.size, cy + sy / 2), Z.ravel()], axis=1))
    # Left + right
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    pts.append(np.stack([np.full(Y.size, cx - sx / 2), Y.ravel(), Z.ravel()], axis=1))
    pts.append(np.stack([np.full(Y.size, cx + sx / 2), Y.ravel(), Z.ravel()], axis=1))
    return np.concatenate(pts, axis=0)


def _floor_points(x_range, y_range, spacing=0.04, z=0.0):
    xs = np.arange(x_range[0], x_range[1], spacing)
    ys = np.arange(y_range[0], y_range[1], spacing)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)


def _wall_points(x_range, z_range, y, spacing=0.04):
    xs = np.arange(x_range[0], x_range[1], spacing)
    zs = np.arange(z_range[0], z_range[1], spacing)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], axis=1)


def build_synthetic_room(seed: int = 0):
    """Mini-room: floor + back wall + side wall + bed + chair + table.

    Returns (coords [M,3], colors [M,3] uint8, gt_labels [M], pred_labels [M]).
    Colors mimic indoor RGB: warm beige floor, off-white walls, brown furniture.
    Pred labels = GT with a few realistic confusions (chair↔sofa, table↔bed).
    """
    rng = np.random.default_rng(seed)

    parts = []  # list of (xyz, rgb_uint8, class_idx)

    # Floor: 4 m × 4 m, warm beige
    fp = _floor_points((-2.0, 2.0), (-2.0, 2.0), spacing=0.05, z=0.0)
    fc = np.tile(np.array([[210, 195, 175]], dtype=np.uint8), (fp.shape[0], 1))
    parts.append((fp, fc, FLOOR))

    # Back wall y = 2.0, height 2.4 m, off-white
    bw = _wall_points((-2.0, 2.0), (0.0, 2.4), y=2.0, spacing=0.05)
    bwc = np.tile(np.array([[235, 230, 220]], dtype=np.uint8), (bw.shape[0], 1))
    parts.append((bw, bwc, WALL))

    # Side wall x = -2.0
    sw = _wall_points((-2.0, 2.0), (0.0, 2.4), y=-2.0, spacing=0.05)
    sw[:, [0, 1]] = sw[:, [1, 0]]  # rotate to x = const wall
    sw[:, 0] = -2.0
    swc = np.tile(np.array([[230, 225, 215]], dtype=np.uint8), (sw.shape[0], 1))
    parts.append((sw, swc, WALL))

    # Bed: 1.6 × 2.0 × 0.5 box centered at (0.6, 1.0)
    bed = _box_surface_points(0.6, 1.0, 0.25, 1.6, 2.0, 0.5, spacing=0.05)
    bed_color = np.tile(np.array([[180, 140, 110]], dtype=np.uint8), (bed.shape[0], 1))
    parts.append((bed, bed_color, BED))

    # Table: 1.0 × 0.6 × 0.7 box at (-1.0, -0.5)
    tab = _box_surface_points(-1.0, -0.5, 0.35, 1.0, 0.6, 0.7, spacing=0.05)
    tc = np.tile(np.array([[140, 100, 70]], dtype=np.uint8), (tab.shape[0], 1))
    parts.append((tab, tc, TABLE))

    # Chair: 0.5 × 0.5 × 1.0 box at (0.0, -1.2)
    ch = _box_surface_points(0.0, -1.2, 0.5, 0.5, 0.5, 1.0, spacing=0.04)
    cc = np.tile(np.array([[110, 80, 60]], dtype=np.uint8), (ch.shape[0], 1))
    parts.append((ch, cc, CHAIR))

    # Concatenate
    coords = np.concatenate([p[0] for p in parts], axis=0).astype(np.float32)
    colors = np.concatenate([p[1] for p in parts], axis=0).astype(np.uint8)
    gt_labels = np.concatenate([np.full(p[0].shape[0], p[2], dtype=np.int64) for p in parts])

    # Add small noise to coords so voxelization isn't perfectly aligned.
    coords += rng.normal(0.0, 0.005, coords.shape).astype(np.float32)

    # Predictions: copy GT, then introduce realistic confusions.
    pred_labels = gt_labels.copy()
    # 12% of chair voxels get mislabeled as sofa
    chair_idx = np.where(gt_labels == CHAIR)[0]
    flip_chair = rng.choice(chair_idx, size=int(chair_idx.size * 0.12), replace=False)
    pred_labels[flip_chair] = SOFA
    # 8% of bed voxels get mislabeled as sofa
    bed_idx = np.where(gt_labels == BED)[0]
    flip_bed = rng.choice(bed_idx, size=int(bed_idx.size * 0.08), replace=False)
    pred_labels[flip_bed] = SOFA
    # 3% of floor voxels mislabeled as wall (sensor noise / ambiguous boundary)
    floor_idx = np.where(gt_labels == FLOOR)[0]
    flip_floor = rng.choice(floor_idx, size=int(floor_idx.size * 0.03), replace=False)
    pred_labels[flip_floor] = WALL
    # 6% of table voxels get mislabeled as bed (similar low boxy shape)
    table_idx = np.where(gt_labels == TABLE)[0]
    flip_table = rng.choice(table_idx, size=int(table_idx.size * 0.06), replace=False)
    pred_labels[flip_table] = BED

    return (
        torch.from_numpy(coords),
        torch.from_numpy(colors),
        torch.from_numpy(gt_labels),
        torch.from_numpy(pred_labels),
    )


# ---------------------------------------------------------------- #
# Spin up viser + push synthetic data + register a default camera
# ---------------------------------------------------------------- #
def start_visualizer():
    viz = ScanNetViserVisualizer(
        voxel_size=VOXEL_SIZE,
        num_classes=20,
        ignore_index=255,
        port=PORT,
        interval_seconds=0.0,  # never throttle
    )

    @viz.server.on_client_connect
    def _on_connect(client):
        # Frame the side-by-side cube layout. Three panels span ~13 m along +x
        # (each ~4 m wide + 0.4 m gap), centered around x ≈ 4.4 m. Camera sits
        # in front + above, looking back at the row of panels.
        client.camera.position = (4.5, -9.5, 4.5)
        client.camera.look_at = (4.5, 0.0, 1.0)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.fov = 0.9  # radians (~52°), tighter than default for closer crop

    coords, colors, gt, pred = build_synthetic_room(seed=0)
    viz.maybe_update(coords, colors, gt, pred, epoch=12, step=1234)
    return viz


# ---------------------------------------------------------------- #
# Drive headless Chromium via playwright
# ---------------------------------------------------------------- #
async def _capture(out_paths: dict[str, str]) -> None:
    from playwright.async_api import async_playwright

    url = f"http://127.0.0.1:{PORT}"
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--use-gl=angle",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--enable-features=Vulkan",
            ],
        )
        try:
            context = await browser.new_context(viewport={"width": 1700, "height": 950})
            page = await context.new_page()
            print(f"loading {url} …")
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            # Give the WebGL scene + GUI time to render and settle.
            await page.wait_for_timeout(6_000)

            print("full UI screenshot")
            await page.screenshot(path=out_paths["full"], full_page=False)

            # Cube-only crop (drop the right-side GUI panel ~360 px wide).
            print("cube-only crop")
            await page.screenshot(
                path=out_paths["cubes"],
                clip={"x": 0, "y": 0, "width": 1340, "height": 950},
            )

            # GUI sidebar crop
            print("sidebar crop")
            await page.screenshot(
                path=out_paths["sidebar"],
                clip={"x": 1340, "y": 0, "width": 360, "height": 950},
            )
        finally:
            await browser.close()


def main() -> None:
    print("starting viser visualizer …")
    viz = start_visualizer()
    print(f"viser server up at http://127.0.0.1:{PORT}")

    out_paths = {
        "full": os.path.join(OUT_DIR, "01_full_ui.png"),
        "cubes": os.path.join(OUT_DIR, "02_cube_panels.png"),
        "sidebar": os.path.join(OUT_DIR, "03_sidebar_legend.png"),
    }

    # Brief warmup so server is fully ready before browser connects.
    time.sleep(1.5)
    asyncio.run(_capture(out_paths))

    print("\nsaved screenshots:")
    for name, path in out_paths.items():
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {name:8s} {path} ({size/1024:.1f} KB)")

    # ScanNetViserVisualizer doesn't expose a stop method; just exit.
    print("\ndone — exiting (viser server torn down with process)")


if __name__ == "__main__":
    main()
