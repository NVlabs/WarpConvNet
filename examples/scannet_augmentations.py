# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data augmentations for the ScanNet example.

Ported from the recipe in
https://github.com/chrischoy/SpatioTemporalSegmentation, which adds ~5–10
mIoU on ScanNet over the un-augmented baseline.

All transforms operate on `(coords, colors, labels)` numpy arrays:

    coords: (N, 3) float32 — 3D point positions in meters
    colors: (N, 3) float32 — RGB in [0, 255]
    labels: (N,)   int     — semantic label per point

Wrap a base ScanNet dataset with `AugmentedScanNetDataset` to apply a
`Compose([...])` pipeline at sample time. The wrapper handles the
[-1, 1] ↔ [0, 255] color conversion expected by the chromatic transforms.

Parameter bounds for `RandomRotation3D`, `RandomScale`,
`RandomTranslationRatio`, and `ElasticDistortion` come directly from the
ScanNet recipe in
https://github.com/chrischoy/SpatioTemporalSegmentation
(`lib/datasets/scannet.py` and `lib/dataset.py`).
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------- #
# Color (feature) transformations  — colors expected in [0, 255] floats.
# --------------------------------------------------------------------- #
class ChromaticTranslation:
    """Add a single random color offset to all RGB values (whole-scene tint)."""

    def __init__(self, trans_range_ratio: float = 0.10, prob: float = 0.95):
        self.trans_range_ratio = trans_range_ratio
        self.prob = prob

    def __call__(self, coords, colors, labels):
        if random.random() < self.prob:
            tr = (np.random.rand(1, 3).astype(np.float32) - 0.5) * 255 * 2 * self.trans_range_ratio
            colors[:, :3] = np.clip(tr + colors[:, :3], 0.0, 255.0)
        return coords, colors, labels


class ChromaticAutoContrast:
    """Per-scene auto-contrast blended with the original colors."""

    def __init__(
        self, prob: float = 0.20, randomize_blend_factor: bool = True, blend_factor: float = 0.5
    ):
        self.prob = prob
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, colors, labels):
        if random.random() < self.prob:
            lo = colors[:, :3].min(0, keepdims=True)
            hi = colors[:, :3].max(0, keepdims=True)
            scale = 255.0 / np.maximum(hi - lo, 1e-6)
            contrast = (colors[:, :3] - lo) * scale
            blend = random.random() if self.randomize_blend_factor else self.blend_factor
            colors[:, :3] = (1 - blend) * colors[:, :3] + blend * contrast
            colors[:, :3] = np.clip(colors[:, :3], 0.0, 255.0)
        return coords, colors, labels


class ChromaticJitter:
    """Per-point Gaussian noise on RGB."""

    def __init__(self, std: float = 0.01, prob: float = 0.95):
        self.std = std  # std of Gaussian noise as a fraction of [0, 255] range
        self.prob = prob

    def __call__(self, coords, colors, labels):
        if random.random() < self.prob:
            noise = np.random.randn(colors.shape[0], 3).astype(np.float32) * self.std * 255.0
            colors[:, :3] = np.clip(noise + colors[:, :3], 0.0, 255.0)
        return coords, colors, labels


class ChromaticDrop:
    """Occasionally zero out RGB to teach the model not to depend on color."""

    def __init__(self, prob: float = 0.20):
        self.prob = prob

    def __call__(self, coords, colors, labels):
        if random.random() < self.prob:
            colors[:, :3] = 127.5  # mid-gray in [0, 255]
        return coords, colors, labels


# --------------------------------------------------------------------- #
# Coordinate transformations
# --------------------------------------------------------------------- #
class RandomDropout:
    """Drop a random subset of points (simulates partial sensor coverage)."""

    def __init__(self, dropout_ratio: float = 0.20, prob: float = 0.50):
        self.dropout_ratio = dropout_ratio
        self.prob = prob

    def __call__(self, coords, colors, labels):
        if random.random() < self.prob:
            n = coords.shape[0]
            keep = int(n * (1 - self.dropout_ratio))
            inds = np.random.choice(n, keep, replace=False)
            coords = coords[inds]
            colors = colors[inds]
            labels = labels[inds]
        return coords, colors, labels


class RandomHorizontalFlip:
    """Mirror coords across one or more horizontal axes (NOT the up-axis)."""

    def __init__(self, upright_axis: str = "z", prob: float = 0.95):
        self.upright_axis = {"x": 0, "y": 1, "z": 2}[upright_axis.lower()]
        self.horz_axes = [i for i in range(3) if i != self.upright_axis]
        self.prob = prob

    def __call__(self, coords, colors, labels):
        if random.random() < self.prob:
            for ax in self.horz_axes:
                if random.random() < 0.5:
                    coords[:, ax] = coords[:, ax].max() - coords[:, ax]
        return coords, colors, labels


class RandomScale:
    """Uniformly rescale coordinates."""

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1)):
        self.lo, self.hi = scale_range

    def __call__(self, coords, colors, labels):
        s = np.random.uniform(self.lo, self.hi)
        coords = coords * s
        return coords, colors, labels


class RandomRotation3D:
    """Per-axis rotation with independent bounds.

    Default matches SpatioTemporalSegmentation ScanNet recipe:
    small wobble around x and y, full rotation around the up-axis (z).
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-np.pi / 64, np.pi / 64),
        y_range: tuple[float, float] = (-np.pi / 64, np.pi / 64),
        z_range: tuple[float, float] = (-np.pi, np.pi),
    ):
        self.bounds = (x_range, y_range, z_range)

    def __call__(self, coords, colors, labels):
        rx, ry, rz = (np.random.uniform(lo, hi) for (lo, hi) in self.bounds)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rot = (Rz @ Ry @ Rx).astype(np.float32)
        coords = (coords @ rot.T).astype(np.float32)
        return coords, colors, labels


class RandomTranslationRatio:
    """Random whole-scene translation as a ratio of scene extent per axis.

    Matches SpatioTemporalSegmentation TRANSLATION_AUGMENTATION_RATIO_BOUND.
    """

    def __init__(
        self,
        x_ratio: tuple[float, float] = (-0.2, 0.2),
        y_ratio: tuple[float, float] = (-0.2, 0.2),
        z_ratio: tuple[float, float] = (0.0, 0.0),
    ):
        self.bounds = (x_ratio, y_ratio, z_ratio)

    def __call__(self, coords, colors, labels):
        extent = coords.max(0) - coords.min(0)  # (3,)
        t = np.empty(3, dtype=np.float32)
        for i, (lo, hi) in enumerate(self.bounds):
            t[i] = np.random.uniform(lo, hi) * float(extent[i])
        coords = coords + t
        return coords, colors, labels


class ElasticDistortion:
    """Smooth random warp on coordinate space.

    Algorithm from SpatioTemporalSegmentation. ``distortion_params`` is a
    list of ``(granularity, magnitude)`` tuples in METERS — each pass
    samples a Gaussian noise grid at ``granularity`` spacing, smooths it
    with three orthogonal box filters, then trilinearly interpolates the
    noise field back onto the input coordinates and adds it scaled by
    ``magnitude``.

    Default ``((0.2, 0.4), (0.8, 1.6))`` matches the upstream ScanNet
    recipe. Two passes of increasing granularity warp the cloud at both
    fine and coarse scales — this is what makes flat walls bow and chair
    legs curl in the augmented view.
    """

    def __init__(
        self,
        distortion_params: Sequence[tuple[float, float]] = ((0.2, 0.4), (0.8, 1.6)),
        prob: float = 0.95,
    ):
        self.distortion_params = list(distortion_params)
        self.prob = prob

    @staticmethod
    def _elastic_distort(coords: np.ndarray, granularity: float, magnitude: float) -> np.ndarray:
        # Lazy scipy import so callers without scipy can still use the rest of
        # the module.
        import scipy.ndimage
        import scipy.interpolate

        blurx = np.ones((3, 1, 1, 1), dtype=np.float32) / 3
        blury = np.ones((1, 3, 1, 1), dtype=np.float32) / 3
        blurz = np.ones((1, 1, 3, 1), dtype=np.float32) / 3
        coords_min = coords.min(0)
        # Build noise grid sized so it covers the cloud + 3 cells of margin.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)
        for _ in range(2):
            noise = scipy.ndimage.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blurz, mode="constant", cval=0)
        # World-space coordinates of each grid cell along each axis.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        return coords + interp(coords) * magnitude

    def __call__(self, coords, colors, labels):
        if random.random() < self.prob:
            coords = coords.astype(np.float32)
            for granularity, magnitude in self.distortion_params:
                coords = self._elastic_distort(coords, granularity, magnitude).astype(np.float32)
        return coords, colors, labels


# --------------------------------------------------------------------- #
# Composition + factory
# --------------------------------------------------------------------- #
class Compose:
    """Run a sequence of transforms in order."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, coords, colors, labels):
        for t in self.transforms:
            coords, colors, labels = t(coords, colors, labels)
        return coords, colors, labels


def default_train_augmentations() -> Compose:
    """Standard ScanNet recipe — drop-in for ``data.augmentations=true``.

    Mirrors SpatioTemporalSegmentation's ScanNet config:
    https://github.com/chrischoy/SpatioTemporalSegmentation/blob/master/lib/datasets/scannet.py
    """
    return Compose(
        [
            # Coordinate (run BEFORE elastic distortion so warps act on the
            # rotated/scaled cloud and remain meaningful in metric units).
            RandomRotation3D(),
            RandomScale((0.9, 1.1)),
            RandomHorizontalFlip(upright_axis="z"),
            RandomTranslationRatio(),
            ElasticDistortion(distortion_params=((0.2, 0.4), (0.8, 1.6))),
            RandomDropout(dropout_ratio=0.20, prob=0.20),
            # Color (operate on [0, 255] floats)
            ChromaticAutoContrast(prob=0.20),
            ChromaticTranslation(trans_range_ratio=0.10),
            ChromaticJitter(std=0.01),
            ChromaticDrop(prob=0.20),
        ]
    )


# --------------------------------------------------------------------- #
# Dataset wrapper — handles [-1, 1] ↔ [0, 255] color conversion.
# --------------------------------------------------------------------- #
class AugmentedScanNetDataset(Dataset):
    """Wrap a ScanNet dataset and apply augmentations per sample.

    The ScanNet base dataset returns colors in [-1, 1] (OpenScene
    preprocessing). Internally we convert to [0, 255] floats for the
    chromatic transforms, then back to [-1, 1] for the model.
    """

    def __init__(self, base: Dataset, transform: Compose | None = None):
        self.base = base
        self.transform = transform or default_train_augmentations()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        coords = np.asarray(item["coords"], dtype=np.float32).copy()
        colors = np.asarray(item["colors"], dtype=np.float32).copy()
        labels = np.asarray(item["labels"]).copy()

        # OpenScene RGB is normalized to [-1, 1]; lift to [0, 255] for
        # chromatic transforms which assume that range.
        colors_255 = (colors + 1.0) * 127.5
        colors_255 = np.clip(colors_255, 0.0, 255.0)

        coords, colors_255, labels = self.transform(coords, colors_255, labels)

        # Back to [-1, 1] for the model.
        colors = (colors_255 / 127.5) - 1.0

        # Return numpy (matches base ScanNetDataset; collate_fn calls
        # torch.tensor on each value).
        return {
            "coords": coords,
            "colors": colors,
            "labels": np.asarray(labels),
        }
