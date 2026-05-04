# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic point-cloud transforms for dataset augmentation.

Every transform takes and returns a single ``sample`` dict. Known
per-point keys are sliced (and rotated, when directional) together:

    coords   : (N, 3) float32  — required
    colors   : (N, 3) float32  — chromatic transforms expect [0, 255];
                                 wrap with ``LiftColorsFromUnitRange`` /
                                 ``NormalizeColorsToUnitRange`` if your
                                 dataset ships colors in [-1, 1]
    normals  : (N, 3) float32  — rotated and mirrored with coords
    labels   : (N,)   int      — single semantic label per point
    segment  : (N,)   int      — semantic id (instance segmentation)
    instance : (N,)   int      — instance id per point

Any other keys (e.g. ``name``, ``coord_int``) pass through untouched.

Pass a ``Compose`` to ``ScanNetDataset(transform=...)`` /
``ScanNetInstanceDataset(transform=...)`` (or any user dataset that
returns the same dict shape) to apply augmentations in the dataloader.

Parameter bounds for ``RandomRotation3D``, ``RandomScale``,
``RandomTranslationRatio``, and ``ElasticDistortion`` come from
SpatioTemporalSegmentation
(https://github.com/chrischoy/SpatioTemporalSegmentation), but generalize
to any indoor / outdoor scene-scale point cloud.
"""
from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

import numpy as np


Sample = dict
Transform = Callable[[Sample], Sample]


# Per-point arrays sliced together when a transform drops points.
PER_POINT_KEYS = ("coords", "colors", "normals", "labels", "segment", "instance")
# Per-point arrays that rotate / mirror with coords (have a 3-vector last dim).
DIRECTIONAL_KEYS = ("coords", "normals")


def _slice_per_point(sample: Sample, keep: np.ndarray) -> Sample:
    """Return a new sample with every per-point array sliced by ``keep``.

    ``keep`` is a bool mask or int index array. Non-per-point fields
    (e.g. ``name``) are passed through unchanged.
    """
    out = dict(sample)
    for k in PER_POINT_KEYS:
        v = out.get(k)
        if v is not None:
            out[k] = v[keep]
    return out


def _apply_rotation(sample: Sample, rot: np.ndarray) -> Sample:
    """Apply a 3×3 rotation matrix to every directional per-point array."""
    out = dict(sample)
    for k in DIRECTIONAL_KEYS:
        v = out.get(k)
        if v is not None:
            out[k] = (v @ rot.T).astype(np.float32, copy=False)
    return out


# --------------------------------------------------------------------- #
# Color (feature) transformations  — colors expected in [0, 255] floats.
# --------------------------------------------------------------------- #
class ChromaticTranslation:
    """Add a single random color offset to all RGB values (whole-scene tint)."""

    def __init__(self, trans_range_ratio: float = 0.10, prob: float = 0.95):
        self.trans_range_ratio = float(trans_range_ratio)
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        colors = sample.get("colors")
        if colors is None or random.random() >= self.prob:
            return sample
        colors = colors.copy()
        tr = (np.random.rand(1, 3).astype(np.float32) - 0.5) * 255 * 2 * self.trans_range_ratio
        colors[:, :3] = np.clip(tr + colors[:, :3], 0.0, 255.0)
        return {**sample, "colors": colors}


class ChromaticAutoContrast:
    """Per-scene auto-contrast blended with the original colors."""

    def __init__(
        self,
        prob: float = 0.20,
        randomize_blend_factor: bool = True,
        blend_factor: float = 0.5,
    ):
        self.prob = float(prob)
        self.randomize_blend_factor = bool(randomize_blend_factor)
        self.blend_factor = float(blend_factor)

    def __call__(self, sample: Sample) -> Sample:
        colors = sample.get("colors")
        if colors is None or random.random() >= self.prob:
            return sample
        colors = colors.copy()
        lo = colors[:, :3].min(0, keepdims=True)
        hi = colors[:, :3].max(0, keepdims=True)
        scale = 255.0 / np.maximum(hi - lo, 1e-6)
        contrast = (colors[:, :3] - lo) * scale
        blend = random.random() if self.randomize_blend_factor else self.blend_factor
        colors[:, :3] = np.clip((1 - blend) * colors[:, :3] + blend * contrast, 0.0, 255.0)
        return {**sample, "colors": colors}


class ChromaticJitter:
    """Per-point Gaussian noise on RGB."""

    def __init__(self, std: float = 0.01, prob: float = 0.95):
        self.std = float(std)
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        colors = sample.get("colors")
        if colors is None or random.random() >= self.prob:
            return sample
        noise = np.random.randn(colors.shape[0], 3).astype(np.float32) * self.std * 255.0
        colors = colors.copy()
        colors[:, :3] = np.clip(noise + colors[:, :3], 0.0, 255.0)
        return {**sample, "colors": colors}


class ChromaticDrop:
    """Occasionally zero out RGB to teach the model not to depend on color."""

    def __init__(self, prob: float = 0.20, value: float = 127.5):
        self.prob = float(prob)
        self.value = float(value)

    def __call__(self, sample: Sample) -> Sample:
        colors = sample.get("colors")
        if colors is None or random.random() >= self.prob:
            return sample
        colors = colors.copy()
        colors[:, :3] = self.value
        return {**sample, "colors": colors}


# --------------------------------------------------------------------- #
# Coordinate transformations
# --------------------------------------------------------------------- #
class RandomDropout:
    """Drop a random subset of points (simulates partial sensor coverage)."""

    def __init__(self, dropout_ratio: float = 0.20, prob: float = 0.50):
        self.dropout_ratio = float(dropout_ratio)
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        n = sample["coords"].shape[0]
        keep = int(n * (1.0 - self.dropout_ratio))
        if keep <= 0 or keep >= n:
            return sample
        inds = np.random.choice(n, keep, replace=False)
        return _slice_per_point(sample, inds)


class RandomHorizontalFlip:
    """Mirror coords (and normals, if present) across one or more horizontal axes."""

    def __init__(self, upright_axis: str = "z", prob: float = 0.95):
        self.upright_axis = {"x": 0, "y": 1, "z": 2}[upright_axis.lower()]
        self.horz_axes = [i for i in range(3) if i != self.upright_axis]
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        coords = sample["coords"].copy()
        normals = sample.get("normals")
        normals = normals.copy() if normals is not None else None
        for ax in self.horz_axes:
            if random.random() < 0.5:
                coords[:, ax] = coords[:, ax].max() - coords[:, ax]
                if normals is not None:
                    normals[:, ax] = -normals[:, ax]
        out = {**sample, "coords": coords}
        if normals is not None:
            out["normals"] = normals
        return out


class RandomScale:
    """Uniformly rescale coordinates."""

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1)):
        self.lo, self.hi = float(scale_range[0]), float(scale_range[1])

    def __call__(self, sample: Sample) -> Sample:
        s = float(np.random.uniform(self.lo, self.hi))
        return {**sample, "coords": sample["coords"] * s}


class RandomRotation3D:
    """Per-axis rotation with independent bounds.

    Default matches the ScanNet recipe: small wobble around x and y, full
    rotation around the up-axis (z).
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-np.pi / 64, np.pi / 64),
        y_range: tuple[float, float] = (-np.pi / 64, np.pi / 64),
        z_range: tuple[float, float] = (-np.pi, np.pi),
    ):
        self.bounds = (x_range, y_range, z_range)

    def __call__(self, sample: Sample) -> Sample:
        rx, ry, rz = (np.random.uniform(lo, hi) for (lo, hi) in self.bounds)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rot = (Rz @ Ry @ Rx).astype(np.float32)
        return _apply_rotation(sample, rot)


class RandomTranslationRatio:
    """Random whole-scene translation as a ratio of scene extent per axis."""

    def __init__(
        self,
        x_ratio: tuple[float, float] = (-0.2, 0.2),
        y_ratio: tuple[float, float] = (-0.2, 0.2),
        z_ratio: tuple[float, float] = (0.0, 0.0),
    ):
        self.bounds = (x_ratio, y_ratio, z_ratio)

    def __call__(self, sample: Sample) -> Sample:
        coords = sample["coords"]
        extent = coords.max(0) - coords.min(0)
        t = np.empty(3, dtype=np.float32)
        for i, (lo, hi) in enumerate(self.bounds):
            t[i] = np.random.uniform(lo, hi) * float(extent[i])
        return {**sample, "coords": coords + t}


class ElasticDistortion:
    """Smooth random warp on coordinate space (does not move normals).

    ``distortion_params`` is a list of ``(granularity, magnitude)`` pairs in
    METERS — each pass samples a Gaussian noise grid, smooths it with three
    orthogonal box filters, and trilinearly interpolates the field back
    onto the input coordinates.
    """

    def __init__(
        self,
        distortion_params: Sequence[tuple[float, float]] = ((0.2, 0.4), (0.8, 1.6)),
        prob: float = 0.95,
    ):
        self.distortion_params = list(distortion_params)
        self.prob = float(prob)

    @staticmethod
    def _elastic_distort(coords: np.ndarray, granularity: float, magnitude: float) -> np.ndarray:
        # Lazy scipy import so callers without scipy can still use the rest of
        # the module.
        import scipy.interpolate
        import scipy.ndimage

        blurx = np.ones((3, 1, 1, 1), dtype=np.float32) / 3
        blury = np.ones((1, 3, 1, 1), dtype=np.float32) / 3
        blurz = np.ones((1, 1, 3, 1), dtype=np.float32) / 3
        coords_min = coords.min(0)
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)
        for _ in range(2):
            noise = scipy.ndimage.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blurz, mode="constant", cval=0)
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

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        coords = sample["coords"].astype(np.float32, copy=True)
        for granularity, magnitude in self.distortion_params:
            coords = self._elastic_distort(coords, granularity, magnitude).astype(np.float32)
        return {**sample, "coords": coords}


class RandomCrop:
    """Random metric box crop to bound the point count.

    Picks a random origin point in the cloud, then keeps every point inside
    an axis-aligned cube of side ``crop_size`` meters centered on that
    origin. Skips the crop if the result would have fewer than
    ``min_points`` points.

    Useful for managing GPU memory when full ScanNet scenes overflow the
    budget — Mask3D / SoftGroup style cropping.
    """

    def __init__(self, crop_size: float = 6.0, min_points: int = 5000, prob: float = 1.0):
        self.crop_size = float(crop_size)
        self.min_points = int(min_points)
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        coords = sample["coords"]
        n = coords.shape[0]
        if n == 0:
            return sample
        half = self.crop_size * 0.5
        origin = coords[np.random.randint(0, n)]
        keep = np.all(np.abs(coords - origin) <= half, axis=1)
        if int(keep.sum()) < self.min_points:
            return sample
        return _slice_per_point(sample, keep)


# --------------------------------------------------------------------- #
# Color-range helpers — for datasets that ship colors in [-1, 1] (e.g.
# the OpenScene-preprocessed ScanNetDataset). The chromatic transforms
# work in [0, 255] floats; lift on the way in, normalize on the way out.
# --------------------------------------------------------------------- #
class LiftColorsFromUnitRange:
    """Map ``colors`` from [-1, 1] floats to [0, 255] floats."""

    def __call__(self, sample: Sample) -> Sample:
        colors = sample.get("colors")
        if colors is None:
            return sample
        return {**sample, "colors": np.clip((colors + 1.0) * 127.5, 0.0, 255.0)}


class NormalizeColorsToUnitRange:
    """Map ``colors`` from [0, 255] floats back to [-1, 1] floats."""

    def __call__(self, sample: Sample) -> Sample:
        colors = sample.get("colors")
        if colors is None:
            return sample
        return {**sample, "colors": (colors / 127.5) - 1.0}


# --------------------------------------------------------------------- #
# Composition + factories
# --------------------------------------------------------------------- #
class Compose:
    """Run a sequence of dict-in / dict-out transforms in order."""

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


def default_train_augmentations(
    crop_size: float | None = None,
    min_points: int = 5000,
    colors_in_unit_range: bool = False,
) -> Compose:
    """Reasonable default training pipeline for indoor scene point clouds.

    Parameters
    ----------
    crop_size : float, optional
        Side length in meters of an axis-aligned ``RandomCrop`` box
        inserted after the coordinate transforms. Pass ``None`` to skip
        cropping (full-scene training); set to a metric value (e.g.
        ``6.0``) to bound point count for memory-heavy backbones.
    min_points : int
        Skip the crop if it would leave fewer than this many points.
    colors_in_unit_range : bool
        Set ``True`` when the dataset returns colors in [-1, 1] (e.g. the
        OpenScene-preprocessed ``ScanNetDataset``). The pipeline lifts
        colors to [0, 255] for the chromatic transforms and normalizes
        back to [-1, 1] on the way out.
    """
    transforms: list[Transform] = []
    if colors_in_unit_range:
        transforms.append(LiftColorsFromUnitRange())
    transforms += [
        # Coordinate (run BEFORE elastic distortion so warps act on the
        # rotated/scaled cloud and remain meaningful in metric units).
        RandomRotation3D(),
        RandomScale((0.9, 1.1)),
        RandomHorizontalFlip(upright_axis="z"),
        RandomTranslationRatio(),
        ElasticDistortion(distortion_params=((0.2, 0.4), (0.8, 1.6))),
        RandomDropout(dropout_ratio=0.20, prob=0.20),
    ]
    if crop_size is not None:
        transforms.append(RandomCrop(crop_size=crop_size, min_points=min_points))
    transforms += [
        ChromaticAutoContrast(prob=0.20),
        ChromaticTranslation(trans_range_ratio=0.10),
        ChromaticJitter(std=0.01),
        ChromaticDrop(prob=0.05),
    ]
    if colors_in_unit_range:
        transforms.append(NormalizeColorsToUnitRange())
    return Compose(transforms)
