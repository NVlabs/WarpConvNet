# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

SampleTransform = Callable[[Dict[str, Any]], Dict[str, Any]]

from warpconvnet.dataset.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)
from warpconvnet.geometry.coords.ops.voxel import voxel_downsample_np

SCANNET_URL = "https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip"


class ScanNetDataset(Dataset):
    """
    Dataset from the OpenScene project.
    """

    def __init__(
        self,
        root: str = "./data/scannet",
        split: str = "train",
        voxel_size: Optional[float] = None,
        out_type: Literal["point", "voxel"] = "voxel",
        min_coord: Optional[Tuple[float, float, float]] = None,
        transform: Optional[SampleTransform] = None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.out_type = out_type
        if min_coord is not None:
            min_coord = torch.tensor(min_coord)
        self.min_coord = min_coord
        self.transform = transform
        self.prepare_data()

    def prepare_data(self):
        # If data is not downloaded, download it
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
            os.system(f"wget {SCANNET_URL} -O {self.root}/scannet_3d.zip")
            os.system(f"unzip {self.root}/scannet_3d.zip -d {self.root}")
            os.system(f"mv {self.root}/scannet_3d/* {self.root}")
            os.system(f"rmdir {self.root}/scannet_3d")

        # Get split txts
        self.files = []
        with open(os.path.join(self.root, f"scannetv2_{self.split}.txt")) as f:
            self.files = sorted(f.readlines())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        coords, colors, labels = torch.load(
            os.path.join(self.root, self.split, file.strip() + "_vh_clean_2.pth"),
            weights_only=False,
        )
        if self.min_coord is not None:
            coords -= self.min_coord
        # All to tensor
        if self.voxel_size is not None:
            # Use cpu for downsampling in dataloader. Should use multiple workers.
            unique_coords, to_unique_indices = voxel_downsample_np(coords, self.voxel_size)
            if self.out_type == "point":
                unique_coords = coords[to_unique_indices]
            sample = {
                "coords": unique_coords,
                "colors": colors[to_unique_indices],
                "labels": labels[to_unique_indices],
            }
        else:
            sample = {
                "coords": coords,
                "colors": colors,
                "labels": labels,
            }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class ScanNetInstanceDataset(Dataset):
    """ScanNet / ScanNet200 instance + semantic segmentation dataset.

    Reads the per-scene Pointcept-style preprocessed layout::

        root/
            train/
                scene0000_00/
                    coord.npy        # (N, 3)  float32
                    color.npy        # (N, 3)  uint8 or float32
                    normal.npy       # (N, 3)  float32
                    segment20.npy    # (N,)    int32   (-1 = ignore)
                    segment200.npy   # (N,)    int32   (-1 = ignore)
                    instance.npy     # (N,)    int32   (-1 = ignore)
                ...
            val/...
            test/...

    The raw ScanNet meshes are gated by the official Terms of Use:
    https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf. Once downloaded,
    preprocess into the layout above with the Pointcept or Mask3D scripts:
        - https://github.com/Pointcept/Pointcept (datasets/preprocessing/scannet)
        - https://github.com/JonasSchult/Mask3D (datasets/preprocessing)

    Parameters
    ----------
    root : str
        Path to the preprocessed dataset root.
    split : str or sequence of str
        Split name(s) inside ``root`` to load (e.g. ``"train"``, ``"val"``,
        ``["train", "val"]``).
    label_set : {"scannet20", "scannet200"}
        Which semantic label set to load. Selects ``segment20.npy`` or
        ``segment200.npy``.
    voxel_size : float, optional
        If set, voxel-downsample each scan to this resolution on the CPU.
    """

    NUM_CLASSES = {"scannet20": 20, "scannet200": 200}
    IGNORE_INDEX = -1

    def __init__(
        self,
        root: str = "./data/scannet_preprocessed",
        split: Union[str, Sequence[str]] = "train",
        label_set: Literal["scannet20", "scannet200"] = "scannet200",
        voxel_size: Optional[float] = None,
        transform: Optional[SampleTransform] = None,
    ):
        super().__init__()
        if label_set not in ("scannet20", "scannet200"):
            raise ValueError(f"label_set must be scannet20 or scannet200, got {label_set}")
        self.root = os.path.expanduser(root)
        self.split = split
        self.label_set = label_set
        self.voxel_size = voxel_size
        self.transform = transform
        self._segment_asset = "segment20" if label_set == "scannet20" else "segment200"
        self.class2id = np.array(
            VALID_CLASS_IDS_20 if label_set == "scannet20" else VALID_CLASS_IDS_200
        )
        self.scenes = self._collect_scenes()
        if not self.scenes:
            raise FileNotFoundError(
                f"No scenes found under {root!r} for split={split!r}. "
                "See class docstring for the expected layout and preprocessing."
            )

    def _collect_scenes(self):
        splits = [self.split] if isinstance(self.split, str) else list(self.split)
        scenes = []
        for s in splits:
            scenes.extend(sorted(glob.glob(os.path.join(self.root, s, "*"))))
        return scenes

    def __len__(self):
        return len(self.scenes)

    def _load_scene(self, scene_dir: str):
        out = {}
        for fname in os.listdir(scene_dir):
            if not fname.endswith(".npy"):
                continue
            out[fname[:-4]] = np.load(os.path.join(scene_dir, fname))
        return out

    def __getitem__(self, idx):
        scene_dir = self.scenes[idx]
        raw = self._load_scene(scene_dir)

        coord = raw["coord"].astype(np.float32)
        color = raw.get("color", np.zeros_like(coord)).astype(np.float32)
        normal = raw.get("normal", np.zeros_like(coord)).astype(np.float32)

        n = coord.shape[0]
        if self._segment_asset in raw:
            segment = raw[self._segment_asset].reshape(-1).astype(np.int32)
        else:
            segment = np.full(n, self.IGNORE_INDEX, dtype=np.int32)
        if "instance" in raw:
            instance = raw["instance"].reshape(-1).astype(np.int32)
        else:
            instance = np.full(n, self.IGNORE_INDEX, dtype=np.int32)

        out = {
            "name": os.path.basename(scene_dir),
            "coords": coord,
            "colors": color,
            "normals": normal,
            "segment": segment,
            "instance": instance,
        }

        if self.voxel_size is not None:
            int_coords, keep = voxel_downsample_np(coord, self.voxel_size)
            for k in ("coords", "colors", "normals", "segment", "instance"):
                out[k] = out[k][keep]
            out["coord_int"] = int_coords
        if self.transform is not None:
            out = self.transform(out)
        return out
